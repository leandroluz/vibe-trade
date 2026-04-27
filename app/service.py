from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.ai import AIInterpretation, build_ai_payload, interpret_with_openai
from app.analyzer import AnalysisResult, analyze_market
from app.config import Settings, load_settings
from app.data import (
    AnalysisHistoryError,
    CandleDataError,
    build_replay_window,
    load_candles_csv,
    save_candles_csv,
)
from app.indicators import add_indicators
from app.mt5_client import MT5Client, MT5ConnectionError
from app.stores import build_analysis_history_store


class AnalysisServiceError(Exception):
    """Raised when an analysis request cannot be fulfilled."""


@dataclass(frozen=True)
class AnalysisRequest:
    symbol: str
    timeframe: str
    candles_count: int
    profile: str
    data_file: str | None = None
    save_data: str | None = None
    log_file: str | None = None
    replay_step: int = 0
    ai_context_window: int = 10
    with_ai: bool = False
    openai_api_key: str | None = None


@dataclass(frozen=True)
class AnalysisExecution:
    analysis: AnalysisResult
    source: str
    candle_status: str
    change_message: str | None
    ai_payload: dict | None = None
    ai_interpretation: AIInterpretation | None = None
    replay_step: int | None = None


def get_available_symbols(
    *,
    query: str | None = None,
    settings: Settings | None = None,
) -> list[str]:
    active_settings = settings or load_settings()
    client = MT5Client(host=active_settings.mt5_host, port=active_settings.mt5_port)
    try:
        client.initialize()
        return client.list_symbols(query=query)
    except MT5ConnectionError as exc:
        raise AnalysisServiceError(str(exc)) from exc
    finally:
        client.shutdown()


def execute_analysis(
    request: AnalysisRequest,
    *,
    settings: Settings | None = None,
    previous_analysis: AnalysisResult | None = None,
    persist_log: bool = True,
) -> AnalysisExecution:
    active_settings = settings or load_settings()
    log_path = Path(request.log_file).expanduser() if request.log_file else None
    history_store = build_analysis_history_store(log_path)

    try:
        candles, source = _load_candles_source(request, active_settings)
        resolved_symbol = candles.attrs.get("resolved_symbol", request.symbol) or request.symbol
        enriched_candles = add_indicators(candles)
        analysis = analyze_market(
            resolved_symbol,
            request.timeframe.upper(),
            enriched_candles,
            profile=request.profile,
        )
    except (CandleDataError, MT5ConnectionError, AnalysisHistoryError) as exc:
        raise AnalysisServiceError(str(exc)) from exc
    except Exception as exc:
        raise AnalysisServiceError(f"Erro inesperado ao executar análise: {exc}") from exc

    candle_status = _describe_candle_status(analysis, previous_analysis)
    change_message = _build_change_message(analysis, previous_analysis)
    ai_payload = None
    ai_interpretation = None

    if request.with_ai:
        ai_payload = build_ai_payload(
            analysis=analysis,
            log_path=log_path,
            context_window=request.ai_context_window,
            candle_status=candle_status,
            change_message=change_message,
        )
        try:
            ai_interpretation = interpret_with_openai(
                payload=ai_payload,
                model=active_settings.openai_model,
                api_key=request.openai_api_key,
            )
        except Exception as exc:
            raise AnalysisServiceError(f"Falha na integração de IA: {exc}") from exc

    if persist_log and history_store is not None:
        history_store.append_analysis_run(
            analysis=analysis,
            request_metadata={
                "mode": "single-run",
                "symbol_requested": request.symbol,
                "timeframe_requested": request.timeframe.upper(),
                "profile_requested": request.profile,
                "candles_requested": request.candles_count,
                "watch_interval_seconds": 0,
                "data_file": request.data_file,
                "save_data": request.save_data,
                "replay_step": request.replay_step if request.data_file else None,
                "event_history": [],
            },
            source=source,
            candle_status=candle_status,
            change_message=change_message,
            ai_interpretation=ai_interpretation,
        )

    return AnalysisExecution(
        analysis=analysis,
        source=source,
        candle_status=candle_status,
        change_message=change_message,
        ai_payload=ai_payload,
        ai_interpretation=ai_interpretation,
        replay_step=request.replay_step if request.data_file else None,
    )


def _load_candles_source(
    request: AnalysisRequest,
    settings: Settings,
) -> tuple[object, str]:
    if request.data_file:
        loaded = load_candles_csv(request.data_file)
        if request.replay_step > 0 or len(loaded) > request.candles_count:
            candles = build_replay_window(
                loaded,
                count=request.candles_count,
                step=request.replay_step,
            )
            return candles, "csv-replay"
        return build_replay_window(loaded, count=request.candles_count, step=0), "csv-file"

    client = MT5Client(host=settings.mt5_host, port=settings.mt5_port)
    try:
        client.initialize()
        candles = client.get_candles(
            symbol=request.symbol,
            timeframe=request.timeframe,
            count=request.candles_count,
        )
        if request.save_data:
            save_candles_csv(candles, request.save_data)
        return candles, "mt5"
    finally:
        client.shutdown()


def _describe_candle_status(current: AnalysisResult, previous: AnalysisResult | None) -> str:
    if previous is None:
        return f"Candle atual: {current.last_candle_time}"
    if current.last_candle_time != previous.last_candle_time:
        return f"Candle novo detectado: {previous.last_candle_time} -> {current.last_candle_time}"
    return f"Sem candle novo. Último candle permanece em {current.last_candle_time}"


def _build_change_message(current: AnalysisResult, previous: AnalysisResult | None) -> str | None:
    if previous is None:
        return "Primeiro ciclo do monitoramento."

    previous_snapshot = (previous.setup, previous.setup_type, previous.summary)
    current_snapshot = (current.setup, current.setup_type, current.summary)
    if previous_snapshot == current_snapshot:
        return "Sem mudança de setup desde o ciclo anterior."

    became_actionable = previous.setup == "sem trade" and current.setup != "sem trade"
    setup_changed = previous.setup != current.setup
    type_changed = previous.setup_type != current.setup_type

    fragments = []
    if setup_changed:
        fragments.append(f"setup: {previous.setup} -> {current.setup}")
    if type_changed:
        fragments.append(f"tipo: {previous.setup_type} -> {current.setup_type}")
    if previous.summary != current.summary:
        fragments.append(f"leitura: {current.summary}")

    prefix = "ALERTA: novo sinal." if became_actionable else "Mudança detectada."
    return f"{prefix} {' | '.join(fragments)}"
