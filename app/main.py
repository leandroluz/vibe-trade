from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

from app.ai import (
    AIIntegrationError,
    AIInterpretation,
    build_ai_payload,
    format_ai_interpretation,
    interpret_with_openai,
)
from app.analyzer import AnalysisResult, analyze_market
from data.config import load_settings
from app.data import (
    AnalysisHistoryError,
    CandleDataError,
    build_replay_window,
    load_candles_csv,
    save_candles_csv,
)
from app.indicators import add_indicators
from app.mt5_client import MT5Client, MT5ConnectionError


def build_parser() -> argparse.ArgumentParser:
    settings = load_settings()

    parser = argparse.ArgumentParser(
        description="Copiloto técnico V0 para leitura de candles via MetaTrader 5."
    )
    parser.add_argument("--symbol", default=settings.default_symbol, help="Ativo a analisar.")
    parser.add_argument(
        "--timeframe",
        default=settings.default_timeframe,
        help="Timeframe do MT5, por exemplo: M5, M15, H1, D1.",
    )
    parser.add_argument(
        "--candles",
        type=int,
        default=settings.candles_count,
        help="Quantidade de candles para análise.",
    )
    parser.add_argument(
        "--profile",
        choices=["conservador", "equilibrado", "agressivo"],
        default=settings.analysis_profile,
        help="Perfil de leitura do setup.",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        help="Reexecuta a análise a cada N segundos. Use 0 para executar uma vez.",
    )
    parser.add_argument(
        "--data-file",
        help="Lê candles de um arquivo CSV local em vez do MT5.",
    )
    parser.add_argument(
        "--save-data",
        help="Salva os candles coletados em um arquivo CSV.",
    )
    parser.add_argument(
        "--log-file",
        default=settings.analysis_log_path,
        help="Arquivo JSONL para persistir o histórico de análises.",
    )
    parser.add_argument(
        "--print-ai-payload",
        action="store_true",
        help="Imprime o payload estruturado para futura integracao com IA.",
    )
    parser.add_argument(
        "--ai-context-window",
        type=int,
        default=10,
        help="Quantidade de registros recentes do JSONL para incluir no payload da IA.",
    )
    parser.add_argument(
        "--with-ai",
        action="store_true",
        help="Solicita uma interpretacao adicional via OpenAI usando o payload estruturado.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.candles < 50:
        parser.error("--candles deve ser pelo menos 50 para suportar a EMA 50.")
    if args.watch < 0:
        parser.error("--watch não pode ser negativo.")
    if args.ai_context_window < 0:
        parser.error("--ai-context-window não pode ser negativo.")
    if args.watch > 0 and args.with_ai:
        parser.error("--with-ai nesta versao suporta apenas execucao unica.")
    if args.watch > 0 and args.data_file and not Path(args.data_file).exists():
        parser.error("--data-file precisa apontar para um arquivo existente.")

    settings = load_settings()
    log_path = Path(args.log_file).expanduser() if args.log_file else None

    if args.watch == 0:
        return _run_once(args, settings)

    replay_candles = None
    if args.data_file:
        try:
            replay_candles = load_candles_csv(args.data_file)
        except CandleDataError as exc:
            print(f"Erro de dados: {exc}")
            return 1

    previous_analysis: AnalysisResult | None = None
    event_history: list[str] = []
    replay_step = 0
    try:
        while True:
            result = _run_once(args, settings, replay_candles=replay_candles, replay_step=replay_step)
            if isinstance(result, int):
                return result
            candle_status = _describe_candle_status(result, previous_analysis)
            change_message = _build_change_message(result, previous_analysis)
            _render_watch_cycle(result, candle_status, change_message, event_history)
            _append_jsonl_log(
                log_path=log_path,
                analysis=result,
                args=args,
                source="csv-replay" if replay_candles is not None else ("csv-file" if args.data_file else "mt5"),
                candle_status=candle_status,
                change_message=change_message,
                replay_step=replay_step if replay_candles is not None else None,
                event_history=event_history,
            )
            previous_analysis = result
            if replay_candles is not None:
                replay_step += 1
                max_step = len(replay_candles) - args.candles
                if replay_step > max_step:
                    print("\nReplay concluído: fim do arquivo de candles.")
                    return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nMonitoramento interrompido pelo usuário.")
        return 0


def _run_once(
    args: argparse.Namespace,
    settings,
    replay_candles=None,
    replay_step: int = 0,
) -> AnalysisResult | int:
    candle_status = None
    change_message = None
    try:
        candles = _load_candles_source(args, settings, replay_candles=replay_candles, replay_step=replay_step)
        resolved_symbol = candles.attrs.get("resolved_symbol", args.symbol) or args.symbol
        enriched_candles = add_indicators(candles)
        analysis = analyze_market(
            resolved_symbol,
            args.timeframe.upper(),
            enriched_candles,
            profile=args.profile,
        )
    except CandleDataError as exc:
        print(f"Erro de dados: {exc}")
        return 1
    except AnalysisHistoryError as exc:
        print(f"Erro de histórico: {exc}")
        return 1
    except MT5ConnectionError as exc:
        print(f"Erro de conexão MT5: {exc}")
        return 1
    except Exception as exc:
        print(f"Erro inesperado: {exc}")
        return 1

    print_output = format_analysis(analysis)
    if args.watch == 0:
        print(print_output)
        candle_status = f"Execução única para candle {analysis.last_candle_time}"
        _append_jsonl_log(
            log_path=Path(args.log_file).expanduser() if args.log_file else None,
            analysis=analysis,
            args=args,
            source="csv-file" if args.data_file else "mt5",
            candle_status=candle_status,
            change_message=None,
            replay_step=replay_step if replay_candles is not None else None,
            event_history=[],
        )
        payload = None
        if args.print_ai_payload or args.with_ai:
            payload = build_ai_payload(
                analysis=analysis,
                log_path=Path(args.log_file).expanduser() if args.log_file else None,
                context_window=args.ai_context_window,
                candle_status=candle_status,
                change_message=change_message,
            )
        if args.print_ai_payload and payload is not None:
            print(json.dumps(payload, ensure_ascii=True, indent=2))
        if args.with_ai and payload is not None:
            try:
                ai_result = _run_ai_interpretation(payload=payload, settings=settings)
            except AIIntegrationError as exc:
                print(f"Erro de IA: {exc}")
                return 1
            print(format_ai_interpretation(ai_result))
        return 0

    return analysis


def format_analysis(analysis: AnalysisResult) -> str:
    return "\n".join(
        [
            "=" * 50,
            "VIBE-TRADE | RESUMO TÉCNICO",
            "=" * 50,
            f"Ativo: {analysis.symbol}",
            f"Timeframe: {analysis.timeframe}",
            f"Perfil: {analysis.profile}",
            f"Último candle: {analysis.last_candle_time}",
            f"Candles analisados: {analysis.candles_analyzed}",
            f"Preço atual: {_format_price_value(analysis.current_price)}",
            f"EMA 9: {_format_price_value(analysis.ema_9)}",
            f"EMA 20: {_format_price_value(analysis.ema_20)}",
            f"EMA 50: {_format_price_value(analysis.ema_50)}",
            f"Tendência: {analysis.trend}",
            f"RSI 14: {analysis.rsi:.2f}",
            f"ATR 14: {_format_range_value(analysis.atr, analysis.current_price)}",
            f"ATR 14 (%): {analysis.atr_percent:.3f}%",
            f"Máxima últimos 20 candles: {_format_price_value(analysis.recent_high)}",
            f"Mínima últimos 20 candles: {_format_price_value(analysis.recent_low)}",
            f"Faixa últimos 20 candles: {_format_range_value(analysis.range_20, analysis.current_price)}",
            f"Distância da EMA 20: {_format_signed_range_value(analysis.distance_to_ema_20, analysis.current_price)}",
            f"Distância da EMA 50: {_format_signed_range_value(analysis.distance_to_ema_50, analysis.current_price)}",
            f"Viés: {analysis.bias}",
            f"Setup: {analysis.setup}",
            f"Tipo de setup: {analysis.setup_type}",
            f"Leitura: {analysis.summary}",
            "=" * 50,
        ]
    )


def _render_watch_cycle(
    current: AnalysisResult,
    candle_status: str,
    change_message: str | None,
    event_history: list[str],
) -> None:
    print("\033[2J\033[H", end="")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")

    status_line = f"Monitorando {current.symbol} | {current.timeframe} | perfil {current.profile}"
    print(status_line)
    print(candle_status)

    if change_message:
        print(change_message)
        _append_event_history(event_history, change_message)

    if event_history:
        print("Histórico recente:")
        for item in event_history:
            print(f"- {item}")

    print(format_analysis(current))


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


def _describe_candle_status(current: AnalysisResult, previous: AnalysisResult | None) -> str:
    if previous is None:
        return f"Candle atual: {current.last_candle_time}"
    if current.last_candle_time != previous.last_candle_time:
        return f"Candle novo detectado: {previous.last_candle_time} -> {current.last_candle_time}"
    return f"Sem candle novo. Último candle permanece em {current.last_candle_time}"


def _append_event_history(event_history: list[str], message: str, limit: int = 5) -> None:
    stamped_message = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    event_history.append(stamped_message)
    if len(event_history) > limit:
        del event_history[:-limit]


def _append_jsonl_log(
    *,
    log_path: Path | None,
    analysis: AnalysisResult,
    args: argparse.Namespace,
    source: str,
    candle_status: str,
    change_message: str | None,
    replay_step: int | None,
    event_history: list[str],
) -> None:
    if log_path is None:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "mode": "watch" if args.watch > 0 else "single-run",
        "symbol_requested": args.symbol,
        "timeframe_requested": args.timeframe.upper(),
        "profile_requested": args.profile,
        "candles_requested": args.candles,
        "watch_interval_seconds": args.watch,
        "data_file": args.data_file,
        "save_data": args.save_data,
        "replay_step": replay_step,
        "candle_status": candle_status,
        "change_message": change_message,
        "event_history": list(event_history),
        "analysis": asdict(analysis),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _run_ai_interpretation(*, payload: dict, settings) -> AIInterpretation:
    try:
        return interpret_with_openai(payload=payload, model=settings.openai_model)
    except AIIntegrationError as exc:
        raise AIIntegrationError(f"Falha na integracao de IA: {exc}") from exc


def _load_candles_source(
    args: argparse.Namespace,
    settings,
    replay_candles=None,
    replay_step: int = 0,
):
    if replay_candles is not None:
        return build_replay_window(replay_candles, count=args.candles, step=replay_step)

    if args.data_file:
        loaded = load_candles_csv(args.data_file)
        return build_replay_window(loaded, count=args.candles, step=0)

    client = MT5Client(host=settings.mt5_host, port=settings.mt5_port)
    try:
        client.initialize()
        candles = client.get_candles(
            symbol=args.symbol,
            timeframe=args.timeframe,
            count=args.candles,
        )
        if args.save_data:
            save_candles_csv(candles, args.save_data)
        return candles
    finally:
        client.shutdown()


def _format_price_value(value: float) -> str:
    decimals = _infer_price_decimals(value)
    return f"{value:.{decimals}f}"


def _format_range_value(value: float, reference_price: float) -> str:
    decimals = max(_infer_price_decimals(reference_price), 4)
    return f"{value:.{decimals}f}"


def _format_signed_range_value(value: float, reference_price: float) -> str:
    decimals = max(_infer_price_decimals(reference_price), 4)
    return f"{value:+.{decimals}f}"


def _infer_price_decimals(value: float) -> int:
    absolute = abs(value)
    if absolute >= 1000:
        return 2
    if absolute >= 100:
        return 2
    if absolute >= 10:
        return 3
    if absolute >= 1:
        return 5
    if absolute >= 0.1:
        return 5
    return 6


if __name__ == "__main__":
    raise SystemExit(main())
