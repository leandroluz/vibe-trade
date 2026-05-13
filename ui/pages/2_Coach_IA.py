from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_settings
from app.mt5_client import MT5ConnectionError
from app.services.jarvis_ai import ask_jarvis_coach
from app.services.mt5_service import ensure_mt5, get_account_snapshot, get_positions_snapshot
from app.services.position_manager import analyze_open_positions, build_position_management_summary
from app.services.risk import calculate_position_risk
from app.services.scanner import scan_market


DEFAULT_SYMBOLS = [
    "AUDUSD.pro",
    "USDJPY.pro",
    "EURUSD.pro",
    "GBPUSD.pro",
    "USDCAD.pro",
    "USDCHF.pro",
    "AUDJPY.pro",
    "GBPJPY.pro",
    "XAUUSD.pro",
]
COACH_LOG_PATH = ROOT / "logs" / "coach_ai_history.jsonl"
POSITION_MANAGEMENT_LOG_PATH = ROOT / "logs" / "position_management_history.jsonl"
TIMEFRAME_OPTIONS = ["M5", "M15", "M30", "H1"]
RISK_PER_TRADE_OPTIONS = [0.25, 0.50, 0.75, 1.00]
TOTAL_RISK_OPTIONS = [1.0, 1.5, 2.0, 3.0]
AUTO_INTERVAL_OPTIONS = {
    "15 segundos": 15_000,
    "30 segundos": 30_000,
    "60 segundos": 60_000,
    "5 minutos": 300_000,
}
AUTO_INTERVAL_REVERSE = {value: key for key, value in AUTO_INTERVAL_OPTIONS.items()}


st.set_page_config(
    page_title="Coach IA",
    page_icon="J",
    layout="wide",
)


def main() -> None:
    settings = load_settings()
    defaults = _load_sidebar_defaults(settings)
    st.title("Coach IA — O que fazer agora?")
    st.caption("Leitura operacional da conta MT5 com foco em risco, contexto técnico e orientação conservadora.")

    with st.sidebar:
        st.header("Parâmetros")
        symbols = st.multiselect(
            "Ativos para escanear",
            options=DEFAULT_SYMBOLS,
            default=defaults["symbols"],
            key="coach_symbols",
        )
        timeframe = st.selectbox(
            "Timeframe principal",
            options=TIMEFRAME_OPTIONS,
            index=TIMEFRAME_OPTIONS.index(defaults["timeframe"])
            if defaults["timeframe"] in TIMEFRAME_OPTIONS
            else 0,
            key="coach_timeframe",
        )
        account_size = st.number_input(
            "Tamanho da conta Alpha",
            min_value=1000.0,
            value=defaults["account_size"],
            step=1000.0,
            key="coach_account_size",
        )
        risk_per_trade_pct = st.selectbox(
            "Risco por trade",
            options=RISK_PER_TRADE_OPTIONS,
            index=RISK_PER_TRADE_OPTIONS.index(defaults["risk_per_trade_pct"]),
            format_func=lambda value: f"{value:.2f}%",
            key="coach_risk_per_trade",
        )
        max_total_risk_pct = st.selectbox(
            "Risco máximo total aberto",
            options=TOTAL_RISK_OPTIONS,
            index=TOTAL_RISK_OPTIONS.index(defaults["max_total_risk_pct"]),
            format_func=lambda value: f"{value:.1f}%",
            key="coach_max_total_risk",
        )
        auto_evaluate = st.checkbox("Autoavaliar", value=defaults["auto_evaluate"], key="coach_auto_evaluate")
        auto_interval_label = st.selectbox(
            "Intervalo",
            options=list(AUTO_INTERVAL_OPTIONS.keys()),
            index=list(AUTO_INTERVAL_OPTIONS.keys()).index(defaults["auto_interval_label"]),
            disabled=not auto_evaluate,
            key="coach_auto_interval",
        )
        manual_model = st.text_input(
            "Modelo manual",
            value=defaults["manual_model"],
            help="Modelo usado ao clicar manualmente em `Analisar agora`.",
            key="coach_manual_model",
        )
        auto_model = st.text_input(
            "Modelo auto",
            value=defaults["auto_model"],
            help="Modelo usado nas reavaliações automáticas.",
            key="coach_auto_model",
        )
        analyze_now = st.button("Analisar agora", type="primary", use_container_width=True)

    _persist_sidebar_state(
        symbols=symbols,
        timeframe=timeframe,
        account_size=account_size,
        risk_per_trade_pct=risk_per_trade_pct,
        max_total_risk_pct=max_total_risk_pct,
        auto_evaluate=auto_evaluate,
        auto_interval_label=auto_interval_label,
        manual_model=manual_model,
        auto_model=auto_model,
    )

    if auto_evaluate:
        _trigger_autorefresh(AUTO_INTERVAL_OPTIONS[auto_interval_label])
        analyze_now = True

    if not analyze_now:
        st.info("Selecione os parâmetros na barra lateral e clique em `Analisar agora`.")
        return

    if not symbols:
        st.warning("Selecione ao menos um ativo para o scanner.")
        return

    try:
        result = run_coach_analysis(
            symbols=symbols,
            timeframe=timeframe,
            account_size=account_size,
            risk_per_trade_pct=risk_per_trade_pct,
            max_total_risk_pct=max_total_risk_pct,
            model_name=auto_model if auto_evaluate else manual_model,
        )
    except MT5ConnectionError as exc:
        st.error(f"Não foi possível conectar ao MT5: {exc}")
        return
    except Exception as exc:
        st.error(f"Falha ao executar a análise do Coach IA: {exc}")
        return

    render_result(result)


def run_coach_analysis(
    *,
    symbols: list[str],
    timeframe: str,
    account_size: float,
    risk_per_trade_pct: float,
    max_total_risk_pct: float,
    model_name: str,
) -> dict[str, Any]:
    log_error = None
    management_log_error = None
    client = ensure_mt5()
    try:
        account = get_account_snapshot(client)
        positions = get_positions_snapshot(client)
        enriched_positions = [calculate_position_risk(client, position) for position in positions]
        positions_metrics = analyze_open_positions(client, positions)
        position_management = build_position_management_summary(positions_metrics)
        open_risk_usd = sum(float(item.get("risk_to_sl") or 0.0) for item in enriched_positions)
        risk_per_trade_usd = account_size * (risk_per_trade_pct / 100.0)
        max_total_risk_usd = account_size * (max_total_risk_pct / 100.0)
        scan_symbols = list(
            dict.fromkeys(
                [*symbols, *[str(item.get("symbol") or "") for item in positions if item.get("symbol")]]
            )
        )
        market_scan = scan_market(
            client,
            symbols=scan_symbols,
            risk_usd=risk_per_trade_usd,
            timeframe=timeframe,
        )
    finally:
        client.shutdown()

    warnings = _build_risk_warnings(
        positions=enriched_positions,
        open_risk_usd=open_risk_usd,
        max_total_risk_usd=max_total_risk_usd,
    )
    warnings.extend(position_management.get("risk_alerts") or [])

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "account": {
            **account,
            "reference_account_size": account_size,
        },
        "risk_config": {
            "risk_per_trade_pct": risk_per_trade_pct,
            "risk_per_trade_usd": risk_per_trade_usd,
            "max_total_open_risk_pct": max_total_risk_pct,
            "max_total_open_risk_usd": max_total_risk_usd,
            "current_open_risk_usd": open_risk_usd,
        },
        "open_positions": enriched_positions,
        "position_management": position_management,
        "market_scan": market_scan,
        "_model_override": model_name,
        "instruction": (
            "Avalie a conta, risco aberto, scanner técnico e a gestão ativa das posições. "
            "Responda qual a melhor ação agora entre open_trade, manage_positions, wait e avoid_trading. "
            "Inclua um plano operacional conservador, sem executar nada automaticamente."
        ),
    }

    ai_error = None
    ai_response: dict[str, Any] | None = None
    ai_raw = None
    try:
        ai_result = ask_jarvis_coach(payload)
        ai_response = ai_result.parsed
        ai_raw = {
            "parsed": ai_result.parsed,
            "raw_text": ai_result.raw_text,
            "model": ai_result.model,
            "response_id": ai_result.response_id,
            "parse_error": ai_result.parse_error,
        }
        if ai_result.parse_error:
            ai_error = (
                "A IA respondeu, mas o parsing estruturado exigiu fallback. "
                "A resposta bruta foi preservada no log."
            )
    except Exception as exc:
        ai_error = str(exc)
        ai_raw = {"parsed": None, "raw_text": "", "error": ai_error}

    try:
        _append_coach_log(payload=payload, ai_response=ai_raw)
    except Exception as exc:
        log_error = str(exc)

    try:
        _append_position_management_log(
            account=account,
            positions_metrics=position_management["positions_metrics"],
            ai_action_plan=(ai_response or {}).get("action_plan"),
        )
    except Exception as exc:
        management_log_error = str(exc)

    return {
        "payload": payload,
        "account": account,
        "positions": enriched_positions,
        "positions_management": position_management["positions_metrics"],
        "position_management_summary": position_management,
        "market_scan": market_scan,
        "open_risk_usd": open_risk_usd,
        "risk_per_trade_usd": risk_per_trade_usd,
        "max_total_risk_usd": max_total_risk_usd,
        "warnings": warnings,
        "ai_response": ai_response,
        "ai_error": ai_error,
        "log_error": log_error,
        "management_log_error": management_log_error,
        "timeframe": timeframe,
        "model_name": model_name,
    }


def render_result(result: dict[str, Any]) -> None:
    account = result["account"]
    positions = result["positions"]
    positions_management = result["positions_management"]
    position_management_summary = result["position_management_summary"]
    scan_rows = result["market_scan"]
    ai_response = result["ai_response"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Balance", _fmt_money(account.get("balance")))
    col2.metric("Equity", _fmt_money(account.get("equity")))
    col3.metric("Margem Livre", _fmt_money(account.get("free_margin")))
    col4.metric("Risco Aberto", _fmt_money(result["open_risk_usd"]))
    col5.metric("Risco por Trade", _fmt_money(result["risk_per_trade_usd"]))

    for warning in result["warnings"]:
        st.warning(warning)

    if result["ai_error"]:
        st.info(f"Status da IA: {result['ai_error']}")
    if result.get("log_error"):
        st.warning(f"Falha ao gravar log do Coach IA: {result['log_error']}")
    if result.get("management_log_error"):
        st.warning(f"Falha ao gravar log de gestão de posições: {result['management_log_error']}")
    st.caption(f"Modelo em uso nesta análise: `{result['model_name']}`")

    st.subheader("Posições abertas")
    if positions:
        positions_df = pd.DataFrame(positions)
        st.dataframe(
            positions_df[
                [
                    "ticket",
                    "symbol",
                    "type",
                    "volume",
                    "entry",
                    "sl",
                    "tp",
                    "price_current",
                    "profit",
                    "swap",
                    "risk_to_sl",
                    "risk_status",
                    "comment",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.info("Nenhuma posição aberta na conta neste momento.")

    st.subheader("Gestão ativa das posições")
    if positions_management:
        management_df = pd.DataFrame(positions_management)
        st.dataframe(
            management_df[
                [
                    "symbol",
                    "type",
                    "volume",
                    "entry",
                    "current_price",
                    "sl",
                    "tp",
                    "profit",
                    "risk_usd",
                    "current_r",
                    "position_status",
                    "suggested_action",
                    "severity",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.info("Sem posições abertas para gestão neste momento.")

    for symbol in position_management_summary.get("positions_without_sl") or []:
        st.error(f"{symbol}: posição sem SL. Ajuste o risco manualmente ou avalie encerrar.")

    if result["max_total_risk_usd"] > 0 and result["open_risk_usd"] > result["max_total_risk_usd"]:
        st.warning("O risco aberto está acima do limite configurado.")

    st.subheader("Scanner técnico")
    scan_df = pd.DataFrame(scan_rows)
    if not scan_df.empty:
        visible_columns = [
            column
            for column in [
                "symbol",
                "setup",
                "direction",
                "score",
                "close",
                "spread",
                "ema20",
                "ema50",
                "rsi14",
                "atr14",
                "entry",
                "sl",
                "tp1",
                "tp2",
                "volume",
                "reason",
                "status",
            ]
            if column in scan_df.columns
        ]
        st.dataframe(scan_df[visible_columns], use_container_width=True)
    else:
        st.info("Sem resultados do scanner.")

    st.subheader("Resposta da IA")
    if ai_response is None:
        st.warning("Não foi possível obter uma resposta estruturada da IA nesta execução.")
    else:
        decision = ai_response.get("decision", "indefinido")
        if decision == "open_trade":
            st.success(f"Decisão da IA: {decision}")
        elif decision in {"manage_positions", "wait"}:
            st.info(f"Decisão da IA: {decision}")
        else:
            st.warning(f"Decisão da IA: {decision}")

        st.write(ai_response.get("account_reading", ""))
        st.write(ai_response.get("risk_reading", ""))
        st.write(ai_response.get("best_action_now", ""))

        warnings = ai_response.get("warnings") or []
        for item in warnings:
            st.warning(item)

        opportunities = ai_response.get("opportunities") or []
        st.markdown("**Oportunidades aprovadas pela IA**")
        if opportunities:
            st.dataframe(pd.DataFrame(opportunities), use_container_width=True)
        else:
            st.info("Nenhuma oportunidade aprovada pela IA.")

        ai_positions_management = ai_response.get("positions_management") or []
        st.markdown("**Sugestões de gestão das posições abertas**")
        if ai_positions_management:
            st.dataframe(pd.DataFrame(ai_positions_management), use_container_width=True)
        else:
            st.info("Sem sugestões específicas de gestão nesta rodada.")

        action_plan = ai_response.get("action_plan") or {}
        st.subheader("Plano de ação objetivo")
        st.write(action_plan.get("now", ""))

        do_not_do = action_plan.get("do_not_do") or []
        if do_not_do:
            st.markdown("**O que não fazer**")
            for item in do_not_do:
                st.write(f"- {item}")

        position_triggers = action_plan.get("position_triggers") or []
        if position_triggers:
            st.markdown("**Gatilhos por posição**")
            st.dataframe(pd.DataFrame(position_triggers), use_container_width=True)

        next_review = action_plan.get("next_review")
        if next_review:
            st.info(f"Próxima reavaliação: {next_review}")

    with st.expander("Payload enviado para a IA", expanded=False):
        st.json(result["payload"])


def _build_risk_warnings(
    *,
    positions: list[dict[str, Any]],
    open_risk_usd: float,
    max_total_risk_usd: float,
) -> list[str]:
    warnings: list[str] = []

    if any(item.get("risk_status") == "missing_sl" for item in positions):
        warnings.append("Há posições abertas sem stop loss. Isso exige atenção imediata.")

    if max_total_risk_usd > 0 and open_risk_usd > max_total_risk_usd:
        warnings.append(
            f"O risco aberto ({_fmt_money(open_risk_usd)}) excede o limite configurado ({_fmt_money(max_total_risk_usd)})."
        )

    if any(item.get("risk_status") == "missing_sl" for item in positions):
        warnings.append("O risco aberto exibido não inclui posições sem SL definido.")

    usd_exposure = _summarize_usd_exposure(positions)
    if len(usd_exposure) >= 2:
        warnings.append(
            "Há correlação relevante em posições ligadas ao dólar: "
            + ", ".join(sorted(usd_exposure))
            + "."
        )

    return warnings


def _summarize_usd_exposure(positions: list[dict[str, Any]]) -> set[str]:
    exposure: set[str] = set()
    for position in positions:
        symbol = str(position.get("symbol") or "").upper()
        if "USD" not in symbol:
            continue
        direction = str(position.get("type") or "").lower()
        if direction not in {"buy", "sell"}:
            continue
        if symbol.startswith("USD"):
            exposure.add(f"{symbol}:{'usd_long' if direction == 'buy' else 'usd_short'}")
        elif symbol.endswith("USD"):
            exposure.add(f"{symbol}:{'usd_short' if direction == 'buy' else 'usd_long'}")
    return exposure


def _append_coach_log(*, payload: dict[str, Any], ai_response: dict[str, Any] | None) -> None:
    COACH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COACH_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "logged_at": datetime.now().isoformat(timespec="seconds"),
                    "payload": payload,
                    "ai_response": ai_response,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _append_position_management_log(
    *,
    account: dict[str, Any],
    positions_metrics: list[dict[str, Any]],
    ai_action_plan: dict[str, Any] | None,
) -> None:
    POSITION_MANAGEMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with POSITION_MANAGEMENT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "account": account,
                    "positions_metrics": positions_metrics,
                    "ai_action_plan": ai_action_plan,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _trigger_autorefresh(interval_ms: int) -> None:
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=interval_ms, key="coach_ia_autorefresh")
    except Exception:
        components.html(
            f"""
            <script>
            setTimeout(function() {{
                window.parent.location.reload();
            }}, {interval_ms});
            </script>
            """,
            height=0,
        )
        st.sidebar.caption(
            "Autoavaliação via refresh simples. Para um comportamento mais suave, instale `streamlit-autorefresh`."
        )


def _load_sidebar_defaults(settings) -> dict[str, Any]:
    params = st.query_params
    raw_symbols = params.get("symbols", "")
    parsed_symbols = [item for item in raw_symbols.split(",") if item] if raw_symbols else DEFAULT_SYMBOLS
    valid_symbols = [item for item in parsed_symbols if item in DEFAULT_SYMBOLS] or DEFAULT_SYMBOLS

    timeframe = params.get("timeframe", settings.default_timeframe)
    if timeframe not in TIMEFRAME_OPTIONS:
        timeframe = settings.default_timeframe if settings.default_timeframe in TIMEFRAME_OPTIONS else "M5"

    risk_per_trade_pct = _coerce_option(params.get("risk_per_trade_pct"), RISK_PER_TRADE_OPTIONS, 0.50)
    max_total_risk_pct = _coerce_option(params.get("max_total_risk_pct"), TOTAL_RISK_OPTIONS, 1.5)
    auto_interval_ms = _coerce_option(params.get("auto_interval_ms"), list(AUTO_INTERVAL_REVERSE.keys()), 60_000)

    return {
        "symbols": valid_symbols,
        "timeframe": timeframe,
        "account_size": _coerce_float(params.get("account_size"), 100000.0),
        "risk_per_trade_pct": risk_per_trade_pct,
        "max_total_risk_pct": max_total_risk_pct,
        "auto_evaluate": str(params.get("auto_evaluate", "false")).lower() == "true",
        "auto_interval_label": AUTO_INTERVAL_REVERSE.get(auto_interval_ms, "60 segundos"),
        "manual_model": str(params.get("manual_model", settings.openai_manual_model)).strip() or settings.openai_manual_model,
        "auto_model": str(params.get("auto_model", settings.openai_auto_model)).strip() or settings.openai_auto_model,
    }


def _persist_sidebar_state(
    *,
    symbols: list[str],
    timeframe: str,
    account_size: float,
    risk_per_trade_pct: float,
    max_total_risk_pct: float,
    auto_evaluate: bool,
    auto_interval_label: str,
    manual_model: str,
    auto_model: str,
) -> None:
    st.query_params.update(
        {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "account_size": f"{float(account_size):.2f}",
            "risk_per_trade_pct": str(risk_per_trade_pct),
            "max_total_risk_pct": str(max_total_risk_pct),
            "auto_evaluate": "true" if auto_evaluate else "false",
            "auto_interval_ms": str(AUTO_INTERVAL_OPTIONS[auto_interval_label]),
            "manual_model": manual_model,
            "auto_model": auto_model,
        }
    )


def _coerce_float(raw_value: Any, default: float) -> float:
    try:
        return float(raw_value)
    except Exception:
        return default


def _coerce_option(raw_value: Any, options: list[Any], default: Any) -> Any:
    try:
        if raw_value is None:
            return default
        if isinstance(default, float):
            value = float(raw_value)
            return value if value in options else default
        if isinstance(default, int):
            value = int(raw_value)
            return value if value in options else default
        return raw_value if raw_value in options else default
    except Exception:
        return default


def _fmt_money(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return "-"


if __name__ == "__main__":
    main()
