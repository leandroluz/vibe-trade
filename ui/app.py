from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ai import format_ai_interpretation
from app.config import load_settings
from app.data import CandleDataError, load_candles_csv
from app.service import (
    AnalysisRequest,
    AnalysisServiceError,
    execute_analysis,
    get_available_symbols,
)
from app.stores import build_analysis_history_store


st.set_page_config(
    page_title="Jarvis Trader",
    page_icon="J",
    layout="wide",
)


def main() -> None:
    settings = load_settings()
    st.title("Jarvis Trader")
    st.caption("Interface funcional inicial em Streamlit sobre o motor Python do projeto.")

    with st.sidebar:
        st.header("Configuração")
        source_mode = st.radio(
            "Fonte de dados",
            options=["MT5 ao vivo", "Replay por CSV"],
            index=0,
        )
        symbol = settings.default_symbol
        if source_mode == "MT5 ao vivo":
            symbol = _render_mt5_symbol_selector(settings=settings)
        else:
            symbol = st.text_input("Ativo", value=settings.default_symbol)
        timeframe = st.text_input("Timeframe", value=settings.default_timeframe)
        profile = st.selectbox(
            "Perfil",
            options=["conservador", "equilibrado", "agressivo"],
            index=["conservador", "equilibrado", "agressivo"].index(settings.analysis_profile),
        )
        candles_count = st.number_input(
            "Candles analisados",
            min_value=50,
            max_value=5000,
            value=settings.candles_count,
            step=50,
        )
        log_file = st.text_input(
            "Histórico JSONL",
            value=settings.analysis_log_path or "logs/analysis_history.jsonl",
        )
        with_ai = st.checkbox("Interpretar com IA", value=False)
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value="",
            type="password",
            disabled=not with_ai,
            help="Usada apenas nesta sessão da interface. Não é salva no projeto.",
        )
        ai_context_window = st.number_input(
            "Janela de contexto da IA",
            min_value=0,
            max_value=100,
            value=10,
            step=1,
            disabled=not with_ai,
        )

        data_file = None
        save_data = None
        replay_step = 0
        if source_mode == "Replay por CSV":
            data_file = st.text_input("Arquivo CSV", value="data/eurusd_m5.csv")
            replay_step = _render_replay_controls(data_file=data_file, candles_count=int(candles_count))
        else:
            auto_save_snapshot = st.checkbox("Salvar snapshot CSV automaticamente", value=False)
            if auto_save_snapshot:
                save_data = _default_snapshot_path(symbol=symbol, timeframe=timeframe)
                st.caption(f"Snapshot será salvo em `{save_data}`")

        run_clicked = st.button("Rodar análise", type="primary", use_container_width=True)

    history_col, main_col = st.columns([1, 2], gap="large")

    with history_col:
        st.subheader("Histórico recente")
        _render_history_panel(
            log_file=log_file,
            symbol=symbol,
            timeframe=timeframe,
            profile=profile,
        )

    with main_col:
        if run_clicked:
            request = AnalysisRequest(
                symbol=symbol,
                timeframe=timeframe,
                candles_count=int(candles_count),
                profile=profile,
                data_file=data_file or None,
                save_data=save_data or None,
                log_file=log_file or None,
                replay_step=replay_step,
                ai_context_window=int(ai_context_window),
                with_ai=with_ai,
                openai_api_key=openai_api_key or None,
            )
            _render_execution(request)
        else:
            st.info("Preencha os parâmetros na barra lateral e clique em `Rodar análise`.")


def _render_replay_controls(*, data_file: str, candles_count: int) -> int:
    try:
        candles = load_candles_csv(data_file)
    except CandleDataError as exc:
        st.warning(str(exc))
        return 0

    max_step = max(len(candles) - candles_count, 0)
    st.caption(
        f"Arquivo com {len(candles)} candles. "
        f"Último símbolo detectado: {candles.attrs.get('resolved_symbol', 'desconhecido')}"
    )
    if max_step == 0:
        st.caption("Sem passos adicionais para replay com a janela atual.")
        return 0

    return int(
        st.slider(
            "Passo do replay",
            min_value=0,
            max_value=max_step,
            value=max_step,
            help="0 usa o primeiro bloco do arquivo; o máximo usa o trecho mais recente.",
        )
    )


def _render_mt5_symbol_selector(*, settings) -> str:
    search_query = st.text_input(
        "Filtrar ativos do MT5",
        value="",
        help="Digite parte do nome para filtrar os símbolos disponíveis na conta.",
    )
    try:
        symbols = get_available_symbols(query=search_query or None, settings=settings)
    except AnalysisServiceError as exc:
        st.warning(f"Não foi possível carregar símbolos do MT5: {exc}")
        return st.text_input("Ativo", value=settings.default_symbol)

    if not symbols:
        st.warning("Nenhum símbolo encontrado com esse filtro.")
        return st.text_input("Ativo", value=settings.default_symbol)

    default_symbol = settings.default_symbol
    if default_symbol not in symbols:
        default_symbol = symbols[0]

    default_index = symbols.index(default_symbol)
    return st.selectbox(
        "Ativo",
        options=symbols,
        index=default_index,
        help="Lista carregada do MetaTrader 5 com os símbolos disponíveis na conta.",
    )


def _render_history_panel(
    *,
    log_file: str,
    symbol: str,
    timeframe: str,
    profile: str,
) -> None:
    if not log_file:
        st.caption("Nenhum arquivo de log configurado.")
        return

    history_store = build_analysis_history_store(log_file)
    history = history_store.load_recent(
        limit=15,
        symbol=symbol,
        timeframe=timeframe,
        profile=profile,
    )
    if not history:
        st.caption("Ainda não há histórico filtrado para este ativo/timeframe/perfil.")
        return

    rows = []
    for entry in history:
        analysis = entry.get("analysis", {})
        rows.append(
            {
                "logged_at": entry.get("logged_at"),
                "symbol": analysis.get("symbol"),
                "setup": analysis.get("setup"),
                "type": analysis.get("setup_type"),
                "trend": analysis.get("trend"),
                "price": analysis.get("current_price"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_execution(request: AnalysisRequest) -> None:
    if request.with_ai and not request.openai_api_key and not os.getenv("OPENAI_API_KEY"):
        st.error(
            "Informe uma OpenAI API Key no campo da barra lateral ou configure `OPENAI_API_KEY` no ambiente."
        )
        return

    try:
        execution = execute_analysis(request)
    except AnalysisServiceError as exc:
        st.error(str(exc))
        return

    analysis = execution.analysis

    st.subheader("Diagnóstico")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ativo", analysis.symbol)
    col2.metric("Setup", analysis.setup)
    col3.metric("Tipo", analysis.setup_type)
    col4.metric("Tendência", analysis.trend)

    st.caption(execution.candle_status)
    if execution.change_message:
        st.write(execution.change_message)

    price_cols = st.columns(4)
    price_cols[0].metric("Preço", _fmt_price(analysis.current_price))
    price_cols[1].metric("EMA 20", _fmt_price(analysis.ema_20))
    price_cols[2].metric("EMA 50", _fmt_price(analysis.ema_50))
    price_cols[3].metric("RSI 14", f"{analysis.rsi:.2f}")

    range_cols = st.columns(4)
    range_cols[0].metric("ATR 14", _fmt_range(analysis.atr, analysis.current_price))
    range_cols[1].metric("ATR %", f"{analysis.atr_percent:.3f}%")
    range_cols[2].metric("Máxima 20", _fmt_price(analysis.recent_high))
    range_cols[3].metric("Mínima 20", _fmt_price(analysis.recent_low))

    st.subheader("Leitura")
    st.write(f"**Viés:** {analysis.bias}")
    st.write(f"**Resumo:** {analysis.summary}")

    with st.expander("Snapshot técnico completo", expanded=False):
        st.json(analysis.__dict__)

    if execution.ai_interpretation is not None:
        st.subheader("Jarvis IA")
        st.code(format_ai_interpretation(execution.ai_interpretation), language="text")

    if execution.ai_payload is not None:
        with st.expander("Payload da IA", expanded=False):
            st.json(execution.ai_payload)


def _fmt_price(value: float) -> str:
    absolute = abs(value)
    if absolute >= 100:
        return f"{value:.2f}"
    if absolute >= 1:
        return f"{value:.5f}"
    return f"{value:.6f}"


def _default_snapshot_path(*, symbol: str, timeframe: str) -> str:
    normalized_symbol = re.sub(r"[^a-z0-9]+", "_", symbol.strip().lower()).strip("_") or "snapshot"
    normalized_timeframe = re.sub(r"[^a-z0-9]+", "_", timeframe.strip().lower()).strip("_") or "tf"
    return f"data/{normalized_symbol}_{normalized_timeframe}.csv"


def _fmt_range(value: float, reference: float) -> str:
    absolute = abs(reference)
    if absolute >= 1:
        return f"{value:.5f}"
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
