from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ai import AIIntegrationError, build_ai_payload, chat_with_openai, summarize_chat_memory
from app.chat_memory import build_chat_memory_context
from app.config import load_settings
from app.service import (
    AnalysisRequest,
    AnalysisServiceError,
    execute_analysis,
    get_available_symbols,
)
from app.stores import build_chat_history_store


st.set_page_config(
    page_title="Chat Jarvis",
    page_icon="J",
    layout="wide",
)


def main() -> None:
    settings = load_settings()
    st.title("Chat Jarvis")
    st.caption("Página separada para conversar com o Jarvis sobre o snapshot técnico atual.")

    with st.sidebar:
        st.header("Contexto")
        source_mode = st.radio(
            "Fonte de dados",
            options=["MT5 ao vivo", "Replay por CSV"],
            index=0,
            key="chat_source_mode",
        )

        symbol = settings.default_symbol
        if source_mode == "MT5 ao vivo":
            symbol = _render_mt5_symbol_selector(settings=settings)
        else:
            symbol = st.text_input("Ativo", value=settings.default_symbol, key="chat_symbol")

        timeframe = st.text_input("Timeframe", value=settings.default_timeframe, key="chat_timeframe")
        profile = st.selectbox(
            "Perfil",
            options=["conservador", "equilibrado", "agressivo"],
            index=["conservador", "equilibrado", "agressivo"].index(settings.analysis_profile),
            key="chat_profile",
        )
        candles_count = st.number_input(
            "Candles analisados",
            min_value=50,
            max_value=5000,
            value=settings.candles_count,
            step=50,
            key="chat_candles_count",
        )
        log_file = st.text_input(
            "Histórico JSONL",
            value=settings.analysis_log_path or "logs/analysis_history.jsonl",
            key="chat_log_file",
        )
        chat_log_file = st.text_input(
            "Histórico do chat",
            value="logs/jarvis_chat_history.jsonl",
            key="chat_history_log_file",
        )
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value="",
            type="password",
            help="Usada apenas nesta sessão do chat. Não é salva no projeto.",
            key="chat_openai_api_key",
        )
        ai_context_window = st.number_input(
            "Janela de contexto da IA",
            min_value=0,
            max_value=100,
            value=10,
            step=1,
            key="chat_ai_context_window",
        )

        data_file = None
        replay_step = 0
        if source_mode == "Replay por CSV":
            data_file = st.text_input("Arquivo CSV", value="data/eurusd_m5.csv", key="chat_data_file")
        update_context = st.button("Atualizar contexto", type="primary", use_container_width=True)

    if update_context:
        _refresh_context(
            AnalysisRequest(
                symbol=symbol,
                timeframe=timeframe,
                candles_count=int(candles_count),
                profile=profile,
                data_file=data_file or None,
                log_file=log_file or None,
                replay_step=replay_step,
                ai_context_window=int(ai_context_window),
                with_ai=False,
            )
        )

    execution = st.session_state.get("chat_execution")
    payload = st.session_state.get("chat_payload")

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.subheader("Contexto atual")
        if execution is None or payload is None:
            st.info("Atualize o contexto para habilitar o chat.")
        else:
            analysis = execution.analysis
            st.metric("Ativo", analysis.symbol)
            st.metric("Setup", analysis.setup)
            st.metric("Tipo", analysis.setup_type)
            st.metric("Tendência", analysis.trend)
            st.caption(execution.candle_status)
            if execution.change_message:
                st.write(execution.change_message)
            with st.expander("Resumo da sessão", expanded=False):
                summary = st.session_state.get("jarvis_session_summary")
                if summary:
                    st.markdown(summary)
                else:
                    st.caption("Ainda não há resumo acumulado desta sessão.")
            _render_recent_chat_history(
                chat_log_file=chat_log_file,
                symbol=analysis.symbol,
                timeframe=analysis.timeframe,
                profile=analysis.profile,
            )
            with st.expander("Snapshot técnico", expanded=False):
                st.json(payload["current_snapshot"])
            with st.expander("Histórico recente filtrado", expanded=False):
                st.json(payload["recent_history"])

    with right_col:
        st.subheader("Conversa")
        if execution is None or payload is None:
            st.info("Sem contexto carregado para o Jarvis.")
            return

        _render_chat_messages()

        question = st.chat_input("Pergunte ao Jarvis sobre o contexto atual")
        if question:
            if not openai_api_key and not os.getenv("OPENAI_API_KEY"):
                st.error("Informe uma OpenAI API Key na barra lateral ou configure `OPENAI_API_KEY`.")
                return

            chat_store = build_chat_history_store(chat_log_file)
            conversation_context = build_chat_memory_context(
                chat_store=chat_store,
                session_messages=st.session_state.get("jarvis_chat_messages", []),
                session_summary=st.session_state.get("jarvis_session_summary"),
                symbol=execution.analysis.symbol,
                timeframe=execution.analysis.timeframe,
                profile=execution.analysis.profile,
            )
            _append_chat_message("user", question)
            try:
                response = chat_with_openai(
                    payload=payload,
                    question=question,
                    model=settings.openai_model,
                    api_key=openai_api_key or None,
                    conversation_context=conversation_context,
                )
            except AIIntegrationError as exc:
                st.error(str(exc))
                return

            assistant_text = "\n".join(
                [
                    response.answer,
                    "",
                    f"Modelo: {response.model or 'desconhecido'}",
                    f"Response ID: {response.response_id or 'indisponivel'}",
                ]
            )
            _append_chat_message("assistant", assistant_text)
            try:
                st.session_state["jarvis_session_summary"] = summarize_chat_memory(
                    payload=payload,
                    previous_summary=st.session_state.get("jarvis_session_summary"),
                    question=question,
                    answer=response.answer,
                    model=settings.openai_model,
                    api_key=openai_api_key or None,
                )
            except AIIntegrationError as exc:
                st.warning(str(exc))
            try:
                if chat_store is not None:
                    chat_store.append_chat_turn(
                        symbol=execution.analysis.symbol,
                        timeframe=execution.analysis.timeframe,
                        profile=execution.analysis.profile,
                        question=question,
                        answer=response.answer,
                        model=response.model,
                        response_id=response.response_id,
                        snapshot=payload["current_snapshot"],
                        history_summary=payload["history_summary"],
                    )
            except Exception as exc:
                st.warning(str(exc))
            st.rerun()


def _refresh_context(request: AnalysisRequest) -> None:
    try:
        execution = execute_analysis(request, persist_log=False)
    except AnalysisServiceError as exc:
        st.error(str(exc))
        return

    payload = build_ai_payload(
        analysis=execution.analysis,
        log_path=request.log_file,
        context_window=request.ai_context_window,
        candle_status=execution.candle_status,
        change_message=execution.change_message,
    )
    st.session_state["chat_execution"] = execution
    st.session_state["chat_payload"] = payload
    st.session_state["jarvis_chat_messages"] = []
    st.session_state["jarvis_session_summary"] = ""
    st.success("Contexto do Jarvis atualizado.")


def _append_chat_message(role: str, content: str) -> None:
    messages = st.session_state.setdefault("jarvis_chat_messages", [])
    messages.append({"role": role, "content": content})


def _render_chat_messages() -> None:
    messages = st.session_state.setdefault("jarvis_chat_messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _render_recent_chat_history(
    *,
    chat_log_file: str,
    symbol: str,
    timeframe: str,
    profile: str,
) -> None:
    try:
        chat_store = build_chat_history_store(chat_log_file)
        if chat_store is None:
            st.caption("Nenhum histórico de chat configurado.")
            return
        entries = chat_store.load_recent(
            limit=5,
            symbol=symbol,
            timeframe=timeframe,
            profile=profile,
        )
    except Exception as exc:
        st.warning(str(exc))
        return

    if not entries:
        st.caption("Ainda não há histórico de conversa para este ativo/timeframe/perfil.")
        return

    with st.expander("Histórico recente do chat", expanded=False):
        for entry in reversed(entries):
            st.markdown(f"**{entry.get('logged_at')}**")
            st.markdown(f"**Pergunta:** {entry.get('question')}")
            st.markdown(f"**Resposta:** {entry.get('answer')}")
            st.caption(
                f"Modelo: {entry.get('model') or 'desconhecido'} | "
                f"Response ID: {entry.get('response_id') or 'indisponivel'}"
            )
            st.divider()


def _render_mt5_symbol_selector(*, settings) -> str:
    search_query = st.text_input(
        "Filtrar ativos do MT5",
        value="",
        help="Digite parte do nome para filtrar os símbolos disponíveis na conta.",
        key="chat_symbol_search",
    )
    try:
        symbols = get_available_symbols(query=search_query or None, settings=settings)
    except AnalysisServiceError as exc:
        st.warning(f"Não foi possível carregar símbolos do MT5: {exc}")
        return st.text_input("Ativo", value=settings.default_symbol, key="chat_symbol_manual")

    if not symbols:
        st.warning("Nenhum símbolo encontrado com esse filtro.")
        return st.text_input("Ativo", value=settings.default_symbol, key="chat_symbol_empty")

    default_symbol = settings.default_symbol
    if default_symbol not in symbols:
        default_symbol = symbols[0]

    return st.selectbox(
        "Ativo",
        options=symbols,
        index=symbols.index(default_symbol),
        help="Lista carregada do MetaTrader 5 com os símbolos disponíveis na conta.",
        key="chat_symbol_select",
    )


if __name__ == "__main__":
    main()
