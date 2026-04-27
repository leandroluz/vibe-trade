from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.analyzer import AnalysisResult
from app.stores import build_analysis_history_store


class AIIntegrationError(Exception):
    """Raised when the AI integration cannot build or execute a request."""


@dataclass(frozen=True)
class AIInterpretation:
    market_summary: str
    setup_explanation: str
    risk_flags: list[str]
    action_note: str
    response_id: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class AIChatResponse:
    answer: str
    response_id: str | None = None
    model: str | None = None


def build_ai_payload(
    *,
    analysis: AnalysisResult,
    log_path: str | Path | None,
    context_window: int = 10,
    candle_status: str | None = None,
    change_message: str | None = None,
) -> dict:
    recent_history = []
    if log_path:
        history_store = build_analysis_history_store(log_path)
        if history_store is not None:
            recent_history = history_store.load_recent(
                limit=context_window,
                symbol=analysis.symbol,
                timeframe=analysis.timeframe,
                profile=analysis.profile,
            )

    compact_history = [_compact_history_entry(entry) for entry in recent_history]

    return {
        "schema_version": "v1",
        "objective": "Interpretar o contexto tecnico atual sem recalcular indicadores.",
        "guidance": {
            "use_only_provided_data": True,
            "do_not_invent_market_data": True,
            "do_not_issue_orders": True,
            "explain_conflicts_when_present": True,
        },
        "current_snapshot": asdict(analysis),
        "current_context": {
            "candle_status": candle_status,
            "change_message": change_message,
        },
        "recent_history": compact_history,
        "history_summary": _summarize_history(compact_history),
    }


def interpret_with_openai(
    *,
    payload: dict,
    model: str,
    api_key: str | None = None,
) -> AIInterpretation:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AIIntegrationError(
            "Pacote `openai` nao encontrado. Instale as dependencias do projeto para usar `--with-ai`."
        ) from exc
    try:
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise AIIntegrationError(
            "Pacote `pydantic` nao encontrado. Instale as dependencias do projeto para usar `--with-ai`."
        ) from exc

    class AIInterpretationSchema(BaseModel):
        market_summary: str = Field(description="Resumo curto do contexto atual do mercado.")
        setup_explanation: str = Field(description="Explicacao objetiva do setup atual.")
        risk_flags: list[str] = Field(description="Lista curta de riscos ou conflitos observados.")
        action_note: str = Field(
            description="Nota operacional curta, sem tratar a resposta como ordem de execucao."
        )

    try:
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception as exc:
        raise AIIntegrationError(
            "OPENAI_API_KEY nao configurada ou cliente OpenAI nao inicializado corretamente."
        ) from exc
    instructions = _build_system_instructions()
    user_input = json.dumps(payload, ensure_ascii=True, indent=2)

    try:
        response = client.responses.parse(
            model=model,
            instructions=instructions,
            input=[
                {
                    "role": "user",
                    "content": (
                        "Analise o payload JSON abaixo e produza apenas a interpretacao estruturada.\n\n"
                        f"{user_input}"
                    ),
                }
            ],
            text_format=AIInterpretationSchema,
        )
    except Exception as exc:
        raise AIIntegrationError(f"Falha ao consultar a OpenAI: {exc}") from exc

    parsed = response.output_parsed
    if parsed is None:
        raise AIIntegrationError("A resposta da OpenAI nao retornou payload estruturado.")

    return AIInterpretation(
        market_summary=parsed.market_summary,
        setup_explanation=parsed.setup_explanation,
        risk_flags=parsed.risk_flags,
        action_note=parsed.action_note,
        response_id=getattr(response, "id", None),
        model=getattr(response, "model", None),
    )


def chat_with_openai(
    *,
    payload: dict,
    question: str,
    model: str,
    api_key: str | None = None,
    conversation_context: dict | None = None,
) -> AIChatResponse:
    client = _build_openai_client(api_key=api_key)
    instructions = _build_chat_instructions()
    context_json = json.dumps(payload, ensure_ascii=True, indent=2)
    conversation_json = json.dumps(conversation_context or {}, ensure_ascii=True, indent=2)

    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=[
                {
                    "role": "user",
                    "content": (
                        "Contexto tecnico estruturado:\n"
                        f"{context_json}\n\n"
                        "Memoria curta da conversa:\n"
                        f"{conversation_json}\n\n"
                        "Pergunta do usuario:\n"
                        f"{question}"
                    ),
                }
            ],
        )
    except Exception as exc:
        raise AIIntegrationError(f"Falha ao consultar a OpenAI: {exc}") from exc

    answer = getattr(response, "output_text", "").strip()
    if not answer:
        raise AIIntegrationError("A resposta da OpenAI veio vazia para a pergunta do chat.")

    return AIChatResponse(
        answer=answer,
        response_id=getattr(response, "id", None),
        model=getattr(response, "model", None),
    )


def format_ai_interpretation(result: AIInterpretation) -> str:
    risk_flags = result.risk_flags or ["Nenhum risco adicional destacado."]
    return "\n".join(
        [
            "=" * 50,
            "JARVIS | INTERPRETACAO IA",
            "=" * 50,
            f"Resumo: {result.market_summary}",
            f"Setup: {result.setup_explanation}",
            f"Riscos: {' | '.join(risk_flags)}",
            f"Nota: {result.action_note}",
            f"Modelo: {result.model or 'desconhecido'}",
            f"Response ID: {result.response_id or 'indisponivel'}",
            "=" * 50,
        ]
    )


def _build_system_instructions() -> str:
    return (
        "Voce e o Jarvis Trader, um interprete tecnico. "
        "Use apenas os dados fornecidos no payload JSON. "
        "Nao invente preco, candle, indicador ou historico ausente. "
        "Nao recalcule indicadores. "
        "Nao trate a resposta como ordem de execucao. "
        "Se houver conflito entre setup, tendencia e historico, explique o conflito objetivamente. "
        "Responda em Portugues do Brasil."
    )


def _build_chat_instructions() -> str:
    return (
        "Voce e o Jarvis Trader, um assistente tecnico conversacional. "
        "Responda a pergunta do usuario usando apenas o contexto tecnico fornecido. "
        "Use a memoria curta da conversa para manter continuidade quando ela for relevante. "
        "Nao invente preco, candle, indicador, historico ou ativo ausente do payload. "
        "Nao recalcule indicadores. "
        "Nao trate a resposta como ordem de execucao. "
        "Se a pergunta pedir algo fora do payload, deixe isso explicito. "
        "Responda em Portugues do Brasil, de forma objetiva."
    )


def _build_openai_client(api_key: str | None = None):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AIIntegrationError(
            "Pacote `openai` nao encontrado. Instale as dependencias do projeto para usar `--with-ai`."
        ) from exc

    try:
        return OpenAI(api_key=api_key) if api_key else OpenAI()
    except Exception as exc:
        raise AIIntegrationError(
            "OPENAI_API_KEY nao configurada ou cliente OpenAI nao inicializado corretamente."
        ) from exc


def _compact_history_entry(entry: dict) -> dict:
    analysis = entry.get("analysis", {})
    return {
        "logged_at": entry.get("logged_at"),
        "source": entry.get("source"),
        "mode": entry.get("mode"),
        "candle_status": entry.get("candle_status"),
        "change_message": entry.get("change_message"),
        "analysis": {
            "symbol": analysis.get("symbol"),
            "timeframe": analysis.get("timeframe"),
            "profile": analysis.get("profile"),
            "last_candle_time": analysis.get("last_candle_time"),
            "trend": analysis.get("trend"),
            "setup": analysis.get("setup"),
            "setup_type": analysis.get("setup_type"),
            "bias": analysis.get("bias"),
            "summary": analysis.get("summary"),
            "current_price": analysis.get("current_price"),
            "rsi": analysis.get("rsi"),
            "atr_percent": analysis.get("atr_percent"),
        },
    }


def _summarize_history(history: list[dict]) -> dict:
    if not history:
        return {
            "entries": 0,
            "last_setup": None,
            "last_change_message": None,
            "unique_setups": [],
        }

    unique_setups = sorted(
        {
            entry.get("analysis", {}).get("setup")
            for entry in history
            if entry.get("analysis", {}).get("setup")
        }
    )
    return {
        "entries": len(history),
        "last_setup": history[-1].get("analysis", {}).get("setup"),
        "last_change_message": history[-1].get("change_message"),
        "unique_setups": unique_setups,
    }
