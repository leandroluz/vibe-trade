from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


SYSTEM_PROMPT = """Você é o Jarvis Trader, um coach de trading para conta Alpha Capital Group.

Regras:
- Use apenas os dados fornecidos pelo sistema.
- Não invente preço.
- Não invente notícias.
- Não prometa lucro.
- Não recomende overtrading.
- Priorize preservação de conta, drawdown baixo e gestão de risco.
- Considere correlação entre posições, especialmente exposição ao dólar.
- Se já houver risco aberto alto, prefira gerenciar posições em vez de sugerir nova entrada.
- Se não houver oportunidade clara, diga para esperar.
- A resposta deve ser operacional, objetiva e conservadora.
- A IA não executa ordens.
- A IA deve responder em português.
- Você também atua como gestor de posições abertas.
- Use os campos current_r, risk_usd, distance_to_sl, distance_to_tp e position_status.
- Não sugira nova entrada se o risco aberto estiver próximo do limite configurado.
- Se posição atingir +1R, pode sugerir mover SL para entrada.
- Se posição atingir +1.5R ou mais, pode sugerir parcial.
- Se posição estiver sem SL, priorize alerta de risco.
- Se não houver gatilho claro, diga para manter e reavaliar no próximo candle.
- Nunca execute nada automaticamente."""


class OpportunitySchema(BaseModel):
    symbol: str
    direction: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    volume: float
    confidence: float
    reason: str


class PositionManagementSchema(BaseModel):
    symbol: str
    action: str
    reason: str


class PositionTriggerSchema(BaseModel):
    symbol: str
    condition: str
    level: float
    suggested_action: str
    reason: str


class ActionPlanSchema(BaseModel):
    now: str
    do_not_do: list[str]
    position_triggers: list[PositionTriggerSchema]
    next_review: str


class JarvisCoachSchema(BaseModel):
    decision: str = Field(description="open_trade | manage_positions | wait | avoid_trading")
    account_reading: str
    risk_reading: str
    best_action_now: str
    warnings: list[str]
    opportunities: list[OpportunitySchema]
    positions_management: list[PositionManagementSchema]
    action_plan: ActionPlanSchema


@dataclass(frozen=True)
class JarvisCoachResult:
    parsed: dict[str, Any] | None
    raw_text: str
    model: str | None
    response_id: str | None
    parse_error: str | None = None


def ask_jarvis_coach(payload: dict[str, Any]) -> JarvisCoachResult:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = str(payload.get("_model_override") or "").strip() or os.getenv("OPENAI_MODEL", "").strip()

    if not api_key:
        raise RuntimeError("Defina a variável de ambiente OPENAI_API_KEY para usar o Coach IA.")
    if not model:
        raise RuntimeError("Defina a variável de ambiente OPENAI_MODEL para usar o Coach IA.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Pacote `openai` não encontrado no ambiente do projeto.") from exc

    client = OpenAI(api_key=api_key)
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)

    try:
        response = client.responses.parse(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": (
                        "Analise o snapshot abaixo e retorne apenas o JSON estruturado solicitado.\n\n"
                        f"{payload_json}"
                    ),
                }
            ],
            text_format=JarvisCoachSchema,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("Structured output vazio.")
        parsed_payload = _sanitize_ai_response(parsed.model_dump(), payload)
        return JarvisCoachResult(
            parsed=parsed_payload,
            raw_text=json.dumps(parsed_payload, ensure_ascii=False),
            model=getattr(response, "model", model),
            response_id=getattr(response, "id", None),
        )
    except Exception as structured_exc:
        fallback = _ask_jarvis_fallback(client=client, model=model, payload_json=payload_json)
        if fallback.parsed is not None:
            return fallback
        return JarvisCoachResult(
            parsed=None,
            raw_text=fallback.raw_text,
            model=fallback.model,
            response_id=fallback.response_id,
            parse_error=f"Estruturado: {structured_exc}; fallback: {fallback.parse_error}",
        )


def _ask_jarvis_fallback(*, client: Any, model: str, payload_json: str) -> JarvisCoachResult:
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT
        + "\nRetorne JSON puro, válido e sem markdown seguindo exatamente o schema pedido.",
        input=[
            {
                "role": "user",
                "content": (
                    "Analise o snapshot abaixo e retorne apenas JSON puro.\n\n"
                    f"{payload_json}"
                ),
            }
        ],
    )
    raw_text = getattr(response, "output_text", "").strip()
    if not raw_text:
        return JarvisCoachResult(
            parsed=None,
            raw_text="",
            model=getattr(response, "model", model),
            response_id=getattr(response, "id", None),
            parse_error="Resposta vazia da OpenAI.",
        )

    cleaned = _strip_code_fences(raw_text)
    try:
        parsed = json.loads(cleaned)
        validated = JarvisCoachSchema.model_validate(parsed).model_dump()
        sanitized = _sanitize_ai_response(validated, json.loads(payload_json))
        return JarvisCoachResult(
            parsed=sanitized,
            raw_text=raw_text,
            model=getattr(response, "model", model),
            response_id=getattr(response, "id", None),
        )
    except Exception as exc:
        return JarvisCoachResult(
            parsed=None,
            raw_text=raw_text,
            model=getattr(response, "model", model),
            response_id=getattr(response, "id", None),
            parse_error=str(exc),
        )


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    return cleaned


def _sanitize_ai_response(response: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    market_scan = payload.get("market_scan") or []
    canonical_by_symbol = {
        str(item.get("symbol")): item
        for item in market_scan
        if item.get("status") == "ok" and item.get("direction") in {"buy", "sell"}
    }

    safe_opportunities: list[dict[str, Any]] = []
    warnings = list(response.get("warnings") or [])

    for item in response.get("opportunities") or []:
        symbol = str(item.get("symbol") or "")
        canonical = canonical_by_symbol.get(symbol)
        if canonical is None:
            warnings.append(
                f"Oportunidade para {symbol or 'símbolo vazio'} ignorada: não existe candidato técnico válido no payload."
            )
            continue

        direction = str(item.get("direction") or "")
        if direction != canonical.get("direction"):
            warnings.append(
                f"Oportunidade para {symbol} ignorada: direção divergente do snapshot técnico."
            )
            continue

        safe_opportunities.append(
            {
                "symbol": symbol,
                "direction": canonical["direction"],
                "entry": float(canonical["entry"]),
                "sl": float(canonical["sl"]),
                "tp1": float(canonical["tp1"]),
                "tp2": float(canonical["tp2"]),
                "volume": float(canonical["volume"]),
                "confidence": float(item.get("confidence") or 0.0),
                "reason": str(item.get("reason") or ""),
            }
        )

    sanitized = {
        **response,
        "warnings": warnings,
        "opportunities": safe_opportunities,
    }
    return JarvisCoachSchema.model_validate(sanitized).model_dump()
