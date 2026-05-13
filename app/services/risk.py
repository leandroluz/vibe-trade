from __future__ import annotations

import math
from typing import Any

from app.mt5_client import MT5Client, MT5ConnectionError
from app.services.mt5_service import get_symbol_info, order_calc_profit


def normalize_volume(client: MT5Client, symbol: str, volume: float) -> float:
    symbol_info = get_symbol_info(client, symbol)
    min_volume = float(symbol_info.volume_min or 0.0)
    max_volume = float(symbol_info.volume_max or 0.0)
    step = float(symbol_info.volume_step or 0.0)

    if volume <= 0 or min_volume <= 0:
        return 0.0

    capped = max(min_volume, min(float(volume), max_volume if max_volume > 0 else float(volume)))
    if step <= 0:
        return round(capped, 2)

    steps = math.floor(((capped - min_volume) / step) + 1e-9)
    normalized = min_volume + (steps * step)
    normalized = max(min_volume, min(normalized, max_volume if max_volume > 0 else normalized))

    decimals = _step_decimals(step)
    return round(normalized, decimals)


def calculate_volume_by_risk(
    client: MT5Client,
    symbol: str,
    direction: str,
    entry: float,
    sl: float,
    risk_usd: float,
) -> float:
    if risk_usd <= 0 or entry <= 0 or sl <= 0 or entry == sl:
        return 0.0

    loss_one_lot = order_calc_profit(
        client,
        direction=direction,
        symbol=symbol,
        volume=1.0,
        entry=entry,
        target_price=sl,
    )
    loss_abs = abs(float(loss_one_lot))
    if loss_abs <= 0:
        raise MT5ConnectionError(
            f"Não foi possível estimar perda de 1 lote para {symbol} com entry={entry} e sl={sl}."
        )

    raw_volume = float(risk_usd) / loss_abs
    return normalize_volume(client, symbol, raw_volume)


def calculate_position_risk(client: MT5Client, position: dict[str, Any]) -> dict[str, Any]:
    sl = float(position.get("sl") or 0.0)
    entry = float(position.get("entry") or 0.0)
    current = float(position.get("price_current") or 0.0)
    volume = float(position.get("volume") or 0.0)
    direction = str(position.get("type") or "")
    symbol = str(position.get("symbol") or "")

    if sl <= 0:
        return {
            **position,
            "risk_to_sl": None,
            "risk_status": "missing_sl",
            "distance_to_sl": None,
        }

    risk_value = order_calc_profit(
        client,
        direction=direction,
        symbol=symbol,
        volume=volume,
        entry=current if current > 0 else entry,
        target_price=sl,
    )

    distance = abs(current - sl) if current > 0 else abs(entry - sl)
    return {
        **position,
        "risk_to_sl": abs(float(risk_value)),
        "risk_status": "ok",
        "distance_to_sl": distance,
    }


def build_position_management_snapshot(
    position: dict[str, Any],
    technical_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    direction = str(position.get("type") or "").lower()
    entry = float(position.get("entry") or 0.0)
    current = float(position.get("price_current") or 0.0)
    sl = float(position.get("sl") or 0.0)
    tp = float(position.get("tp") or 0.0)
    profit = float(position.get("profit") or 0.0)
    risk_remaining = position.get("risk_to_sl")

    initial_risk_distance = _initial_risk_distance(direction=direction, entry=entry, sl=sl)
    distance_to_tp = abs(tp - current) if tp > 0 and current > 0 else None
    r_current = _current_r_multiple(
        direction=direction,
        entry=entry,
        current=current,
        initial_risk_distance=initial_risk_distance,
    )
    suggestion, suggestion_reason = _build_management_suggestion(
        position=position,
        technical_context=technical_context,
        r_current=r_current,
    )

    return {
        "ticket": position.get("ticket"),
        "symbol": position.get("symbol"),
        "type": position.get("type"),
        "volume": position.get("volume"),
        "entry": entry,
        "price_current": current,
        "sl": sl,
        "tp": tp,
        "distance_to_sl": position.get("distance_to_sl"),
        "distance_to_tp": distance_to_tp,
        "profit": profit,
        "risk_remaining": risk_remaining,
        "r_current": r_current,
        "technical_setup": technical_context.get("setup") if technical_context else None,
        "technical_direction": technical_context.get("direction") if technical_context else None,
        "technical_reason": technical_context.get("reason") if technical_context else None,
        "management_suggestion": suggestion,
        "management_reason": suggestion_reason,
    }


def _step_decimals(step: float) -> int:
    text = f"{step:.8f}".rstrip("0")
    if "." not in text:
        return 0
    return len(text.split(".")[1])


def _initial_risk_distance(*, direction: str, entry: float, sl: float) -> float | None:
    if entry <= 0 or sl <= 0 or direction not in {"buy", "sell"}:
        return None
    distance = abs(entry - sl)
    if distance <= 0:
        return None
    return distance


def _current_r_multiple(
    *,
    direction: str,
    entry: float,
    current: float,
    initial_risk_distance: float | None,
) -> float | None:
    if initial_risk_distance is None or current <= 0 or entry <= 0:
        return None
    movement = current - entry if direction == "buy" else entry - current
    return movement / initial_risk_distance


def _build_management_suggestion(
    *,
    position: dict[str, Any],
    technical_context: dict[str, Any] | None,
    r_current: float | None,
) -> tuple[str, str]:
    direction = str(position.get("type") or "").lower()
    sl = float(position.get("sl") or 0.0)

    if sl <= 0:
        return (
            "evitar mexer",
            "Posição sem SL definido. Antes de qualquer ajuste tático, a prioridade é revisar a proteção de risco manualmente.",
        )

    if technical_context:
        tech_direction = str(technical_context.get("direction") or "").lower()
        tech_setup = str(technical_context.get("setup") or "").lower()
        if technical_context.get("status") == "ok" and tech_direction in {"buy", "sell"} and tech_direction != direction:
            return (
                "encerrar se a estrutura técnica invalidou",
                "O scanner atual aponta direção oposta à posição aberta, sugerindo estrutura técnica enfraquecida.",
            )
        if tech_setup in {"sem trade", "lateral"} and (r_current is None or r_current < 0.5):
            return (
                "evitar mexer",
                "A leitura técnica está neutra/lateral. Sem confirmação de continuação, ajustes agressivos tendem a piorar a gestão.",
            )

    if r_current is None:
        return (
            "manter",
            "Não foi possível calcular o múltiplo R atual com confiança, então a sugestão permanece conservadora.",
        )

    if r_current < -0.75:
        return (
            "encerrar se a estrutura técnica invalidou",
            "A operação já devolveu parte relevante do risco inicial e merece revisão manual imediata da tese.",
        )
    if r_current >= 1.5:
        return (
            "reduzir parcial",
            "A posição já percorreu mais de 1.5R. Faz sentido avaliar realização parcial manual se isso estiver no seu plano.",
        )
    if r_current >= 1.0:
        return (
            "mover SL para entrada",
            "A operação atingiu pelo menos 1R. Vale considerar proteger o capital movendo o stop para a entrada, manualmente.",
        )
    if 0.0 <= r_current < 1.0:
        return (
            "manter",
            "A posição ainda está evoluindo dentro da estrutura inicial e não atingiu um marco claro para ajuste defensivo.",
        )
    return (
        "evitar mexer",
        "A posição está negativa, mas ainda não há evidência suficiente para sugerir ajuste tático além da disciplina no plano.",
    )
