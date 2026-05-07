from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.mt5_client import MT5Client
from app.services.risk import calculate_position_risk


def calculate_position_metrics(
    client: MT5Client,
    position: dict[str, Any],
) -> dict[str, Any]:
    enriched = calculate_position_risk(client, position)
    entry = float(enriched.get("entry") or 0.0)
    current_price = float(enriched.get("price_current") or 0.0)
    sl = float(enriched.get("sl") or 0.0)
    tp = float(enriched.get("tp") or 0.0)
    profit = float(enriched.get("profit") or 0.0)
    risk_usd = enriched.get("risk_to_sl")
    position_type = str(enriched.get("type") or "").lower()
    has_sl = sl > 0
    has_tp = tp > 0
    distance_to_sl = abs(current_price - sl) if has_sl and current_price > 0 else None
    distance_to_tp = abs(tp - current_price) if has_tp and current_price > 0 else None
    current_r = (profit / float(risk_usd)) if risk_usd not in (None, 0) else None
    initial_risk_distance = _initial_risk_distance(position_type=position_type, entry=entry, sl=sl)
    remaining_risk_ratio = _distance_ratio(distance_to_sl, initial_risk_distance)
    tp_progress_ratio = _distance_ratio(distance_to_tp, initial_risk_distance)
    is_sl_protected = _is_sl_protected(position_type=position_type, entry=entry, sl=sl)

    metrics = {
        "ticket": enriched.get("ticket"),
        "symbol": str(enriched.get("symbol") or ""),
        "type": position_type,
        "volume": float(enriched.get("volume") or 0.0),
        "entry": entry,
        "current_price": current_price,
        "sl": sl,
        "tp": tp,
        "profit": profit,
        "risk_usd": float(risk_usd) if risk_usd is not None else None,
        "current_r": current_r,
        "distance_to_sl": distance_to_sl,
        "distance_to_tp": distance_to_tp,
        "distance_to_sl_pct": _distance_pct(current_price, distance_to_sl),
        "distance_to_tp_pct": _distance_pct(current_price, distance_to_tp),
        "initial_risk_distance": initial_risk_distance,
        "remaining_risk_ratio": remaining_risk_ratio,
        "tp_progress_ratio": tp_progress_ratio,
        "has_sl": has_sl,
        "has_tp": has_tp,
        "is_sl_protected": is_sl_protected,
        "alerts": [],
    }

    if not has_sl:
        metrics["alerts"].append("Posição sem SL definido.")

    return metrics


def classify_position_status(position_metrics: dict[str, Any]) -> dict[str, str]:
    if not position_metrics.get("has_sl"):
        return {
            "position_status": "no_sl",
            "suggested_action": "adicionar SL ou encerrar manualmente",
            "severity": "critical",
        }

    current_r = position_metrics.get("current_r")
    remaining_risk_ratio = position_metrics.get("remaining_risk_ratio")
    tp_progress_ratio = position_metrics.get("tp_progress_ratio")
    is_sl_protected = bool(position_metrics.get("is_sl_protected"))

    if current_r is not None and current_r >= 2.0:
        return {
            "position_status": "positive_2r",
            "suggested_action": "proteger lucro agressivamente ou considerar parcial",
            "severity": "high",
        }
    if current_r is not None and current_r >= 1.5:
        return {
            "position_status": "positive_1_5r",
            "suggested_action": "considerar parcial e mover SL para região protegida",
            "severity": "medium",
        }
    if current_r is not None and current_r >= 1.0:
        return {
            "position_status": "positive_1r",
            "suggested_action": "considerar mover SL para entrada",
            "severity": "medium",
        }

    if not is_sl_protected and remaining_risk_ratio is not None and remaining_risk_ratio <= 0.20:
        return {
            "position_status": "near_sl",
            "suggested_action": "evitar aumentar exposição; aceitar invalidação ou reduzir manualmente",
            "severity": "high",
        }
    if tp_progress_ratio is not None and tp_progress_ratio <= 0.20:
        return {
            "position_status": "near_tp",
            "suggested_action": "evitar mexer sem gatilho claro; reavaliar realização ou proteção",
            "severity": "medium",
        }
    if current_r is None:
        return {
            "position_status": "near_breakeven",
            "suggested_action": "não mexer ainda; posição próxima do breakeven",
            "severity": "low",
        }
    if current_r >= 0.3:
        return {
            "position_status": "positive_light",
            "suggested_action": "manter e aguardar desenvolvimento",
            "severity": "low",
        }
    if -0.3 <= current_r < 0.3:
        return {
            "position_status": "near_breakeven",
            "suggested_action": "não mexer ainda; posição próxima do breakeven",
            "severity": "low",
        }
    if current_r < -0.5:
        return {
            "position_status": "negative",
            "suggested_action": "cautela; posição consumindo risco",
            "severity": "medium",
        }
    return {
        "position_status": "negative",
        "suggested_action": "evitar mexer; aguardar invalidação ou reação objetiva",
        "severity": "low",
    }


def generate_position_triggers(position_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    symbol = str(position_metrics.get("symbol") or "")
    current_r = position_metrics.get("current_r")
    triggers: list[dict[str, Any]] = []

    if not position_metrics.get("has_sl"):
        return [
            {
                "symbol": symbol,
                "condition": "posição sem stop loss",
                "level": 0.0,
                "suggested_action": "adicionar SL ou encerrar manualmente",
                "reason": "Risco não está limitado pela estrutura atual.",
            }
        ]

    if current_r is not None and current_r < 1.0:
        triggers.append(
            {
                "symbol": symbol,
                "condition": "atingir +1R",
                "level": 1.0,
                "suggested_action": "considerar mover SL para entrada",
                "reason": "Atingindo 1R, a operação já justifica proteção de capital.",
            }
        )
    if current_r is not None and current_r < 1.5:
        triggers.append(
            {
                "symbol": symbol,
                "condition": "atingir +1.5R",
                "level": 1.5,
                "suggested_action": "considerar parcial",
                "reason": "Em 1.5R, a operação pode justificar realização parcial manual.",
            }
        )
    if current_r is not None and current_r < 2.0:
        triggers.append(
            {
                "symbol": symbol,
                "condition": "atingir +2R",
                "level": 2.0,
                "suggested_action": "proteger lucro agressivamente",
                "reason": "Em 2R, a prioridade passa a ser preservação do lucro.",
            }
        )
    if not position_metrics.get("is_sl_protected") and position_metrics.get("sl"):
        triggers.append(
            {
                "symbol": symbol,
                "condition": "atingir SL",
                "level": float(position_metrics["sl"]),
                "suggested_action": "aceitar invalidação e não reabrir automaticamente",
                "reason": "Se o stop for atingido, a prioridade é preservar capital e evitar revenge trade.",
            }
        )
    return triggers


def analyze_open_positions(
    client: MT5Client,
    positions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    analyzed: list[dict[str, Any]] = []
    for position in positions:
        metrics = calculate_position_metrics(client, position)
        classification = classify_position_status(metrics)
        triggers = generate_position_triggers({**metrics, **classification})
        analyzed.append(
            {
                **metrics,
                **classification,
                "triggers": triggers,
                "review_timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    return analyzed


def build_position_management_summary(positions_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    total_open_risk_usd = sum(float(item.get("risk_usd") or 0.0) for item in positions_metrics)
    total_floating_profit = sum(float(item.get("profit") or 0.0) for item in positions_metrics)
    positions_without_sl = [item["symbol"] for item in positions_metrics if not item.get("has_sl")]

    highest_risk_position = None
    with_risk = [item for item in positions_metrics if item.get("risk_usd") is not None]
    if with_risk:
        highest_risk_position = max(with_risk, key=lambda item: float(item.get("risk_usd") or 0.0))

    risk_alerts: list[str] = []
    if positions_without_sl:
        risk_alerts.append("Há posições sem SL: " + ", ".join(positions_without_sl) + ".")
    if highest_risk_position is not None:
        risk_alerts.append(
            f"Maior risco remanescente: {highest_risk_position['symbol']} com {float(highest_risk_position['risk_usd'] or 0.0):.2f} USD."
        )

    return {
        "positions_metrics": positions_metrics,
        "total_open_risk_usd": total_open_risk_usd,
        "total_floating_profit": total_floating_profit,
        "highest_risk_position": highest_risk_position,
        "positions_without_sl": positions_without_sl,
        "risk_alerts": risk_alerts,
    }


def _distance_pct(current_price: float, absolute_distance: float | None) -> float | None:
    if current_price <= 0 or absolute_distance is None:
        return None
    return absolute_distance / current_price


def _distance_ratio(distance: float | None, reference: float | None) -> float | None:
    if distance is None or reference is None or reference <= 0:
        return None
    return distance / reference


def _initial_risk_distance(*, position_type: str, entry: float, sl: float) -> float | None:
    if entry <= 0 or sl <= 0 or position_type not in {"buy", "sell"}:
        return None
    distance = abs(entry - sl)
    return distance if distance > 0 else None


def _is_sl_protected(*, position_type: str, entry: float, sl: float) -> bool:
    if position_type == "buy":
        return sl >= entry > 0
    if position_type == "sell":
        return sl <= entry and sl > 0
    return False
