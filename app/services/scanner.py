from __future__ import annotations

from typing import Any

from app.mt5_client import MT5Client
from app.services.indicators import enrich_indicators, get_support_resistance
from app.services.mt5_service import get_rates, get_tick
from app.services.risk import calculate_volume_by_risk


def analyze_symbol(
    client: MT5Client,
    symbol: str,
    risk_usd: float,
    timeframe: str,
    candles_count: int = 300,
) -> dict[str, Any]:
    try:
        candles = get_rates(client, symbol, timeframe, candles_count)
        enriched = enrich_indicators(candles)
        latest = enriched.iloc[-1]
        tick = get_tick(client, symbol)
        support, resistance = get_support_resistance(enriched, window=20)
        atr_value = float(latest["atr_14"])
        close = float(latest["close"])
        ema20 = float(latest["ema_20"])
        ema50 = float(latest["ema_50"])
        rsi14 = float(latest["rsi_14"])
        spread = float(tick["spread"])
        spread_ok = atr_value <= 0 or spread <= atr_value * 0.20

        setup, direction, score = _classify_setup(
            close=close,
            ema20=ema20,
            ema50=ema50,
            rsi14=rsi14,
            spread_ok=spread_ok,
        )

        entry = 0.0
        sl = 0.0
        tp1 = 0.0
        tp2 = 0.0
        volume = 0.0

        if direction == "buy":
            entry = float(tick["ask"])
            sl = min(support, entry - (2 * atr_value)) if atr_value > 0 else support
        elif direction == "sell":
            entry = float(tick["bid"])
            sl = max(resistance, entry + (2 * atr_value)) if atr_value > 0 else resistance

        if entry > 0 and sl > 0 and entry != sl and direction in {"buy", "sell"}:
            risk_distance = abs(entry - sl)
            tp1 = entry + (1.5 * risk_distance) if direction == "buy" else entry - (1.5 * risk_distance)
            tp2 = entry + (2.5 * risk_distance) if direction == "buy" else entry - (2.5 * risk_distance)
            volume = calculate_volume_by_risk(client, symbol, direction, entry, sl, risk_usd)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "ok",
            "setup": setup,
            "direction": direction or "neutral",
            "score": score,
            "close": close,
            "bid": float(tick["bid"]),
            "ask": float(tick["ask"]),
            "spread": spread,
            "spread_ok": spread_ok,
            "ema20": ema20,
            "ema50": ema50,
            "rsi14": rsi14,
            "atr14": atr_value,
            "support_20": support,
            "resistance_20": resistance,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "volume": volume,
            "reason": _build_reason(setup, direction, close, ema20, ema50, rsi14, spread_ok),
            "last_candle_time": str(latest["time"]),
        }
    except Exception as exc:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "error",
            "setup": "erro",
            "direction": "neutral",
            "score": 0,
            "reason": str(exc),
        }


def scan_market(
    client: MT5Client,
    symbols: list[str],
    risk_usd: float,
    timeframe: str,
    candles_count: int = 300,
) -> list[dict[str, Any]]:
    results = [
        analyze_symbol(
            client,
            symbol=symbol,
            risk_usd=risk_usd,
            timeframe=timeframe,
            candles_count=candles_count,
        )
        for symbol in symbols
    ]
    return sorted(results, key=lambda item: (item.get("score", 0), item.get("symbol", "")), reverse=True)


def _classify_setup(
    *,
    close: float,
    ema20: float,
    ema50: float,
    rsi14: float,
    spread_ok: bool,
) -> tuple[str, str | None, int]:
    ema_gap = abs(ema20 - ema50)
    if ema_gap <= max(abs(close) * 0.0002, 1e-9):
        return "sem trade", None, 0

    if close > ema20 > ema50 and 50 <= rsi14 <= 70 and spread_ok:
        score = 60
        score += 15 if close > ema20 else 0
        score += 15 if rsi14 <= 65 else 5
        score += 10 if ema_gap > abs(close) * 0.0005 else 0
        return "continuação de alta", "buy", min(score, 100)

    if close < ema20 < ema50 and 30 <= rsi14 <= 50 and spread_ok:
        score = 60
        score += 15 if close < ema20 else 0
        score += 15 if rsi14 >= 35 else 5
        score += 10 if ema_gap > abs(close) * 0.0005 else 0
        return "continuação de baixa", "sell", min(score, 100)

    if ema20 > ema50 and rsi14 > 70:
        return "sem trade", None, 20
    if ema20 < ema50 and rsi14 < 30:
        return "sem trade", None, 20
    if abs(close - ema20) <= max(abs(close) * 0.0003, 1e-9):
        return "lateral", None, 10
    return "sem trade", None, 0


def _build_reason(
    setup: str,
    direction: str | None,
    close: float,
    ema20: float,
    ema50: float,
    rsi14: float,
    spread_ok: bool,
) -> str:
    if direction == "buy":
        return (
            f"Preço acima das EMAs ({close:.5f} > {ema20:.5f} > {ema50:.5f}), "
            f"RSI em {rsi14:.1f} e spread {'ok' if spread_ok else 'alto'}."
        )
    if direction == "sell":
        return (
            f"Preço abaixo das EMAs ({close:.5f} < {ema20:.5f} < {ema50:.5f}), "
            f"RSI em {rsi14:.1f} e spread {'ok' if spread_ok else 'alto'}."
        )
    return f"Setup classificado como {setup}; RSI={rsi14:.1f} e alinhamento insuficiente das EMAs."
