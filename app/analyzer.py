from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AnalysisResult:
    symbol: str
    timeframe: str
    profile: str
    last_candle_time: str
    candles_analyzed: int
    current_price: float
    ema_9: float
    ema_20: float
    ema_50: float
    trend: str
    rsi: float
    atr: float
    recent_high: float
    recent_low: float
    range_20: float
    distance_to_ema_20: float
    distance_to_ema_50: float
    atr_percent: float
    bias: str
    setup: str
    setup_type: str
    summary: str


def analyze_market(
    symbol: str,
    timeframe: str,
    candles: pd.DataFrame,
    profile: str = "equilibrado",
) -> AnalysisResult:
    if candles.empty:
        raise ValueError("Não há candles suficientes para análise.")

    latest = candles.iloc[-1]
    recent_window = candles.tail(20)
    previous_candles = candles.iloc[:-1]
    breakout_window = previous_candles.tail(20) if not previous_candles.empty else recent_window
    last_candle_time = pd.Timestamp(latest["time"]).strftime("%Y-%m-%d %H:%M:%S")

    trend = _infer_trend(latest)
    bias = _infer_bias(trend=trend, rsi=float(latest["rsi_14"]))
    ema_9 = float(latest["ema_9"])
    ema_20 = float(latest["ema_20"])
    ema_50 = float(latest["ema_50"])
    current_price = float(latest["close"])
    atr = float(latest["atr_14"])
    recent_high = float(recent_window["high"].max())
    recent_low = float(recent_window["low"].min())
    previous_high = float(breakout_window["high"].max())
    previous_low = float(breakout_window["low"].min())
    range_20 = recent_high - recent_low
    distance_to_ema_20 = current_price - ema_20
    distance_to_ema_50 = current_price - ema_50
    atr_percent = (atr / current_price * 100) if current_price else 0.0
    setup, setup_type, summary = _infer_setup(
        profile=profile,
        trend=trend,
        bias=bias,
        rsi=float(latest["rsi_14"]),
        current_price=current_price,
        candle_low=float(latest["low"]),
        candle_high=float(latest["high"]),
        ema_20=ema_20,
        ema_50=ema_50,
        previous_high=previous_high,
        previous_low=previous_low,
        atr_percent=atr_percent,
    )

    return AnalysisResult(
        symbol=symbol,
        timeframe=timeframe,
        profile=profile,
        last_candle_time=last_candle_time,
        candles_analyzed=len(candles),
        current_price=current_price,
        ema_9=ema_9,
        ema_20=ema_20,
        ema_50=ema_50,
        trend=trend,
        rsi=float(latest["rsi_14"]),
        atr=atr,
        recent_high=recent_high,
        recent_low=recent_low,
        range_20=range_20,
        distance_to_ema_20=distance_to_ema_20,
        distance_to_ema_50=distance_to_ema_50,
        atr_percent=atr_percent,
        bias=bias,
        setup=setup,
        setup_type=setup_type,
        summary=summary,
    )


def _infer_trend(latest: pd.Series) -> str:
    ema_9 = float(latest["ema_9"])
    ema_20 = float(latest["ema_20"])
    ema_50 = float(latest["ema_50"])

    if ema_9 > ema_20 > ema_50:
        return "alta"
    if ema_9 < ema_20 < ema_50:
        return "baixa"
    return "lateral"


def _infer_bias(trend: str, rsi: float) -> str:
    if trend == "alta" and 50 <= rsi < 70:
        return "compra"
    if trend == "baixa" and 30 < rsi <= 50:
        return "venda"
    return "neutro"


def _infer_setup(
    *,
    profile: str,
    trend: str,
    bias: str,
    rsi: float,
    current_price: float,
    candle_low: float,
    candle_high: float,
    ema_20: float,
    ema_50: float,
    previous_high: float,
    previous_low: float,
    atr_percent: float,
) -> tuple[str, str, str]:
    thresholds = _profile_thresholds(profile)
    distance_to_ema20_pct = _percent_distance(current_price, ema_20)
    distance_to_ema50_pct = _percent_distance(current_price, ema_50)
    touched_ema20 = candle_low <= ema_20 <= candle_high
    broke_recent_high = current_price > previous_high
    broke_recent_low = current_price < previous_low

    if trend == "lateral":
        return "sem trade", "lateral", "EMAs sem alinhamento claro; mercado lateral."

    if atr_percent < thresholds["min_atr_percent"]:
        return "sem trade", "volatilidade baixa", "Volatilidade muito baixa para justificar entrada."

    if trend == "alta":
        if rsi >= thresholds["high_overbought_rsi"]:
            return "sem trade", "sobrecompra", "Tendência de alta, mas RSI em sobrecompra."
        if current_price < ema_20:
            return "sem trade", "perdeu suporte", "Tendência de alta, porém preço abaixo da EMA 20."
        if distance_to_ema20_pct > thresholds["max_extension_pct"]:
            return "sem trade", "esticado", "Preço esticado acima da EMA 20; risco de pullback."
        if touched_ema20 and current_price >= ema_20 and rsi >= thresholds["bull_pullback_rsi"]:
            return "favor compra", "pullback", "Alta alinhada com pullback na EMA 20 e defesa compradora."
        if broke_recent_high and rsi < thresholds["high_breakout_rsi"]:
            return "favor compra", "rompimento", "Alta alinhada com rompimento da máxima recente."
        return "favor compra", "continuação", "Alta alinhada, preço acima da EMA 20 e RSI saudável."

    if trend == "baixa":
        if rsi <= thresholds["low_oversold_rsi"]:
            return "sem trade", "sobrevenda", "Tendência de baixa, mas RSI em sobrevenda."
        if current_price > ema_20:
            return "sem trade", "perdeu resistência", "Tendência de baixa, porém preço acima da EMA 20."
        if distance_to_ema20_pct < -thresholds["max_extension_pct"]:
            return "sem trade", "esticado", "Preço esticado abaixo da EMA 20; risco de repique."
        if touched_ema20 and current_price <= ema_20 and rsi <= thresholds["bear_pullback_rsi"]:
            return "favor venda", "pullback", "Baixa alinhada com pullback na EMA 20 e defesa vendedora."
        if broke_recent_low and rsi > thresholds["low_breakout_rsi"]:
            return "favor venda", "rompimento", "Baixa alinhada com rompimento da mínima recente."
        return "favor venda", "continuação", "Baixa alinhada, preço abaixo da EMA 20 e RSI saudável."

    if abs(distance_to_ema50_pct) < 0.05:
        return "sem trade", "indefinido", "Preço muito próximo da EMA 50, sem vantagem clara."

    return "sem trade", "indefinido", "Condições insuficientes para um setup objetivo."


def _percent_distance(value: float, reference: float) -> float:
    if not reference:
        return 0.0
    return (value - reference) / reference * 100


def _profile_thresholds(profile: str) -> dict[str, float]:
    if profile == "conservador":
        return {
            "min_atr_percent": 0.020,
            "max_extension_pct": 0.20,
            "high_overbought_rsi": 67.0,
            "low_oversold_rsi": 33.0,
            "bull_pullback_rsi": 55.0,
            "bear_pullback_rsi": 45.0,
            "high_breakout_rsi": 64.0,
            "low_breakout_rsi": 36.0,
        }
    if profile == "agressivo":
        return {
            "min_atr_percent": 0.005,
            "max_extension_pct": 0.45,
            "high_overbought_rsi": 73.0,
            "low_oversold_rsi": 27.0,
            "bull_pullback_rsi": 50.0,
            "bear_pullback_rsi": 50.0,
            "high_breakout_rsi": 70.0,
            "low_breakout_rsi": 30.0,
        }
    return {
        "min_atr_percent": 0.010,
        "max_extension_pct": 0.30,
        "high_overbought_rsi": 70.0,
        "low_oversold_rsi": 30.0,
        "bull_pullback_rsi": 52.0,
        "bear_pullback_rsi": 48.0,
        "high_breakout_rsi": 68.0,
        "low_breakout_rsi": 32.0,
    }
