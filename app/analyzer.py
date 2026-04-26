from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AnalysisResult:
    symbol: str
    timeframe: str
    candles_analyzed: int
    current_price: float
    trend: str
    rsi: float
    atr: float
    recent_high: float
    recent_low: float
    bias: str


def analyze_market(symbol: str, timeframe: str, candles: pd.DataFrame) -> AnalysisResult:
    if candles.empty:
        raise ValueError("Não há candles suficientes para análise.")

    latest = candles.iloc[-1]
    recent_window = candles.tail(20)

    trend = _infer_trend(latest)
    bias = _infer_bias(trend=trend, rsi=float(latest["rsi_14"]))

    return AnalysisResult(
        symbol=symbol,
        timeframe=timeframe,
        candles_analyzed=len(candles),
        current_price=float(latest["close"]),
        trend=trend,
        rsi=float(latest["rsi_14"]),
        atr=float(latest["atr_14"]),
        recent_high=float(recent_window["high"].max()),
        recent_low=float(recent_window["low"].min()),
        bias=bias,
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
