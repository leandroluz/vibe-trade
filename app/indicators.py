from __future__ import annotations

import pandas as pd


def add_indicators(candles: pd.DataFrame) -> pd.DataFrame:
    data = candles.copy()

    data["ema_9"] = _ema(data["close"], span=9)
    data["ema_20"] = _ema(data["close"], span=20)
    data["ema_50"] = _ema(data["close"], span=50)
    data["rsi_14"] = _rsi(data["close"], period=14)
    data["atr_14"] = _atr(data, period=14)

    return data


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    average_loss = average_loss.mask(average_loss == 0)
    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)


def _atr(candles: pd.DataFrame, period: int) -> pd.Series:
    previous_close = candles["close"].shift(1)

    high_low = candles["high"] - candles["low"]
    high_close = (candles["high"] - previous_close).abs()
    low_close = (candles["low"] - previous_close).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
