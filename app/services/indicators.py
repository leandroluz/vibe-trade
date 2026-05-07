from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = average_loss.mask(average_loss == 0)
    relative_strength = average_gain / average_loss
    value = 100 - (100 / (1 + relative_strength))
    return value.fillna(0)


def atr(candles: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = candles["close"].shift(1)
    ranges = pd.concat(
        [
            candles["high"] - candles["low"],
            (candles["high"] - previous_close).abs(),
            (candles["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    true_range = ranges.max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def get_support_resistance(candles: pd.DataFrame, window: int = 20) -> tuple[float, float]:
    sample = candles.tail(window)
    return float(sample["low"].min()), float(sample["high"].max())


def enrich_indicators(candles: pd.DataFrame) -> pd.DataFrame:
    data = candles.copy()
    data["ema_20"] = ema(data["close"], period=20)
    data["ema_50"] = ema(data["close"], period=50)
    data["rsi_14"] = rsi(data["close"], period=14)
    data["atr_14"] = atr(data, period=14)
    support, resistance = get_support_resistance(data, window=20)
    data.attrs["support_20"] = support
    data.attrs["resistance_20"] = resistance
    return data
