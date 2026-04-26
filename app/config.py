from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    default_symbol: str = "EURUSD"
    default_timeframe: str = "M5"
    analysis_profile: str = "equilibrado"
    analysis_log_path: str = ""
    openai_model: str = "gpt-4.1"
    candles_count: int = 300
    mt5_host: str = "127.0.0.1"
    mt5_port: int = 18812


def load_settings() -> Settings:
    load_dotenv()

    default_symbol = os.getenv("DEFAULT_SYMBOL", "EURUSD").strip() or "EURUSD"
    default_timeframe = os.getenv("DEFAULT_TIMEFRAME", "M5").strip().upper() or "M5"
    analysis_profile = os.getenv("ANALYSIS_PROFILE", "equilibrado").strip().lower() or "equilibrado"
    analysis_log_path = os.getenv("ANALYSIS_LOG_PATH", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1").strip() or "gpt-4.1"

    if analysis_profile not in {"conservador", "equilibrado", "agressivo"}:
        analysis_profile = "equilibrado"

    raw_candles_count = os.getenv("CANDLES_COUNT", "300").strip() or "300"
    try:
        candles_count = int(raw_candles_count)
    except ValueError:
        candles_count = 300

    if candles_count <= 0:
        candles_count = 300

    mt5_host = os.getenv("MT5_HOST", "127.0.0.1").strip() or "127.0.0.1"

    raw_mt5_port = os.getenv("MT5_PORT", "18812").strip() or "18812"
    try:
        mt5_port = int(raw_mt5_port)
    except ValueError:
        mt5_port = 18812

    if mt5_port <= 0:
        mt5_port = 18812

    return Settings(
        default_symbol=default_symbol,
        default_timeframe=default_timeframe,
        analysis_profile=analysis_profile,
        analysis_log_path=analysis_log_path,
        openai_model=openai_model,
        candles_count=candles_count,
        mt5_host=mt5_host,
        mt5_port=mt5_port,
    )
