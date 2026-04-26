from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("time", "open", "high", "low", "close")


class CandleDataError(Exception):
    """Raised when local candle data cannot be loaded or persisted."""


def load_candles_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise CandleDataError(f"Arquivo de candles não encontrado: {file_path}")

    try:
        candles = pd.read_csv(file_path)
    except Exception as exc:
        raise CandleDataError(f"Falha ao ler arquivo de candles: {file_path}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in candles.columns]
    if missing:
        missing_columns = ", ".join(missing)
        raise CandleDataError(
            f"Arquivo de candles inválido: faltam colunas obrigatórias: {missing_columns}"
        )

    candles = candles.copy()
    candles["time"] = pd.to_datetime(candles["time"], errors="coerce")
    if candles["time"].isna().any():
        raise CandleDataError("Coluna `time` contém valores inválidos.")

    candles = candles.sort_values("time").reset_index(drop=True)
    candles.attrs["resolved_symbol"] = _infer_symbol_from_dataframe(candles, file_path)
    return candles


def save_candles_csv(candles: pd.DataFrame, path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    export_frame = candles.copy()
    resolved_symbol = candles.attrs.get("resolved_symbol")
    if resolved_symbol and "symbol" not in export_frame.columns:
        export_frame["symbol"] = resolved_symbol
    export_frame["time"] = pd.to_datetime(export_frame["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    export_frame.to_csv(file_path, index=False)


def build_replay_window(candles: pd.DataFrame, count: int, step: int) -> pd.DataFrame:
    if len(candles) < count:
        raise CandleDataError(
            f"Arquivo tem apenas {len(candles)} candles; são necessários pelo menos {count}."
        )

    max_step = len(candles) - count
    if step < 0 or step > max_step:
        raise CandleDataError(
            f"Replay fora do intervalo: passo {step}, máximo permitido {max_step}."
        )

    window = candles.iloc[step : step + count].copy()
    window.attrs["resolved_symbol"] = candles.attrs.get("resolved_symbol")
    window.attrs["replay_step"] = step
    window.attrs["replay_total_steps"] = max_step + 1
    return window


def _infer_symbol_from_dataframe(candles: pd.DataFrame, file_path: Path) -> str:
    symbol_column = candles.get("symbol")
    if symbol_column is not None:
        valid_symbols = symbol_column.dropna().astype(str).unique().tolist()
        if len(valid_symbols) == 1:
            return valid_symbols[0]

    return file_path.stem
