from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("time", "open", "high", "low", "close")


class CandleDataError(Exception):
    """Raised when local candle data cannot be loaded or persisted."""


class AnalysisHistoryError(Exception):
    """Raised when persisted analysis history cannot be read."""


class ChatHistoryError(Exception):
    """Raised when persisted chat history cannot be read or written."""


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


def load_recent_analysis_history(
    path: str | Path,
    limit: int = 10,
    symbol: str | None = None,
    timeframe: str | None = None,
    profile: str | None = None,
) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        return []

    if limit <= 0:
        return []

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except Exception as exc:
        raise AnalysisHistoryError(f"Falha ao ler histórico de análise: {file_path}") from exc

    records: list[dict] = []
    for raw_line in reversed(lines):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnalysisHistoryError(
                f"Histórico JSONL inválido em {file_path}: {exc}"
            ) from exc
        if not _history_record_matches(
            record,
            symbol=symbol,
            timeframe=timeframe,
            profile=profile,
        ):
            continue
        records.append(record)
        if len(records) >= limit:
            break

    records.reverse()
    return records


def append_chat_history_entry(
    path: str | Path,
    *,
    symbol: str,
    timeframe: str,
    profile: str,
    question: str,
    answer: str,
    model: str | None,
    response_id: str | None,
    snapshot: dict,
    history_summary: dict,
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        "symbol": symbol,
        "timeframe": timeframe,
        "profile": profile,
        "question": question,
        "answer": answer,
        "model": model,
        "response_id": response_id,
        "snapshot": snapshot,
        "history_summary": history_summary,
    }

    try:
        with file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception as exc:
        raise ChatHistoryError(f"Falha ao gravar histórico de chat: {file_path}") from exc


def load_recent_chat_history(
    path: str | Path,
    limit: int = 10,
    symbol: str | None = None,
    timeframe: str | None = None,
    profile: str | None = None,
) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists() or limit <= 0:
        return []

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except Exception as exc:
        raise ChatHistoryError(f"Falha ao ler histórico de chat: {file_path}") from exc

    records: list[dict] = []
    for raw_line in reversed(lines):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ChatHistoryError(f"Histórico de chat JSONL inválido em {file_path}: {exc}") from exc
        if not _chat_record_matches(
            record,
            symbol=symbol,
            timeframe=timeframe,
            profile=profile,
        ):
            continue
        records.append(record)
        if len(records) >= limit:
            break

    records.reverse()
    return records


def _history_record_matches(
    record: dict,
    *,
    symbol: str | None,
    timeframe: str | None,
    profile: str | None,
) -> bool:
    analysis = record.get("analysis", {})

    if symbol:
        if str(analysis.get("symbol", "")).upper() != symbol.upper():
            return False

    if timeframe:
        if str(analysis.get("timeframe", "")).upper() != timeframe.upper():
            return False

    if profile:
        if str(analysis.get("profile", "")).lower() != profile.lower():
            return False

    return True


def _chat_record_matches(
    record: dict,
    *,
    symbol: str | None,
    timeframe: str | None,
    profile: str | None,
) -> bool:
    if symbol and str(record.get("symbol", "")).upper() != symbol.upper():
        return False

    if timeframe and str(record.get("timeframe", "")).upper() != timeframe.upper():
        return False

    if profile and str(record.get("profile", "")).lower() != profile.lower():
        return False

    return True


def _infer_symbol_from_dataframe(candles: pd.DataFrame, file_path: Path) -> str:
    symbol_column = candles.get("symbol")
    if symbol_column is not None:
        valid_symbols = symbol_column.dropna().astype(str).unique().tolist()
        if len(valid_symbols) == 1:
            return valid_symbols[0]

    return file_path.stem
