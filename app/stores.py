from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from app.analyzer import AnalysisResult
from app.data import load_recent_analysis_history, load_recent_chat_history

if TYPE_CHECKING:
    from app.ai import AIInterpretation


class AnalysisHistoryStore(Protocol):
    def append_analysis_run(
        self,
        *,
        analysis: AnalysisResult,
        request_metadata: dict,
        source: str,
        candle_status: str,
        change_message: str | None,
        ai_interpretation: AIInterpretation | None = None,
    ) -> None: ...

    def load_recent(
        self,
        *,
        limit: int = 10,
        symbol: str | None = None,
        timeframe: str | None = None,
        profile: str | None = None,
    ) -> list[dict]: ...


class ChatHistoryStore(Protocol):
    def append_chat_turn(
        self,
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
    ) -> None: ...

    def load_recent(
        self,
        *,
        limit: int = 10,
        symbol: str | None = None,
        timeframe: str | None = None,
        profile: str | None = None,
    ) -> list[dict]: ...


class JSONLAnalysisHistoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def append_analysis_run(
        self,
        *,
        analysis: AnalysisResult,
        request_metadata: dict,
        source: str,
        candle_status: str,
        change_message: str | None,
        ai_interpretation: AIInterpretation | None = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "logged_at": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "mode": request_metadata.get("mode", "single-run"),
            "symbol_requested": request_metadata.get("symbol_requested"),
            "timeframe_requested": request_metadata.get("timeframe_requested"),
            "profile_requested": request_metadata.get("profile_requested"),
            "candles_requested": request_metadata.get("candles_requested"),
            "watch_interval_seconds": request_metadata.get("watch_interval_seconds", 0),
            "data_file": request_metadata.get("data_file"),
            "save_data": request_metadata.get("save_data"),
            "replay_step": request_metadata.get("replay_step"),
            "candle_status": candle_status,
            "change_message": change_message,
            "event_history": request_metadata.get("event_history", []),
            "analysis": {
                **analysis.__dict__,
            },
        }
        if ai_interpretation is not None:
            record["ai_interpretation"] = {
                "market_summary": ai_interpretation.market_summary,
                "setup_explanation": ai_interpretation.setup_explanation,
                "risk_flags": ai_interpretation.risk_flags,
                "action_note": ai_interpretation.action_note,
                "model": ai_interpretation.model,
                "response_id": ai_interpretation.response_id,
            }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def load_recent(
        self,
        *,
        limit: int = 10,
        symbol: str | None = None,
        timeframe: str | None = None,
        profile: str | None = None,
    ) -> list[dict]:
        return load_recent_analysis_history(
            self.path,
            limit=limit,
            symbol=symbol,
            timeframe=timeframe,
            profile=profile,
        )


class JSONLChatHistoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def append_chat_turn(
        self,
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def load_recent(
        self,
        *,
        limit: int = 10,
        symbol: str | None = None,
        timeframe: str | None = None,
        profile: str | None = None,
    ) -> list[dict]:
        return load_recent_chat_history(
            self.path,
            limit=limit,
            symbol=symbol,
            timeframe=timeframe,
            profile=profile,
        )


def build_analysis_history_store(path: str | Path | None) -> AnalysisHistoryStore | None:
    if not path:
        return None
    return JSONLAnalysisHistoryStore(path)


def build_chat_history_store(path: str | Path | None) -> ChatHistoryStore | None:
    if not path:
        return None
    return JSONLChatHistoryStore(path)
