from __future__ import annotations

from typing import Any

from app.stores import ChatHistoryStore


def build_chat_memory_context(
    *,
    chat_store: ChatHistoryStore | None,
    session_messages: list[dict[str, Any]] | None,
    symbol: str,
    timeframe: str,
    profile: str,
    session_limit: int = 6,
    persisted_limit: int = 5,
) -> dict:
    return {
        "recent_session_messages": _compact_session_messages(
            session_messages or [],
            limit=session_limit,
        ),
        "recent_persisted_turns": _load_persisted_turns(
            chat_store=chat_store,
            symbol=symbol,
            timeframe=timeframe,
            profile=profile,
            limit=persisted_limit,
        ),
    }


def _compact_session_messages(messages: list[dict[str, Any]], *, limit: int) -> list[dict[str, str]]:
    compacted: list[dict[str, str]] = []
    for message in messages[-limit:]:
        role = str(message.get("role", "")).strip()
        content = _clean_message_content(str(message.get("content", "")).strip())
        if not role or not content:
            continue
        compacted.append({"role": role, "content": content})
    return compacted


def _load_persisted_turns(
    *,
    chat_store: ChatHistoryStore | None,
    symbol: str,
    timeframe: str,
    profile: str,
    limit: int,
) -> list[dict[str, str | None]]:
    if chat_store is None or limit <= 0:
        return []

    turns = chat_store.load_recent(
        limit=limit,
        symbol=symbol,
        timeframe=timeframe,
        profile=profile,
    )
    compacted: list[dict[str, str | None]] = []
    for turn in turns:
        compacted.append(
            {
                "logged_at": turn.get("logged_at"),
                "question": str(turn.get("question", "")).strip(),
                "answer": _clean_message_content(str(turn.get("answer", "")).strip()),
            }
        )
    return compacted


def _clean_message_content(content: str) -> str:
    marker = "\n\nModelo:"
    if marker in content:
        return content.split(marker, 1)[0].strip()
    return content.strip()
