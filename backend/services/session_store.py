"""
================================================================================
模块名称：backend/services/session_store.py
职责：极简的进程内会话存储（够用即可，不上数据库）。

存什么：每个 session 的对话轮次，用于界面右侧"本次会话记录"和工作人员回复。
限制：进程重启即清空；多人并发演示完全足够。真要持久化再换 SQLite。
================================================================================
"""

from __future__ import annotations

import threading
import time
import uuid

_LOCK = threading.Lock()
_SESSIONS: dict[str, dict] = {}

MAX_TURNS = 50


def ensure_session(session_id: str | None) -> str:
    """没有 session_id 就新建一个，有就复用。"""
    with _LOCK:
        sid = (session_id or "").strip()
        if not sid:
            sid = f"s-{uuid.uuid4().hex[:8]}"
        if sid not in _SESSIONS:
            _SESSIONS[sid] = {"created_at": time.time(), "turns": []}
        return sid


def append_turn(session_id: str, turn: dict) -> None:
    with _LOCK:
        session = _SESSIONS.setdefault(session_id, {"created_at": time.time(), "turns": []})
        turn = {**turn, "ts": time.strftime("%H:%M:%S")}
        session["turns"].append(turn)
        if len(session["turns"]) > MAX_TURNS:
            session["turns"] = session["turns"][-MAX_TURNS:]


def get_turns(session_id: str) -> list[dict]:
    with _LOCK:
        return list(_SESSIONS.get(session_id, {}).get("turns", []))


def reset(session_id: str) -> None:
    with _LOCK:
        _SESSIONS.pop(session_id, None)
