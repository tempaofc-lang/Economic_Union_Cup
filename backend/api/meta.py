"""
================================================================================
模块名称：backend/api/meta.py
接口：GET /api/health, GET /api/vocab, GET /api/session/{id}, POST /api/session/{id}/reset
================================================================================
"""

from __future__ import annotations

from fastapi import APIRouter

from config import APP_VERSION
from schemas import HealthResponse, VocabEntry, VocabResponse
from services import kb as kb_service
from services.agent_service import health_info
from services.session_store import get_turns, reset

# pinyin_codec 由 agent_service 提前塞进 sys.path，这里可以安全导入
import pinyin_codec  # noqa: E402

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse, summary="健康检查")
def get_health() -> HealthResponse:
    info = health_info()
    return HealthResponse(
        ok=True,
        version=APP_VERSION,
        llm_available=info["llm_available"],
        model=info["model"],
        kb_items=info["kb_items"],
        thinking=bool(info.get("thinking", False)),
        default_strategy=str(info.get("default_strategy", "llm")),
        env_file=info.get("env_file"),
        api_key_present=bool(info.get("api_key_present", False)),
    )


@router.get("/vocab", response_model=VocabResponse, summary="当前支持的词表")
def get_vocab() -> VocabResponse:
    vocab = pinyin_codec.load_vocab()
    return VocabResponse(
        version=str(vocab["raw"].get("version", "")),
        confirm_pinyin=vocab["confirm_pinyin"],
        confirm_hanzi=vocab["confirm_hanzi"],
        classes=[VocabEntry(id=e["id"], pinyin=e["pinyin"], hanzi=e["hanzi"])
                 for e in vocab["entries"]],
    )


@router.get("/kb", summary="政务知识库条目（调试用）")
def get_kb():
    return {"items": kb_service.list_items()}


@router.get("/session/{session_id}", summary="读取会话记录")
def get_session(session_id: str):
    return {"session_id": session_id, "turns": get_turns(session_id)}


@router.post("/session/{session_id}/reset", summary="清空会话记录")
def post_reset(session_id: str):
    reset(session_id)
    return {"ok": True, "session_id": session_id}
