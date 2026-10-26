"""
================================================================================
模块名称：backend/services/agent_service.py
职责：把解译引擎（手语识别系统/agent.py v3）适配成 /api/agent 与 /api/agent/stream。

分层：
    api/agent_api.py            ← 只做 HTTP 校验
        ↓
    services/agent_service.py   ← 编排：本地查表(交叉校验) → 查知识库(工具) → 调 LLM → 组装契约
        ↓
    手语识别系统/agent.py        ← LLM 主导的拼音解译 + 流式产出
    backend/services/kb.py      ← 政务知识库工具

两种策略（由请求的 strategy 决定）：
    llm   —— LLM 直接读原始拼音解码（默认）。词表只作参考，错拼/词表外词可以推断。
    local —— 先用封闭词表硬过滤，再交给 LLM 组句（v2 的旧行为，保留用于对比）。
    无论哪种，本地查表都会跑一遍，结果写进 mapping.local_word，用来交叉校验。
================================================================================
"""

from __future__ import annotations

import sys
import time
from typing import Iterator, Optional

from config import LEGACY_DIR

if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

import agent as legacy_agent          # noqa: E402
import pinyin_codec                   # noqa: E402

from schemas import (
    AgentResponse,
    Clarify,
    ConfirmCard,
    DecodeItem,
    TokenMapping,
    ToolCall,
)
from services import kb as kb_service
from services.session_store import append_turn, ensure_session


def _build_decode_detail(token_list: list[str]) -> list[DecodeItem]:
    """本地查表的逐 token 明细（用于交叉校验和界面溯源）。"""
    vocab = pinyin_codec.load_vocab()
    items: list[DecodeItem] = []
    for token in token_list:
        r = pinyin_codec.match_one(token, vocab)
        items.append(
            DecodeItem(raw=str(r["raw"]), hanzi=r["hanzi"], hit=r["hit"], score=float(r["score"]))
        )
    return items


def _build_mapping(detail: dict, token_list: list[str]) -> list[TokenMapping]:
    """
    合并「LLM 的解码结果」与「本地查表结果」，生成可对照的明细表。

    agree 只在两边都有词时才判定，避免"本地压根没匹配上"被误报成冲突。
    """
    vocab = pinyin_codec.load_vocab()
    local_by_raw: dict[str, Optional[str]] = {}
    for token in token_list:
        local_by_raw.setdefault(str(token), pinyin_codec.match_one(token, vocab)["hanzi"])

    rows: list[TokenMapping] = []
    llm_tokens = detail.get("tokens") or []

    if llm_tokens:
        for t in llm_tokens:
            raw = t.get("raw", "")
            word = t.get("word")
            local_word = local_by_raw.get(raw)
            agree = (word == local_word) if (word and local_word) else None
            rows.append(TokenMapping(
                raw=raw, word=word, source=t.get("source", "vocab"),
                note=t.get("note"), local_word=local_word, agree=agree,
            ))
    else:
        # 规则兜底路径：没有 LLM 解码结果，用本地查表顶替
        for token in token_list:
            hit = pinyin_codec.match_one(token, vocab)
            rows.append(TokenMapping(
                raw=str(token), word=hit["hanzi"], source="local",
                note=None, local_word=hit["hanzi"], agree=True,
            ))

    return rows


def _merge_card(llm_card: Optional[dict], kb_item: Optional[dict]) -> Optional[ConfirmCard]:
    """办事卡片以知识库为准：模型只贡献措辞，窗口/材料/费用由知识库提供。"""
    if kb_item:
        return ConfirmCard(
            title=kb_item.get("title", ""),
            materials=list(kb_item.get("materials", [])),
            window=kb_item.get("window"),
            fee=kb_item.get("fee"),
            duration=kb_item.get("duration"),
        )
    if not llm_card:
        return None
    return ConfirmCard(
        title=str(llm_card.get("title") or "业务办理"),
        materials=list(llm_card.get("materials") or []),
        window=llm_card.get("window"),
        fee=llm_card.get("fee"),
        duration=None,
    )


def _lookup_kb(token_list: list[str]) -> tuple[Optional[dict], list[ToolCall], str]:
    """工具调用：查政务知识库。返回 (条目, 轨迹, 注入给模型的上下文)。"""
    calls: list[ToolCall] = []
    decoded_words = pinyin_codec.decode_tokens(token_list)["words"]

    t0 = time.perf_counter()
    try:
        item = kb_service.lookup(decoded_words)
        calls.append(ToolCall(
            name="query_gov_kb",
            args={"words": decoded_words},
            ok=item is not None,
            ms=int((time.perf_counter() - t0) * 1000),
            result=item["title"] if item else "未命中知识库",
        ))
        context = kb_service.as_context(item) if item else ""
        return item, calls, context
    except Exception as exc:  # 知识库坏了不能拖垮整条链路
        calls.append(ToolCall(
            name="query_gov_kb", args={"words": decoded_words}, ok=False,
            ms=int((time.perf_counter() - t0) * 1000), result=f"知识库调用失败：{exc}",
        ))
        return None, calls, ""


def _assemble(detail: dict, sid: str, token_list: list[str], kb_item: Optional[dict],
              tool_calls: list[ToolCall], strategy: str, staff_reply: Optional[str],
              t0: float) -> AgentResponse:
    """把引擎返回值组装成 05 §3.3 契约 + 调试字段。"""
    clarify = None
    if detail.get("clarify"):
        clarify = Clarify(
            question=str(detail["clarify"].get("question", "")),
            options=list(detail["clarify"].get("options") or []),
        )

    return AgentResponse(
        session_id=sid,
        raw_pinyin=token_list,
        words=detail.get("words") or [],
        unknown=detail.get("unknown_pinyin") or [],
        decode=_build_decode_detail(token_list),
        mapping=_build_mapping(detail, token_list),
        local_words=detail.get("local_words") or [],
        strategy=strategy,
        translation=detail.get("translation", ""),
        answer=detail.get("answer", ""),
        confirm_card=_merge_card(detail.get("confirm_card"), kb_item),
        clarify=clarify,
        tool_calls=tool_calls,
        mode="llm" if detail.get("source") == "llm" else "rule",
        confidence=float(detail.get("confidence") or 0.0),
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
        cached=bool(detail.get("cached", False)),
        error=detail.get("error"),
        staff_reply=staff_reply,
    )


def _record(sid: str, resp: AgentResponse) -> None:
    append_turn(sid, {
        "raw_pinyin": resp.raw_pinyin,
        "words": resp.words,
        "translation": resp.translation,
        "answer": resp.answer,
        "mode": resp.mode,
        "strategy": resp.strategy,
        "elapsed_ms": resp.elapsed_ms,
    })


# ==============================================================================
# 非流式
# ==============================================================================
def run_agent(words: list[str], session_id: Optional[str] = None,
              staff_reply: Optional[str] = None,
              strategy: str = legacy_agent.DEFAULT_STRATEGY) -> AgentResponse:
    t0 = time.perf_counter()
    sid = ensure_session(session_id)

    token_list = [str(w).strip() for w in (words or []) if str(w).strip()]
    kb_item, tool_calls, kb_context = _lookup_kb(token_list)

    detail = legacy_agent.parse_semantics_detail(
        token_list, kb_context=kb_context, strategy=strategy
    )
    resp = _assemble(detail, sid, token_list, kb_item, tool_calls, strategy, staff_reply, t0)
    _record(sid, resp)
    return resp


# ==============================================================================
# 流式（SSE）
# ==============================================================================
def stream_agent(words: list[str], session_id: Optional[str] = None,
                 staff_reply: Optional[str] = None,
                 strategy: str = legacy_agent.DEFAULT_STRATEGY) -> Iterator[dict]:
    """
    逐条产出 SSE 事件字典：
        {"type": "meta",  ...}    会话、知识库命中、本地查表结果（可先渲染左侧）
        {"type": "delta", "text": "..."}   模型输出增量（原始 JSON 文本片段）
        {"type": "error", "message": "..."}
        {"type": "final", "data": {...}}   完整 AgentResponse
    """
    t0 = time.perf_counter()
    sid = ensure_session(session_id)

    token_list = [str(w).strip() for w in (words or []) if str(w).strip()]
    kb_item, tool_calls, kb_context = _lookup_kb(token_list)

    yield {
        "type": "meta",
        "session_id": sid,
        "strategy": strategy,
        "raw_pinyin": token_list,
        "local_words": pinyin_codec.decode_tokens(token_list)["words"] if token_list else [],
        "decode": [d.model_dump() for d in _build_decode_detail(token_list)],
        "tool_calls": [t.model_dump() for t in tool_calls],
    }

    final_detail = None
    for kind, payload in legacy_agent.stream_semantics_detail(
            token_list, kb_context=kb_context, strategy=strategy):
        if kind == "meta":
            # 引擎侧的 meta 与这里重复，忽略（我们已经发过）
            continue
        if kind == "delta":
            yield {"type": "delta", "text": payload.get("text", "")}
        elif kind == "error":
            yield {"type": "error", "message": payload.get("message", "未知错误")}
        elif kind == "final":
            final_detail = payload

    if final_detail is None:
        final_detail = {
            "tokens": [], "words": [], "translation": "", "answer": "",
            "confirm_card": None, "clarify": None, "confidence": 0.0,
            "source": "rule", "cached": False, "error": "流式未产出结果",
            "raw_pinyin": token_list, "local_words": [], "unknown_pinyin": [],
        }

    resp = _assemble(final_detail, sid, token_list, kb_item, tool_calls, strategy, staff_reply, t0)
    _record(sid, resp)
    yield {"type": "final", "data": resp.model_dump()}


def health_info() -> dict:
    diag = legacy_agent.env_diagnostics()
    return {
        "llm_available": legacy_agent.llm_available(),
        "model": legacy_agent.DEEPSEEK_MODEL,
        "thinking": legacy_agent.LLM_THINKING,
        "default_strategy": legacy_agent.DEFAULT_STRATEGY,
        "kb_items": len(kb_service.list_items()),
        "env_file": diag["env_file"],
        "api_key_present": diag["api_key_present"],
        "decode": "拼音解码：LLM 主导，本地查表仅作交叉校验",
    }
