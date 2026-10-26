"""
================================================================================
模块名称：backend/services/agent_service.py
职责：把现有的解译引擎（手语识别系统/agent.py）适配成 /api/agent 的响应。

分层：
    api/agent_api.py   ← 只做 HTTP 校验
        ↓
    services/agent_service.py  ← 编排：解码 → 查知识库（工具）→ 调 LLM → 组装契约
        ↓
    手语识别系统/pinyin_codec.py（拼音→中文）
    手语识别系统/agent.py（DeepSeek 连词成句）
    backend/services/kb.py（政务知识库工具）

为什么先 import 而不是把代码搬进 backend/：
    05 §2.2 的目标结构是把引擎迁到 backend/agent/ 下，但那是一次性重构；
    当前阶段先"接线"，保证接口能跑通，迁移留到接口稳定之后再做，避免边改结构边调逻辑。
================================================================================
"""

from __future__ import annotations

import sys
import time
from typing import Optional

from config import LEGACY_DIR

# 把现有引擎所在目录加入模块搜索路径，然后按顶层模块导入
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

import agent as legacy_agent          # noqa: E402  手语识别系统/agent.py
import pinyin_codec                   # noqa: E402  手语识别系统/pinyin_codec.py

from schemas import AgentResponse, ConfirmCard, Clarify, DecodeItem, ToolCall
from services import kb as kb_service
from services.session_store import append_turn, ensure_session


HIT_LABEL = {
    "exact": "精确匹配",
    "alias": "别名匹配",
    "fuzzy": "模糊匹配",
    "cjk": "已是中文",
    "unknown": "词表外",
}


def _build_decode_detail(token_list: list[str]) -> list[DecodeItem]:
    """逐 token 的解码明细，供界面展示"可追溯"的解码过程。"""
    vocab = pinyin_codec.load_vocab()
    items: list[DecodeItem] = []
    for token in token_list:
        r = pinyin_codec.match_one(token, vocab)
        items.append(
            DecodeItem(
                raw=str(r["raw"]),
                hanzi=r["hanzi"],
                hit=r["hit"],
                score=float(r["score"]),
            )
        )
    return items


def _merge_card(llm_card: Optional[dict], kb_item: Optional[dict]) -> Optional[ConfirmCard]:
    """
    办事卡片以知识库为准。
    知识库命中时，窗口/材料/费用/时长直接覆盖模型输出——这些是"有出处的数字"；
    模型只贡献措辞（translation/answer）。
    """
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


def run_agent(words: list[str], session_id: Optional[str] = None,
              staff_reply: Optional[str] = None) -> AgentResponse:
    """执行一次完整的解译流程。"""
    t0 = time.perf_counter()
    sid = ensure_session(session_id)

    token_list = [str(w).strip() for w in (words or []) if str(w).strip()]
    tool_calls: list[ToolCall] = []

    # ---- ① 拼音解码（确定性，本地） ----
    decoded = pinyin_codec.decode_tokens(token_list)
    hanzi_words = decoded["words"]

    # ---- ② 工具：查政务知识库 ----
    kb_item = None
    decode_ms = int((time.perf_counter() - t0) * 1000)
    t_kb = time.perf_counter()
    try:
        kb_item = kb_service.lookup(hanzi_words)
        ok = kb_item is not None
        tool_calls.append(ToolCall(
            name="query_gov_kb",
            args={"words": hanzi_words},
            ok=ok,
            ms=int((time.perf_counter() - t_kb) * 1000),
            result=kb_item["title"] if kb_item else "未命中知识库",
        ))
    except Exception as exc:  # 知识库坏了不能拖垮整条链路
        tool_calls.append(ToolCall(
            name="query_gov_kb",
            args={"words": hanzi_words},
            ok=False,
            ms=int((time.perf_counter() - t_kb) * 1000),
            result=f"知识库调用失败：{exc}",
        ))

    # ---- ③ 大模型解译（把知识库信息一并注入） ----
    kb_context = kb_service.as_context(kb_item) if kb_item else ""
    detail = legacy_agent.parse_semantics_detail(token_list, kb_context=kb_context)

    # ---- ④ 组装 05 §3.3 契约 ----
    clarify = None
    if detail.get("clarify"):
        clarify = Clarify(
            question=str(detail["clarify"].get("question", "")),
            options=list(detail["clarify"].get("options") or []),
        )

    response = AgentResponse(
        session_id=sid,
        raw_pinyin=token_list,
        words=hanzi_words,
        unknown=decoded["unknown"],
        decode=_build_decode_detail(token_list),
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

    append_turn(sid, {
        "raw_pinyin": token_list,
        "words": hanzi_words,
        "translation": response.translation,
        "answer": response.answer,
        "mode": response.mode,
        "elapsed_ms": response.elapsed_ms,
    })

    return response


def health_info() -> dict:
    """给 /api/health 用的运行时信息。"""
    return {
        "llm_available": legacy_agent.llm_available(),
        "model": legacy_agent.DEEPSEEK_MODEL,
        "kb_items": len(kb_service.list_items()),
        "decode": "拼音解码为本地确定性算法（不经网络）",
    }
