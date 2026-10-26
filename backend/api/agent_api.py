"""
================================================================================
模块名称：backend/api/agent_api.py
接口：POST /api/agent —— 词序列 → 译文 + 导办回答（05 §3.3 冻结契约）
================================================================================
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from schemas import AgentRequest, AgentResponse
from services.agent_service import run_agent

router = APIRouter(tags=["agent"])


@router.post("/agent", response_model=AgentResponse, summary="词序列解译为完整句子")
def post_agent(req: AgentRequest) -> AgentResponse:
    """
    输入拼音序列（或已解码的中文），返回：
    - `translation`：第一人称中文问句
    - `answer`：政务导办回答
    - `confirm_card`：结构化办事卡片（窗口/材料/费用，数据来自本地政务知识库）
    - `clarify`：词太少时的主动反问
    - `tool_calls`：Agent 决策轨迹（目前是 `query_gov_kb`）
    """
    try:
        return run_agent(
            words=req.words,
            session_id=req.session_id,
            staff_reply=req.staff_reply,
        )
    except Exception as exc:  # 兜底：任何内部异常都返回结构化错误，不让前端拿到 500 白页
        raise HTTPException(status_code=500, detail=f"解译失败：{type(exc).__name__}: {exc}")
