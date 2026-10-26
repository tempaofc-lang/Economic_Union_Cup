"""
================================================================================
模块名称：backend/api/agent_api.py
接口：
    POST /api/agent         非流式，一次性返回完整结果
    POST /api/agent/stream  流式（SSE），边生成边推，用于实时渲染
================================================================================
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from schemas import AgentRequest, AgentResponse
from services.agent_service import run_agent, stream_agent

router = APIRouter(tags=["agent"])


@router.post("/agent", response_model=AgentResponse, summary="词序列解译为完整句子（非流式）")
def post_agent(req: AgentRequest) -> AgentResponse:
    """
    输入拼音序列（或已解码的中文），返回译文、导办回答、办事卡片、反问、工具轨迹。

    `strategy`：
    - `llm`（默认）LLM 直接解码原始拼音，可纠正错拼、推断词表外词；
    - `local` 先用封闭词表过滤，再交给 LLM 组句（用于对比两种策略的差异）。
    """
    try:
        return run_agent(
            words=req.words,
            session_id=req.session_id,
            staff_reply=req.staff_reply,
            strategy=req.strategy,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"解译失败：{type(exc).__name__}: {exc}")


def _sse(obj: dict) -> str:
    """打包一条 SSE 事件。"""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@router.post("/agent/stream", summary="词序列解译（SSE 流式，实时输出）")
def post_agent_stream(req: AgentRequest) -> StreamingResponse:
    """
    与 `/api/agent` 输入相同，但以 SSE 逐条推送：

    | 事件 | 内容 |
    | :-- | :-- |
    | `meta` | 会话 id、本地查表结果、知识库命中、工具轨迹（可先渲染输入侧） |
    | `delta` | 模型输出增量文本（`{"text": "..."}`） |
    | `error` | 出错提示 |
    | `final` | 完整结果，结构与非流式接口一致 |

    前端用 fetch + ReadableStream 读，按 `\\n\\n` 切帧即可。
    """
    generator = stream_agent(
        words=req.words,
        session_id=req.session_id,
        staff_reply=req.staff_reply,
        strategy=req.strategy,
    )

    def event_stream():
        try:
            for event in generator:
                yield _sse(event)
        except Exception as exc:  # 任何意外都以 event 形式告知前端，避免连接静默断开
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # 关掉反向代理缓冲，否则流式会被攒成一坨
        },
    )
