"""
================================================================================
模块名称：backend/api/recognize.py
接口：POST /api/recognize —— 视频/拼音串 → token 列表（05 §3.2）

当前是**假视觉**：把手动输入的拼音串包装成带置信度和检测框的 token 列表。
A 组的 best.pt 到手后，把这个文件里的 `recognize()` 换成真实推理即可，
下游 /api/agent 与前端一行都不用改 —— 这就是接口契约的价值。
================================================================================
"""

from __future__ import annotations

from fastapi import APIRouter

from schemas import RecognizeRequest, RecognizeResponse
from services.mock_vision import recognize as mock_recognize
from services.session_store import ensure_session

router = APIRouter(tags=["recognize"])


@router.post("/recognize", response_model=RecognizeResponse, summary="识别（当前为模拟实现）")
def post_recognize(req: RecognizeRequest) -> RecognizeResponse:
    """
    把一段拼音串转成 YOLO 风格的 token 列表：

    ```
    {"text": "nihao shenfenzheng diushi buban"}
    →
    {"tokens": [{"word": "nihao", "conf": 0.93, "bbox": [180, 120, 330, 310]}, ...],
     "source": "mock", "warning": "模拟识别结果..."}
    ```

    在词尾加 `*` 可以模拟低置信度识别（返回 0.62），用来演示阈值过滤与失败兜底。
    """
    sid = ensure_session(req.session_id)
    return mock_recognize(req.text, sid)
