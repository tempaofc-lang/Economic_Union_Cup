"""
================================================================================
模块名称：backend/services/mock_vision.py
职责：假视觉服务——把手动输入的拼音串变成"看起来像 YOLO 输出"的 token 列表。

用途：
    在 A 组的 best.pt 到手之前，让前端、后端、Agent 三方可以并行开发和联调。
    这是 05 §2.2 里 `mock_vision.py` 的职责。

⚠️ 重要：这里产生的置信度是**模拟值**，不是真实模型输出。
    接口会带 warning 字段、界面会打「模拟数据」标签，避免演示时把假数据当成果
    （05 §1 坑 2 明确禁止这件事）。
================================================================================
"""

from __future__ import annotations

import hashlib

from schemas import RecognizeResponse, Token


def _stable_conf(word: str) -> float:
    """
    由词本身算出一个稳定的置信度（0.88~0.97）。
    用哈希而不是 random，是为了同一个词每次得到同样的值，便于对比和复现。
    """
    h = int(hashlib.md5(word.encode("utf-8")).hexdigest()[:6], 16)
    return round(0.88 + (h % 10) / 100.0, 2)


def _stable_bbox(word: str, index: int) -> tuple[int, int, int, int]:
    """生成一个稳定的、落在 760x480 画面内的检测框。"""
    h = int(hashlib.md5(word.encode("utf-8")).hexdigest()[6:12], 16)
    x1 = 120 + (h % 180) + (index % 3) * 8
    y1 = 90 + (h // 7) % 70
    w = 150 + (h // 13) % 80
    hgt = 190 + (h // 17) % 60
    return (x1, y1, min(x1 + w, 750), min(y1 + hgt, 470))


def split_pinyin(text: str) -> list[str]:
    """
    把用户输入的一串拼音拆成 token。
    支持空格、逗号、顿号、换行分隔；支持在词尾加 `*` 表示"模拟低置信度"。
    """
    raw = (text or "").replace(",", " ").replace("，", " ").replace("、", " ")
    raw = raw.replace("\n", " ")
    return [t for t in raw.split() if t]


def recognize(text: str, session_id: str) -> RecognizeResponse:
    """把拼音串变成 token 列表。"""
    words = split_pinyin(text)
    tokens: list[Token] = []

    for i, w in enumerate(words):
        low_conf = w.endswith("*")
        clean = w.rstrip("*")
        tokens.append(Token(
            word=clean,
            conf=0.62 if low_conf else _stable_conf(clean),
            bbox=_stable_bbox(clean, i),
        ))

    return RecognizeResponse(
        tokens=tokens,
        source="mock",
        session_id=session_id,
        warning="模拟识别结果：置信度为占位值，非真实模型输出。带 * 的词会返回 0.62 的低置信度，用于演示阈值过滤。",
    )
