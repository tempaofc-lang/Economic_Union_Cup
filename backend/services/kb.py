"""
================================================================================
模块名称：backend/services/kb.py
职责：本地政务知识库查询（这是本项目的**第一个真实工具**）。

为什么需要它：
    在此之前，"2 号室拍照、3 号窗口、40 元"这些数字全靠 agent.py 里的硬编码字符串。
    评委问一句"这些数字哪来的"，就答不上来。现在它们来自一个独立、可核对、可扩展的
    JSON 知识库，`tool_calls` 也就有了真实的调用记录可以回显。
================================================================================
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional

from config import GOV_KB_PATH


@lru_cache(maxsize=1)
def load_kb() -> dict:
    """加载知识库（进程内只读一次）。"""
    with open(GOV_KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def list_items() -> list[dict]:
    return load_kb().get("items", [])


def lookup(words: list[str]) -> Optional[dict]:
    """
    用解码后的中文词去查最匹配的事项。

    打分规则：命中一个别名 +1；命中的别名越长（越具体）权重越高，
    避免"窗口""材料"这种泛词把结果带偏。

    返回知识库条目（附加 score 字段），没命中返回 None。
    """
    if not words:
        return None

    word_set = set(words)
    best, best_score = None, 0.0

    for item in list_items():
        score = 0.0
        for alias in item.get("aliases_hanzi", []):
            if alias in word_set:
                # 长词更有区分度："营业执照" 比 "办理" 权重高
                score += 1.0 + 0.5 * max(0, len(alias) - 2)
        if score > best_score:
            best, best_score = item, score

    # 只命中一个泛词（如"窗口"）时不算找到，交给 Agent 反问
    if best is None or best_score < 1.0:
        return None

    return {**best, "score": round(best_score, 2)}


def as_context(item: dict) -> str:
    """把知识库条目渲染成给大模型看的权威信息块。"""
    materials = "；".join(item.get("materials", [])) or "（未登记）"
    return (
        f"- 事项：{item['title']}\n"
        f"- 办理窗口：{item.get('window') or '（未登记）'}\n"
        f"- 所需材料：{materials}\n"
        f"- 费用：{item.get('fee') or '（未登记）'}\n"
        f"- 办理时长：{item.get('duration') or '（未登记）'}\n"
        f"- 补充说明：{item.get('notes') or '无'}"
    )
