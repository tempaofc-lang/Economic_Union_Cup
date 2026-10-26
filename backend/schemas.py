"""
================================================================================
模块名称：backend/schemas.py
职责：定义 /api/* 的请求与响应模型（pydantic v2）。
      响应结构对齐 05 §3.3 冻结契约，额外字段（decode / elapsed_ms 等）为调试增值项。
================================================================================
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# 通用
# ------------------------------------------------------------------------------
class DecodeItem(BaseModel):
    """单个拼音 token 的解码明细，用于界面展示"可追溯"的解码过程。"""
    raw: str = Field(description="原始拼音，如 shenfenzheng")
    hanzi: Optional[str] = Field(default=None, description="还原出的中文，无法还原时为 null")
    hit: str = Field(description="exact / alias / fuzzy / cjk / unknown")
    score: float = Field(default=0.0, description="匹配置信度，0~1")


class ConfirmCard(BaseModel):
    """结构化办事卡片。字段可能为空，前端需容错。"""
    title: str
    materials: list[str] = Field(default_factory=list)
    window: Optional[str] = None
    fee: Optional[str] = None
    duration: Optional[str] = None


class Clarify(BaseModel):
    """词太少或语义不明时的主动反问。"""
    question: str
    options: list[str] = Field(default_factory=list)


class TokenMapping(BaseModel):
    """
    逐 token 的解译明细，用于界面展示"LLM 到底是怎么理解每个拼音的"。

    source：
      vocab    —— 与词表吻合
      inferred —— LLM 推断（错拼纠正 / 词表外政务词），note 里写了依据
      ignored  —— 判为噪声，已丢弃
      local    —— 规则兜底路径，来自本地查表
    """
    raw: str
    word: Optional[str] = None
    source: str = "vocab"
    note: Optional[str] = None
    local_word: Optional[str] = Field(default=None, description="本地查表结果，用于交叉校验")
    agree: Optional[bool] = Field(default=None, description="与本地查表是否一致；无法比较时为 null")


class ToolCall(BaseModel):
    """工具调用轨迹。界面做成"Agent 决策轨迹"可展开面板。"""
    name: str
    args: dict = Field(default_factory=dict)
    ok: bool = True
    ms: int = 0
    result: Optional[str] = None


# ------------------------------------------------------------------------------
# POST /api/recognize
# ------------------------------------------------------------------------------
class RecognizeRequest(BaseModel):
    text: str = Field(description="手动输入的拼音串，用于模拟 YOLO 识别结果")
    session_id: Optional[str] = None


class Token(BaseModel):
    word: str = Field(description="YOLO 输出的拼音类别名")
    conf: float
    bbox: tuple[int, int, int, int]


class RecognizeResponse(BaseModel):
    tokens: list[Token]
    source: Literal["mock", "model"] = "mock"
    session_id: str
    warning: Optional[str] = Field(
        default=None,
        description="mock 模式固定返回提示，避免把模拟置信度当成真实结果",
    )


# ------------------------------------------------------------------------------
# POST /api/agent
# ------------------------------------------------------------------------------
class AgentRequest(BaseModel):
    words: list[str] = Field(default_factory=list, description="拼音序列（也兼容已解码的中文）")
    session_id: Optional[str] = None
    staff_reply: Optional[str] = Field(default=None, description="窗口工作人员的回复；首轮为 null")
    strategy: Literal["llm", "local"] = Field(
        default="llm",
        description="llm=LLM 主导解码（默认）；local=先用封闭词表查表过滤，再交给 LLM 组句",
    )


class AgentResponse(BaseModel):
    session_id: str

    # —— 输入侧（可追溯） ——
    raw_pinyin: list[str] = Field(default_factory=list)
    words: list[str] = Field(default_factory=list, description="解码后的中文词序列")
    unknown: list[str] = Field(default_factory=list, description="词表外、被忽略的拼音")
    decode: list[DecodeItem] = Field(default_factory=list)
    mapping: list[TokenMapping] = Field(default_factory=list, description="逐 token 解译明细")
    local_words: list[str] = Field(default_factory=list, description="本地查表结果，用于对照")
    strategy: Literal["llm", "local"] = "llm"

    # —— 05 §3.3 冻结契约 ——
    translation: str = ""
    answer: str = ""
    confirm_card: Optional[ConfirmCard] = None
    clarify: Optional[Clarify] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    mode: Literal["llm", "rule"] = "rule"

    # —— 调试增值 ——
    confidence: float = 0.0
    elapsed_ms: int = 0
    cached: bool = Field(default=False, description="是否命中进程内缓存（命中时耗时会接近 0）")
    error: Optional[str] = None
    staff_reply: Optional[str] = Field(
        default=None, description="若本轮携带了工作人员回复，原样回显，供界面渲染双向对话"
    )


# ------------------------------------------------------------------------------
# GET /api/health, GET /api/vocab
# ------------------------------------------------------------------------------
class HealthResponse(BaseModel):
    ok: bool
    version: str
    llm_available: bool
    model: str
    kb_items: int
    thinking: bool = Field(default=False, description="是否开启模型思考模式（关闭时首 token 更快）")
    default_strategy: str = "llm"
    env_file: Optional[str] = Field(default=None, description="实际加载的 .env 路径，null 表示没找到")
    api_key_present: bool = Field(default=False, description="是否读到 DEEPSEEK_API_KEY")


class VocabEntry(BaseModel):
    id: int
    pinyin: str
    hanzi: str


class VocabResponse(BaseModel):
    version: str
    confirm_pinyin: str
    confirm_hanzi: str
    classes: list[VocabEntry]
