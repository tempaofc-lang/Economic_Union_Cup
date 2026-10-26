"""
================================================================================
模块名称：agent.py（Agent 智能体 · 拼音解译引擎）
负责人：技术 B 组 - 古宇轩
版本：v3.0（2026-10-22 晚间重写）

【v3 相对 v2 的关键变化：把"解码"交给 LLM】
    v2 的问题：拼音→中文 由 pinyin_codec 的封闭词表查表完成，LLM 只负责把已解码的词连成句子。
      后果是——词表外的拼写、错拼、同音混淆在进入 LLM 之前就被丢弃了，
      LLM 根本没有机会做模糊匹配，整条链路看起来就是"查表 + 润色"。

    v3 的做法：原始拼音**直接交给 LLM 解码**，词表降级为"参考基线"而不是硬闸门。
      · LLM 可以修正错拼（shenfenzeng → 身份证）、按上下文消歧、推断词表外的政务词；
      · 每个拼音都必须有交代：vocab（词表命中）/ inferred（推断，需写理由）/ ignored（噪声）；
      · 本地查表结果仍会跑一遍，但只用于**交叉校验**，结果以 conflicts 字段暴露出来。

【三种解译策略】
    strategy="llm"   默认。LLM 主导解码 + 成句 + 导办。
    strategy="local" 兼容旧行为：先用封闭词表硬过滤，再交给 LLM 组句。
    两者输出结构完全一致，方便同屏对比、也方便答辩时解释差异。

【流式输出】
    stream_semantics_detail() 逐块产出模型输出，供 /api/agent/stream 做 SSE 实时渲染。
    流式时默认关闭思考模式（thinking=disabled）：首 token 从约 2.5s 降到约 0.8s。

【对外接口（保持向后兼容）】
    parse_semantics(word_list) -> (translation, answer)        # app.py 用
    parse_semantics_detail(token_list, ...) -> dict            # backend / 测试用
    stream_semantics_detail(token_list, ...) -> generator      # 流式
================================================================================
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterator

import requests

_MODULE_DIR = Path(__file__).resolve().parent


def _load_env_explicit() -> str | None:
    """
    显式加载 .env。

    为什么不直接用 load_dotenv()：
        python-dotenv 的 find_dotenv() 依赖调用栈和当前工作目录。当 uvicorn 从
        backend/ 启动、或用的是另一个 Python 环境时，它可能找不到
        手语识别系统/.env —— 而且**失败时不报错**，表现就是"LLM 不可用"，
        排查起来极其费时（我们已经踩过一次）。

    所以这里按固定候选路径自己读，有 python-dotenv 时额外再调一次（无害）。
    """
    candidates = [
        _MODULE_DIR / ".env",                       # 手语识别系统/.env（正常情况）
        Path.cwd() / ".env",                        # 当前工作目录
        Path.cwd().parent / "手语识别系统" / ".env",  # 从 backend/ 启动时
        _MODULE_DIR.parent / ".env",                # 项目根目录
    ]

    loaded_from = None
    for path in candidates:
        try:
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                # 不覆盖已有环境变量，方便部署时用系统变量优先
                if key and key not in os.environ:
                    os.environ[key] = value
            loaded_from = str(path)
            break
        except Exception:
            continue

    # 有 python-dotenv 就再兜一层（不覆盖已设置的值）
    try:
        from dotenv import load_dotenv
        load_dotenv(candidates[0], override=False)
    except Exception:
        pass

    return loaded_from


ENV_FILE_LOADED = _load_env_explicit()

from pinyin_codec import decode_tokens, load_vocab

# ==============================================================================
# 🔧 开关与配置（全部可用环境变量覆盖）
# ==============================================================================
USE_REAL_LLM_API = os.getenv("USE_REAL_LLM_API", "1") == "1"   # 置 0 则只用本地规则引擎
ENABLE_CACHE = True                                             # 非流式调用可缓存

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
LLM_MAX_RETRIES = 1
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 700

# 思考模式：0 = 关闭（默认，首 token 快 3 倍）｜1 = 打开（更谨慎，但慢）
LLM_THINKING = os.getenv("LLM_THINKING", "0") == "1"

STRATEGY_LLM = "llm"
STRATEGY_LOCAL = "local"
DEFAULT_STRATEGY = STRATEGY_LLM

_DEFAULT_CONFIDENCE = 0.6

__all__ = [
    "parse_semantics",
    "parse_semantics_detail",
    "stream_semantics_detail",
    "llm_available",
    "SYSTEM_PROMPT_TEMPLATE",
    "STRATEGY_LLM",
    "STRATEGY_LOCAL",
]


# ==============================================================================
# 一、System Prompt v3 —— 让模型明白"理解拼音"是它的第一职责
# ==============================================================================
SYSTEM_PROMPT_TEMPLATE = """你是「手语视界」政务服务大厅的语义解译引擎，服务对象是听障办事群众。

# 一、你的第一职责：理解这串拼音
用户会给你一串视觉识别模块输出的**无声调拼音**，例如：
    shenfenzeng diushi buban

你的任务是**判断这串拼音想表达什么政务诉求**，而不是去词表里做查表匹配。
词表只是参考基线；查表是备用手段，不是目的。

# 二、拼音的已知缺陷（你自己纠正，不要向用户抱怨）
1. **无声调**：shi 可以是 是/事/十/时/市/识，必须靠上下文判断。
2. **有拼写错误**：可能漏字母、多字母、n/l 不分、zh/z ch/c sh/s 不分、前后鼻音混淆。
   例如 shenfenzeng 很可能是 shenfenzheng（身份证）。
3. **连写分写不统一**：shenfenzheng / shen fen zheng / shen_fen_zheng 是同一个词。
4. **语序是手语语序**：话题在前、动作在后（"身份证 补办"而不是"补办身份证"）。
5. **有重复**：同一个手势可能被连续识别多次（buban buban buban）。
6. **有噪声**：可能混入与本次办事无关的孤立拼音。

# 三、已知词表（参考基线，不是硬性边界）
__VOCAB_LINES__

怎么用这张表：
- 拼音与表中某一项吻合（含去声调、去空格、常见拼写差异）→ `source` 标 `"vocab"`。
- 拼音不在表里，但结合其他词能在政务语境下合理推断 → `source` 标 `"inferred"`，并在 `note` 里写清推断依据。
  **允许你推断，但必须留下痕迹**，不要假装它是词表命中的。
- 拼音明显是噪声（如 tiyuguan 体育馆，与办事无关）→ `source` 标 `"ignored"`。
- `queren` 是"确认"手势，属于**控制信号**，触发成句但不参与语义，不要放进 words。

# 四、完成解码后，继续做三件事
1. **成句**：把词按汉语习惯重排，补全省略的主语，写成一句第一人称、自然礼貌的中文问句。
   可以补"我""请问"这类衔接词，但**严禁补充原词里没有的业务事项**（没说预约就不能写"我已预约"）。
2. **导办**：以政务引导员口吻回答，必须给出具体窗口 + 具体材料；不确定的费用不写，**绝不编造**。
   回答控制在 **80 字以内**（要放大显示在群众屏幕上）。
3. **反问**：有效词少于 2 个、或判断不出办什么业务时，不要硬猜，用 clarify 主动反问并给 2~3 个选项。
   格式 `{"question": "一句反问", "options": ["选项1", "选项2"]}`，其余情况填 null。

# 五、输出格式
只输出一个 json 对象，不要 markdown 代码块，不要任何解释文字：

{
  "tokens": [
    {"raw": "原始拼音", "word": "还原的中文，忽略时填 null",
     "source": "vocab | inferred | ignored", "note": "inferred 时写推断依据，其余留空"}
  ],
  "words": ["按语序排好的中文词"],
  "translation": "第一人称中文问句",
  "answer": "导办回答（不要加【政务导办回答】前缀）",
  "confirm_card": {"title": "业务名称", "materials": ["材料"], "window": "窗口", "fee": "费用或 null"} 或 null,
  "clarify": {"question": "反问", "options": ["选项"]} 或 null,
  "confidence": 0.0
}

- `tokens` 必须覆盖输入的**每一个**拼音，顺序与输入一致，不许漏。
- `confidence` 是你对 translation 忠实度的自评（0~1）；有 inferred 的词就适当降低。

# 六、示例

示例 1（标准场景，全部词表命中）
输入：nihao shenfenzheng diushi buban buban
输出：{"tokens":[{"raw":"nihao","word":"你好","source":"vocab","note":""},{"raw":"shenfenzheng","word":"身份证","source":"vocab","note":""},{"raw":"diushi","word":"丢失","source":"vocab","note":""},{"raw":"buban","word":"补办","source":"vocab","note":""},{"raw":"buban","word":"补办","source":"ignored","note":"与上一个重复"}],"words":["你好","身份证","丢失","补办"],"translation":"工作人员您好，我的身份证丢失了，请问要怎么补办？","answer":"请携带居民户口簿原件，先到 2 号人像采集室免费拍照，再到 3 号户政窗口受理，工本费 40 元。","confirm_card":{"title":"身份证补办","materials":["居民户口簿原件"],"window":"3 号户政窗口","fee":"40 元"},"clarify":null,"confidence":0.95}

示例 2（拼写错误 + 词表外推断）
输入：shenfenzeng diushi le buban
输出：{"tokens":[{"raw":"shenfenzeng","word":"身份证","source":"inferred","note":"标准拼写应为 shenfenzheng，少了一个 h，结合丢失补办语境推断"},{"raw":"diushi","word":"丢失","source":"vocab","note":""},{"raw":"le","word":"了","source":"inferred","note":"语气助词，非词表项"},{"raw":"buban","word":"补办","source":"vocab","note":""}],"words":["身份证","丢失","了","补办"],"translation":"我的身份证丢失了，请问要怎么补办？","answer":"请携带居民户口簿原件到 3 号户政窗口办理补办。","confirm_card":{"title":"身份证补办","materials":["居民户口簿原件"],"window":"3 号户政窗口","fee":null},"clarify":null,"confidence":0.78}

示例 3（噪声 + 泛词 -> 反问）
输入：chuangkou tiyuguan
输出：{"tokens":[{"raw":"chuangkou","word":"窗口","source":"vocab","note":""},{"raw":"tiyuguan","word":null,"source":"ignored","note":"体育馆，与政务办事无关"}],"words":["窗口"],"translation":"您好，请问我要去哪个窗口办理？","answer":"请告诉我您要办理的业务，我为您指引对应窗口。","confirm_card":null,"clarify":{"question":"请问您要办理哪项业务？","options":["身份证补办","社保卡激活","营业执照办理"]},"confidence":0.3}

# 七、硬性约束（违反即视为失败）
1. 必须是**合法 json**，且只包含 tokens / words / translation / answer / confirm_card / clarify / confidence 七个键。
2. answer 不得以"【政务导办回答】"开头，该前缀由系统统一添加。
3. 不得输出政治敏感、医疗诊断、法律结论等超出政务导办范围的内容。
4. 若输入完全无法理解，translation 固定写"您好，我没能完整表达，请再问我一次。"，confidence 低于 0.3。
"""


def _build_vocab_lines() -> str:
    """把词表渲染成对照表，并标注"确认"是控制信号。"""
    vocab = load_vocab()
    lines = []
    for e in vocab["entries"]:
        if e["hanzi"] == vocab["confirm_hanzi"]:
            continue
        lines.append(f"- `{e['pinyin']}` = {e['hanzi']}")
    lines.append(f"- `{vocab['confirm_pinyin']}` = {vocab['confirm_hanzi']}（控制信号，见下）")
    return "\n".join(lines)


# 用占位符替换而不是 str.format()，避免 prompt 里的 JSON 花括号要写成 {{ }}
_SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.replace("__VOCAB_LINES__", _build_vocab_lines())


# ==============================================================================
# 二、调用大模型
# ==============================================================================
def llm_available() -> bool:
    return bool(USE_REAL_LLM_API and DEEPSEEK_API_KEY)


def env_diagnostics() -> dict:
    """
    环境诊断：把"为什么 LLM 不可用"变成可查的事实，而不是靠猜。
    /api/health 会返回这组信息。
    """
    return {
        "env_file": ENV_FILE_LOADED,
        "api_key_present": bool(DEEPSEEK_API_KEY),
        "use_real_llm": USE_REAL_LLM_API,
        "thinking": LLM_THINKING,
        "model": DEEPSEEK_MODEL,
    }


if not DEEPSEEK_API_KEY:
    print(
        "[agent] ⚠️ 未读到 DEEPSEEK_API_KEY：LLM 解译不可用，将只用本地规则引擎。\n"
        f"        已尝试的 .env 候选路径均未命中（最近一次：{ENV_FILE_LOADED or '无'}）\n"
        f"        请确认 {_MODULE_DIR / '.env'} 存在且含 DEEPSEEK_API_KEY=sk-..."
    )


def _request_body(messages: list, stream: bool) -> dict:
    body = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "stream": stream,
    }
    # 关闭思考模式可把首 token 延迟从 ~2.5s 降到 ~0.8s，实时演示必须关
    body["thinking"] = {"type": "enabled" if LLM_THINKING else "disabled"}
    return body


def _headers() -> dict:
    return {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}


def _call_deepseek(messages: list) -> str:
    """一次性调用，返回完整 content。失败抛异常。"""
    resp = requests.post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers=_headers(),
        json=_request_body(messages, stream=False),
        timeout=LLM_TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

    content = (resp.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
    if not content or not content.strip():
        raise RuntimeError("模型返回空内容")
    return content


def _stream_deepseek(messages: list) -> Iterator[str]:
    """流式调用，逐块 yield 文本增量。"""
    with requests.post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers=_headers(),
        json=_request_body(messages, stream=True),
        timeout=LLM_TIMEOUT,
        stream=True,
    ) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

        got_any = False
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            payload = raw_line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue

            choices = obj.get("choices") or [{}]
            delta = (choices[0].get("delta") or {}).get("content") or ""
            if delta:
                got_any = True
                yield delta

        if not got_any:
            raise RuntimeError("流式返回为空")


# ==============================================================================
# 三、结果解析与清洗
# ==============================================================================
def _parse_json_loose(text: str) -> dict:
    """宽容解析：剥 ```json 围栏、截取首个 { 到末个 }。"""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("无法从模型输出中解析出 JSON")


def _to_confidence(value) -> float:
    """宽容解析 confidence：null / "95%" / "0.9（较高）" 都不能让整条结果作废。"""
    if value is None:
        return _DEFAULT_CONFIDENCE
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))

    text = str(value).strip()
    if not text:
        return _DEFAULT_CONFIDENCE
    percent = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if percent:
        return max(0.0, min(1.0, float(percent.group(1)) / 100))
    number = re.search(r"\d+(?:\.\d+)?", text)
    if number:
        v = float(number.group())
        return max(0.0, min(1.0, v / 100 if v > 1 else v))
    return _DEFAULT_CONFIDENCE


def _to_str_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        value = re.split(r"[、,，;；\n]+", value)
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _normalize_confirm_card(data: dict):
    card = data.get("confirm_card")
    if not isinstance(card, dict):
        return None
    title = str(card.get("title") or "").strip()
    window = str(card.get("window") or "").strip()
    fee = card.get("fee")
    fee = str(fee).strip() if fee not in (None, "", "null") else None
    materials = _to_str_list(card.get("materials"))
    if not (title or window or materials):
        return None
    return {"title": title or "业务办理", "materials": materials,
            "window": window or None, "fee": fee}


def _normalize_clarify(data: dict):
    clarify = data.get("clarify")
    if not isinstance(clarify, dict):
        return None
    question = str(clarify.get("question") or "").strip()
    if not question:
        return None
    return {"question": question, "options": _to_str_list(clarify.get("options"))[:4]}


def _normalize_tokens(data: dict, raw_tokens: list) -> list:
    """
    规整模型给出的逐 token 解码结果。
    若模型没给 tokens（或给了个空的），返回空列表，由上层回退到本地解码结果。
    """
    items = data.get("tokens")
    if not isinstance(items, list) or not items:
        return []

    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        word = it.get("word")
        word = str(word).strip() if word not in (None, "", "null") else None
        source = str(it.get("source") or "").strip().lower()
        if source not in {"vocab", "inferred", "ignored"}:
            source = "vocab" if word else "ignored"
        out.append({
            "raw": str(it.get("raw") or "").strip(),
            "word": word,
            "source": source,
            "note": str(it.get("note") or "").strip() or None,
        })
    return out


def _words_from_tokens(tokens: list) -> list:
    """从 token 映射里提取词序列：去 ignored、去空、去连续重复。"""
    words = []
    for t in tokens:
        if t["source"] == "ignored" or not t["word"]:
            continue
        if words and words[-1] == t["word"]:
            continue
        words.append(t["word"])
    return words


def _normalize_result(data: dict, raw_tokens: list) -> dict:
    def as_text(key: str) -> str:
        v = data.get(key, "")
        return str(v).strip() if v is not None else ""

    translation = re.sub(r"^\s*(【手语翻译】|手语翻译[:：]|翻译[:：])\s*", "", as_text("translation"))
    answer = re.sub(r"^\s*【政务导办回答】\s*", "", as_text("answer"))

    tokens = _normalize_tokens(data, raw_tokens)
    words = _words_from_tokens(tokens) or _to_str_list(data.get("words"))

    if not translation:
        raise ValueError("translation 为空")
    if len(translation) > 160:
        translation = translation[:160].rstrip("，。、") + "。"
    if len(answer) > 150:
        cut = answer[:150]
        answer = cut[:cut.rfind("。") + 1] if "。" in cut else cut.rstrip("，、；") + "。"
    if not answer:
        answer = "请携带本人有效身份证件前往一号综合导办台取号办理。"

    return {
        "tokens": tokens,
        "words": words,
        "ignored": [t["raw"] for t in tokens if t["source"] == "ignored"],
        "translation": translation,
        "answer": answer,
        "confirm_card": _normalize_confirm_card(data),
        "clarify": _normalize_clarify(data),
        "confidence": _to_confidence(data.get("confidence")),
    }


# ==============================================================================
# 四、本地规则引擎（兜底）
# ==============================================================================
def _rule_based(words: list) -> dict:
    s = set(words)

    if {"身份证", "补办"} & s or {"身份证", "丢失"} & s:
        return {"tokens": [], "words": words, "ignored": [],
                "translation": "工作人员您好，我的身份证丢失了，请问怎么办理补办？",
                "answer": "补办身份证请携带户口本或驾驶证原件，先到 2 号人像采集室免费拍照，"
                          "随后前往 3 号户政窗口办理，工本费 40 元。",
                "confirm_card": None, "clarify": None, "confidence": 0.6}

    if {"社保卡", "医保", "激活"} & s:
        return {"tokens": [], "words": words, "ignored": [],
                "translation": "请问社保卡激活和医保业务在哪个窗口办理？",
                "answer": "请前往大厅东侧 B 区 6～8 号「社保医保综合窗口」，凭身份证在自助取号机取号即可办理。",
                "confirm_card": None, "clarify": None, "confidence": 0.6}

    if {"营业执照", "材料"} & s:
        return {"tokens": [], "words": words, "ignored": [],
                "translation": "我想办理个体工商户营业执照，请问需要准备什么材料？",
                "answer": "请准备经营者身份证原件与经营场所使用证明，材料齐全可在 1 号商事登记窗口受理。",
                "confirm_card": None, "clarify": None, "confidence": 0.55}

    joined = "、".join(words) if words else "（未识别到有效词）"
    return {"tokens": [], "words": words, "ignored": [],
            "translation": f"您好，我想咨询关于「{joined}」的政务服务事项。",
            "answer": "已识别到您的诉求关键词，请携带本人有效身份证件前往一号综合导办台取号办理。",
            "confirm_card": None, "clarify": None, "confidence": 0.35}


# ==============================================================================
# 五、消息组装
# ==============================================================================
def _build_user_message(raw_tokens: list, local_words: list, unknown: list,
                        kb_context: str, strategy: str) -> str:
    lines = []

    if strategy == STRATEGY_LLM:
        lines.extend([
            f"输入拼音：{' '.join(raw_tokens)}",
            "",
            "本地查表结果（仅供参考，**可以推翻**；若你认为是错拼或词表外词，"
            "请按你的判断输出并把 source 标为 inferred）："
            + (("、".join(local_words)) if local_words else "（本地未匹配到任何词）"),
        ])
        if unknown:
            lines.append(f"本地判定为词表外的拼音：{'、'.join(unknown)}（请自行判断是有意义还是噪声）")
    else:
        # local 策略的对照组：**不给原始拼音**，模型只能基于词表过滤后的词成句。
        # 这样两种策略的差异才是真实可比的（否则模型照样会把 wo/yao 推断出来）。
        lines.extend([
            "词表过滤后的词序列：" + ("、".join(local_words) if local_words else "（空）"),
            "",
            "注意：本次**不提供原始拼音**，请严格基于上述词序列成句，不要自行补充未列出的词。",
        ])

    if kb_context:
        lines.extend([
            "",
            "# 权威办事信息（来自本地政务知识库）",
            "# answer 与 confirm_card 中的窗口、材料、费用必须严格以此为准，",
            "# 不得改写，也不得补充这里没有的收费或政策细节",
            kb_context,
        ])

    lines.append("")
    if strategy == STRATEGY_LLM:
        lines.append("请按 system 指令输出 json，tokens 必须覆盖输入的每一个拼音。")
    else:
        lines.append("请按 system 指令输出 json。")
    return "\n".join(lines)


def _prepare(token_list, strategy: str):
    """公共预处理：拿原始 token、跑本地查表（用于交叉校验）。"""
    tokens = [str(t).strip() for t in (token_list or []) if str(t).strip()]
    local = decode_tokens(tokens) if tokens else {
        "words": [], "unknown": [], "fuzzy": [], "confirmed": False, "items": []
    }
    return tokens, local


def _base_fields(tokens: list, local: dict, strategy: str) -> dict:
    return {
        "raw_pinyin": tokens,
        "unknown_pinyin": local["unknown"],
        "local_words": local["words"],
        "fuzzy": local["fuzzy"],
        "confirmed": local["confirmed"],
        "strategy": strategy,
        "error": None,
    }


# ==============================================================================
# 六、非流式主接口
# ==============================================================================
_cache: dict = {}


def _fallback_from_rules(local_words: list, base: dict, error=None) -> dict:
    result = _rule_based(local_words)
    result.update(base)
    result.setdefault("confirm_card", None)
    result.setdefault("clarify", None)
    result["source"] = "rule"
    result["cached"] = False
    result["error"] = error
    return result


def parse_semantics_detail(token_list, kb_context: str = "",
                           strategy: str = DEFAULT_STRATEGY) -> dict:
    """
    完整版解译。

    参数
        token_list : list[str] 原始拼音（也兼容已是中文的情况）
        kb_context : str       知识库给的权威办事信息
        strategy   : "llm"（默认，LLM 主导解码）| "local"（本地查表过滤后再组句）
    返回字典包含 translation/answer/confirm_card/clarify/confidence/tokens/words/... 
    """
    tokens, local = _prepare(token_list, strategy)

    if not tokens:
        return {
            "tokens": [], "words": [], "ignored": [], "translation": "", "answer": "",
            "confirm_card": None, "clarify": None, "confidence": 0.0,
            "source": "rule", "raw_pinyin": [], "unknown_pinyin": [], "local_words": [],
            "fuzzy": [], "confirmed": False, "strategy": strategy, "error": None, "cached": False,
        }

    base = _base_fields(tokens, local, strategy)

    if not llm_available():
        return _fallback_from_rules(local["words"], base)

    cache_key = (tuple(tokens), kb_context, strategy)
    if ENABLE_CACHE and cache_key in _cache:
        cached = dict(_cache[cache_key])
        cached.update(base)
        cached["cached"] = True
        return cached

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(
            tokens, local["words"], local["unknown"], kb_context, strategy)},
    ]

    last_err = None
    for _ in range(LLM_MAX_RETRIES + 1):
        try:
            result = _normalize_result(_parse_json_loose(_call_deepseek(messages)), tokens)
            result.update(base)
            result["source"] = "llm"
            result["cached"] = False
            if ENABLE_CACHE:
                _cache[cache_key] = dict(result)
            return result
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            continue

    return _fallback_from_rules(local["words"], base, error=last_err)


# ==============================================================================
# 七、流式接口
# ==============================================================================
def stream_semantics_detail(token_list, kb_context: str = "",
                            strategy: str = DEFAULT_STRATEGY,
                            emit_raw: bool = True) -> Iterator[tuple]:
    """
    流式解译。依次产出：
        ("meta",  {raw_pinyin, local_words, unknown, confirmed, strategy, kb_hit})
        ("delta", {"text": "增量文本"})      # 模型原始输出片段（前端可边收边解析）
        ("final", 完整结果字典)
        ("error", {"message": str, "fallback": 结果字典})

    说明：流式不做缓存，避免"看起来是实时，其实命中缓存"。
    """
    tokens, local = _prepare(token_list, strategy)
    base = _base_fields(tokens, local, strategy)

    yield ("meta", {
        "raw_pinyin": tokens,
        "local_words": local["words"],
        "unknown_pinyin": local["unknown"],
        "confirmed": local["confirmed"],
        "strategy": strategy,
        "kb_hit": bool(kb_context),
    })

    if not tokens:
        result = {
            "tokens": [], "words": [], "ignored": [], "translation": "", "answer": "",
            "confirm_card": None, "clarify": None, "confidence": 0.0,
            "source": "rule", "cached": False, **base,
        }
        yield ("final", result)
        return

    if not llm_available():
        yield ("final", _fallback_from_rules(local["words"], base, error="LLM 未启用"))
        return

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(
            tokens, local["words"], local["unknown"], kb_context, strategy)},
    ]

    buf = ""
    try:
        for chunk in _stream_deepseek(messages):
            buf += chunk
            if emit_raw:
                yield ("delta", {"text": chunk})
        result = _normalize_result(_parse_json_loose(buf), tokens)
        result.update(base)
        result["source"] = "llm"
        result["cached"] = False
        yield ("final", result)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        yield ("error", {"message": err})
        yield ("final", _fallback_from_rules(local["words"], base, error=err))


# ==============================================================================
# 八、app.py 兼容接口
# ==============================================================================
def parse_semantics(word_list):
    """返回 (中文翻译句, 政务导办回答)。签名与 v1 完全一致。"""
    detail = parse_semantics_detail(word_list)
    translation = detail["translation"]
    answer = detail["answer"]
    if answer and not answer.startswith("【政务导办回答】"):
        answer = "【政务导办回答】" + answer
    return translation, answer


# ==============================================================================
# 自测：python agent.py
# ==============================================================================
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(f"模型 {DEEPSEEK_MODEL} | 思考模式 {'开' if LLM_THINKING else '关'} "
          f"| LLM 可用 {llm_available()}\n")

    cases = [
        ("标准", ["nihao", "shenfenzheng", "diushi", "buban", "queren"]),
        ("错拼 shenfenzeng", ["shenfenzeng", "diushi", "buban"]),
        ("词表外 + 噪声", ["wo", "yao", "buban", "shenfenzeng", "tiyuguan"]),
        ("连写变体", ["shen fen zheng", "diu_shi"]),
        ("泛词", ["chuangkou"]),
    ]
    for label, toks in cases:
        d = parse_semantics_detail(toks)
        print(f"[{label}] {' '.join(toks)}")
        print("  tokens :", [(t["raw"], t["word"], t["source"]) for t in d["tokens"]])
        print("  本地   :", d["local_words"])
        print("  译文   :", d["translation"])
        print("  导办   :", d["answer"])
        print(f"  置信度 : {d['confidence']:.2f} | {d['source']} | cached={d['cached']}\n")
