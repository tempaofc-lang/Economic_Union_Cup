"""
================================================================================
模块名称：agent.py（Agent 智能体 · 拼音 → 完整句子 的语义解译引擎）
负责人：技术 B 组 - 古宇轩
版本：v2.0（2026-10-22 重写：输入由「中文词组」改为「无声调拼音」）

【这条链路在干什么】
    YOLO 识别出的原始结果：  ["nihao", "shenfenzheng", "diushi", "buban"]
            ↓ pinyin_codec.py（确定性解码，封闭词表 17 类）
    中文词序列：             ["你好", "身份证", "丢失", "补办"]
            ↓ agent.py（本文件：DeepSeek 大模型）
    输出：                   翻译句 = "工作人员您好，我的身份证丢失了，请问怎么补办？"
                             导办答 = "【政务导办回答】请携带户口本原件，先到 2 号人像采集室…"

【为什么不把拼音直接丢给大模型？】
    无声调拼音歧义极大（"shi" 能是 是/事/十/时/市/识），让大模型自由发挥必然编造业务事项。
    所以我们先用封闭词表做 100% 确定的解码，再让大模型做它真正擅长的两件事：
    ① 按汉语语序重排、补全省略成分、连成通顺句子；② 生成具体可执行的政务导办回答。

【app.py 无需任何改动】
    对外仍然只暴露 parse_semantics(word_list) -> (translation, ai_answer)，
    与旧版签名完全一致，app.py 会自动挂载。
================================================================================
"""

from __future__ import annotations

import json
import os
import re

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # python-dotenv 没装也不影响运行（可直接用系统环境变量）
    pass

from pinyin_codec import decode_tokens, load_vocab

# ==============================================================================
# 🔧 开关与配置（全部可用环境变量覆盖，方便答辩前一键切换）
# ==============================================================================
USE_REAL_LLM_API = os.getenv("USE_REAL_LLM_API", "1") == "1"  # 置 0 则只用本地规则引擎
ENABLE_CACHE = True                                            # 同样的词序列只调用一次，省时省钱

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "25"))   # 秒；超时立刻回退规则引擎，界面不能卡住
LLM_MAX_RETRIES = 2                                 # JSON 模式偶发返回空内容，必须重试
LLM_TEMPERATURE = 0.2                               # 低温度，减少发挥
LLM_MAX_TOKENS = 600

# 模型没给出可解析的 confidence 时的中性取值（宁可保守，也不显示 0）
_DEFAULT_CONFIDENCE = 0.6

__all__ = ["parse_semantics", "parse_semantics_detail", "llm_available", "SYSTEM_PROMPT_TEMPLATE"]


# ==============================================================================
# 一、System Prompt —— 这是整个模块最值钱的部分
# ==============================================================================
SYSTEM_PROMPT_TEMPLATE = """你是「手语视界」政务服务大厅的语义解译引擎，服务对象是听障办事群众。

# 一、你的输入
用户会给你一串由视觉识别模块输出的**无声调拼音**，例如：
    nihao shenfenzheng diushi buban

这些拼音有三个已知缺陷，你必须在心里自动纠正，不要向用户抱怨：
1. **语序不符合汉语习惯**——手语常把"话题"放前面、动作放后面（如"身份证 补办"）。
2. **有重复**——同一个手势会被连续识别多次（如 buban buban buban）。
3. **有误识别**——偶尔混入与本次办事无关的孤立词，或少量词表外的拼音。

# 二、允许词表（你只能在这 17 个词的范围里工作）
{vocab_lines}

# 三、你必须依次完成三件事
**第一步 解码**
把每个拼音还原为上面的中文词。重复的只保留一个。

**第二步 成句**
把中文词按汉语习惯重新排序，补全省略的主语和语气词，写成**一句第一人称的、自然礼貌的中文问句**。
- 可以补充"我""请问""您好""想要"这类**衔接词**，它们不改变原意。
- 严禁补充原词里没有的**业务事项**。例如原词里没有"预约"，就绝不能写"我已预约"。

**第三步 导办**
以政务大厅引导员的口吻回答，给出可执行的指引，必须包含：
- 具体窗口（如"3 号户政窗口""B 区 6～8 号社保医保窗口"）
- 具体材料（如"户口本原件""经营者身份证原件"）
- 能确定时才写费用和时长，不确定就不写，**绝不编造收费金额或政策细节**。
- **长度控制在 80 字以内**：这句话要放大显示在听障群众的屏幕上，太长会看不完。
  宁可只说最关键的一到两个动作，也不要罗列所有可能情况。

# 四、输出格式
只输出一个 json 对象，不要 markdown 代码块，不要任何解释文字。结构严格为：
{{
  "words": ["你还原出的中文词，按你重排后的语序"],
  "ignored": ["被你判定为误识别而忽略的拼音或词"],
  "translation": "第二步得到的第一人称中文问句",
  "answer": "第三步得到的导办回答（不要加【政务导办回答】前缀，系统会自动加）",
  "confirm_card": {{
    "title": "业务名称，如「身份证补办」",
    "materials": ["所需材料，一条一项"],
    "window": "具体窗口，如「3 号户政窗口」",
    "fee": "费用，如「40 元」；不确定就填 null"
  }},
  "clarify": null,
  "confidence": 0.0
}}

其中 confidence 是你对 translation 忠实度的自评（0~1）：
- 词全部还原且业务指向明确 → 0.85~0.98
- 有词被忽略、或存在同音歧义 → 0.5~0.8
- 几乎无法成句 → 0.1~0.3

关于 confirm_card 与 clarify：
- confirm_card 是把 answer 里的关键信息结构化，方便界面做成卡片展示；**判断不出业务时整个字段填 null**，不要瞎填。
- clarify 是**主动反问**，只在两种情况下使用：有效词只有 1 个及以下，或无法判断对方要办什么业务。
  格式为 {{"question": "一句反问", "options": ["选项1", "选项2"]}}，选项给 2~3 个。
  其余情况一律填 null。**宁可反问，也不要硬猜**——猜错比不答更糟。

# 五、示例

示例 1（标准场景）
输入拼音：nihao shenfenzheng diushi buban buban
输出 json：
{{"words":["你好","身份证","丢失","补办"],"ignored":["buban(重复)"],"translation":"工作人员您好，我的身份证丢失了，请问要怎么补办？","answer":"补办身份证请携带户口本或驾驶证原件，先到 2 号人像采集室免费拍照，随后到 3 号户政窗口办理，工本费 40 元。","confidence":0.95}}

示例 2（语序倒置、成分缺失）
输入拼音：buban shenfenzheng
输出 json：
{{"words":["身份证","补办"],"ignored":[],"translation":"您好，我的身份证要补办，请问在哪里办理？","answer":"请携带户口本原件到 3 号户政窗口办理身份证补办，现场采集人像后即可受理。","confidence":0.72}}

示例 3（混入无关词 + 词表外拼音）
输入拼音：qingwen shebaoka jihuo chuangkou nali tiyuguan
输出 json：
{{"words":["请问","社保卡","激活","窗口","哪里"],"ignored":["tiyuguan"],"translation":"请问社保卡激活在哪个窗口办理？","answer":"请前往大厅东侧 B 区 6～8 号「社保医保综合窗口」，凭身份证在自助取号机取号后办理，无需预约。","confidence":0.9}}

示例 4（无法成句）
输入拼音：xiexie
输出 json：
{{"words":["谢谢"],"ignored":[],"translation":"谢谢您！","answer":"不客气，还有其他需要请随时告诉我。","confirm_card":null,"clarify":null,"confidence":0.4}}

示例 5（词太少，主动反问而不是硬猜）
输入拼音：chuangkou
输出 json：
{{"words":["窗口"],"ignored":[],"translation":"您好，请问我要去哪个窗口办理？","answer":"请告诉我您要办理的业务，我为您指引对应窗口。","confirm_card":null,"clarify":{{"question":"请问您要办理哪项业务？","options":["身份证补办","社保卡激活","营业执照办理"]}},"confidence":0.25}}

# 六、硬性约束（违反即视为失败）
1. 输出必须是**合法 json**，且**只包含** words / ignored / translation / answer / confirm_card / clarify / confidence 七个键。
2. answer 不得以"【政务导办回答】"开头，该前缀由系统统一添加。
3. 不得输出政治敏感、医疗诊断、法律结论等超出政务导办范围的内容。
4. 若输入里没有任何可用信息，translation 固定写"您好，我没能完整表达，请再问我一次。"，confidence 低于 0.3。
"""


def _build_vocab_lines() -> str:
    """把词表渲染成 prompt 里的对照表：- `shenfenzheng` = 身份证"""
    vocab = load_vocab()
    lines = []
    for e in vocab["entries"]:
        if e["hanzi"] == vocab["confirm_hanzi"]:
            continue  # 「确认」是控制信号，不会进入语义
        lines.append(f"- `{e['pinyin']}` = {e['hanzi']}")
    return "\n".join(lines)


_SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(vocab_lines=_build_vocab_lines())


# ==============================================================================
# 二、调用大模型
# ==============================================================================
def llm_available() -> bool:
    """给界面 / 测试脚本判断当前能否真的调 API。"""
    return bool(USE_REAL_LLM_API and DEEPSEEK_API_KEY)


def _call_deepseek(messages: list) -> str:
    """发一次 chat/completions 请求，返回 content 字符串。失败抛异常。"""
    resp = requests.post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "response_format": {"type": "json_object"},  # 强制 JSON 输出
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
            "stream": False,
        },
        timeout=LLM_TIMEOUT,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    if not content or not content.strip():
        # DeepSeek 官方文档提示：JSON 模式下有小概率返回空 content
        raise RuntimeError("模型返回空内容")
    return content


def _parse_json_loose(text: str) -> dict:
    """
    宽容解析：模型偶尔会用 ```json 包裹，或在前后多写几个字。
    先直接 loads；失败则截取第一个 { 到最后一个 } 再试。
    """
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
    """
    宽容解析 confidence。模型偶尔会返回 null / "95%" / "0.9（较高）" 这类脏数据，
    任何一种都不应该让整条结果作废，也不应该把 0 显示给用户。
    """
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
    """把可能是字符串、可能是列表的字段，统一成干净的字符串列表。"""
    if value is None:
        return []
    if isinstance(value, str):
        value = re.split(r"[、,，;；\n]+", value)
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _normalize_confirm_card(data: dict):
    """
    结构化「办事卡片」。全空则返回 None —— 宁可没有卡片，也不要给用户一张空卡片。
    """
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
    """
    主动反问。没有 question 就视为无效 —— 反问必须有内容，否则不如不问。
    """
    clarify = data.get("clarify")
    if not isinstance(clarify, dict):
        return None
    question = str(clarify.get("question") or "").strip()
    if not question:
        return None
    options = _to_str_list(clarify.get("options"))[:4]
    return {"question": question, "options": options}


def _normalize_result(data: dict) -> dict:
    """校验 + 清洗 + 补默认值，保证下游拿到的一定是 5 个键的干净字典。"""
    def as_text(key: str) -> str:
        v = data.get(key, "")
        return str(v).strip() if v is not None else ""

    translation = as_text("translation")
    answer = as_text("answer")

    # 去掉模型自己加的标记，统一由本模块添加
    translation = re.sub(r"^\s*(【手语翻译】|手语翻译[:：]|翻译[:：])\s*", "", translation)
    answer = re.sub(r"^\s*【政务导办回答】\s*", "", answer)

    words = data.get("words") or []
    if isinstance(words, str):
        words = [w for w in re.split(r"[、,，\s]+", words) if w]
    words = [str(w).strip() for w in words if str(w).strip()]

    ignored = data.get("ignored") or []
    if isinstance(ignored, str):
        ignored = [i for i in re.split(r"[、,，\s]+", ignored) if i]
    ignored = [str(i).strip() for i in ignored if str(i).strip()]

    confidence = _to_confidence(data.get("confidence"))

    if not translation:
        raise ValueError("translation 为空")
    if len(translation) > 160:
        translation = translation[:160].rstrip("，。、") + "。"
    if len(answer) > 150:
        # 大字屏放不下过长的回答，按句号截断
        cut = answer[:150]
        answer = (cut[:cut.rfind("。") + 1] if "。" in cut else cut.rstrip("，、；") + "。")
    if not answer:
        answer = "请携带本人有效身份证件前往一号综合导办台取号办理。"

    return {
        "words": words,
        "ignored": ignored,
        "translation": translation,
        "answer": answer,
        "confirm_card": _normalize_confirm_card(data),
        "clarify": _normalize_clarify(data),
        "confidence": confidence,
    }


# ==============================================================================
# 三、本地规则引擎（兜底）——断网 / 无额度 / 超时 / 解析失败时使用
# ==============================================================================
def _rule_based(words: list) -> dict:
    """不依赖任何网络的老逻辑，保证界面永远有输出。"""
    s = set(words)

    if {"身份证", "补办"} & s or {"身份证", "丢失"} & s:
        return {
            "words": words,
            "ignored": [],
            "translation": "工作人员您好，我的身份证丢失了，请问怎么办理补办？",
            "answer": "补办身份证请携带户口本或驾驶证原件，先到右侧 2 号人像采集室免费拍照，"
                      "随后前往 3 号户政窗口办理，工本费 40 元。",
            "confidence": 0.6,
        }

    if {"社保卡", "医保", "激活"} & s:
        return {
            "words": words,
            "ignored": [],
            "translation": "请问社保卡激活和医保业务在哪个窗口办理？",
            "answer": "请前往大厅东侧 B 区 6～8 号「社保医保综合窗口」，凭身份证在自助取号机取号即可办理。",
            "confidence": 0.6,
        }

    if {"营业执照", "材料"} & s:
        return {
            "words": words,
            "ignored": [],
            "translation": "我想办理个体工商户营业执照，请问需要准备什么材料？",
            "answer": "请准备：① 经营者身份证原件；② 经营场所租赁合同或产权证明。材料齐全可在 1 号商事登记窗口受理。",
            "confidence": 0.55,
        }

    joined = "、".join(words) if words else "（未识别到有效词）"
    return {
        "words": words,
        "ignored": [],
        "translation": f"您好，我想咨询关于「{joined}」的政务服务事项。",
        "answer": "已识别到您的诉求关键词，请携带本人有效身份证件前往一号综合导办台取号办理。",
        "confidence": 0.35,
    }


# ==============================================================================
# 四、对外主接口
# ==============================================================================
_cache = {}


def _build_user_message(raw_tokens: list, decoded_words: list, unknown: list,
                        kb_context: str = "") -> str:
    """
    把「原始拼音 + 规则预解码结果 + 可选的知识库权威信息」一起交给模型。

    kb_context 由调用方（backend 的 gov_kb 工具）注入。有它时模型只负责措辞，
    窗口号/材料/费用一律以知识库为准，从根上消除"硬编码数字"和"编造收费"的问题。
    """
    lines = [
        f"输入拼音：{' '.join(raw_tokens)}",
        "",
        f"规则引擎预解码：{'、'.join(decoded_words) if decoded_words else '（无）'}",
    ]
    if unknown:
        lines.append(f"词表外拼音（请判断是否有意义，无意义请放进 ignored）：{'、'.join(unknown)}")
    if kb_context:
        lines.extend([
            "",
            "# 权威办事信息（来自本地政务知识库，answer 与 confirm_card 中的窗口、材料、费用",
            "# 必须严格以此为准，不得改写，也不得补充这里没有的收费或政策细节）",
            kb_context,
        ])
    lines.append("")
    lines.append("请按 system 指令输出 json。")
    return "\n".join(lines)


def parse_semantics_detail(token_list, kb_context: str = "") -> dict:
    """
    完整版解译，返回包含调试信息的字典（界面调试面板、测试脚本、backend API 用）。

    参数 token_list : list[str]，元素是 YOLO 输出的拼音（也兼容已是中文的情况）
         kb_context : str，可选。知识库给出的权威办事信息，会注入到用户消息里。
    返回:
        {
          "translation": str,   "answer": str,     "confidence": float,
          "words": list[str],   "ignored": list[str],
          "confirm_card": dict | None,   # 结构化办事卡片（业务/材料/窗口/费用）
          "clarify": dict | None,        # 主动反问（词太少时不硬猜）
          "source": "llm" | "rule",  "raw_pinyin": list[str],
          "unknown_pinyin": list[str], "fuzzy": list[str], "confirmed": bool,
          "error": str | None
        }
    """
    tokens = [str(t).strip() for t in (token_list or []) if str(t).strip()]
    if not tokens:
        return {
            "translation": "", "answer": "", "confidence": 0.0,
            "words": [], "ignored": [], "confirm_card": None, "clarify": None,
            "source": "rule", "raw_pinyin": [],
            "unknown_pinyin": [], "fuzzy": [], "confirmed": False, "error": None,
        }

    # ---- 第一层：确定性拼音解码（不经网络，毫秒级） ----
    decoded = decode_tokens(tokens)
    words = decoded["words"]

    base = {
        "raw_pinyin": tokens,
        "unknown_pinyin": decoded["unknown"],
        "fuzzy": decoded["fuzzy"],
        "confirmed": decoded["confirmed"],
        "error": None,
    }

    if not words and not decoded["unknown"]:
        return {**base, "words": [], "ignored": [], "translation": "",
                "answer": "", "confirm_card": None, "clarify": None,
                "confidence": 0.0, "source": "rule"}

    # 命中缓存（演示时同一段手语会重复触发，避免重复烧钱和等待）
    # 注意：缓存键必须带上知识库上下文，否则"同一串拼音、不同权威信息"会串味
    cache_key = (tuple(words), kb_context)
    if ENABLE_CACHE and cache_key in _cache:
        cached = dict(_cache[cache_key])
        cached.update(base)
        cached["cached"] = True          # 告诉上层"这是缓存命中"，界面才不会把 0ms 当成真实耗时
        return cached

    # ---- 第二层：大模型解译 ----
    if llm_available():
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",
             "content": _build_user_message(tokens, words, decoded["unknown"], kb_context)},
        ]

        last_err = None
        for _ in range(LLM_MAX_RETRIES + 1):
            try:
                content = _call_deepseek(messages)
                result = _normalize_result(_parse_json_loose(content))
                result.update(base)
                result["source"] = "llm"
                result["cached"] = False
                if ENABLE_CACHE:
                    _cache[cache_key] = dict(result)
                return result
            except Exception as exc:  # 网络 / 额度 / JSON 解析 全部兜住
                last_err = f"{type(exc).__name__}: {exc}"
                continue

        # ---- 第三层：全失败 → 规则引擎兜底 ----
        return _fallback_from_rules(words, base, error=last_err)

    # 未启用 API（离线演示 / 断网 / 没配 Key）
    return _fallback_from_rules(words, base)


def _fallback_from_rules(words: list, base: dict, error=None) -> dict:
    """规则引擎结果 + 统一补齐字段，保证下游拿到的字典结构与 LLM 路径完全一致。"""
    result = _rule_based(words)
    result.update(base)
    result.setdefault("confirm_card", None)
    result.setdefault("clarify", None)
    result.setdefault("cached", False)
    result["source"] = "rule"
    result["error"] = error
    return result


def parse_semantics(word_list):
    """
    app.py 调用的兼容接口：返回 (中文翻译句, 政务导办回答)。
    签名与旧版完全一致，app.py 一行都不用改。
    """
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

    print(f"模型: {DEEPSEEK_MODEL} | Key 已配置: {'是' if bool(DEEPSEEK_API_KEY) else '否'} "
          f"| LLM 可用: {llm_available()}\n")

    for raw in (["nihao", "shenfenzheng", "diushi", "buban", "queren"],
                ["buban", "shenfenzheng"],
                ["qingwen", "shebaoka", "jihuo", "chuangkou", "nali", "tiyuguan"]):
        d = parse_semantics_detail(raw)
        print("输入拼音:", " ".join(raw))
        print("  解码中文:", d["words"], "| 确认手势:", d["confirmed"])
        print("  翻译    :", d["translation"])
        print("  导办    :", d["answer"])
        print(f"  置信度  : {d['confidence']:.2f} | 来源: {d['source']}"
              + (f" | 错误: {d['error']}" if d["error"] else ""))
        print()
