"""
================================================================================
模块名称：pinyin_codec.py（拼音解码层）
负责人：技术 B 组 - 古宇轩
职责：
    把 YOLO 输出的**无声调拼音**，还原成【封闭词表】内的中文词。

为什么要单独一层，而不是直接丢给大模型？
    1. 无声调拼音歧义极大（"shi" 可以是 是/事/十/时/市/识…），
       交给大模型"自由发挥"必然编造业务事项，答辩时无法自圆其说。
    2. 我们的词表是封闭的（17 类），完全可以 100% 确定性还原。
    3. 先解码成中文再让大模型「连词成句」，准确率和可解释性都更高；
       大模型只负责"排序 + 补全 + 写导办回答"，这是它擅长的部分。

对外接口：
    decode_tokens(tokens)  -> dict       完整解码报告（推荐，给 agent.py 用）
    to_hanzi(tokens)       -> list[str]  只要中文词（给界面显示用）

独立测试：
    python pinyin_codec.py
================================================================================
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache

VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab.json")

# 拼音声调符号 → 无调字母
_TONE_TABLE = str.maketrans({
    "ā": "a", "á": "a", "ǎ": "a", "à": "a",
    "ē": "e", "é": "e", "ě": "e", "è": "e",
    "ī": "i", "í": "i", "ǐ": "i", "ì": "i",
    "ō": "o", "ó": "o", "ǒ": "o", "ò": "o",
    "ū": "u", "ú": "u", "ǔ": "u", "ù": "u",
    "ǖ": "v", "ǘ": "v", "ǚ": "v", "ǜ": "v", "ü": "v",
    "ń": "n", "ň": "n", "ǹ": "n", "ḿ": "m",
})

# 模糊匹配阈值：低于它认为"不是同一个词"，宁可不认也不瞎认
_FUZZY_THRESHOLD = 0.85
_FUZZY_MIN_LEN = 4


def normalize(token) -> str:
    """
    归一化一个拼音 token：
        "Shēn Fèn Zhèng" / "shen_fen_zheng" / "shen-fen-zheng"  →  "shenfenzheng"

    规则：转小写 → 去声调 → 只保留英文字母（去掉分隔符、数字、空格）。
    """
    if token is None:
        return ""
    s = unicodedata.normalize("NFC", str(token)).strip().lower()
    s = s.translate(_TONE_TABLE)
    return re.sub(r"[^a-z]", "", s)


def has_cjk(token) -> bool:
    """判断 token 里是否已经是汉字（A 组如果改回输出中文，此函数负责兜底）。"""
    if token is None:
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", str(token)))


def _fuzzy_key(s: str) -> str:
    """
    构造"发音近似键"，用来兜住模型类别名的书写差异：
        zh/ch/sh → z/c/s ；v → u ；ng → n
    只作为第二道保险，精确命中永远优先。
    """
    s = s.replace("zh", "z").replace("ch", "c").replace("sh", "s")
    s = s.replace("v", "u").replace("ng", "n")
    return s


@lru_cache(maxsize=1)
def load_vocab(path: str = VOCAB_PATH) -> dict:
    """加载并索引词表，返回 {exact: {...}, fuzzy: {...}, entries: [...]}"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    entries = raw["classes"]
    exact: dict[str, dict] = {}
    fuzzy: dict[str, dict] = {}

    for e in entries:
        hanzi = e["hanzi"]
        # 主拼音 + 所有别名，全部归一化后建索引
        for key in [e["pinyin"], *e.get("aliases", [])]:
            nk = normalize(key)
            if nk:
                exact.setdefault(nk, e)
        fuzzy.setdefault(_fuzzy_key(normalize(e["pinyin"])), e)

    return {
        "raw": raw,
        "entries": entries,
        "exact": exact,
        "fuzzy": fuzzy,
        "confirm_pinyin": normalize(raw.get("confirm", {}).get("pinyin", "queren")),
        "confirm_hanzi": raw.get("confirm", {}).get("hanzi", "确认"),
    }


def match_one(token, vocab: dict | None = None) -> dict:
    """
    单个 token 的解码结果：
        {"raw", "norm", "hanzi" 或 None, "hit", "score"}

    hit 取值：
        cjk     —— 输入本来就是汉字，直接用
        exact   —— 精确命中主拼音或别名
        fuzzy   —— 模糊命中（书写差异），会记入告警
        unknown —— 词表里没有，忽略
    """
    vocab = vocab or load_vocab()

    if has_cjk(token):
        text = str(token).strip()
        # 已经是汉字，但仍要判断它是否属于"确认"这一类控制信号
        for e in vocab["entries"]:
            if e["hanzi"] == text:
                return {"raw": token, "norm": text, "hanzi": e["hanzi"], "hit": "cjk", "score": 1.0}
        return {"raw": token, "norm": text, "hanzi": text, "hit": "cjk", "score": 1.0}

    norm = normalize(token)
    if not norm:
        return {"raw": token, "norm": "", "hanzi": None, "hit": "unknown", "score": 0.0}

    # ① 精确命中
    e = vocab["exact"].get(norm)
    if e:
        return {"raw": token, "norm": norm, "hanzi": e["hanzi"], "hit": "exact", "score": 1.0}

    # ② 发音近似键命中
    e = vocab["fuzzy"].get(_fuzzy_key(norm))
    if e:
        return {"raw": token, "norm": norm, "hanzi": e["hanzi"], "hit": "fuzzy", "score": 0.9}

    # ③ 编辑距离兜底（只对足够长的 token 生效，避免 "nl"/"sm" 之类误配）
    if len(norm) >= _FUZZY_MIN_LEN:
        best, best_score = None, 0.0
        for key, entry in vocab["exact"].items():
            if abs(len(key) - len(norm)) > 4:
                continue
            score = SequenceMatcher(None, norm, key).ratio()
            if score > best_score:
                best, best_score = entry, score
        if best is not None and best_score >= _FUZZY_THRESHOLD:
            return {"raw": token, "norm": norm, "hanzi": best["hanzi"], "hit": "fuzzy", "score": round(best_score, 3)}

    return {"raw": token, "norm": norm, "hanzi": None, "hit": "unknown", "score": 0.0}


def decode_tokens(tokens) -> dict:
    """
    批量解码，返回完整报告（agent.py 与界面都用它）。

    返回:
        {
          "words":    ["你好", "身份证", "补办"],   # 已去重、已剔除「确认」控制信号
          "items":    [每个 token 的明细],
          "unknown":  ["tiyuguan"],                # 词表外、被忽略的
          "fuzzy":    ["yingyezhao"],              # 模糊命中、需人工确认的
          "confirmed": True/False                  # 是否出现「确认」手势
        }
    """
    vocab = load_vocab()
    items, words, unknown, fuzzy = [], [], [], []
    confirmed = False

    for t in (tokens or []):
        r = match_one(t, vocab)
        items.append(r)

        if r["hit"] == "unknown":
            unknown.append(str(r["raw"]))
            continue
        if r["hit"] == "fuzzy":
            fuzzy.append(f'{r["raw"]}→{r["hanzi"]}')

        hanzi = r["hanzi"]
        # 「确认」是控制信号：触发成句，但不进入语义
        if hanzi == vocab["confirm_hanzi"]:
            confirmed = True
            continue

        # 去掉连续重复（同一手势被连续识别多次）
        if words and words[-1] == hanzi:
            continue
        words.append(hanzi)

    return {
        "words": words,
        "items": items,
        "unknown": unknown,
        "fuzzy": fuzzy,
        "confirmed": confirmed,
    }


def to_hanzi(tokens) -> list[str]:
    """只要中文词，给界面显示用（不想看拼音的话）。"""
    return decode_tokens(tokens)["words"]


# ------------------------------------------------------------------------------
# 自测：python pinyin_codec.py
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    vocab = load_vocab()
    print(f"词表版本 {vocab['raw'].get('version')}，共 {len(vocab['entries'])} 类")
    print(f"确认手势：{vocab['confirm_hanzi']}（{vocab['confirm_pinyin']}）\n")

    cases = [
        ["nihao", "shenfenzheng", "diushi", "buban"],
        ["Shēn Fèn Zhèng", "diu_shi", "buban"],
        ["yingyezhao", "xuyao", "shenme", "cailiao"],
        ["sfz", "buban", "queren"],
        ["nihao", "nihao", "身份证", "tiyuguan", "xiexie"],
    ]
    for c in cases:
        d = decode_tokens(c)
        print("输入:", " ".join(map(str, c)))
        print("  解码:", d["words"], "| 确认手势:", d["confirmed"])
        if d["unknown"]:
            print("  忽略:", d["unknown"])
        if d["fuzzy"]:
            print("  模糊:", d["fuzzy"])
        print()
