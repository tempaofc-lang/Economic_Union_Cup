"""
================================================================================
模块名称：letter_pipeline.py（字母流 → 拼音词 → 成句 的装配层）
负责人：技术 B 组 - 古宇轩
依赖：letter_adapter.py（组员提供的单字符去抖模块）

【为什么需要这一层】
    A 组的模型是**逐字母**输出（一帧一个字母），而下游 agent 需要的是**拼音词**
    （如 "shenfenzheng"）。letter_adapter 解决了"同一个字母被连续识别多次"，
    但没有解决两个问题：

      1. 词边界：字母一直往后拼，用户不说话系统就不知道该在哪切词。
      2. 成句触发：什么时候把攒好的词交给 Agent。

    本层补上这两件事，并且不修改 letter_adapter 的内部逻辑：
      · 词边界 —— 连续 PAUSE_FRAMES 帧没有有效字母，视为"手放下了"，收尾当前词；
      · 成句   —— 检出「确认」手势（vocab.json 里的 queren）就把已完成的词序列交出去；
      · 置信度 —— 一个词的置信度取它所有字母的均值，供界面按阈值过滤。

【对接方式】
    vision_engine.predict_sign_tokens() 逐帧拿到 (字母, 置信度) 后：
        event = stream.feed(letter, conf)
        if event["sentence"]:  # 成句了
            tokens = [{"word": p, "conf": c, "bbox": ...} for p, c in ...]
    tokens 的结构与现有 vision_engine 契约完全一致，agent 侧不用改。
================================================================================
"""

from __future__ import annotations

from letter_adapter import LetterAdapter

# 与 vocab.json 中「确认」手势的 pinyin 保持一致
CONFIRM_LABEL = "queren"

# 连续多少帧没有有效字母算"一个词结束"。
# 25fps 下 6 帧约 0.24 秒，是"松手换下一个字"的合理间隔；太短会把词切碎，太长会粘词。
PAUSE_FRAMES = 6


class LetterStream:
    """把逐帧字母流装配成拼音词序列与整句。"""

    def __init__(self, stable_frames: int = 3, release_frames: int = 2,
                 min_conf: float = 0.75, pause_frames: int = PAUSE_FRAMES,
                 confirm_label: str = CONFIRM_LABEL):
        self.adapter = LetterAdapter(stable_frames, release_frames, min_conf)
        self.pause_frames = pause_frames
        self.confirm_label = confirm_label
        self._word_confs: list[float] = []
        self.completed: list[dict] = []      # [{"pinyin": str, "conf": float}]

    # ------------------------------------------------------------------ 帧输入
    def feed(self, label, conf=0.0) -> dict:
        """
        喂一帧。返回事件字典：
            {"letter": 本帧新确认的字母或 None,
             "word":   本次刚收尾的词或 None,
             "sentence": 成句时的拼音列表或 None}

        调用方只需要关心 sentence 非空的那一刻。
        """
        # 「确认」不是字母，必须在这里截住：交给 adapter 只会被当成无效帧。
        if (self.confirm_label and isinstance(label, str)
                and label.strip().lower() == self.confirm_label):
            return self.confirm()

        letter = self.adapter.feed(label, conf)

        if letter:
            self._word_confs.append(float(conf) if isinstance(conf, (int, float)) else 0.0)
            return {"letter": letter, "word": None, "sentence": None}

        # adapter 内部用 empty_count 记连续无效帧，正好可以复用来判断停顿。
        # 用 == 而不是 >=，保证一次停顿只收尾一次。
        if self.pause_frames and self.adapter.empty_count == self.pause_frames:
            word = self._close_word()
            if word:
                return {"letter": None, "word": word, "sentence": None}

        return {"letter": None, "word": None, "sentence": None}

    # -------------------------------------------------------------- 词/句收尾
    def _close_word(self):
        """收尾当前正在拼的词，返回词字符串；没有内容则返回 None。"""
        raw = self.adapter.current_word
        if not raw:
            self._word_confs = []
            return None

        conf = sum(self._word_confs) / len(self._word_confs) if self._word_confs else 0.0
        self.adapter.finish_word()
        self._word_confs = []
        self.completed.append({"pinyin": raw, "conf": round(conf, 3)})
        return raw

    def confirm(self) -> dict:
        """
        检出「确认」手势：收尾当前词并成句。
        即使当前没有词也没关系，已完成的词照样返回。
        """
        word = self._close_word()
        sentence = [item["pinyin"] for item in self.completed]
        self.completed = []
        self.adapter.reset()
        return {"letter": None, "word": word, "sentence": sentence}

    # ------------------------------------------------------------------ 只读态
    def current_word(self) -> str:
        return self.adapter.current_word

    def tokens(self) -> list[dict]:
        """已完成的词 + 正在拼的词，转成 vision_engine 的 token 结构（不含 bbox）。"""
        items = list(self.completed)
        if self.adapter.current_word:
            conf = sum(self._word_confs) / len(self._word_confs) if self._word_confs else 0.0
            items = items + [{"pinyin": self.adapter.current_word, "conf": round(conf, 3)}]
        return [{"word": it["pinyin"], "conf": it["conf"]} for it in items]


def simulate(letters: str, frames_per_letter: int = 3, conf: float = 0.9,
             pause_frames: int = PAUSE_FRAMES, delimiter: str = " ",
             confirm: bool = True) -> tuple[list[dict], list[str]]:
    """
    把一串字母按"每个字母保持 N 帧"回放，模拟真实模型的逐帧输出。
    用于 A 组模型未就绪时跑通链路、以及单元测试。

    letters 里的空格表示词边界（会插入一段停顿）；
    末尾会补一次「确认」手势，返回成句后的 token 列表。

    返回 (tokens, 事件日志)
    """
    stream = LetterStream(pause_frames=pause_frames)
    trace: list[str] = []

    for ch in letters:
        if ch == delimiter:
            for _ in range(pause_frames):
                stream.feed("", 0.0)
            trace.append(f"⏸ 停顿 {pause_frames} 帧收词")
            continue
        for _ in range(frames_per_letter):
            ev = stream.feed(ch, conf)
            if ev["letter"]:
                trace.append(f"+{ev['letter']}")
            if ev["word"]:
                trace.append(f"『{ev['word']}』")

    # 末尾停顿 + 确认，模拟"打完最后一字，比确认手势"
    for _ in range(pause_frames):
        stream.feed("", 0.0)
    if confirm:
        ev = stream.confirm()
        if ev["sentence"]:
            trace.append("✅ 成句: " + " / ".join(ev["sentence"]))
        tokens = [{"word": p, "conf": conf} for p in ev["sentence"]]
        return tokens, trace

    return stream.tokens(), trace


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    for script in ("wo", "wo yao", "shenfenzheng", "wo yao buban shenfenzheng"):
        tokens, trace = simulate(script)
        print(f"脚本: {script!r}")
        print("  过程:", " ".join(trace))
        print("  tokens:", tokens)
        print()
