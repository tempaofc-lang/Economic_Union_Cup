class LetterAdapter:
    def __init__(self, stable_frames=3, release_frames=2,
                 min_conf=0.75):
        self.stable_frames = stable_frames
        self.release_frames = release_frames
        self.min_conf = min_conf
        self.reset()

    def reset(self):
        self.current_word = ""
        self.words = []
        self.candidate = None
        self.count = 0
        self.last_letter = None
        self.empty_count = 0

    def feed(self, label, conf=0.0):
        """每帧输入 YOLO 预测的字母和置信度。"""
        # 置信度可能来自空检测（None）或被上游转成了字符串，直接比较会抛 TypeError，
        # 而这里的异常会打断整个识别循环，所以统一兜成 0（当作无效帧处理）。
        if not isinstance(conf, (int, float)):
            conf = 0.0
        if (
            not isinstance(label, str)
            or len(label) != 1
            or not ("a" <= label.lower() <= "z")
            or conf < self.min_conf
        ):
            self.empty_count += 1
            if self.empty_count >= self.release_frames:
                self.candidate = None
                self.count = 0
                self.last_letter = None
            return None

        label = label.lower()
        self.empty_count = 0

        if label == self.candidate:
            self.count += 1
        else:
            self.candidate = label
            self.count = 1

        if (self.count >= self.stable_frames
                and label != self.last_letter):
            self.current_word += label
            self.last_letter = label
            return label

        return None

    def finish_word(self):
        """用户确认当前词已经拼写完毕。"""
        if self.current_word:
            self.words.append(self.current_word)
        self.current_word = ""
        self.candidate = None
        self.count = 0
        self.last_letter = None

    def finish_sentence(self):
        """用户确认成句，返回 Agent 所需的拼音列表。"""
        self.finish_word()
        result = self.words.copy()
        self.reset()
        return result