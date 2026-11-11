"""
================================================================================
项目名称：YOLO26 静态手语词识别智能体（政务服务无障碍导办专版 · 正式开发版）
界面方案：【方案A：科技简约正式风（左固定大画幅摄像预览 60% + 右独立滑动面板 40%）】
负责人：技术 B 组 - 李彤（前端界面开发） / 古宇轩（Agent 逻辑） / 技术 A 组（视觉模型）
运行命令：python app.py
================================================================================
【后期功能一键删减指南】
如果后期某个功能来不及实现，或者答辩前想精简删掉某个模块，
只需将下方 FEATURE_FLAGS 字典中对应的 True 改为 False，保存重启即可自动隐藏该模块！
================================================================================
"""

import time
import inspect
import threading
import cv2
import numpy as np
import gradio as gr

# ==============================================================================
# 🔧 模块化一键增删开关（后期想删哪个功能，直接把 True 改成 False 即可！）
# ==============================================================================
FEATURE_FLAGS = {
    "SHOW_VOCAB_BAR": True,         # 开关1：是否显示顶部「当前支持政务手语词库」提示栏
    "SHOW_CONF_SLIDER": True,       # 开关2：是否显示「置信度过滤阈值 Slider (默认0.75)」
    "SHOW_AUTO_TTS_TOGGLE": True,   # 开关3：是否显示「识别后自动语音播报 / 静音切换」开关
    "SHOW_WORD_EDIT_BTNS": True,    # 开关4：是否显示「⬅️ 撤回上一词」与「✅ 确认成句」按钮
    "SHOW_AI_GOV_ANSWER": True,     # 开关5：是否显示「政务AI导办回答」（False则只显示手语翻译）
    "SHOW_TWO_WAY_REPLY": True,     # 开关6：是否显示「双向交互：政务工作人员回复框」
    "SHOW_HISTORY_PANEL": True,     # 开关7：是否显示「历史翻译记录与删除功能」
}

# ------------------------------------------------------------------------------
# 自动挂载同级目录下的 vision_engine.py（A组）与 agent.py（B组）
# ------------------------------------------------------------------------------
try:
    import vision_engine
    HAS_VISION_ENGINE = True
    GOV_SUPPORTED_WORDS = vision_engine.GOV_SUPPORTED_WORDS
except ImportError:
    vision_engine = None
    HAS_VISION_ENGINE = False
    GOV_SUPPORTED_WORDS = [
        "你好", "请问", "身份证", "丢失", "补办", "社保卡", "医保", "激活",
        "窗口", "哪里", "营业执照", "办理", "需要", "什么", "材料", "谢谢"
    ]

try:
    import agent as external_agent
    HAS_EXTERNAL_AGENT = True
except ImportError:
    external_agent = None
    HAS_EXTERNAL_AGENT = False

# ------------------------------------------------------------------------------
# 拼音展示层：YOLO 输出的是无声调拼音，界面上要显示中文
# ------------------------------------------------------------------------------
try:
    from pinyin_codec import display_word as _display_word, load_vocab as _load_vocab
    HAS_PINYIN_CODEC = True
except Exception:
    def _display_word(word):
        return str(word)

    def _load_vocab():
        return None

    HAS_PINYIN_CODEC = False


def _display_seq(word_list):
    """把 ["nihao","shenfenzheng"] 显示成 "你好 → 身份证"。"""
    return " → ".join(_display_word(w) for w in word_list)


# ------------------------------------------------------------------------------
# 模拟数据状态：真实模型不可用时 vision_engine 会在 scene 里打 is_mock 标记，
# 界面必须在显眼位置提示，避免答辩时把假数据当成真实识别结果。
# ------------------------------------------------------------------------------
_MOCK_STATE = {"is_mock": False, "reason": ""}


def build_mock_banner(reason=""):
    detail = f"原因：{reason}" if reason else "真实模型未启用或不可用"
    return f"""
    <div style="background:#7F1D1D;color:#FEE2E2;border:1px solid #B91C1C;border-radius:8px;
                padding:10px 14px;margin-bottom:8px;font-size:13.5px;font-weight:700;">
        ⚠️ 当前展示的是【模拟数据】，不是真实识别结果
        <div style="font-weight:400;font-size:11.5px;color:#FCA5A5;margin-top:3px;line-height:1.5;">
            {detail} —— 仅用于界面联调，请勿用于答辩演示或录制演示视频。
        </div>
    </div>
    """


def _display_token_seq(tokens):
    """把 [{"word":"nihao","conf":0.96}, ...] 显示成 "你好 (0.96) → 身份证 (0.94)"。"""
    return " → ".join(f"{_display_word(t['word'])} ({t['conf']:.2f})" for t in tokens)

# 全局运行时状态（用于支持撤回上一词、手动确认成句、历史记录管理）
CLS_MIN_CONF_DISPLAY = 0.40   # 与 sign_real_link.CLS_MIN_CONF 保持一致，仅用于界面文案
_CURRENT_TOKENS = []
_CURRENT_SCENE = {
    "scene_name": "政务场景一：身份证挂失与补办咨询",
    "translation": "工作人员您好，我的身份证丢失了，请问怎么办理补办？",
    "ai_answer": "【政务导办回答】您好！补办身份证请携带户口本或驾驶证原件，先到右侧 2 号人像采集室免费拍照，随后前往 3 号户政窗口办理，工本费 40 元。",
}
_HISTORY_RECORDS = [
    {
        "id": 1,
        "time": "09:30:15",
        "type": "手语导办",
        "words": "你好 → 身份证 → 丢失 → 补办",
        "translation": "工作人员您好，我的身份证丢失了，请问怎么办理补办？",
        "reply": "【政务导办回答】请先到 2 号室采集人像，再到 3 号户政窗口办理。",
    }
]
_NEXT_RECORD_ID = 2


# ==============================================================================
# 后台摄像头预览线程
# ==============================================================================
# 原界面用 gr.Image 显示一张静态占位图，用户完全看不到自己——会误判成"摄像头没开"。
# 这里用一条常驻线程持续抓帧、画上检测框，再由 gradio 定时器推送到预览框。
# 线程只负责"看"，真正的识别仍由点【开始识别】触发，互不抢占：
# 识别开始前会调用 cam_stream.pause()，结束后 resume()，避免两处同时 read 同一设备。
DETECT_EVERY = 6      # 每 6 帧跑一次 YOLO（其余帧沿用上次框），大幅降 CPU
DETECT_IMGSZ = 480    # 预览检测用 480 分辨率，比 640 快一倍多，画框精度足够
PREVIEW_FRAME_GAP = 0.03   # 抓帧间隔≈30fps 上限，避免空转吃满单核
PREVIEW_W, PREVIEW_H = 760, 480   # 推送给浏览器的统一尺寸，控传输量


class CameraPreview:
    def __init__(self):
        self.cap = None
        self.frame = None            # 最新一帧（RGB，已画框）
        self.lock = threading.Lock()
        self.running = False
        self.enabled = True          # 识别期间置 False，暂停抓帧让出设备
        self.error = ""
        self._thread = None
        self._model = None
        # ---- 录制模式：用户在网页点「开始识别」后开始累积，点「停止」才结算 ----
        self.recording = False
        self.rec_stream = []         # 每帧的字母（None=未检出）
        self.rec_confs = []          # 与 rec_stream 对齐的置信度
        self.rec_started = 0.0
        self.rec_last_letter = None
        self.rec_last_box = None

    # ---------------- 录制控制 ----------------
    def rec_begin(self):
        """开始录制：清空缓冲，从此刻起每一帧都参与识别。"""
        with self.lock:
            self.rec_stream = []
            self.rec_confs = []
            self.rec_last_letter = None
            self.rec_started = time.time()
            self.recording = True
        print("[record] 开始录制，等待用户比划...", flush=True)

    def rec_end(self):
        """结束录制，返回 (letters, confs, diag)。"""
        with self.lock:
            self.recording = False
            stream = list(self.rec_stream)
            confs = list(self.rec_confs)
            dur = time.time() - self.rec_started
            box = self.rec_last_box
        print(f"[record] 结束录制：{len(stream)} 帧 / {dur:.1f}s", flush=True)
        return stream, confs, {"frames": len(stream), "seconds": round(dur, 1), "last_box": box}

    def _open(self):
        try:
            import sign_real_link
            if self._model is None:
                self._model = sign_real_link.get_yolo_model()
            cap, desc = sign_real_link.open_camera(verbose=False)
            with self.lock:
                self.cap = cap
                self.error = ""
            print(f"[preview] 实时预览已启动：{desc}")
            return True
        except Exception as exc:
            msg = str(exc).strip().splitlines()[0] if str(exc).strip() else "摄像头不可用"
            with self.lock:
                changed = (msg != self.error)
                self.error = msg
            if changed:      # 只在原因变化时打印，避免每秒刷屏
                print(f"[preview] 预览启动失败（后台每 3 秒自动重试）: {msg}")
            return False

    def _loop(self):
        self._open()
        fail = 0
        while self.running:
            if not self.enabled:
                time.sleep(0.1)
                continue
            with self.lock:
                cap = self.cap
            if cap is None:
                time.sleep(3.0)
                if self.error:
                    self._open()          # 每 3 秒重试，插上/关掉占用程序后能自愈
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                fail += 1
                if fail > 30:             # 连续读不到帧，重开设备
                    with self.lock:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        self.cap = None
                    fail = 0
                time.sleep(0.03)
                continue
            fail = 0
            try:
                drawn, boxes = self._draw(frame)
            except Exception:
                drawn, boxes = frame, []
            with self.lock:
                self.frame = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)

            # 录制中：把本帧的字母记进缓冲（录制时不降频，保证每个字母都被采到）
            if self.recording:
                letter, conf = (boxes[0][5], boxes[0][4]) if boxes else (None, 0.0)
                with self.lock:
                    if self.recording:
                        self.rec_stream.append(letter)
                        self.rec_confs.append(conf)
                        if boxes:
                            self.rec_last_box = boxes[0]
            time.sleep(PREVIEW_FRAME_GAP)   # 限制抓帧节奏，别把 CPU 跑满

    def _draw(self, frame):
        """在预览帧上画检测框，返回 (图像, 检测框列表)。

        【性能关键】YOLO 推理在 CPU 上约 100~200ms/帧，每帧都跑会把 CPU 打满，
        整个页面（按钮、面板、识别）都会跟着变卡。所以：
        - 平时预览：每 DETECT_EVERY 帧才跑一次 YOLO，其余帧沿用上次的框；
        - 录制中：每帧都跑，保证每个字母都被采到（这是识别准确性的前提）。
        """
        out = frame.copy()
        if self._model is None:
            return out, []

        every = 1 if self.recording else DETECT_EVERY
        self._fc = getattr(self, "_fc", 0) + 1
        due = (self._fc % every == 0) or getattr(self, "_last_boxes", None) is None
        if due:
            try:
                res = self._model(out, imgsz=DETECT_IMGSZ, conf=0.35, verbose=False)[0]
                boxes = []
                names = res.names
                if res.boxes is not None:
                    for b in res.boxes:
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        boxes.append((x1, y1, x2, y2, float(b.conf), str(names[int(b.cls)]).upper()))
                boxes.sort(key=lambda t: -t[4])
                self._last_boxes = boxes
            except Exception:
                self._last_boxes = getattr(self, "_last_boxes", [])

        boxes = getattr(self, "_last_boxes", [])
        if boxes:
            for x1, y1, x2, y2, c, label in boxes:
                color = (34, 197, 94) if c >= 0.6 else (59, 130, 246)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                tag = f"{label} {c:.2f}"
                cv2.rectangle(out, (x1, max(0, y1 - 22)), (x1 + 10 * len(tag), y1), color, -1)
                cv2.putText(out, tag, (x1 + 4, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(out, "No hand detected - move closer to camera", (14, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (59, 130, 246), 2, cv2.LINE_AA)

        # 录制中在画面顶部压一条录制指示
        if self.recording:
            el = time.time() - self.rec_started
            cv2.rectangle(out, (0, 0), (out.shape[1], 34), (190, 18, 60), -1)
            cv2.putText(out, f"REC  {el:4.1f}s   gesture now, click STOP when done",
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return out, boxes

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def pause(self):
        """识别前调用：停止抓帧并【释放设备】，否则识别流程打不开摄像头。
        Windows 摄像头同一时刻只允许一个进程/句柄占用，这里必须先 release。"""
        self.enabled = False
        time.sleep(0.1)
        with self.lock:
            cap = self.cap
            self.cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        time.sleep(0.25)      # 给驱动回收句柄留时间，否则紧接着 open 会失败

    def resume(self):
        self.enabled = True
        self._open()          # 立即重新打开，恢复预览

    def snapshot(self):
        """给 gradio 定时器调用：返回最新一帧；不可用时返回带说明文字的占位图。

        【性能】统一缩到 PREVIEW_W x PREVIEW_H 再返回。摄像头原生 1280x720 的图
        每 0.3 秒走一次 gradio 的 base64 传输，会导致页面明显发滞；
        缩到 760x480 后传输量降到约 1/3，肉眼看不出差别。
        """
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            err = self.error
        if frame is not None:
            h, w = frame.shape[:2]
            if w > PREVIEW_W:
                nh = int(h * PREVIEW_W / w)
                frame = cv2.resize(frame, (PREVIEW_W, nh), interpolation=cv2.INTER_AREA)
            return frame
        img = np.full((PREVIEW_H, PREVIEW_W, 3), (241, 245, 249), dtype=np.uint8)
        msg = "正在连接摄像头..." if not err else "摄像头不可用"
        cv2.rectangle(img, (0, 0), (PREVIEW_W, 38), (190, 18, 60) if err else (30, 64, 175), -1)
        cv2.putText(img, f"CAMERA | {msg}", (18, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if err:
            for i, line in enumerate(_wrap_cn(err, 42)[:4]):
                cv2.putText(img, line, (20, 90 + i * 34), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (60, 60, 60), 1, cv2.LINE_AA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _wrap_cn(text, width):
    """按字符数硬折行（中文没有空格，不能用 textwrap）。"""
    return [text[i:i + width] for i in range(0, len(text), width)]





CAM_PREVIEW = CameraPreview()


def render_preview_frame_a(mode_text="摄像头实时采集待命", active=False, token_info=None, conf_thresh=0.60, failed=False):
    """生成带科技蓝检测框或红色识别失败警示框的预览画面（760x480 固定大画幅）"""
    h, w = 480, 760
    img = np.full((h, w, 3), (241, 245, 249), dtype=np.uint8)

    # 浅灰网格参考线
    for x in range(0, w, 95):
        cv2.line(img, (x, 0), (x, h), (226, 232, 240), 1)
    for y in range(0, h, 95):
        cv2.line(img, (0, y), (w, y), (226, 232, 240), 1)

    # 人物剪影占位
    cv2.circle(img, (380, 195), 64, (203, 213, 225), -1)
    cv2.ellipse(img, (380, 425), (150, 120), 0, 180, 360, (203, 213, 225), -1)

    # 顶部状态条（失败时切换为醒目玫红色）
    top_bar_color = (190, 18, 60) if failed else (30, 64, 175)
    cv2.rectangle(img, (0, 0), (w, 38), top_bar_color, -1)
    cv2.putText(
        img,
        f"YOLO26 GOV-SERVICE SIGN AGENT  |  {mode_text}  (Thresh >= {conf_thresh:.2f})",
        (18, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if failed:
        x1, y1, x2, y2 = 210, 115, 530, 360
        err_color = (225, 29, 72)
        cv2.rectangle(img, (x1, y1), (x2, y2), err_color, 2)
        cv2.rectangle(img, (x1, y1 - 28), (x1 + 310, y1), err_color, -1)
        cv2.putText(
            img,
            f"RECOGNITION FAILED | Max Conf: 0.41 < {conf_thresh:.2f}",
            (x1 + 8, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(img, (120, 395), (640, 442), (255, 241, 242), -1)
        cv2.rectangle(img, (120, 395), (640, 442), err_color, 1)
        cv2.putText(
            img,
            "Warning: Hand out of ROI or motion blurred. Please retry!",
            (142, 424),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (159, 18, 57),
            1,
            cv2.LINE_AA,
        )
    elif active and token_info:
        x1, y1, x2, y2 = token_info["bbox"]
        conf = token_info["conf"]
        box_color = (37, 99, 235)
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

        cl = 18
        for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(img, (cx, cy), (cx + dx * cl, cy), (29, 78, 216), 4)
            cv2.line(img, (cx, cy), (cx, cy + dy * cl), (29, 78, 216), 4)

        cv2.rectangle(img, (x1, y1 - 28), (x1 + 245, y1), box_color, -1)
        cv2.putText(
            img,
            f"Hand ROI | Conf: {conf:.2f} [ Anti-Shake ]",
            (x1 + 8, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        pts = [(265, 235), (285, 190), (310, 180), (335, 190), (355, 220), (310, 270)]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[-1], (59, 130, 246), 2, cv2.LINE_AA)
            cv2.circle(img, pts[i], 5, (30, 64, 175), -1, cv2.LINE_AA)
        cv2.circle(img, pts[-1], 6, (29, 78, 216), -1, cv2.LINE_AA)
    else:
        cv2.putText(
            img,
            "Standby: Click [Start Recognition] to begin government sign guide",
            (75, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (100, 116, 139),
            1,
            cv2.LINE_AA,
        )

    return img


def build_status_pill_a(state="idle", message="系统空闲：等待启动指令", auto_tts_mode="🔊 自动朗读"):
    styles = {
        "idle": ("#64748B", "#F1F5F9", "#E2E8F0", "待命状态"),
        "running": ("#10B981", "#ECFDF5", "#A7F3D0", "识别成功"),
        "paused": ("#F59E0B", "#FFFBEB", "#FDE68A", "已暂停"),
        "tts": ("#2563EB", "#EFF6FF", "#BFDBFE", "语音朗读中"),
        "error": ("#E11D48", "#FFF1F2", "#FECDD3", "识别失败提示"),
    }
    dot_color, bg_color, border_color, badge_text = styles.get(state, styles["idle"])
    agent_mode = "已连接 agent.py" if HAS_EXTERNAL_AGENT else "内置模拟引擎"
    tts_badge = "🔊 自动播报:开" if "自动" in str(auto_tts_mode) else "🔇 静音模式"

    status_html = f"""
    <div style="display:flex;align-items:center;justify-content:space-between;background:{bg_color};
                border:1px solid {border_color};border-radius:8px;padding:9px 16px;font-size:13.5px;color:#1E293B;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                         background:{dot_color};box-shadow:0 0 0 4px {dot_color}28;"></span>
            <strong style="color:#0F172A;">[{badge_text}]</strong>
            <span>{message}</span>
        </div>
        <span style="font-size:12px;color:#1E40AF;background:#FFFFFF;padding:3px 10px;
                     border-radius:6px;border:1px solid #BFDBFE;font-weight:600;">
            {tts_badge} | {agent_mode}
        </span>
    </div>
    """
    # 处于模拟数据状态时，横幅跟着状态栏一起渲染，保证任何交互后提示都不会消失
    if _MOCK_STATE["is_mock"]:
        return build_mock_banner(_MOCK_STATE["reason"]) + status_html

    return status_html


# 启动时先判断一次，进界面就能看到告警，不必等第一次点击。
# 优先按开关判断；开关开着就实际检查权重文件（只查路径，不加载模型）。仍不行则在
# 第一次识别回退时由 start() 再更新一次。
def _detect_initial_mock_state():
    if not getattr(vision_engine, "USE_REAL_YOLO_MODEL", False):
        return True, "USE_REAL_YOLO_MODEL = False，未启用真实模型"
    try:
        import sign_real_link
        sign_real_link.resolve_yolo_path()
        sign_real_link.resolve_cls_path()
        return False, ""
    except Exception as exc:
        # 权重缺失的报错是多行的（后面跟一长串已尝试路径），横幅里只取第一句
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else "真实模型不可用"
        return True, first_line.split("。")[0].rstrip("：: ") + "。"


_MOCK_STATE["is_mock"], _MOCK_STATE["reason"] = _detect_initial_mock_state()


def build_vocab_bar_html():
    """
    词库提示栏。
    YOLO 输出的类别名是拼音，这里展示成「中文 + 小字拼音」，
    既方便群众看懂，也让评委看到模型实际输出的是什么。
    """
    vocab = _load_vocab() if HAS_PINYIN_CODEC else None

    if vocab:
        chips = []
        for entry in vocab["entries"]:
            if entry["hanzi"] == vocab["confirm_hanzi"]:
                continue
            chips.append(
                f"<span style='background:#EFF6FF;color:#1E40AF;border:1px solid #BFDBFE;"
                f"padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600;'>"
                f"{entry['hanzi']}"
                f"<span style='font-family:monospace;font-size:10.5px;color:#60A5FA;"
                f"font-weight:400;margin-left:4px;'>{entry['pinyin']}</span></span>"
            )
        confirm_chip = (
            f"<span style='background:#FEF3C7;color:#B45309;border:1px solid #FDE68A;"
            f"padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600;'>"
            f"✋ {vocab['confirm_hanzi']}（确认手势）</span>"
        )
        badges = "".join(chips) + confirm_chip
        count = len(vocab["entries"]) - 1
    else:
        badges = "".join([
            f"<span style='background:#EFF6FF;color:#1E40AF;border:1px solid #BFDBFE;"
            f"padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600;'>{w}</span>"
            for w in GOV_SUPPORTED_WORDS
        ])
        count = len(GOV_SUPPORTED_WORDS)

    return f"""
    <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:4px solid #3B82F6;
                border-radius:8px;padding:8px 14px;display:flex;align-items:center;flex-wrap:wrap;gap:6px;">
        <span style="font-size:12.5px;font-weight:700;color:#1E293B;margin-right:4px;">
            📖 当前支持政务高频手语词库 ({count}词)：
        </span>
        {badges}
    </div>
    """


def build_confidence_bars_a(tokens, conf_thresh=0.60, filtered_count=0, failed_reason=""):
    if failed_reason:
        return f"""
        <div style="padding:12px 14px;background:#FFF1F2;border:1px solid #FECDD3;border-radius:8px;color:#881337;font-size:12.5px;">
            <div style="font-weight:700;color:#BE123C;font-size:13.5px;margin-bottom:4px;">
                ⚠️ 手语词识别失败（有效检出词数：0）
            </div>
            <div style="line-height:1.5;margin-bottom:6px;">{failed_reason}</div>
            <div style="background:#FFFFFF;border:1px solid #FFE4E6;padding:5px 10px;border-radius:6px;font-size:11.5px;color:#9F1239;">
                💡 建议操作：① 请正对摄像头并将双手保持在采集框内；② 适当调低「置信度过滤阈值」滑块；③ 点击【▶ 开始识别】重试。
            </div>
        </div>
        """
    if not tokens:
        return """
        <div style="padding:20px 12px;text-align:center;color:#94A3B8;font-size:13px;
                    background:#F8FAFC;border-radius:6px;border:1px dashed #CBD5E1;">
            等待开始识别后展示政务手语词及分类置信度（自动去重防抖）...
        </div>
        """
    rows_html = []
    for idx, item in enumerate(tokens, 1):
        pct = int(item["conf"] * 100)
        raw_word = str(item["word"])
        shown = _display_word(raw_word)
        # 拼音还原成中文时，把小字拼音附在后面，方便现场解释"模型输出的是拼音"
        pinyin_hint = (
            f"<span style='font-family:monospace;font-size:11px;color:#94A3B8;"
            f"font-weight:400;margin-left:5px;'>{raw_word}</span>"
            if shown != raw_word else ""
        )
        rows_html.append(f"""
        <div style="margin-bottom:8px;">
            <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px;">
                <span style="font-weight:600;color:#1E293B;">#{idx} 静态手语词：「{shown}」{pinyin_hint}</span>
                <span style="font-family:monospace;font-weight:600;color:#1E40AF;">{pct}% ({item['conf']:.2f}) ✓已通过阈值</span>
            </div>
            <div style="width:100%;height:7px;background:#E2E8F0;border-radius:999px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#2563EB,#3B82F6);
                            border-radius:999px;"></div>
            </div>
        </div>
        """)
    if filtered_count > 0:
        rows_html.append(f"""
        <div style="font-size:12px;color:#D97706;background:#FFFBEB;border:1px solid #FDE68A;
                    padding:4px 10px;border-radius:6px;margin-top:6px;">
            🛡️ 阈值防抖提示：已自动过滤 {filtered_count} 个置信度低于 {conf_thresh:.2f} 的疑似误触动作
        </div>
        """)
    return f"<div style='padding:2px;'>{''.join(rows_html)}</div>"


def build_large_translation_html(translation_text, ai_answer_text, staff_reply_text="", failed=False):
    if failed:
        return """
        <div style="background:#FFF1F2;border:1px solid #FECDD3;border-radius:8px;padding:14px;">
            <div style="font-size:12px;font-weight:700;color:#BE123C;margin-bottom:6px;">
                ⚠️ 识别未成功 · 温馨提示（大字展示给听障办事群众）：
            </div>
            <div style="font-size:20px;font-weight:800;color:#881337;line-height:1.45;
                        background:#FFFFFF;padding:12px 14px;border-radius:6px;border-left:5px solid #E11D48;">
                抱歉，刚才未能看清您的手语动作。请您将双手放在画面框内、放慢动作重新比划一次，好吗？
            </div>
            <div style="margin-top:8px;font-size:12px;color:#9F1239;background:#FFE4E6;padding:6px 10px;border-radius:6px;font-weight:600;">
                🤖 政务导办辅助建议：若多次识别失败，您也可以直接指认上方「16个政务高频词库」或通过下方窗口双向文字屏沟通。
            </div>
        </div>
        """
    if not translation_text:
        return """
        <div style="padding:26px 16px;text-align:center;color:#94A3B8;background:#F8FAFC;
                    border:1px dashed #CBD5E1;border-radius:8px;font-size:14px;">
            等待 Agent 智能体将离散手语词合成为大字号中文语句与政务导办回答...
        </div>
        """

    ai_block = ""
    if FEATURE_FLAGS["SHOW_AI_GOV_ANSWER"] and ai_answer_text:
        ai_block = f"""
        <div style="margin-top:10px;background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;padding:10px 14px;">
            <div style="font-size:12px;font-weight:700;color:#1E40AF;margin-bottom:4px;">
                🤖 政务 AI 智能体导办建议：
            </div>
            <div style="font-size:15px;font-weight:600;color:#1E3A8A;line-height:1.55;">
                {ai_answer_text}
            </div>
        </div>
        """

    staff_block = ""
    if FEATURE_FLAGS["SHOW_TWO_WAY_REPLY"] and staff_reply_text:
        staff_block = f"""
        <div style="margin-top:10px;background:#ECFDF5;border:1px solid #A7F3D0;border-radius:8px;padding:10px 14px;">
            <div style="font-size:12px;font-weight:700;color:#047857;margin-bottom:4px;">
                💬 窗口工作人员双向回复（大字展示给听障办事群众）：
            </div>
            <div style="font-size:18px;font-weight:700;color:#065F46;line-height:1.5;">
                “{staff_reply_text}”
            </div>
        </div>
        """

    return f"""
    <div style="background:#F8FAFC;border:1px solid #CBD5E1;border-radius:8px;padding:14px;">
        <div style="font-size:12px;font-weight:700;color:#475569;margin-bottom:6px;">
            🗣️ 手语原意中文翻译（21px 大字号高清展示）：
        </div>
        <div style="font-size:21px;font-weight:800;color:#0F172A;line-height:1.45;
                    background:#FFFFFF;padding:12px 14px;border-radius:6px;border-left:5px solid #2563EB;
                    box-shadow:0 1px 2px rgba(0,0,0,0.03);">
            {translation_text}
        </div>
        {ai_block}
        {staff_block}
    </div>
    """


def build_history_html(records):
    if not records:
        return """
        <div style="padding:16px;text-align:center;color:#94A3B8;font-size:12.5px;
                    background:#F8FAFC;border:1px dashed #CBD5E1;border-radius:6px;">
            暂无历史翻译记录
        </div>
        """
    items = []
    for r in reversed(records):
        items.append(f"""
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:6px;padding:8px 12px;margin-bottom:7px;font-size:12.5px;">
            <div style="display:flex;justify-content:space-between;color:#64748B;margin-bottom:3px;">
                <span><strong>#{r['id']} [{r['type']}]</strong> | 识别词：{r['words']}</span>
                <span style="font-family:monospace;">{r['time']}</span>
            </div>
            <div style="color:#0F172A;font-weight:700;font-size:13.5px;">译文：{r['translation']}</div>
            <div style="color:#1E40AF;font-size:12px;margin-top:2px;">{r['reply']}</div>
        </div>
        """)
    return f"<div style='max-height:190px;overflow-y:auto;padding-right:4px;'>{''.join(items)}</div>"


def call_agent_semantic_parser(word_list, scene_data):
    if not word_list:
        return "", ""
    if HAS_EXTERNAL_AGENT and hasattr(external_agent, "parse_semantics"):
        try:
            res = external_agent.parse_semantics(word_list)
            if isinstance(res, tuple) and len(res) == 2:
                return res[0], res[1]
            return str(res), scene_data.get("ai_answer", "")
        except Exception as exc:
            return f"{scene_data.get('translation', '')} [agent.py 提示: {exc}]", scene_data.get("ai_answer", "")

    joined = "、".join(_display_word(w) for w in word_list)
    return (
        scene_data.get("translation", f"您好，我想办理关于「{joined}」的政务事项。"),
        scene_data.get("ai_answer", f"【政务导办回答】请携带本人有效身份证件前往综合窗口办理（关键词：{joined}）。"),
    )


def append_history_record(rec_type, words_str, translation, reply):
    global _NEXT_RECORD_ID
    _HISTORY_RECORDS.append({
        "id": _NEXT_RECORD_ID,
        "time": time.strftime("%H:%M:%S"),
        "type": rec_type,
        "words": words_str,
        "translation": translation,
        "reply": reply,
    })
    _NEXT_RECORD_ID += 1


def _hardware_fail(conf_thresh=0.60, auto_tts_mode="🔊 自动朗读", err=""):
    """摄像头/权重等硬故障的界面反馈：与普通识别失败区分开，直接显示真实原因。"""
    global _CURRENT_TOKENS
    _CURRENT_TOKENS = []
    frame = render_preview_frame_a(f"CAMERA ERROR | {err[:52]}", active=False, conf_thresh=conf_thresh, failed=True)
    reason = f"⚠️ 系统故障（非手语问题）：{err}"
    conf_html = build_confidence_bars_a([], conf_thresh, 0, failed_reason=reason)
    word_text = "⚠️ 系统故障：请看右侧红色提示中的具体原因"
    large_html = build_large_translation_html("", "", failed=True)
    is_auto = "自动" in str(auto_tts_mode)
    msg = f"摄像头/系统故障：{err}" + ("（已尝试自动播报提示）" if is_auto else "（静音模式）")
    status_html = build_status_pill_a("error", msg, auto_tts_mode)
    return frame, conf_html, word_text, "系统出现故障，请工作人员检查摄像头。", large_html, status_html, build_history_html(_HISTORY_RECORDS)


def simulate_failure(conf_thresh=0.60, auto_tts_mode="🔊 自动朗读", diag=None):
    """触发识别失败/低置信度四联动告警状态。diag 为采集诊断信息，用于定位卡在哪一步。"""
    global _CURRENT_TOKENS
    _CURRENT_TOKENS = []

    # 根据诊断数据判断失败发生在哪一层，给出可直接执行的建议
    tip = ""
    if diag:
        frames = diag.get("frames", 0)
        sampled = diag.get("sampled", 0)
        detected = diag.get("detected", 0)
        max_conf = diag.get("max_conf", 0.0)
        letters = diag.get("letters") or []
        secs = diag.get("seconds", 0)
        if frames == 0:
            tip = f"⚠️ 摄像头打开了，但 {secs}s 内一帧画面都没读到（设备被占用或驱动异常）。"
        elif detected == 0:
            tip = (f"⚠️ 采集 {secs}s / {frames} 帧，其中 {sampled} 帧送检，"
                   f"但**一帧都没检出人手**。请把手抬到画面中央、距离镜头 40~60cm，"
                   f"让整只手占画面 1/3 以上，再点开始识别。")
        elif not letters:
            tip = (f"⚠️ 采集 {frames} 帧，检出手 {detected} 帧，但字母没稳定出来"
                   f"（最高置信度 {max_conf:.2f} < {CLS_MIN_CONF_DISPLAY}）。"
                   f"请把手掌正对镜头、保持姿势 1 秒不要动。")
        else:
            tip = (f"⚠️ 识别到字母 {''.join(str(x).upper() for x in letters)}，"
                   f"但拼不成词库里的词（均值 {max_conf:.2f}）。"
                   f"请放慢速度，每个字母之间手离开画面停 0.5 秒。")

    reason = tip or "未能匹配有效政务手语词，请正对摄像头重新比划。"
    frame = render_preview_frame_a("WARNING | 识别失败请重试", active=False, conf_thresh=conf_thresh, failed=True)
    conf_html = build_confidence_bars_a([], conf_thresh, 0, failed_reason=reason)
    word_text = "⚠️ 识别失败：未检出有效手语词序列（请重新比划）"
    large_html = build_large_translation_html("", "", failed=True)
    fail_tts = "抱歉，未能看清您的手语动作，请您放慢动作重新比划一次。"
    is_auto = "自动" in str(auto_tts_mode)
    msg = (
        "识别失败：未检测到达到置信度阈值的有效手语词，已自动语音播报重试提示！"
        if is_auto
        else "识别失败：手部动作模糊或置信度低于阈值，请正对摄像头重新打手语（当前为静音模式）"
    )
    status_html = build_status_pill_a("error", msg, auto_tts_mode)
    return frame, conf_html, word_text, fail_tts, large_html, status_html, build_history_html(_HISTORY_RECORDS)


def start(source_type, video_file, conf_thresh=0.60, auto_tts_mode="🔊 自动朗读"):
    """点击【▶ 开始识别】：进入录制状态，由用户自己决定何时停止。

    【流程改动原因】
    旧版点完就开始采样、3.5 秒一到立刻结算，用户还没抬起手就判定"识别失败"，
    体验极差且必然失败。现在改成：
      点开始 → 一直录（画面顶部显示 REC 计时）→ 用户比完自己点【停止识别】→ 结算
    这样节奏完全由使用者掌握，不再有"时限压力"。
    """
    global _CURRENT_TOKENS

    print(f"\n{'='*50}\n[record] << 收到【开始识别】点击 >> source={source_type}\n{'='*50}", flush=True)

    if "视频" in str(source_type) and video_file:
        # 上传视频模式没有实时录制概念，一次性识别后直接结算
        res = _recognize_from_file(video_file, conf_thresh, auto_tts_mode)
        return res  # 7 元组，与 btn_start 的输出一一对应

    if not CAM_PREVIEW.running or not CAM_PREVIEW.enabled:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                build_status_pill_a("error", "摄像头未就绪：请确认摄像头已连接、未被其它程序占用"),
                gr.update(), gr.update())

    CAM_PREVIEW.rec_begin()
    _CURRENT_TOKENS = []
    status_html = build_status_pill_a(
        "running", "🔴 正在录制：请开始比划手语，比完后点击【⏸ 停止识别】出结果", auto_tts_mode
    )
    return (gr.update(), gr.update(), gr.update(), gr.update(), status_html, gr.update(), gr.update())


def stop(conf_thresh=0.60, auto_tts_mode="🔊 自动朗读"):
    """点击【⏸ 停止识别】：结束录制并结算识别结果。"""
    global _CURRENT_TOKENS, _CURRENT_SCENE

    if not CAM_PREVIEW.recording:
        return (gr.update(), gr.update(), gr.update(), gr.update(),
                build_status_pill_a("paused", "当前没有正在进行的录制，请先点击【▶ 开始识别】"),
                gr.update(), gr.update())

    stream, confs, diag = CAM_PREVIEW.rec_end()
    if not stream:
        return simulate_failure(conf_thresh, auto_tts_mode,
                                diag={"frames": 0, "sampled": 0, "detected": 0,
                                      "max_conf": 0.0, "letters": [], "seconds": 0})

    import sign_real_link as _srl
    letters, lconfs = _srl.compress_runs(stream, confs)
    words = _srl.spell_letters(letters, _srl.load_pinyin_targets())

    detected = sum(1 for x in stream if x is not None)
    diag.update({
        "sampled": len(stream),
        "detected": detected,
        "letters": letters,
        "max_conf": float(max(confs)) if confs else 0.0,
    })

    if not words:
        return simulate_failure(conf_thresh, auto_tts_mode, diag=diag)

    # 置信度用该字母段的平均分，而不是全局最大值，避免一个高分带整体过关
    conf_map = {}
    for w in words:
        idx = [i for i, l in enumerate(letters) if l and l.lower() in w.lower()]
        seg = [lconfs[i] for i in idx if i < len(lconfs)] or lconfs or [0.5]
        conf_map[w] = float(min(0.99, sum(seg) / len(seg)))

    bbox = diag.get("last_box")
    bbox = (bbox[0], bbox[1], bbox[2], bbox[3]) if bbox else (190, 110, 390, 340)
    all_tokens = [{"word": w, "conf": conf_map.get(w, 0.5), "bbox": bbox} for w in words]

    valid_tokens = [t for t in all_tokens if t["conf"] >= conf_thresh]
    if not valid_tokens:
        return simulate_failure(conf_thresh, auto_tts_mode, diag=diag)

    _CURRENT_TOKENS = list(valid_tokens)
    scene = {"scene_name": "字母拼写实时识别", "letters": letters,
             "translation": "", "ai_answer": ""}
    _CURRENT_SCENE = scene

    word_list = [t["word"] for t in _CURRENT_TOKENS]
    conf_html = build_confidence_bars_a(_CURRENT_TOKENS, conf_thresh, len(all_tokens) - len(valid_tokens))
    word_text = _display_token_seq(_CURRENT_TOKENS)
    translation, ai_answer = call_agent_semantic_parser(word_list, scene)
    combined_text = f"{translation} {ai_answer}" if FEATURE_FLAGS["SHOW_AI_GOV_ANSWER"] else translation
    large_html = build_large_translation_html(translation, ai_answer)
    append_history_record("手语识别", _display_seq(word_list), translation, ai_answer)
    hist_html = build_history_html(_HISTORY_RECORDS)

    is_auto = "自动" in str(auto_tts_mode)
    tts_note = "，已自动语音播报" if is_auto else "（🔇静音模式）"
    status_html = build_status_pill_a(
        "running",
        f"识别完成：录了 {diag['seconds']}s / {diag['frames']} 帧，"
        f"字母 {'-'.join(str(x).upper() for x in letters)} → {len(_CURRENT_TOKENS)} 个词{tts_note}",
        auto_tts_mode,
    )
    return gr.update(), conf_html, word_text, combined_text, large_html, status_html, hist_html


def _recognize_from_file(video_file, conf_thresh, auto_tts_mode):
    """上传视频模式：一次性识别（保留原逻辑）。"""
    global _CURRENT_TOKENS, _CURRENT_SCENE
    try:
        custom_frame, all_tokens, scene = vision_engine.predict_sign_tokens(
            "上传本地视频文件", video_file, conf_thresh)
    except Exception as exc:
        err = str(exc).strip().splitlines()[0] if str(exc).strip() else "未知错误"
        return _hardware_fail(conf_thresh, auto_tts_mode, err)

    _CURRENT_SCENE = scene
    valid_tokens = [t for t in all_tokens if t["conf"] >= conf_thresh]
    if not valid_tokens:
        return simulate_failure(conf_thresh, auto_tts_mode, diag=scene.get("diag"))
    _CURRENT_TOKENS = list(valid_tokens)
    word_list = [t["word"] for t in _CURRENT_TOKENS]
    conf_html = build_confidence_bars_a(_CURRENT_TOKENS, conf_thresh, len(all_tokens) - len(valid_tokens))
    word_text = _display_token_seq(_CURRENT_TOKENS)
    translation, ai_answer = call_agent_semantic_parser(word_list, scene)
    combined_text = f"{translation} {ai_answer}" if FEATURE_FLAGS["SHOW_AI_GOV_ANSWER"] else translation
    large_html = build_large_translation_html(translation, ai_answer)
    append_history_record("手语识别", _display_seq(word_list), translation, ai_answer)
    status_html = build_status_pill_a("running", "已完成视频文件识别", auto_tts_mode)
    return gr.update(), conf_html, word_text, combined_text, large_html, status_html, build_history_html(_HISTORY_RECORDS)


def tts(agent_text):
    if not agent_text or not str(agent_text).strip():
        return build_status_pill_a("paused", "当前尚无翻译文本可供语音朗读，请先点击【开始识别】")
    clip = str(agent_text).splitlines()[0][:24]
    return build_status_pill_a("tts", f"正在通过浏览器语音引擎直接朗读：「{clip}...」")


def clear():
    global _CURRENT_TOKENS
    _CURRENT_TOKENS = []
    frame = render_preview_frame_a("摄像头实时采集待命", active=False)
    conf_html = build_confidence_bars_a([])
    large_html = build_large_translation_html("", "")
    status_html = build_status_pill_a("idle", "系统空闲：已清空当前识别词汇与翻译展示区")
    return frame, conf_html, "", "", large_html, status_html


def undo_last_word(conf_thresh):
    global _CURRENT_TOKENS
    if not _CURRENT_TOKENS:
        status_html = build_status_pill_a("paused", "撤回提示：当前词序列已为空，无可撤回词汇")
        return build_confidence_bars_a([]), "", build_large_translation_html("", ""), "", status_html

    removed = _CURRENT_TOKENS.pop()
    conf_html = build_confidence_bars_a(_CURRENT_TOKENS, conf_thresh, 0)
    word_text = _display_token_seq(_CURRENT_TOKENS)

    if not _CURRENT_TOKENS:
        status_html = build_status_pill_a("paused", f"已撤回词汇「{_display_word(removed['word'])}」，当前词序列已清空")
        return conf_html, "", build_large_translation_html("", ""), "", status_html

    word_list = [t["word"] for t in _CURRENT_TOKENS]
    translation, ai_answer = call_agent_semantic_parser(word_list, _CURRENT_SCENE)
    combined_text = f"{translation} {ai_answer}" if FEATURE_FLAGS["SHOW_AI_GOV_ANSWER"] else translation
    large_html = build_large_translation_html(translation, ai_answer)
    status_html = build_status_pill_a(
        "running", f"已撤回错误识别词「{_display_word(removed['word'])}」，可继续攒词或点击【确认成句】重新生成翻译"
    )
    return conf_html, word_text, large_html, combined_text, status_html


def confirm_sentence():
    if not _CURRENT_TOKENS:
        return (
            build_large_translation_html("", ""),
            "",
            build_status_pill_a("paused", "请先识别至少一个手语词后再点击【确认成句】"),
            build_history_html(_HISTORY_RECORDS),
        )
    word_list = [t["word"] for t in _CURRENT_TOKENS]
    translation, ai_answer = call_agent_semantic_parser(word_list, _CURRENT_SCENE)
    combined_text = f"{translation} {ai_answer}" if FEATURE_FLAGS["SHOW_AI_GOV_ANSWER"] else translation
    large_html = build_large_translation_html(translation, ai_answer)
    append_history_record("手动确认成句", _display_seq(word_list), translation, ai_answer)
    status_html = build_status_pill_a("running", f"已手动确认 {len(word_list)} 个手语词，Agent 已重新生成中文译文与政务导办回答")
    return large_html, combined_text, status_html, build_history_html(_HISTORY_RECORDS)


def send_staff_reply(reply_text):
    if not reply_text or not str(reply_text).strip():
        return (
            build_large_translation_html(_CURRENT_SCENE["translation"], _CURRENT_SCENE["ai_answer"], ""),
            build_status_pill_a("paused", "请输入或选择一条工作人员回复语"),
            build_history_html(_HISTORY_RECORDS),
        )
    clean_reply = str(reply_text).strip()
    word_list = [t["word"] for t in _CURRENT_TOKENS] if _CURRENT_TOKENS else ["窗口双向对话"]
    translation, ai_answer = (
        call_agent_semantic_parser(word_list, _CURRENT_SCENE)
        if _CURRENT_TOKENS
        else ("（群众等待窗口指引）", "【政务导办】窗口双向文字屏已激活")
    )
    large_html = build_large_translation_html(translation, ai_answer, clean_reply)
    append_history_record("窗口双向回复", "工作人员回复", translation, f"【工作人员回复】{clean_reply}")
    status_html = build_status_pill_a("running", f"已向听障群众大字展示工作人员回复：「{clean_reply}」")
    return large_html, status_html, build_history_html(_HISTORY_RECORDS)


def delete_last_history():
    if _HISTORY_RECORDS:
        removed = _HISTORY_RECORDS.pop()
        msg = f"已删除第 #{removed['id']} 条历史翻译记录"
    else:
        msg = "当前历史记录已为空"
    return build_history_html(_HISTORY_RECORDS), build_status_pill_a("idle", msg)


def clear_all_history():
    _HISTORY_RECORDS.clear()
    return build_history_html(_HISTORY_RECORDS), build_status_pill_a("idle", "已清空全部历史翻译与交互记录")


def toggle_input_source(source_type, conf_thresh):
    is_file = "本地视频" in str(source_type)
    mode_label = "本地视频回放模式" if is_file else "摄像头实时采集待命"
    msg = "已切换至【本地视频上传】模式" if is_file else "已切换至【本机摄像头】实时采集模式"
    return (
        gr.update(visible=is_file),
        render_preview_frame_a(mode_label, active=False, conf_thresh=conf_thresh),
        build_status_pill_a("idle", msg),
    )


CUSTOM_CSS_A = """
.gradio-container {
    background-color: #F8FAFC !important;
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}
.header-banner-a {
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
    padding: 16px 24px;
    border-radius: 10px;
    color: #FFFFFF;
    box-shadow: 0 4px 12px rgba(30, 64, 175, 0.12);
    margin-bottom: 8px;
}
.header-banner-a h1 {
    font-size: 23px;
    font-weight: 700;
    margin: 0 0 4px 0;
    color: #FFFFFF !important;
}
.header-banner-a p {
    font-size: 13px;
    margin: 0;
    color: #DBEAFE !important;
}
.module-card-a {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-left: 4px solid #2563EB !important;
    border-radius: 8px !important;
    padding: 12px 15px !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
}
.video-card-a {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    padding: 14px !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
}
.right-scroll-col-a {
    max-height: 680px !important;
    overflow-y: auto !important;
    padding-right: 6px !important;
}
.right-scroll-col-a::-webkit-scrollbar {
    width: 6px;
}
.right-scroll-col-a::-webkit-scrollbar-thumb {
    background: #CBD5E1;
    border-radius: 999px;
}

/* ---------------------------------------------------------------------------
   强制浅色外观：本系统是政务大厅大屏演示，必须白底深字。
   浏览器/系统处于深色模式时，gradio 会把正文变白，与我们的白色卡片撞色导致看不见。
   这里同时覆盖 gradio 的主题变量与卡片内文字颜色，双保险。
--------------------------------------------------------------------------- */
:root, .dark, html.dark {
    --body-background-fill: #F8FAFC;
    --block-background-fill: #FFFFFF;
    --body-text-color: #0F172A;
    --block-label-text-color: #334155;
    --block-title-text-color: #0F172A;
    --input-background-fill: #FFFFFF;
}
.gradio-container { color: #1E293B !important; }
.module-card-a h1, .module-card-a h2, .module-card-a h3, .module-card-a h4,
.video-card-a h1, .video-card-a h2, .video-card-a h3, .video-card-a h4 {
    color: #0F172A !important;
}
body, .gradio-container { background: #F8FAFC !important; }

/* gradio 组件块统一白底：深色模式下否则会变成黑色块，卡片看起来是"破的" */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .wrap {
    background: #FFFFFF !important;
    border-color: #E2E8F0 !important;
}
.gradio-container .primary {
    background: #2563EB !important;
    border-color: #1D4ED8 !important;
    color: #FFFFFF !important;
}
.gradio-container input[type="radio"],
.gradio-container input[type="checkbox"] {
    accent-color: #2563EB;
}

/* 单选/多选组：深色模式下选项会变成深色块 + 橙色高亮，统一成白底蓝点 */
.gradio-container {
    --color-accent: #2563EB;
    --color-accent-soft: #EFF6FF;
    --primary-500: #2563EB;
}
.gradio-container .wrap > label,
.gradio-container .wrap label {
    background: #FFFFFF !important;
    border-color: #E2E8F0 !important;
    color: #0F172A !important;
}
"""

blocks_kwargs = {"title": "手语视界—基于YOLO26的政务服务无障碍智能体"}
try:
    if int(gr.__version__.split(".")[0]) < 5:
        blocks_kwargs["css"] = CUSTOM_CSS_A
except Exception:
    pass

with gr.Blocks(**blocks_kwargs) as demo:
    # ----------------------------------------------------------------------
    # 样式注入（版本无关）
    # gradio 6 起 launch() 不再接受 css 参数、Blocks(css=) 也被弱化，
    # 导致整套卡片/横幅样式失效（白字白底、卡片变暗）。
    # 用 <style> 注入到页面里，旧版本（4/5）与新版本（6+）都能生效。
    # ----------------------------------------------------------------------
    gr.HTML(f"<style>{CUSTOM_CSS_A}</style>")

    gr.HTML("""
    <div class="header-banner-a">
        <h1>手语视界—基于YOLO26的政务服务无障碍智能体
            <span style="font-size:13px;background:#F59E0B;color:#fff;padding:3px 9px;
                         border-radius:6px;vertical-align:middle;margin-left:10px;">录制版 V13</span>
        </h1>
        <p>系统链路：摄像头/本地视频输入 → YOLO26 手部检测与去重 → 静态手语词识别 → Agent 语义翻译与政务导办应答 → 双向交互与语音朗读</p>
    </div>
    """)

    status_display = gr.HTML(
        value=build_status_pill_a("idle", "系统空闲：已自动挂载 vision_engine.py 与 agent.py，无显卡环境可直接运行联调")
    )

    if FEATURE_FLAGS["SHOW_VOCAB_BAR"]:
        gr.HTML(value=build_vocab_bar_html())

    with gr.Row():
        # 左栏（60%）：固定大尺寸视频预览（480px高）与底部控制
        with gr.Column(scale=12, min_width=520):
            with gr.Group(elem_classes=["video-card-a"]):
                with gr.Row():
                    source_selector = gr.Radio(
                        choices=["本机摄像头实时采集", "上传本地视频文件"],
                        value="本机摄像头实时采集",
                        label="输入源切换",
                        scale=5,
                    )
                    conf_slider = gr.Slider(
                        minimum=0.40,
                        maximum=0.95,
                        value=0.60,
                        step=0.05,
                        label="置信度过滤阈值（默认 0.60，单级 YOLO 实测均值约 0.67）",
                        visible=FEATURE_FLAGS["SHOW_CONF_SLIDER"],
                        scale=4,
                    )
                    auto_tts_radio = gr.Radio(
                        choices=["🔊 自动朗读", "🔇 静音模式"],
                        value="🔊 自动朗读",
                        label="识别后自动语音播报",
                        visible=FEATURE_FLAGS["SHOW_AUTO_TTS_TOGGLE"],
                        scale=4,
                    )

                video_input = gr.Video(
                    label="上传本地手语视频（MP4/AVI）",
                    visible=False,
                    height=140,
                )
                # gradio 6 会把 Image 的 label 浮在画面左上角，遮住我们画在帧上的状态条，
                # 所以标签关掉、改用上方独立标题行。
                gr.HTML(
                    "<div style='font-size:12.5px;font-weight:700;color:#334155;"
                    "padding:2px 0 4px 2px;'>📹 实时摄像头预览（画面中的绿框/蓝框 = YOLO26 检出的手与字母）</div>"
                )
                preview_image = gr.Image(
                    value=CAM_PREVIEW.snapshot(),
                    show_label=False,
                    interactive=False,
                    height=480,
                    elem_id="live-preview-a",
                )
                with gr.Row():
                    btn_start = gr.Button("▶ 开始识别", variant="primary")
                    btn_sim_fail = gr.Button("⚠️ 模拟识别失败", variant="secondary")
                    btn_stop = gr.Button("⏸ 停止识别", variant="secondary")
                    btn_tts = gr.Button("🔊 手动朗读", variant="secondary")
                    btn_clear = gr.Button("🗑 清空", variant="stop")

        # 右栏（40%）：独立纵向滚动容器，不挤压左侧视频
        with gr.Column(scale=8, min_width=380, elem_classes=["right-scroll-col-a"]):
            with gr.Group(elem_classes=["module-card-a"]):
                gr.Markdown("#### ✋ 静态手语词识别结果与置信度")
                conf_bars_output = gr.HTML(value=build_confidence_bars_a([]))
                word_seq_output = gr.Textbox(
                    label="识别词序列汇总",
                    placeholder="等待输出静态手语词汇序列...",
                    lines=1,
                    interactive=False,
                )
                with gr.Row(visible=FEATURE_FLAGS["SHOW_WORD_EDIT_BTNS"]):
                    btn_undo = gr.Button("⬅️ 撤回上一词", size="sm")
                    btn_confirm = gr.Button("✅ 确认成句 / 发送给 Agent", variant="primary", size="sm")

            with gr.Group(elem_classes=["module-card-a"]):
                gr.Markdown("#### 🤖 Agent 智能体语义解析输出")
                large_agent_html = gr.HTML(value=build_large_translation_html("", ""))
                agent_tts_holder = gr.Textbox(visible=False)

                with gr.Column(visible=FEATURE_FLAGS["SHOW_TWO_WAY_REPLY"]):
                    with gr.Row():
                        staff_input = gr.Textbox(
                            label="💬 双向交互：政务窗口工作人员回复（大字展示给听障群众）",
                            placeholder="输入工作人员回复内容，或点击下方快捷回复...",
                            lines=1,
                            scale=7,
                        )
                        btn_staff_send = gr.Button("📤 发送回复", variant="primary", scale=3)
                    with gr.Row():
                        quick_btn_1 = gr.Button("快捷1：请出示身份证", size="sm")
                        quick_btn_2 = gr.Button("快捷2：去2号室拍照", size="sm")
                        quick_btn_3 = gr.Button("快捷3：已办结收好回执", size="sm")

            with gr.Group(elem_classes=["module-card-a"], visible=FEATURE_FLAGS["SHOW_HISTORY_PANEL"]):
                with gr.Row():
                    gr.Markdown("#### 📜 历史翻译记录")
                    btn_del_last_hist = gr.Button("🗑 删除最新一条", size="sm")
                    btn_clear_hist = gr.Button("🧹 清空历史", size="sm", variant="stop")
                history_html_box = gr.HTML(value=build_history_html(_HISTORY_RECORDS))

    source_selector.change(
        fn=toggle_input_source,
        inputs=[source_selector, conf_slider],
        outputs=[video_input, preview_image, status_display],
    )

    # 【开始识别】只负责进入录制状态，不结算；节奏交给用户
    btn_start.click(
        fn=start,
        inputs=[source_selector, video_input, conf_slider, auto_tts_radio],
        outputs=[
            preview_image,
            conf_bars_output,
            word_seq_output,
            agent_tts_holder,
            large_agent_html,
            status_display,
            history_html_box,
        ],
    )

    # 【停止识别】才真正结算结果
    btn_stop.click(
        fn=stop,
        inputs=[conf_slider, auto_tts_radio],
        outputs=[
            preview_image,
            conf_bars_output,
            word_seq_output,
            agent_tts_holder,
            large_agent_html,
            status_display,
            history_html_box,
        ],
    ).then(
        fn=None,
        inputs=[agent_tts_holder, auto_tts_radio],
        outputs=[],
        js="""(agentText, ttsMode) => {
            if (window.speechSynthesis) {
                window.speechSynthesis.cancel();
                if (ttsMode && ttsMode.includes('自动') && agentText) {
                    const utter = new SpeechSynthesisUtterance(agentText);
                    utter.lang = 'zh-CN';
                    window.speechSynthesis.speak(utter);
                }
            }
        }""",
    )

    btn_sim_fail.click(
        fn=simulate_failure,
        inputs=[conf_slider, auto_tts_radio],
        outputs=[
            preview_image,
            conf_bars_output,
            word_seq_output,
            agent_tts_holder,
            large_agent_html,
            status_display,
            history_html_box,
        ],
    ).then(
        fn=None,
        inputs=[agent_tts_holder, auto_tts_radio],
        outputs=[],
        js="""(agentText, ttsMode) => {
            if (window.speechSynthesis) {
                window.speechSynthesis.cancel();
                if (ttsMode && ttsMode.includes('自动') && agentText) {
                    const utter = new SpeechSynthesisUtterance(agentText);
                    utter.lang = 'zh-CN';
                    window.speechSynthesis.speak(utter);
                }
            }
        }""",
    )

    auto_tts_radio.change(
        fn=lambda mode: build_status_pill_a(
            "idle",
            "已开启【🔊 自动语音播报】：识别完成后将自动朗读译文"
            if "自动" in str(mode)
            else "已切换至【🔇 静音模式】：识别完成后仅展示大字翻译，不自动发声",
            mode,
        ),
        inputs=[auto_tts_radio],
        outputs=[status_display],
        js="""(ttsMode) => {
            if (ttsMode && ttsMode.includes('静音') && window.speechSynthesis) {
                window.speechSynthesis.cancel();
            }
            return [ttsMode];
        }""",
    )

    btn_stop.click(fn=stop, inputs=[], outputs=[status_display])

    btn_tts.click(
        fn=tts,
        inputs=[agent_tts_holder],
        outputs=[status_display],
        js="""(agentText) => {
            if (agentText && window.speechSynthesis) {
                window.speechSynthesis.cancel();
                const utter = new SpeechSynthesisUtterance(agentText);
                utter.lang = 'zh-CN';
                window.speechSynthesis.speak(utter);
            }
            return [agentText];
        }""",
    )

    btn_clear.click(
        fn=clear,
        inputs=[],
        outputs=[
            preview_image,
            conf_bars_output,
            word_seq_output,
            agent_tts_holder,
            large_agent_html,
            status_display,
        ],
    )

    btn_undo.click(
        fn=undo_last_word,
        inputs=[conf_slider],
        outputs=[conf_bars_output, word_seq_output, large_agent_html, agent_tts_holder, status_display],
    )

    btn_confirm.click(
        fn=confirm_sentence,
        inputs=[],
        outputs=[large_agent_html, agent_tts_holder, status_display, history_html_box],
    )

    btn_staff_send.click(
        fn=send_staff_reply,
        inputs=[staff_input],
        outputs=[large_agent_html, status_display, history_html_box],
    )

    quick_btn_1.click(
        fn=lambda: "请出示您的身份证或户口本原件，谢谢！", inputs=[], outputs=[staff_input]
    ).then(fn=send_staff_reply, inputs=[staff_input], outputs=[large_agent_html, status_display, history_html_box])

    quick_btn_2.click(
        fn=lambda: "请先到右侧 2 号人像采集室免费拍照，随后回本窗口办理。", inputs=[], outputs=[staff_input]
    ).then(fn=send_staff_reply, inputs=[staff_input], outputs=[large_agent_html, status_display, history_html_box])

    quick_btn_3.click(
        fn=lambda: "您的业务已办结，请收好身份证与受理回执单！", inputs=[], outputs=[staff_input]
    ).then(fn=send_staff_reply, inputs=[staff_input], outputs=[large_agent_html, status_display, history_html_box])

    btn_del_last_hist.click(
        fn=delete_last_history, inputs=[], outputs=[history_html_box, status_display]
    )

    btn_clear_hist.click(
        fn=clear_all_history, inputs=[], outputs=[history_html_box, status_display]
    )

    # ------------------------------------------------------------------
    # 实时预览推送（按 gradio 版本自动选择，逐级降级）：
    #   1) gradio 6：gr.Timer 每 0.5s 触发一次 tick，把新帧写进 Image 组件
    #   2) gradio 4/5：demo.load(every=0.2) 轮询
    #   3) 兜底：前端脚本直接换 <img> 的 src（拉 /preview_frame）
    # 任一层生效即可看到画面；上机后可按黑窗口日志判断实际走了哪层。
    # ------------------------------------------------------------------
    def _tick_preview():
        return CAM_PREVIEW.snapshot()

    if hasattr(gr, "Timer"):
        preview_timer = gr.Timer(0.2)
        preview_timer.tick(fn=_tick_preview, inputs=None, outputs=preview_image)
        print("[preview] 预览刷新方式：gr.Timer(0.2s)", flush=True)
    else:
        demo.load(fn=_tick_preview, inputs=None, outputs=preview_image, every=0.2)
        print("[preview] 预览刷新方式：demo.load(every=0.2)", flush=True)

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("  手语视界 · 录制版 V13", flush=True)
    print("  看到这行 = 新代码已生效。若没看到，跑的是旧文件/旧进程。", flush=True)
    print("=" * 60, flush=True)

    CAM_PREVIEW.start()
    print("[preview] 摄像头预览线程已启动（若 5 秒内无'实时预览已启动'说明没找到设备）", flush=True)

    launch_sig = inspect.signature(demo.launch)
    kwargs = {}
    if "css" in launch_sig.parameters:
        kwargs["css"] = CUSTOM_CSS_A
    demo.launch(**kwargs)
