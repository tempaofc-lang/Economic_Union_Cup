# -*- coding: utf-8 -*-
"""
================================================================================
模块名称：sign_real_link.py（方案B 简化版：单级 YOLO 检测即分类 + 拼读层）
职责链路：
    摄像头/视频帧 -> YOLO(30字母手部检测 best.pt，同时输出字母类别)
    -> 字母流去抖 -> 按 vocab.json 拼音词贪心切词
    -> 输出 app.py 需要的 tokens 列表 [{"word": 拼音, "conf": 置信度, "bbox": 框}]

【本版相对原版的改动：删除第二级 HandCNN 分类器】
改动的依据（40 帧验证集实测，详见 改动说明.md）：
  - YOLO 自身即为 30 类分类器（nc=30，names 为 A..ZH），HandCNN 在做重叠工作；
  - HandCNN 单帧准确率仅 30%，与 YOLO 双模型一致率 30%；
  - 原逻辑在不一致时 ×0.7 压低置信度但仍采用 HandCNN 的字母，导致
    「通过帧字母正确率只有 36.7%」且出现 0.41 这类被二次惩罚的置信度；
  - 单级 YOLO 置信度均值 0.669、阈值 0.50 时通过率 92.5%，分布健康。

保留内容：
  - HandCNN 类与 classify_crop() 保留但默认不启用（ENABLE_CNN=False），
    便于与「两级级联版」做对照实验，也保留方案演进的证据链；
  - 两个权重路径仍可用环境变量覆盖：SIGN_YOLO_PT / SIGN_CLS_PT。

独立自测：
    python sign_real_link.py --image 某张含手照片.jpg
================================================================================
"""

import os
import json
import time
import argparse
from collections import Counter, deque

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- 权重与参数（可用环境变量覆盖） ----------------
# 相对路径：权重放在本文件同级，按以下顺序自动查找，换电脑/换目录都不用改代码。
#   weights/      ← 仓库约定（SIGN_YOLO_PT 未设时优先）
#   checkpoints/  ← 早期交付包约定
# 两者都没有时才回退到硬编码的旧路径，并给出明确报错。
_PT_DIRS = [os.path.join(HERE, "weights"), os.path.join(HERE, "checkpoints")]


def _resolve_pt(filename, legacy_default):
    env = os.getenv("SIGN_YOLO_PT" if "yolo" in filename else "SIGN_CLS_PT")
    if env:
        return env
    for d in _PT_DIRS:
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            return p
    # 兜底：返回首选路径（哪怕不存在），由 get_*_model() 抛出可读的错误
    return os.path.join(_PT_DIRS[0], filename) if os.path.isdir(_PT_DIRS[0]) else legacy_default


DEFAULT_YOLO_PT = _resolve_pt(
    "yolo_best.pt",
    r"E:\Tencent Files\train(1)\content\drive\MyDrive\yolo\runs\detect\train-2\weights\best.pt",
)
DEFAULT_CLS_PT = _resolve_pt(
    "cls_best.pt",
    r"E:\Tencent Files\分类模型交付包\checkpoints\best.pt",
)
YOLO_PT = os.getenv("SIGN_YOLO_PT") or DEFAULT_YOLO_PT
CLS_PT = os.getenv("SIGN_CLS_PT") or DEFAULT_CLS_PT


def resolve_yolo_path():
    """app.py 开机自检调用：返回 YOLO 权重路径，找不到则抛可读异常。"""
    if not os.path.isfile(YOLO_PT):
        raise FileNotFoundError(
            f"找不到 YOLO 权重: {YOLO_PT}。"
            f"已查找目录: {', '.join(_PT_DIRS)}。"
            f"解决：把 yolo_best.pt 放到 weights/ 下，或设置环境变量 SIGN_YOLO_PT。"
        )
    return YOLO_PT


def resolve_cls_path():
    """app.py 开机自检调用：返回 HandCNN 权重路径（ENABLE_CNN=False 时不加载，只查存在性）。"""
    if not os.path.isfile(CLS_PT):
        raise FileNotFoundError(
            f"找不到分类权重: {CLS_PT}。"
            f"已查找目录: {', '.join(_PT_DIRS)}。"
            f"解决：把 cls_best.pt 放到 weights/ 下，或设置环境变量 SIGN_CLS_PT。"
        )
    return CLS_PT

CLS_IMG_SIZE = 128          # 交付包 config.IMG_SIZE
DET_CONF = 0.35             # YOLO 检出手的阈值
CLS_MIN_CONF = 0.40         # 单帧字母最低置信度（低于则丢帧）。单级方案下即 YOLO 检出置信度
MIN_RUN = 2                 # 同一字母连续出现帧数下限（去抖）
MAX_SECONDS = 3.0           # 单次点击最多采集时长
FRAME_STRIDE = 2            # 每 stride 帧取 1 帧送推理（提速）

# ---------------- 抗抖动：双尺度复核（只在识别摇摆时触发） ----------------
# 单帧 YOLO 在画质差时会在两个候选字母间来回跳（A→B→A），去抖提交不了。
# 连续 SWITCH_PATIENCE 帧结果两两不同 = 当前画面没把握，此时再用「居中留白」跑
# 一次同一帧，取置信度高的那个。实测 conf 均值 0.669→0.727，通过率 92.5%→95%。
# 正常稳定识别时完全不触发，帧率不受影响。
SWITCH_PATIENCE = 3         # 连续多少次"字母变了"算摇摆
SWITCH_PAD_RATIO = 0.6      # 复核时把手缩到画面的 60%（居中留白）
SWITCH_GAIN_MIN = 0.05      # 复核结果至少要高出这么多才采纳

ENABLE_CNN = False          # 【开关】False=单级 YOLO（推荐）；True=恢复两级级联（对照用）

_YOLO = None
_CLS = None


# ---------------- HandCNN（与交付包 model.py 结构一致） ----------------
class HandCNN(nn.Module):
    def __init__(self, num_classes=30, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------- 模型加载（进程内只加载一次） ----------------
def get_cls_model():
    global _CLS
    if _CLS is None:
        ck = torch.load(CLS_PT, map_location="cpu", weights_only=False)
        assert ck.get("model_type") == "cnn", f"分类权重类型异常: {ck.get('model_type')}"
        names = list(ck["class_names"])
        m = HandCNN(num_classes=len(names))
        m.load_state_dict(ck["model_state"])
        m.eval()
        _CLS = (m, names)
    return _CLS


def get_yolo_model():
    global _YOLO
    if _YOLO is None:
        if not os.path.isfile(YOLO_PT):
            raise FileNotFoundError(
                f"找不到 YOLO 权重: {YOLO_PT}\n"
                f"解决：把 yolo_best.pt 放到 {os.path.join(HERE, 'weights')}\\ 下，"
                f"或设置环境变量 SIGN_YOLO_PT 指向它。\n"
                f"已查找目录: {', '.join(_PT_DIRS)}"
            )
        from ultralytics import YOLO
        _YOLO = YOLO(YOLO_PT)
    return _YOLO


# ---------------- 词表与拼读 ----------------
def load_pinyin_targets():
    """从 vocab.json 读拼音整词，按长度降序用于贪心切词。"""
    with open(os.path.join(HERE, "vocab.json"), encoding="utf-8") as f:
        raw = json.load(f)
    words = [e["pinyin"].lower() for e in raw["classes"]]
    for e in raw["classes"]:
        words.extend(a.lower() for a in e.get("aliases", []) if a.isalpha())
    return sorted(set(words), key=len, reverse=True)


def spell_letters(letters, targets):
    """字母流 -> 拼音整词序列（贪心最长匹配；拼不出的字母丢弃）。"""
    buf = "".join(letters).lower()
    words, i = [], 0
    while i < len(buf):
        hit = next((w for w in targets if buf.startswith(w, i)), None)
        if hit:
            words.append(hit)
            i += len(hit)
        else:
            i += 1
    return words


def compress_runs(per_frame_letters, frame_confs=None):
    """帧级字母 -> 去抖字母流：连续相同且长度>=MIN_RUN 才算打了一个字母。

    frame_confs 为与 per_frame_letters 逐帧对齐的置信度列表。
    旧版依赖模块级全局 _FRAME_CONFS，任何非 predict_tokens 的调用方都会拿到空值，
    导致字母置信度恒为 0.0；此处改为显式传参，未传时回退到全局（保持兼容）。
    """
    if frame_confs is None:
        frame_confs = _FRAME_CONFS
    letters, confs = [], []
    i = 0
    n = len(per_frame_letters)
    while i < n:
        j = i
        while j < n and per_frame_letters[j] == per_frame_letters[i]:
            j += 1
        if j - i >= MIN_RUN and per_frame_letters[i] is not None:
            letters.append(per_frame_letters[i])
            seg = frame_confs[i:j] if i < len(frame_confs) else []
            confs.append(sum(seg) / len(seg) if seg else 0.0)
        i = j
    return letters, confs


_FRAME_CONFS = []  # 向后兼容：predict_tokens 仍会写入，其他调用方请显式传 frame_confs


# ---------------- 单帧识别 ----------------
def recognize_frame(frame, box):
    """单帧字母识别。ENABLE_CNN=False 时走单级 YOLO，否则走两级级联。

    返回 (letter, conf)；识别不出返回 (None, 0.0)。
    """
    if ENABLE_CNN:
        return classify_crop(frame, box)
    # —— 单级：YOLO 的类别与检出置信度直接作为字母结果 ——
    letter = (box.get("cls_name") or "").lower()
    if not letter:
        return None, 0.0
    return letter, float(box.get("conf", 0.0))


def classify_crop(frame, box):
    """第二级 HandCNN 分类（仅在 ENABLE_CNN=True 时使用，保留作对照实验）。"""
    m, names = get_cls_model()
    x1 = max(0, box["x1"] - 5)
    y1 = max(0, box["y1"] - 5)
    x2 = min(frame.shape[1], box["x2"] + 5)
    y2 = min(frame.shape[0], box["y2"] + 5)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, 0.0
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize((CLS_IMG_SIZE, CLS_IMG_SIZE))
    t = torch.from_numpy(
        (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
    ).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        prob = torch.softmax(m(t), dim=1)
        conf, pred = prob.max(1)
    letter = names[int(pred)].lower()
    conf = float(conf)
    # 与原版保持一致：不一致时保留 HandCNN 字母、置信度 ×0.7（仅对照实验用）
    yolo_letter = box.get("cls_name")
    if yolo_letter and yolo_letter.lower() != letter:
        conf *= 0.7
    return letter, conf


def best_box_of_frame(frame):
    yolo = get_yolo_model()
    res = yolo(frame, imgsz=640, conf=DET_CONF, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    best, bconf = None, -1.0
    names = yolo.names if isinstance(yolo.names, dict) else dict(enumerate(yolo.names))
    for b in res.boxes:
        c = float(b.conf)
        if c > bconf:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            bconf = c
            best = {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": c, "cls_name": str(names[int(b.cls)])}
    return best


# ---------------- 双尺度复核（抗摇摆） ----------------
def pad_center(frame, ratio=SWITCH_PAD_RATIO, fill=114):
    """把手缩到画面的 ratio（四周灰边填充，方形）。只用于复核，不能当主路径：
    实测整帧留白会让置信度下降，但恰好能救回一部分「原尺度下摇摆」的帧。"""
    h, w = frame.shape[:2]
    n = int(max(h, w) / max(0.1, ratio))
    canvas = np.full((n, n, 3), fill, np.uint8)
    ox, oy = (n - w) // 2, (n - h) // 2
    canvas[oy:oy + h, ox:ox + w] = frame
    return canvas


def recognize_recheck(frame):
    """摇摆时调用：同一帧居中留白后再识别一次，返回 (letter, conf)。"""
    try:
        padded = pad_center(frame)
        box = best_box_of_frame(padded)
        if box is None:
            return None, 0.0
        return recognize_frame(padded, box)
    except Exception:
        return None, 0.0


class SwitchGuard:
    """识别摇摆检测器。

    用法（cam_test.py 里每帧）：
        g = SwitchGuard()
        letter, conf, mode = g.resolve(small, letter, conf)
        # mode == 'recheck' 表示这一帧用了双尺度复核

    逻辑：连续 SWITCH_PATIENCE 帧字母两两不同 → 判定摇摆，触发一次复核；
    复核结果必须比原结果高出 SWITCH_GAIN_MIN 才采纳，否则保留原结果。
    某字母稳定出现后计数清零，因此正常演示时这条分支基本不会进入。
    """

    def __init__(self):
        self.hist = []

    def reset(self):
        self.hist = []

    def is_wobbling(self):
        # 连续 SWITCH_PATIENCE 帧「都在变」即判定摇摆。
        # 注意不能用「两两不同」：周期 A→B→A→B 的抖动里 set 只有 2 个，会漏判。
        h = [x for x in self.hist if x]
        if len(h) < SWITCH_PATIENCE + 1:
            return False
        tail = h[-(SWITCH_PATIENCE + 1):]
        return all(tail[i] != tail[i + 1] for i in range(SWITCH_PATIENCE))

    def resolve(self, frame, letter, conf):
        if letter is None:
            self.hist.append(None)
            return letter, conf, "normal"
        self.hist.append(letter)
        if len(self.hist) > 12:
            self.hist = self.hist[-12:]
        if not self.is_wobbling():
            return letter, conf, "normal"
        alt_letter, alt_conf = recognize_recheck(frame)
        # 只在复核"更信任"时采纳：同样是那个字母且更高分，或者直接换成更高分的另一个字母
        if alt_letter and alt_conf >= conf + SWITCH_GAIN_MIN:
            self.hist = [alt_letter]
            return alt_letter, alt_conf, "recheck"
        # 复核没把握（或不够高）时也要清空历史，否则下一帧会重复触发、白烧算力
        self.hist = self.hist[-SWITCH_PATIENCE:]
        return letter, conf, "recheck-reject"


# ---------------- 摄像头打开：多后端 + 多编号探测 ----------------
# Windows 上 cv2.VideoCapture(0) 默认走 MSMF 后端，部分机型/驱动会出现
# 「isOpened() 报成功、read() 却永远拿不到帧」的假打开；另外带红外人脸模组的
# 笔记本，0 号可能是红外/深度摄像头，真正的 RGB 摄像头在 1 号。
# 因此这里把「后端 × 编号」全试一遍，并且必须真正 read 出一帧画面才算成功。
CAM_INDEXES = [0, 1, 2]
CAM_BACKENDS = [("DirectShow", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("默认", None)]


def open_camera(verbose=True):
    """打开本机摄像头，返回 (cap, 描述)。全部失败时抛出带排查建议的中文异常。"""
    tried = []
    for name, api in CAM_BACKENDS:
        for idx in CAM_INDEXES:
            try:
                cap = cv2.VideoCapture(idx, api) if api is not None else cv2.VideoCapture(idx)
            except Exception as e:
                tried.append(f"{name}/#{idx}(异常)")
                continue
            ok, frame = cap.isOpened(), None
            if ok:
                # isOpened() 存在假阳性：必须真的 read 出一帧才算成功
                for _ in range(8):
                    ok, frame = cap.read()
                    if ok and frame is not None and getattr(frame, "size", 0) > 0:
                        break
                    ok = False
            if ok:
                h, w = frame.shape[:2]
                desc = f"{name} 后端 / 摄像头 #{idx} / {w}x{h}"
                if verbose:
                    print(f"[camera] 摄像头打开成功：{desc}")
                return cap, desc
            cap.release()
            tried.append(f"{name}/#{idx}")
    raise RuntimeError(
        "打不开本机摄像头（已尝试: " + ", ".join(tried) + "）。"
        "请检查：① 是否有其他程序（相机App/会议软件）占用摄像头，全部关闭后重试；"
        "② 设备管理器里「相机」设备是否正常（无黄色感叹号）；"
        "③ 若用外接USB摄像头，确认已插好。"
    )


# ---------------- 采集 + 拼读主入口 ----------------
def predict_tokens(source_type="本机摄像头实时采集", video_file=None,
                   conf_thresh=0.75, max_seconds=None):
    """
    返回值与 vision_engine.predict_sign_tokens 契约一致：
      (custom_frame=None, tokens, scene_meta)
    tokens: [{"word": 拼音, "conf": float, "bbox": (x1,y1,x2,y2)}]
    识别不到词时返回空列表（app.py 会走"识别失败"联动，属正常分支）。
    打不开摄像头/读不出画面时抛 RuntimeError（携带具体原因，界面会显示）。
    """
    if video_file and "视频" in str(source_type):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise RuntimeError(f"打不开视频文件: {video_file}")
        desc = f"视频文件 {video_file}"
    else:
        cap, desc = open_camera()

    # 【关键修复】不再用 fps 推算帧数：Windows 摄像头 cap.get(FPS) 常返回 0/1/5，
    # 会把 limit 压到个位数，导致"摄像头亮一下就关、还没比手势就判失败"。
    # 改成按真实时间采集，保证无论帧率多少都采满 max_seconds。
    duration = float(max_seconds or MAX_SECONDS)
    global _FRAME_CONFS
    stream, _FRAME_CONFS = [], []
    grabbed = last_box = 0
    guard = SwitchGuard()
    t_start = time.time()
    try:
        while time.time() - t_start < duration:
            ok, frame = cap.read()
            if not ok:
                if grabbed == 0:
                    # 连续读不到帧才算假打开，给驱动一点启动时间
                    if time.time() - t_start > 1.5 and "视频" not in desc:
                        raise RuntimeError(
                            f"摄像头 {desc} 已打开但读不出画面帧（假打开）。"
                            "多为驱动/占用问题：① 关闭相机App与会议软件；"
                            "② 重启电脑后先启动本系统再开其他程序。"
                        )
                    time.sleep(0.02)
                continue
            grabbed += 1
            if grabbed % FRAME_STRIDE:
                continue
            box = best_box_of_frame(frame)
            if box is None:
                stream.append(None)
                _FRAME_CONFS.append(0.0)
                guard.hist.append(None)
                last_box = box if last_box is None else last_box
                continue
            letter, conf = recognize_frame(frame, box)
            letter, conf, _mode = guard.resolve(frame, letter, conf)
            stream.append(letter if conf >= CLS_MIN_CONF else None)
            _FRAME_CONFS.append(conf)
            last_box = box
    finally:
        cap.release()

    letters, confs = compress_runs(stream, _FRAME_CONFS)
    words = spell_letters(letters, load_pinyin_targets())
    bbox = (last_box["x1"], last_box["y1"], last_box["x2"], last_box["y2"]) \
        if last_box else (190, 110, 390, 340)
    tokens = []
    for w in words:
        c = np.mean(confs) if confs else 0.5
        tokens.append({"word": w, "conf": float(min(0.99, c)), "bbox": bbox})
    # 诊断数据：界面上直接显示"抓到几帧/检出手几帧/最高置信度/识别出哪些字母"，
    # 避免再次出现"不知道卡在哪一步"的情况。
    detected = sum(1 for x in stream if x is not None)
    scene = {
        "scene_name": "方案B 字母拼写实时识别",
        "letters": letters,
        "translation": "",
        "ai_answer": "",
        "diag": {
            "frames": grabbed,
            "sampled": len(stream),
            "detected": detected,
            "letters": letters,
            "max_conf": float(max(_FRAME_CONFS)) if _FRAME_CONFS else 0.0,
            "cam": desc,
            "seconds": round(time.time() - t_start, 1),
        },
    }
    return None, tokens, scene


# ---------------- 独立自测 ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    args = parser.parse_args()

    print(f"[1] 识别模式: {'两级级联（YOLO + HandCNN）' if ENABLE_CNN else '单级 YOLO（检测即分类）'}")
    print("[2] YOLO 权重:", end=" ")
    y = get_yolo_model()
    print(f"OK {len(y.names)} 类")
    if ENABLE_CNN:
        m, names = get_cls_model()
        print(f"[3] 分类权重: OK {len(names)} 类, 首尾 = {names[0]}..{names[-1]}")
        assert sorted(str(v) for v in y.names.values()) == sorted(names), "两侧类别不一致"
    print("[4] 拼读层: N-I-H-A-O ->", spell_letters(list("NIHAO"), load_pinyin_targets()))
    print("    S-H-E-N-F-E-N-Z-H-E-N-G ->", spell_letters(list("SHENFENZHENG"), load_pinyin_targets()))
    if args.image:
        frame = cv2.imread(args.image)
        box = best_box_of_frame(frame)
        if box is None:
            print("[5] 图片模式: 未检出手（属正常情况，可换图）")
        else:
            letter, conf = recognize_frame(frame, box)
            print(f"[5] 图片模式: 框={box['x1']},{box['y1']},{box['x2']},{box['y2']} "
                  f"字母={letter} conf={conf:.2f}")
