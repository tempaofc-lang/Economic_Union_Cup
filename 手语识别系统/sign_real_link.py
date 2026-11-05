# -*- coding: utf-8 -*-
"""
================================================================================
模块名称：sign_real_link.py（方案B：30 字母模型 + 拼读层 真实推理接入）
职责链路：
    摄像头/视频帧 -> YOLO(30字母手部检测 best.pt) -> 裁剪 -> HandCNN(30字母分类
    checkpoints/best.pt) -> 字母流去抖 -> 按 vocab.json 拼音词贪心切词
    -> 输出 app.py 需要的 tokens 列表 [{"word": 拼音, "conf": 置信度, "bbox": 框}]

说明：
  - word 一律输出【无声调拼音】，由 pinyin_codec / agent.py 查封闭词表还原中文；
  - 两个权重路径可用环境变量覆盖：SIGN_YOLO_PT / SIGN_CLS_PT；
  - 本模块不依赖外部交付包：HandCNN 结构与交付包 model.py 逐层一致，权重键
    model_state/class_names/model_type 已实测匹配。

独立自测：
    python sign_real_link.py --image 某张含手照片.jpg
================================================================================
"""

import os
import json
import argparse
from collections import Counter, deque
from difflib import SequenceMatcher

# Windows + Anaconda 环境下，PATH 里的 conda Library\bin 会先加载一份 libiomp5md.dll，
# 与 pip 安装的 torch 自带 OpenMP 运行时冲突，import torch 直接报 OMP Error #15。
# 这里在导入 torch 之前放行重复加载（本场景只做单张图 CNN 推理，无性能/正确性风险）。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- 权重与参数（可用环境变量覆盖） ----------------
# 查找顺序：环境变量 → 仓库内固定位置 → A 组原始路径。
# 建议把权重放到 手语识别系统/weights/ 下（.gitignore 已排除 *.pt，不会误提交）。
DEFAULT_YOLO_PT = r"E:\Tencent Files\train(1)\content\drive\MyDrive\yolo\runs\detect\train-2\weights\best.pt"
DEFAULT_CLS_PT = r"E:\Tencent Files\分类模型交付包\checkpoints\best.pt"

_YOLO_CANDIDATES = [
    os.path.join(HERE, "weights", "yolo_best.pt"),
    os.path.join(HERE, "weights", "det_best.pt"),
    DEFAULT_YOLO_PT,
]
_CLS_CANDIDATES = [
    os.path.join(HERE, "weights", "cls_best.pt"),
    os.path.join(HERE, "checkpoints", "best.pt"),
    DEFAULT_CLS_PT,
]


def _resolve_weight(env_name, candidates, label):
    """按候选顺序找权重；都不存在时抛出可操作的报错，而不是裸 FileNotFoundError。"""
    paths = [os.getenv(env_name)] + candidates
    for p in paths:
        if p and os.path.isfile(p):
            return p
    raise RuntimeError(
        f"{label}权重不存在。已尝试：\n"
        + "\n".join(f"  - {p}" for p in paths if p)
        + f"\n解决：把权重放到 {os.path.join(HERE, 'weights')}\\ 下，"
          f"或设置环境变量 {env_name} 指向 .pt 文件。"
    )


# 注意：这里**不在导入期做存在性检查**。
# 否则没有权重的机器上连 `from sign_real_link import spell_letters` 都会失败，
# 拼读层就没法单独测（A 组指南 L2 就是这么测的）。真正加载时才检查。
YOLO_PT = os.getenv("SIGN_YOLO_PT") or DEFAULT_YOLO_PT
CLS_PT = os.getenv("SIGN_CLS_PT") or DEFAULT_CLS_PT


def resolve_yolo_path():
    return _resolve_weight("SIGN_YOLO_PT", _YOLO_CANDIDATES, "YOLO 检测")


def resolve_cls_path():
    return _resolve_weight("SIGN_CLS_PT", _CLS_CANDIDATES, "HandCNN 分类")

CLS_IMG_SIZE = 128          # 交付包 config.IMG_SIZE
DET_CONF = 0.35             # YOLO 检出手的阈值
CLS_MIN_CONF = 0.50         # 单帧字母分类最低置信度（低于则丢帧）
MIN_RUN = 2                 # 同一字母连续出现帧数下限（去抖）
MAX_SECONDS = 3.0           # 单次点击最多采集时长
FRAME_STRIDE = 2            # 每 stride 帧取 1 帧送推理（提速）

# ---------------- 拼读容错参数 ----------------
# 单字母分类准确率约 61%，逐字母连乘后长词几乎必然出错（11 字母约 0.2%）。
# 拼读层必须容忍少量错字母，否则"身份证"这类词永远拼不出来。
FUZZY_RATIO = 0.78          # 相似度阈值（0~1），越高越严格
LONGER_RATIO = 0.85         # 短别名命中后，允许被"更长的近似词"覆盖的阈值
RESIDUAL_MIN_LEN = 2        # 拼不出的残余字母至少这么长才透传给下游
RESIDUAL_VOWELS = set("aeiou")   # 残余片段至少含一个元音才像拼音，过滤 qqq 之类噪声

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
        ck = torch.load(resolve_cls_path(), map_location="cpu", weights_only=False)
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
        from ultralytics import YOLO
        _YOLO = YOLO(resolve_yolo_path())
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


def _match_at(buf, i, targets, allow_fuzzy):
    """
    在 buf[i:] 处匹配一个词，返回命中目标或 None。

    规则（按优先级）：
      1. 精确前缀匹配。targets 已按长度降序，所以取到的是**最长**的精确命中。
      2. 长词覆盖：精确命中的可能只是个短别名（如 shenfenz 是 shenfenzheng 的别名）。
         如果更长的词在当前窗口里足够像（>= LONGER_RATIO），取长词，
         否则尾字母打错时会切出 "shenfenz" + "henx" 这种碎片。
      3. 相似度兜底（allow_fuzzy 时）：容忍 1~2 个字母打错。
         allow_fuzzy=False 用于"上一个字符还没匹配上"的情形，
         否则窗口会往后滑，把 WOBUBAN 里的 o 吃掉，拼成 buban 而丢掉 wo。
    """
    exact = next((t for t in targets if buf.startswith(t, i)), None)
    if exact is not None:
        for t in targets:
            if len(t) <= len(exact):
                continue
            window = buf[i:i + len(t)]
            if len(window) < len(t) - 1:
                continue
            if SequenceMatcher(None, window, t).ratio() >= LONGER_RATIO:
                return t
        return exact

    if not allow_fuzzy:
        return None

    for t in targets:
        if len(t) < 3:                   # 短别名（sfz / qw / xx）只接受精确命中
            continue
        window = buf[i:i + len(t)]
        if len(window) < len(t) - 1:
            continue
        if SequenceMatcher(None, window, t).ratio() >= FUZZY_RATIO:
            return t
    return None


def spell_letters(letters, targets):
    """
    字母流 -> 拼音词序列。

    三级策略，按顺序尝试：
      1. 精确最长匹配（原逻辑，无误字时命中率最高）；
      2. 相似度兜底：容忍 1~2 个字母打错（NIXHAO -> nihao）；
      3. 残余透传：连续拼不出的字母攒成片段交给下游大模型推断。
         例如 wo / yao 不在封闭词表里，但 LLM 认得"我要"。

    残余片段要求长度 >= RESIDUAL_MIN_LEN 且含元音，避免把噪声也透传下去。
    """
    buf = "".join(letters).lower()
    words, pending, i = [], [], 0

    def flush_pending():
        if not pending:
            return
        chunk = "".join(pending)
        pending.clear()
        if len(chunk) >= RESIDUAL_MIN_LEN and (RESIDUAL_VOWELS & set(chunk)):
            words.append(chunk)

    while i < len(buf):
        hit = _match_at(buf, i, targets, allow_fuzzy=not pending)
        if hit:
            flush_pending()
            words.append(hit)
            i += len(hit)
        else:
            pending.append(buf[i])
            i += 1

    flush_pending()
    return words


def compress_runs(per_frame_letters):
    """帧级字母 -> 去抖字母流：连续相同且长度>=MIN_RUN 才算打了一个字母。"""
    letters, confs = [], []
    i = 0
    n = len(per_frame_letters)
    while i < n:
        j = i
        while j < n and per_frame_letters[j] == per_frame_letters[i]:
            j += 1
        if j - i >= MIN_RUN and per_frame_letters[i] is not None:
            letters.append(per_frame_letters[i])
            confs.append(sum(_FRAME_CONFS[i:j]) / (j - i))
        i = j
    return letters, confs


_FRAME_CONFS = []  # 与 per_frame_letters 同步记录（模块级缓存，简化传递）


# ---------------- 单帧识别 ----------------
def classify_crop(frame, box):
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
    # YOLO 侧类别与分类侧一致时加分，不一致轻罚（双模型共识）
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


# ---------------- 采集 + 拼读主入口 ----------------
def predict_tokens(source_type="本机摄像头实时采集", video_file=None,
                   conf_thresh=0.75, max_seconds=None):
    """
    返回值与 vision_engine.predict_sign_tokens 契约一致：
      (custom_frame=None, tokens, scene_meta)
    tokens: [{"word": 拼音, "conf": float, "bbox": (x1,y1,x2,y2)}]
    识别不到词时返回空列表（app.py 会走"识别失败"联动，属正常分支）。
    """
    src = video_file if (video_file and "视频" in str(source_type)) else 0
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频/摄像头: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    limit = int((max_seconds or MAX_SECONDS) * fps)
    global _FRAME_CONFS
    stream, _FRAME_CONFS = [], []
    grabbed = last_box = 0
    try:
        while grabbed < limit:
            ok, frame = cap.read()
            if not ok:
                break
            grabbed += 1
            if grabbed % FRAME_STRIDE:
                continue
            box = best_box_of_frame(frame)
            if box is None:
                stream.append(None)
                _FRAME_CONFS.append(0.0)
                last_box = box if last_box is None else last_box
                continue
            letter, conf = classify_crop(frame, box)
            stream.append(letter if conf >= CLS_MIN_CONF else None)
            _FRAME_CONFS.append(conf)
            last_box = box
    finally:
        cap.release()

    letters, confs = compress_runs(stream)
    words = spell_letters(letters, load_pinyin_targets())
    bbox = (last_box["x1"], last_box["y1"], last_box["x2"], last_box["y2"]) \
        if last_box else (190, 110, 390, 340)
    tokens = []
    for w in words:
        c = np.mean(confs) if confs else 0.5
        tokens.append({"word": w, "conf": float(min(0.99, c)), "bbox": bbox})
    scene = {
        "scene_name": "方案B 字母拼写实时识别",
        "letters": letters,
        "translation": "",
        "ai_answer": "",
    }
    return None, tokens, scene


# ---------------- 独立自测 ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    args = parser.parse_args()

    print("[1] 分类权重:", end=" ")
    m, names = get_cls_model()
    print(f"OK {len(names)} 类, 首尾 = {names[0]}..{names[-1]}")
    print("[2] YOLO 权重:", end=" ")
    y = get_yolo_model()
    print(f"OK {len(y.names)} 类")
    assert sorted(str(v) for v in y.names.values()) == sorted(names), "两侧类别不一致"
    print("[3] 拼读层: N-I-H-A-O ->", spell_letters(list("NIHAO"), load_pinyin_targets()))
    print("    S-H-E-N-F-E-N-Z-H-E-N-G ->", spell_letters(list("SHENFENZHENG"), load_pinyin_targets()))
    if args.image:
        frame = cv2.imread(args.image)
        box = best_box_of_frame(frame)
        if box is None:
            print("[4] 图片模式: 未检出手（属正常情况，可换图）")
        else:
            letter, conf = classify_crop(frame, box)
            print(f"[4] 图片模式: 框={box['x1']},{box['y1']},{box['x2']},{box['y2']} "
                  f"YOLO类={box['cls_name']} 分类={letter} conf={conf:.2f}")
