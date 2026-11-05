"""
================================================================================
模块名称：vision_engine.py（YOLO26 手部检测与静态手语词分类接入层）
负责人：技术 A 组 - 邵嫣然（YOLO26 模型训练）、杨李灿（分类模型脚本）、张明慧（数据工程）
对接说明：
  1. 在第 1 阶段（10.17-10.21）无显卡/无权重联调期间，保持 `USE_REAL_YOLO_MODEL = False`，
     系统会自动轮播内置的「政务服务三大典型场景」模拟检测数据。
  2. 当邵嫣然导出 `best.pt` 且杨李灿完成分类脚本后，将 `USE_REAL_YOLO_MODEL` 改为 `True`，
     并在 `predict_sign_tokens()` 中填入真实推理代码即可，无需改动 `app.py`！
  3. **A 组为逐字母输出**：打开 `USE_LETTER_MODE = True` 后，逐帧字母会先经过
     letter_pipeline（去抖 + 停顿分词 + 确认成句）再转成拼音词，
     下游 agent 拿到的仍是 `[{word, conf, bbox}]`，接口不变。
================================================================================
"""

# ------------------------------------------------------------------------------
# 🔧 技术 A 组模型切换开关：接入真实 best.pt 时请改为 True
# ------------------------------------------------------------------------------
USE_REAL_YOLO_MODEL = False

# ------------------------------------------------------------------------------
# 🔤 逐字母模式的开关与参数（A 组的模型是一帧输出一个字母）
#   True  —— 走"字母 → 拼音词"链路（当前用 LETTER_DEMO_SCRIPT 模拟逐帧输出）
#   False —— 保持原来的"直接输出词"的模拟场景
# 等 A 组权重到位后，把 _predict_from_letters() 里的模拟回放换成真实推理即可。
# ------------------------------------------------------------------------------
USE_LETTER_MODE = True

# 无模型时的演示脚本：空格表示词边界（松手停顿），末尾自动补「确认」手势。
# 轮播多个脚本，保留原来"三大场景"的演示效果，同时把字母链路走全。
LETTER_DEMO_SCRIPTS = [
    "shenfenzheng",                  # 身份证（一个长词，12 个字母）
    "shebaoka jihuo",                # 社保卡 激活（两个词，演示停顿分词）
    "yingyezhizhao cailiao",         # 营业执照 材料
]
_LETTER_INDEX = 0

# 送检时画的框（字母模型不返回框，这里给一个固定 ROI 供界面绘制）
LETTER_BOX = (250, 120, 510, 360)

try:
    import letter_pipeline
    HAS_LETTER_PIPELINE = True
except ImportError:            # agent/letter_pipeline 不在同一目录时优雅退化
    letter_pipeline = None
    HAS_LETTER_PIPELINE = False

# 政务服务大厅支持的 16 个核心静态手语词库（张明慧、杨李灿可根据 data.yaml 实际类别修改）
GOV_SUPPORTED_WORDS = [
    "你好", "请问", "身份证", "丢失", "补办", "社保卡", "医保", "激活",
    "窗口", "哪里", "营业执照", "办理", "需要", "什么", "材料", "谢谢"
]

# 内置政务大厅三大典型场景模拟检测数据（含 0.62 低置信度误触动作，用于演示阈值防抖过滤）
MOCK_SCENES = [
    {
        "scene_name": "政务场景一：身份证挂失与补办咨询",
        "tokens": [
            {"word": "你好", "conf": 0.98, "bbox": (190, 110, 390, 340)},
            {"word": "身份证", "conf": 0.96, "bbox": (205, 115, 405, 345)},
            {"word": "丢失", "conf": 0.89, "bbox": (195, 105, 400, 340)},
            {"word": "补办", "conf": 0.95, "bbox": (200, 112, 415, 350)},
            {"word": "误触动作(测试阈值过滤)", "conf": 0.62, "bbox": (210, 120, 410, 350)},
        ],
        "translation": "工作人员您好，我的身份证丢失了，请问怎么办理补办？",
        "ai_answer": "【政务导办回答】您好！补办身份证请携带户口本或驾驶证原件，先到右侧 2 号人像采集室免费拍照，随后前往 3 号户政窗口办理，工本费 40 元。",
    },
    {
        "scene_name": "政务场景二：社保卡激活与医保窗口问询",
        "tokens": [
            {"word": "请问", "conf": 0.97, "bbox": (195, 110, 400, 340)},
            {"word": "社保卡", "conf": 0.95, "bbox": (210, 120, 410, 350)},
            {"word": "医保", "conf": 0.96, "bbox": (185, 105, 400, 345)},
            {"word": "窗口", "conf": 0.91, "bbox": (200, 115, 415, 350)},
            {"word": "哪里", "conf": 0.98, "bbox": (205, 110, 420, 355)},
        ],
        "translation": "请问社保卡激活和医保查询业务在哪个窗口办理？",
        "ai_answer": "【政务导办回答】请前往大厅东侧 B 区 6～8 号「社保医保综合窗口」，无需预约，凭身份证在自助取号机取号即可办理。",
    },
    {
        "scene_name": "政务场景三：个体工商户营业执照材料咨询",
        "tokens": [
            {"word": "办理", "conf": 0.96, "bbox": (200, 108, 410, 340)},
            {"word": "营业执照", "conf": 0.94, "bbox": (190, 115, 405, 350)},
            {"word": "需要", "conf": 0.97, "bbox": (205, 110, 420, 345)},
            {"word": "什么", "conf": 0.88, "bbox": (195, 112, 410, 350)},
            {"word": "材料", "conf": 0.98, "bbox": (185, 105, 410, 355)},
        ],
        "translation": "我想办理个体工商户营业执照，请问需要准备什么材料？",
        "ai_answer": "【政务导办回答】请准备：① 经营者身份证原件；② 经营场所租赁合同或产权证明。材料齐全可在 1 号商事登记窗口当场拿证。",
    },
]

_MOCK_INDEX = 0


def predict_sign_tokens(source_type="本机摄像头实时采集", video_file=None, conf_thresh=0.75):
    """
    统一视觉推理入口函数（由 app.py 点击【▶ 开始识别】时自动调用）

    参数:
        source_type (str): "本机摄像头实时采集" 或 "上传本地视频文件"
        video_file (str | None): 上传的本地 MP4 视频路径
        conf_thresh (float): 前端滑块设定的置信度过滤阈值 (0.50 ~ 0.95)

    返回:
        tuple: (custom_rgb_frame, raw_tokens, scene_dict)
            - custom_rgb_frame (np.ndarray | None): 若返回 None，前端会自动根据 bbox 绘制科技蓝检测框；
              若技术 A 组希望直接返回 OpenCV 画好框的 RGB 图像 (480x760)，可在此返回 numpy 数组。
            - raw_tokens (list[dict]): 识别到的手语词序列，格式如 [{"word": "你好", "conf": 0.96, "bbox": (x1,y1,x2,y2)}]
            - scene_dict (dict): 场景元信息与兜底译文
    """
    global _MOCK_INDEX

    if USE_LETTER_MODE:
        return _predict_from_letters(source_type, video_file, conf_thresh)

    if USE_REAL_YOLO_MODEL:
        # ======================================================================
        # TODO: 邵嫣然、杨李灿在此处接入真实 YOLO26 (best.pt) 与静态手语分类模型
        # ======================================================================
        # 示例结构：
        # from ultralytics import YOLO
        # model = YOLO("best.pt")
        # results = model(video_file if "本地视频" in str(source_type) else 0)
        # ...
        pass

    # 默认使用内置的政务大厅模拟数据（支持无显卡环境直接联调演示）
    scene = MOCK_SCENES[_MOCK_INDEX % len(MOCK_SCENES)]
    _MOCK_INDEX += 1
    return None, list(scene["tokens"]), scene


def _predict_from_letters(source_type, video_file, conf_thresh, script=None):
    """
    逐字母模式的推理入口。

    真实链路应当是：
        for frame in 视频流:
            letter, conf = 字母模型(frame)          # A 组提供
            event = stream.feed(letter, conf)       # 去抖 + 停顿分词 + 确认成句
            if event["sentence"]: 交给下游

    当前 A 组权重未到位，这里用 LETTER_DEMO_SCRIPT 按"每个字母保持 3 帧"回放，
    走的是**同一套** letter_pipeline 逻辑，所以切到真实模型时只需替换数据来源。
    """
    scene = {
        "scene_name": "逐字母识别（去抖 + 停顿分词 + 确认成句）",
        "translation": "",
        "ai_answer": "",
    }

    if not HAS_LETTER_PIPELINE:
        return None, [], {"scene_name": "逐字母模式不可用：未找到 letter_pipeline.py"}

    global _LETTER_INDEX
    if script is None:
        script = LETTER_DEMO_SCRIPTS[_LETTER_INDEX % len(LETTER_DEMO_SCRIPTS)]
        _LETTER_INDEX += 1

    tokens, trace = letter_pipeline.simulate(script)

    # 不在这里做阈值过滤：app.py 会统一按滑块过滤，并展示"过滤了几个"
    raw_tokens = [
        {"word": t["word"], "conf": t["conf"], "bbox": LETTER_BOX} for t in tokens
    ]
    scene["scene_name"] += f"：{' / '.join(t['word'] for t in tokens) or '（未识别到字母）'}"
    return None, raw_tokens, scene
