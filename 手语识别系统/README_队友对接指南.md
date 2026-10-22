# 🚀 YOLO26 静态手语词识别智能体 · 队友对接与修改指南

## 🧭 第二部分：项目完整制作流程与系统架构（全体队友必读）

### 1. 端到端系统数据流向（两段式解耦架构）

本项目聚焦初赛“静态手语词识别 + Agent 语义理解”，采用**视觉感知层、智能体语义层、前端交互层三层完全解耦**的设计。**技术 A 组和技术 B 组的同学只需修改各自对应的独立 `.py` 文件，互不干扰、无需改动前端界面代码**：

~~~text
[本机摄像头实时流 / 本地MP4视频上传]
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. 视觉感知层：vision_engine.py（技术 A 组：邵嫣然、杨李灿、张明慧）│
│    - YOLO26 检测手部边界框 bbox: (x1, y1, x2, y2)             │
│    - 静态手语分类器输出词汇与置信度: {"word": "身份证", "conf": 0.96}│
└─────────────────────────────────────────────────────────────┘
        │
        ▼ (经过 app.py 置信度阈值 Slider >= 0.75 防抖过滤)
        ├─► 若有效检出词数为 0 ──► 自动触发【⚠️ 识别失败四联动提示】
        │                         (红框告警 + 状态灯变红 + 原因分析 + 大字重试引导 + 语音提示)
        │
        ▼ (若检出有效手语词序列，支持自动成句、或手动【⬅️ 撤回上一词】/【✅ 确认成句】)
┌─────────────────────────────────────────────────────────────┐
│ 2. 智能体语义层：agent.py（技术 B 组：古宇轩）                    │
│    - 输入: 过滤后的静态手语词列表 ["你好", "身份证", "丢失", "补办"] │
│    - 输出1 (translation): "工作人员您好，我的身份证丢失了，怎么补办？"│
│    - 输出2 (ai_answer): "【政务导办回答】请带户口本先到2号室拍照..." │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 交互展示层：app.py / demo.html（技术 B 组：李彤）             │
│    - 左栏（60%固定大尺寸）：760×480 视频检测框预览 + 底部五键控制栏 │
│    - 右栏（40%独立上下滑动）：置信度条 + 21px大字翻译 + 双向回复 + 历史│
│    - 语音播报：支持【🔊 自动朗读】与【🔇 静音模式】一键切换         │
└─────────────────────────────────────────────────────────────┘
~~~

### 2. 团队成员分工与各自对接文件清单

| 团队成员 | 角色定位 | 负责查看/修改的文件 | 对接与操作方式说明 |
| :--- | :--- | :--- | :--- |
| **全体队友 / 李宏伟老师** | 预览界面、评审指导 | `demo.html` | **无需安装任何软件或 Python 环境**，直接双击 `demo.html` 即可在浏览器全屏体验所有功能（含模拟识别失败、静音切换、模块删减开关）。 |
| **邵嫣然、杨李灿、张明慧**<br>（技术 A 组 · 有显卡） | YOLO26 训练、手语分类模型、数据集映射 | `vision_engine.py` | 训练出 `best.pt` 和分类器后，打开 `vision_engine.py`，将 `USE_REAL_YOLO_MODEL` 改为 `True`，在 `predict_sign_tokens()` 中返回识别词、置信度和坐标框即可，**无需碰 `app.py`**。 |
| **古宇轩**<br>（技术 B 组 · Agent 负责） | Agent 语义理解、对接大模型 API | `agent.py` | 打开 `agent.py`，将 `USE_REAL_LLM_API` 改为 `True`，在 `parse_semantics(word_list)` 中调用大模型 API 并返回 `(中文翻译句, 政务导办回答)` 元组即可，**无需碰 `app.py`**。 |
| **李彤**<br>（技术 B 组 · 前端负责） | 界面维护、模块增删、前后端联调 | `app.py` | 通过修改 `app.py` 顶部的 `FEATURE_FLAGS` 字典（`True`/`False`），5 秒内即可隐藏或开启任意功能模块。 |
| **杜佳润、吴夏云、刘欣、李东瑞** | 视频录屏、技术文档、PPT 美化、系统测试 | `demo.html` 或 `app.py` | 使用界面进行 11.3 真机录屏、截取高清白底答辩图（可利用界面上的政务三大场景切换、模拟失败、双向回复演示完整故事线）。 |

### 3. 结合团队排期的四阶段开发推进表（10.14 — 11.8）

1. **第 1 阶段（10.17 — 10.21）假数据跑通框架**：
   * 李彤、古宇轩无需等待 A 组训练权重，直接运行 `python app.py`（此时 `USE_REAL_YOLO_MODEL = False`），用内置的政务服务三大场景模拟数据跑通「识别 → 撤回/确认 → Agent 翻译与导办问答 → 自动朗读/静音 → 双向回复 → 历史记录」全流程。
2. **第 2 阶段（10.22 — 10.26）中期评审（李宏伟老师把关）**：
   * 10.25—10.26 周末全员会向李老师展示初版界面与模型出框效果，并根据老师意见通过 `FEATURE_FLAGS` 随时微调模块。
3. **第 2 阶段后半（10.27 — 10.31）接通真实模型与大模型 API**：
   * 邵嫣然、杨李灿将 `best.pt` 接入 `vision_engine.py`；古宇轩将大模型 API 接入 `agent.py`；李彤完成端到端联调。
4. **第 3~5 阶段（11.3 — 11.8）视频录制、文档截图与封版提交**：
   * 杜佳润使用联调好的界面录制 MP4 演示视频；吴夏云、刘欣截取高清白底界面图放入技术文档与路演 PPT。

---

## 🔧 第三部分：队友修改代码与接入模型实操指南（手把手教程）

### 指南 A：技术 A 组（邵嫣然、杨李灿）如何接入真实的 YOLO26 和分类模型？

打开 `vision_engine.py`，你们会看到文件顶部有一个开关 `USE_REAL_YOLO_MODEL = False`。当你们训练好 YOLO26 权重 `best.pt` 和静态手语分类器后，只需按以下三步修改 `vision_engine.py`：

1. 将第 18 行的 `USE_REAL_YOLO_MODEL = False` 改为 `True`；
2. 在全局区域加载你们训练好的模型权重（只加载一次，避免每次点击重复加载卡顿）；
3. 在 `predict_sign_tokens(source_type, video_file, conf_thresh)` 函数中返回检测到的手语词列表，每个元素是一个字典 `{"word": "识别词", "conf": 0.95, "bbox": (x1, y1, x2, y2)}`：

~~~python
# 技术 A 组在 vision_engine.py 中的真实模型接入示例：
from ultralytics import YOLO
import cv2

USE_REAL_YOLO_MODEL = True
yolo_model = YOLO("best.pt")  # 邵嫣然训练的 YOLO26 手部检测权重

def predict_sign_tokens(source_type, video_file=None, conf_thresh=0.75):
    """
    返回值说明:
      - annotated_frame (np.ndarray 或 None): 若返回 RGB 格式的画框图像(480x760)，前端直接显示它；
        若返回 None，前端会自动根据下方 tokens 里的 bbox 坐标绘制蓝色检测框！
      - raw_tokens (list[dict]): 模型识别出的原始手语词序列（无需手动过滤阈值，app.py 会自动根据滑块过滤）
      - scene_meta (dict): 当前演示场景的兜底信息
    """
    # 调用 YOLO26 + 杨李灿的静态手语分类脚本后，按以下格式组装结果即可：
    raw_tokens = [
        {"word": "你好", "conf": 0.96, "bbox": (190, 110, 390, 340)},
        {"word": "身份证", "conf": 0.94, "bbox": (205, 115, 405, 345)},
        {"word": "补办", "conf": 0.91, "bbox": (200, 112, 415, 350)},
    ]
    return None, raw_tokens, {"scene_name": "YOLO26 真实模型推理"}
~~~

---

### 指南 B：技术 B 组（古宇轩）如何接入大模型 Agent？

打开 `agent.py`，找到 `parse_semantics(word_list)` 函数。前端 `app.py` 会把经过置信度过滤（以及用户撤回修改后）的纯中文手语词列表（例如 `["你好", "身份证", "丢失", "补办"]`）传给你，你只需返回 **`(手语原意中文翻译, 政务AI导办回答)`** 两个字符串组成的元组：

~~~python
# 古宇轩在 agent.py 中的大模型 API 接入示例：
from openai import OpenAI

USE_REAL_LLM_API = True
client = OpenAI(api_key="你的API_KEY", base_url="https://api.deepseek.com/v1")

def parse_semantics(word_list: list[str]) -> tuple[str, str]:
    joined = "、".join(word_list)
    prompt = (
        f"听障办事群众在政务服务大厅依次打出了以下静态手语词：【{joined}】。\n"
        "请严格输出两行内容（不要多余解释）：\n"
        "第1行：站在听障群众第一人称视角，连词成句翻译为自然礼貌的中文问句；\n"
        "第2行：站在政务大厅AI导办员角度，给出简明、准确的窗口指引与所需材料回答（以【政务导办回答】开头）。"
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    lines = [line.strip() for line in resp.choices[0].message.content.strip().splitlines() if line.strip()]
    translation = lines[0] if len(lines) > 0 else f"您好，我想办理关于{joined}的业务。"
    ai_answer = lines[1] if len(lines) > 1 else "【政务导办回答】请携带本人身份证原件前往一号综合窗口办理。"
    return translation, ai_answer
~~~

---

### 指南 C：前端负责人（李彤）如何一键删减/隐藏后期来不及做的功能？

如果后期某个模块不想展示，打开 `app.py` 第 25 行左右的 `FEATURE_FLAGS` 字典，把对应开关从 `True` 改为 `False`，保存并重启 `python app.py` 即可自动隐藏该模块，完全不会破坏布局或报错：

~~~python
FEATURE_FLAGS = {
    "SHOW_VOCAB_BAR": True,         # 设为 False 隐藏顶部「16个支持政务手语词库」提示栏
    "SHOW_CONF_SLIDER": True,       # 设为 False 隐藏「置信度过滤阈值 Slider」
    "SHOW_AUTO_TTS_TOGGLE": True,   # 设为 False 隐藏「🔊自动朗读 / 🔇静音模式」切换开关
    "SHOW_WORD_EDIT_BTNS": True,    # 设为 False 隐藏「⬅️ 撤回上一词」与「✅ 确认成句」按钮
    "SHOW_AI_GOV_ANSWER": True,     # 设为 False 只显示手语翻译，隐藏「政务AI导办回答」
    "SHOW_TWO_WAY_REPLY": True,     # 设为 False 隐藏「政务窗口工作人员双向回复输入框」
    "SHOW_HISTORY_PANEL": True,     # 设为 False 隐藏右下角「历史翻译记录与删除」面板
}
~~~
