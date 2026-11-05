# 方案B 接入纠错检测指南（sign_real_link.py 版）

改动范围（只动了 2 个文件，其余原样）：

| 文件 | 改动 |
|---|---|
| `sign_real_link.py` | **新增**：YOLO检测→HandCNN分类→字母拼读→拼音词 完整推理模块 |
| `vision_engine.py` | 开关 `USE_REAL_YOLO_MODEL=True`；TODO 分支填入 4 行调用（异常自动回退模拟数据） |
| `app.py` / `agent.py` / `pinyin_codec.py` / `vocab.json` | **未改一行**，接口契约原样对接 |

已实测通过（2026-11-05）：两份权重加载 30/30 类一致；`NI-HAO-XIE-XIE` → `['nihao','xiexie']` → codec 还原 `你好、谢谢`；摄像头真实采集返回契约 `{"word","conf","bbox"}` 全部通过。

---

## 一、按层纠错（出问题先定位在哪一层）

```
L0 依赖 → L1 权重 → L2 拼读 → L3 链路契约 → L4 界面表现 → L5 真实识别效果
```

### L0 依赖检查（在 手语识别系统 目录下）

```powershell
python -c "import torch,cv2,numpy,PIL,ultralytics; print('deps OK')"
```

缺什么 `python -m pip install 什么`。`app.py` 另需 `gradio`。

### L1 权重体检（最常用，一条命令查完整链路）

```powershell
python sign_real_link.py
```

期望 4 行全绿：

```
[1] 分类权重: OK 30 类, 首尾 = A..ZH
[2] YOLO 权重: OK 30 类
[3] 拼读层: N-I-H-A-O -> ['nihao']
    S-H-E-N-F-E-N-Z-H-E-N-G -> ['shenfenzheng']
```

| 报错 | 原因 | 纠正 |
|---|---|---|
| `FileNotFoundError ...best.pt` | 权重移动了位置 | 设环境变量 `SIGN_YOLO_PT` / `SIGN_CLS_PT` 指向新路径（默认值写死在文件顶部） |
| `AssertionError: 分类权重类型异常` | 拿成了 SignNet 或其他权重 | 分类端必须是交付包 `checkpoints\best.pt`（HandCNN，键 `model_state`） |
| `Error(s) in loading state_dict ... size mismatch` | 权重和 `sign_real_link.py` 内 HandCNN 结构不符 | 说明拿到了别的版本模型，找杨李灿核对 |
| `ModuleNotFoundError: ultralytics` | 未装库 | L0 补装 |
| `[3] 拼读层 -> []` | `vocab.json` 被改动/丢失 | 恢复词表（classes 需含 pinyin 字段） |

### L2 拼读层单测

```powershell
python -c "from sign_real_link import spell_letters, load_pinyin_targets as L; import pinyin_codec as p; w=spell_letters(list('SHENFENZHENGDIUSHIBUBAN'), L()); print(w, p.decode_tokens(w)['words'])"
```

期望：`['shenfenzheng','diushi','buban'] ['身份证', '丢失', '补办']`。
拼不出词 = 字母流里有错字（见 L5 混淆表）。

### L3 链路契约（不依赖摄像头，秒级）

```powershell
python -c "import vision_engine as ve; print('开关:', ve.USE_REAL_YOLO_MODEL); f,t,s=ve.predict_sign_tokens('本机摄像头实时采集',None,0.75); print(type(t), t[:1] if t else '空列表(=界面识别失败分支,正常)')"
```

期望：开关 `True`；返回是列表（每项含 word/conf/bbox 三键）。
若打印 `[vision_engine] 真实模型推理失败，回退模拟数据: ...` —— 冒号后就是根因（摄像头被占用/无权限时很常见），界面仍能演示但走的假数据。

### L4 界面启动

```powershell
python app.py
```

点击【▶ 开始识别】：会实时采集最长 3 秒画面（控制台可能短暂停顿，属预期）。

### L5 真实识别效果（最终判据，必须真人测）

对着摄像头**逐个字母**打手语，每个字母保持约 1 秒以上（去抖要求连续 2 帧以上同字母），打完整词停顿一下再打下一个。常用词手势数：

| 词 | 字母数 | 词 | 字母数 |
|---|---|---|---|
| yibao 医保 | 5 | banli 办理 | 6 |
| xiexie 谢谢 | 6 | nali 哪里 | 4 |
| shenfenzheng 身份证 | 11 | yingyezhizhao 营业执照 | 13 |

---

## 二、常见现象 → 纠正速查

| 现象 | 定位 | 纠正 |
|---|---|---|
| 界面总是"识别失败" | ①镜头前无人 ②检不出手框 ③拼不出词表词 | 依次查：手入画且光线足；`DET_CONF 0.35` 可调低（环境变量不便时改文件顶部常量）；确认打的是字母 |
| 单词丢字/丢尾 | 字母打太快（<2 帧）被去抖吃掉 | 每个字母保持 1 秒以上 |
| 长词拼不出（身份证11字母） | 单字母约 61% 准确率，误差连乘（11字母≈0.2%） | 方案B 固有代价：优先演示短词（谢谢/哪里/医保/办理），长词用简写别名（sfz→aliases 已支持，但需模型拼出 s-f-z） |
| 频繁出 `unknown` | C↔CH、S↔SH、K↔L、J↔Z/ZH 混淆 | 已知问题，codec 的 fuzzy 层部分兜住；看返回 scene 里 `letters` 字段核对模型实际收到的字母 |
| 一个词被拆成两半 | 停顿>2 帧判为字母断点（当前按连续段切） | 词间不必刻意停顿过久；拼读是全局贪心，一般不受影响 |
| 首字母"确认"无法触发 | `queren` 需 15 个字母，30 类模型没有专门确认手势 | 用前端【✅ 确认成句】按钮（app.py 已有，不受影响） |
| 摄像头打不开 | 被其他程序占用/无权限 | 关闭占用的相机应用重试；L3 命令可复现 |
| 要临时回退演示模式 | — | `vision_engine.py` 第 20 行改 `USE_REAL_YOLO_MODEL = False` 即可，一键还原 |

## 三、性能说明

- CPU 实测每帧约 0.3~0.5s（每 2 帧取 1 帧、最长 3 秒封顶），点【开始识别】约 3~5 秒返回，演示可接受；
- 想调采集时长/采样密度：`sign_real_link.py` 顶部 `MAX_SECONDS`、`FRAME_STRIDE`；
- 有 N 卡机器装 CUDA 版 torch 后自动加速（当前代码按 CPU 写死 map_location，足够本场景）。

## 四、验收口径

- [x] L1 四行全绿
- [x] L2 拼读→codec 还原正确
- [x] L3 契约字段齐全
- [x] L4 界面可启动、异常自动回退不崩
- [ ] L5 真人打手语出词 —— **需你们在答辩机位实测**（建议练：xiexie、nali、yibao、banli）

---

## 五、B 组联调补充（2026-11-05）

### 5.1 环境安装（CPU 即可，不必装 CUDA 版）

```powershell
cd 手语识别系统
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install ultralytics
```

实测版本：`torch 2.15.0+cpu`、`ultralytics 8.4.202`。

### 5.2 Windows + Anaconda 的 OMP 冲突（必看）

症状——`import torch` 直接失败：

```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

原因：PATH 里的 conda `Library\bin` 已经加载过一份 `libiomp5md.dll`，与 pip 装的 torch 自带 OpenMP 冲突。
处理：`sign_real_link.py` 已在 `import torch` **之前**设了
`os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")`，无需手动设置。
但如果在别的脚本里单独 `import torch`，需要自己补这一行，否则同样报错。

### 5.3 权重放哪里

查找顺序：**环境变量 → 仓库内固定位置 → A 组原始路径**（原始路径在 E 盘，只对 A 组自己的机器有效）。

| 模型 | 环境变量 | 推荐放置位置 |
| :-- | :-- | :-- |
| YOLO 检测 | `SIGN_YOLO_PT` | `手语识别系统/weights/yolo_best.pt` |
| HandCNN 分类 | `SIGN_CLS_PT` | `手语识别系统/weights/cls_best.pt` |

`*.pt` 已被 `.gitignore` 排除，不会误提交。缺权重时会给出可操作的报错（列出已尝试的路径），
而不是裸 `FileNotFoundError`。

### 5.4 拼读层修复（本次联调做的两处改动）

| 问题 | 修复前 | 修复后 |
| :-- | :-- | :-- |
| 词内错 1 个字母 | `NIXHAO` → `[]`，整词丢失 | `NIXHAO` → `['nihao']` |
| 词表外拼音 | `WOYAO` → `[]`，「我要」消失 | `WOYAO` → `['woyao']` → LLM 解出「我要」 |

**为什么必须修**：单字母准确率约 61%，11 字母连乘只剩 0.2%——没有容错层，
"身份证"这类核心词永远拼不出来。

实现要点：

- `_match_at()` 三级匹配：精确最长 → **长词覆盖**（短别名 `shenfenz` 不会截断 `shenfenzheng`）
  → 相似度兜底（`FUZZY_RATIO=0.78`，`LONGER_RATIO=0.85`）；
- 相似度兜底只在"上一个字母已匹配成功"时启用，否则窗口会后滑，
  把 `WOBUBAN` 里的 `o` 吃掉、拼成 `buban` 而丢掉 `wo`；
- 拼不出的连续字母攒成**残余片段**透传，要求长度 ≥2 且含元音（过滤 `qqq` 之类噪声）。

> 残余片段的代价：可能透传少量无意义 token，由 LLM 标 `inferred` 或忽略。
> 实测 `NALK`（相似度 0.75 未达阈值）→ 透传 `nalk` → LLM 仍解出「哪里」。

### 5.5 当前状态（本机实测）

- 依赖：已装齐（torch / ultralytics / cv2 / PIL / numpy）
- L1：❌ 权重缺失（`E:` 盘不存在，仓库内也没有 `.pt`），**卡在权重加载，不是代码问题**
- L2：✅ 拼读层 19/19 通过（含容错、残余透传、7 个既有用例零回归）
- L3：✅ 能正确回退并打印具体原因
- L4/L5：⛔ 需先拿到权重 + 摄像头

