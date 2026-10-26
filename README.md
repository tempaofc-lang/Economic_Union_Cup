# 手语视界 · 基于 YOLO26 的政务服务无障碍智能体

> 中国数字经济联盟首届实践大赛 · 智能体开发与应用赛道 · 本科生组
> 江西师范大学 · 团队「八仙过海」

让听障群众在政务大厅打一段手语，窗口工作人员的屏幕上立刻出现一句通顺的中文，
以及「该去几号窗口、要带什么材料」的答复 —— 不需要手语翻译员在场。

---

## 〇、仓库与协作入口

- **仓库地址**：<https://github.com/tempaofc-lang/Economic_Union_Cup>（Private）
- **加入方式**：把你的 GitHub 用户名发给组长 → 在 `Settings → Collaborators` 邀请 → 查收邮件点接受。
- **克隆**：`git clone https://github.com/tempaofc-lang/Economic_Union_Cup.git`
- **第一次上手**：先读 [文档/docs/01_B组分工与学习路线.md](文档/docs/01_B组分工与学习路线.md) 第 0 节。

---

## 一、当前状态（2026-10-22 晚）

| 模块 | 状态 | 说明 |
| :-- | :-- | :-- |
| 视觉模型（A 组） | 🟡 训练中 | 尚未交付权重；输出格式已改为**无声调拼音** |
| 视觉封装（B 组） | 🟡 有接口无实现 | `/api/recognize` 契约已冻结，当前走 `mock_vision.py`；换个函数即可接真模型 |
| Agent 解译层（B 组） | 🟢 v3.0 | **LLM 主导解码**（可纠错拼/推断词表外词）+ 本地交叉校验 + 三层兜底，11/11 回归通过 |
| 流式输出 | 🟢 已上线 | `/api/agent/stream`（SSE），首帧 12ms、首字 1.1s、全文 1.8s |
| Backend（B 组） | 🟢 可运行 | agent / agent-stream / recognize / health / vocab / kb / session |
| 政务知识库工具 | 🟢 4 条 | 窗口/材料/费用有了出处，`tool_calls` 真实回显 |
| 前端界面（B 组） | 🟢 两种 | Gradio 保底版 + **拼音解译调试台**（含策略对照） |
| 演示视频 | ⬜ 未开始 | 脚本与分镜待编写 |

**关键死线**：2026-11-08 材料截止 → 实际 **11-07 封版**，线上评审 11-09 起。

---

## 一之二、语义链路（10-22 晚 v3：LLM 接管解码）

```
YOLO 拼音  ["wo","yao","buban","shenfenzeng","tiyuguan"]
   │
   ├─► pinyin_codec（本地查表）── 只做交叉校验，不再拦截数据
   │        结果：["补办","身份证"]
   ▼
LLM（DeepSeek 流式）── 逐 token 判断：词表命中 / 推断 / 噪声
   │        结果：我(inferred) 要(inferred) 补办(vocab) 身份证(inferred) tiyuguan(ignored)
   ▼
翻译句    "您好，我要补办身份证，请问需要带什么材料、去哪个窗口？"
导办回答  "请携带居民户口簿原件，先到 2 号人像采集室免费拍照，再到 3 号户政窗口受理，工本费 40 元。"
```

**为什么改成这样**：v2 是「本地查表解码 → LLM 组句」，真正的难点（拼音→词）由查表算法完成，
LLM 只负责排版，词表外的 `wo/yao` 会被直接丢弃 —— 那叫"查表 + 润色"，不是 LLM 解译。
v3 把原始拼音直接交给 LLM，词表降为参考基线，并且**每个词都必须标注来源和推断理由**。

- 设计与调参说明：[文档/docs/04_Prompt设计与拼音解码说明.md](文档/docs/04_Prompt设计与拼音解码说明.md)
- 回归测试：`cd 手语识别系统 && python test_agent.py`
- 断网 / 没额度时自动回退本地规则引擎，界面不会白屏

---

## 一之三、现在已经有两种跑法

| 方式 | 命令 | 用途 |
| :-- | :-- | :-- |
| **Gradio 单文件版**（保底） | `cd 手语识别系统 && python app.py` → <http://127.0.0.1:7860/> | 中期评审保底演示；整条链路已可视化 |
| **FastAPI + 调试台**（主线） | `.\run_dev.ps1` → <http://127.0.0.1:8000/> | 前后端分离，接口已冻结；**手动输入拼音**即可跑通识别→解译全流程 |

调试台地址 <http://127.0.0.1:8000/>，接口文档 <http://127.0.0.1:8000/docs>。
后端细节见 [backend/README.md](backend/README.md)。

---

## 二、目录结构

```
Economic_Union_Cup/
├─ README.md                          ← 你在这里
├─ run_dev.ps1                        ← 一键启动 backend
├─ backend/                           ← FastAPI 服务端（05 §2.2 目标结构）
│   ├─ main.py                        ← 入口，只做挂载
│   ├─ schemas.py                     ← 请求/响应模型
│   ├─ api/                           ← agent / recognize / meta 路由
│   ├─ services/                      ← 解码编排、知识库工具、假视觉、会话
│   ├─ data/gov_kb.json               ← 政务知识库（窗口/材料/费用，有出处）
│   └─ frontend/                      ← 拼音解译调试台（原生 HTML/CSS/JS）
├─ 手语识别系统/                        ← 解译引擎 + Gradio 保底版
│   ├─ app.py                         ← Gradio 主界面（李彤）
│   ├─ vision_engine.py               ← 视觉接入层（吴夏云）
│   ├─ agent.py                       ← 语义解译层：拼音→句子（古宇轩）
│   ├─ pinyin_codec.py                ← 拼音解码层：封闭词表确定性还原（古宇轩）
│   ├─ vocab.json                     ← 与 A 组共享的 17 类词表（拼音↔中文）
│   ├─ test_agent.py                  ← 解译层回归测试（12 条用例）
│   ├─ demo.html                      ← 免安装静态预览（评委/老师可直接双击）
│   ├─ .env.example                   ← API Key 配置模板（真实 .env 不入库）
│   ├─ requirements.txt
│   └─ README_队友对接指南.md           ← 初版对接手册（部分内容已被 docs 取代）
├─ tests/
│   └─ agent_cases_result.md          ← 最近一次回归测试结果
└─ 文档/                               ← 全部文字材料
    ├─ docs/                          ← B 组工作文档（先读 05 和 06）
    │   ├─ 01_B组分工与学习路线.md       ← 分角色任务卡与学习链接（分工部分已被 05 取代）
    │   ├─ 02_环境搭建与Git协作.md       ← 从零装环境 + 每天怎么提交
    │   ├─ 03_接口契约与A组对接清单.md   ← 向 A 组要什么、拿到后怎么接（端点以 05 为准）
    │   ├─ 04_Prompt设计与拼音解码说明.md ← 拼音链路与 system prompt 逐段拆解
    │   ├─ 05_B组详细分工与项目框架.md   ← ⭐ 分工与架构的最终依据（FastAPI + 静态前端）
    │   └─ 06_学习路径与资源清单.md      ← ⭐ 分角色学习路线、排错决策树、资源表
    ├─ 成员安排方案.docx                 ← 10-14 初版分工
    ├─ 第一次讨论内容汇总.docx            ← 10-16 第一次讨论纪要
    ├─ 1009-中国数字经济联盟首届实践大赛参赛承诺书.docx
    ├─ 手语识别系统.zip                  ← 旧版源码归档（不入库）
    ├─ *.jpg / *.jpeg                  ← 早期界面截图（旧版源码，勿直接用于文档）
    └─ picture/                        ← 早期界面截图合集
```

> `文档/手语识别系统.zip` 与各类截图属于归档，被 `.gitignore` 排除，不会进入仓库。

**文档优先级（2026-10-22 组内决定）**：01–04 与 05–06 **都可以遵循，冲突时以 05 / 06 为准**。
05 决定分工方式、系统架构与接口契约；06 决定学习路线；01 保留分角色任务卡细节，03 保留向 A 组确认的问题清单。

---

## 三、三分钟跑起来

```bash
# 1. 进入代码目录
cd 手语识别系统

# 2. 建虚拟环境（只需一次）
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # macOS / Linux

# 3. 装依赖
pip install -r requirements.txt

# 4. 启动
python app.py
```

浏览器打开 `http://127.0.0.1:7860`，点「▶ 开始识别」即可看到完整流程（此时为模拟数据）。

> 不想装环境？直接双击 `手语识别系统/demo.html` 也能体验全部交互（需联网加载样式）。

详细步骤与常见报错见 [文档/docs/02_环境搭建与Git协作.md](文档/docs/02_环境搭建与Git协作.md)。

---

## 四、架构约定（最重要的一条）

```
A 组的 YOLO 裸模
      ↓  （吴夏云封装）
vision_engine.py ── 输出 [{word, conf, bbox}, ...]
      ↓
app.py（李彤）── 置信度过滤 / 撤回 / 确认成句
      ↓
agent.py（古宇轩）── 词序列 → (中文翻译, 政务导办回答)
      ↓
界面大字展示 + 语音播报 + 双向回复 + 历史记录
```

**换模型不需要改界面。** 三个人各碰各的文件，互不阻塞：

- 古宇轩 → 只改 `agent.py`
- 吴夏云 → 只改 `vision_engine.py`（及新增 `inference.py`）
- 李彤 → 只改 `app.py` / `demo.html`

---

## 五、协作纪律（五条）

1. **每天至少一次 commit**，哪怕是改一个错别字。提交信息格式：`[模块] 做了什么`，例如 `[前端] 修复静音模式仍播报的问题`。
2. **开工先拉取**：`git pull` 之后再改。
3. **API Key、模型权重、录屏原片一律不进仓库**（`.gitignore` 已挡，别用 `-f` 强推）。
4. **卡住 30 分钟就发群里**，不要硬扛一晚上。
5. **每天的三个问题**发组内群：今天做完了什么 / 明天做什么 / 现在卡在哪。

---

## 六、用到的大模型

DeepSeek API（OpenAI 兼容格式）：

- `base_url`: `https://api.deepseek.com`
- `model`: `deepseek-flash`
- 文档：<https://api-docs.deepseek.com/zh-cn/>
- 结构化输出（JSON）：<https://api-docs.deepseek.com/zh-cn/guides/json_mode>

API Key 由组长统一申领，放进本地 `.env`，**不入库**。
