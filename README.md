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

## 一、当前状态（2026-10-22）

| 模块 | 状态 | 说明 |
| :-- | :-- | :-- |
| 视觉模型（A 组） | 🟡 训练中 | 尚未交付权重；B 组先用模拟数据联调 |
| 视觉封装（B 组） | ⬜ 待开工 | `vision_engine.py` 目前是模拟数据，需替换为真实推理 |
| Agent 语义层（B 组） | 🟡 骨架可用 | `agent.py` 现为规则引擎，需接入大模型 API |
| 前端界面（B 组） | 🟢 可运行 | `app.py` / `demo.html` 全流程可演示 |
| 演示视频 | ⬜ 未开始 | 脚本与分镜待编写 |

**关键死线**：2026-11-08 材料截止 → 实际 **11-07 封版**，线上评审 11-09 起。

---

## 二、目录结构

```
Economic_Union_Cup/
├─ README.md                          ← 你在这里
├─ 手语识别系统/                        ← 代码（B 组主战场）
│   ├─ app.py                         ← Gradio 主界面（李彤）
│   ├─ vision_engine.py               ← 视觉接入层（吴夏云）
│   ├─ agent.py                       ← 语义理解层（古宇轩）
│   ├─ vocab.json                     ← 与 A 组共享的 17 类词表
│   ├─ demo.html                      ← 免安装静态预览（评委/老师可直接双击）
│   ├─ .env.example                   ← API Key 配置模板
│   ├─ requirements.txt
│   └─ README_队友对接指南.md           ← 初版对接手册（部分内容已被 docs 取代）
└─ 文档/                               ← 全部文字材料
    ├─ docs/                          ← B 组工作文档（先读这里）
    │   ├─ 01_B组分工与学习路线.md       ← 谁做什么、学什么、什么时候交
    │   ├─ 02_环境搭建与Git协作.md       ← 从零装环境 + 每天怎么提交
    │   └─ 03_接口契约与A组对接清单.md   ← 我们向 A 组要什么、怎么接
    ├─ 成员安排方案.docx                 ← 10-14 初版分工（B 组部分以 01 为准）
    ├─ 第一次讨论内容汇总.docx            ← 10-16 第一次讨论纪要
    ├─ 1009-中国数字经济联盟首届实践大赛参赛承诺书.docx
    ├─ 手语识别系统.zip                  ← 旧版源码归档（不入库）
    ├─ *.jpg / *.jpeg                  ← 早期界面截图（旧版源码，勿直接用于文档）
    └─ picture/                        ← 早期界面截图合集
```

> `文档/手语识别系统.zip` 与各类截图属于归档，被 `.gitignore` 排除，不会进入仓库。

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
