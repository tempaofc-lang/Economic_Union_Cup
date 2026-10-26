# backend · 手语视界 服务端

> 对应 [05 §2.2 / §3](../文档/docs/05_B组详细分工与项目框架.md) 的目标结构。
> 当前状态：**接口跑通，LLM 主导解译 + 流式输出已上线；视觉侧仍为模拟实现**。

---

## 一、启动

```bash
# 项目根目录
pip install -r backend/requirements.txt

# 方式一：脚本（推荐）
.\run_dev.ps1

# 方式二：手动
cd backend
python -m uvicorn main:app --reload --port 8000
```

| 地址 | 用途 |
| :-- | :-- |
| <http://127.0.0.1:8000/> | **拼音解译调试台**（手动输入拼音，流式跑完整链路） |
| <http://127.0.0.1:8000/docs> | FastAPI 自动生成的接口文档，可直接点着测 |

> ⚠️ 两个容易踩的坑，都已在 `run_dev.ps1` 里处理：
> 1. `--reload` 默认只监听启动目录，而解译引擎在 `手语识别系统/` 下 —— 不加 `--reload-dir` 会出现"改了像没生效"。
> 2. `.env` 的查找依赖调用栈和 cwd，换目录/换 Python 环境可能**静默**读不到 Key。
>    现在改为显式路径解析，并可通过 `/api/health` 的 `env_file`、`api_key_present` 自查。

---

## 二、接口

| 方法 | 路径 | 说明 |
| :-- | :-- | :-- |
| GET | `/api/health` | 健康检查：LLM 是否可用、模型、思考模式、知识库条数、`.env` 来源 |
| GET | `/api/vocab` | 当前词表（17 类，拼音 ↔ 中文） |
| GET | `/api/kb` | 政务知识库全部条目（调试用） |
| POST | `/api/recognize` | 拼音串 → YOLO 风格 token（**当前为模拟**） |
| POST | `/api/agent` | 词序列 → 译文 + 导办回答 + 办事卡片 + 反问 + 工具轨迹（非流式） |
| POST | `/api/agent/stream` | 同上，**SSE 流式实时输出** |
| GET | `/api/session/{id}` | 读取会话记录 |
| POST | `/api/session/{id}/reset` | 清空会话记录 |

### POST /api/recognize

```json
{"text": "shenfenzeng diushi buban", "session_id": null}
```

词尾加 `*` 会返回 0.62 的低置信度，用来演示阈值过滤。
响应固定带 `warning` 字段，明确这是**模拟数据**（对应 05 §1 坑 2 的要求）。

### POST /api/agent

请求：

```json
{
  "words": ["wo", "yao", "buban", "shenfenzeng", "tiyuguan"],
  "session_id": "s-001",
  "strategy": "llm"
}
```

- `words`：**原始拼音**（也兼容已是中文的情况）
- `strategy`：`llm`（默认，LLM 主导解码）｜ `local`（先用封闭词表过滤，再交给 LLM 组句）

响应（05 §3.3 契约 + 调试字段）：

```json
{
  "session_id": "s-001",
  "strategy": "llm",
  "raw_pinyin": ["wo", "yao", "buban", "shenfenzeng", "tiyuguan"],
  "words": ["我", "要", "补办", "身份证"],
  "local_words": ["补办", "身份证"],
  "mapping": [
    {"raw": "wo", "word": "我", "source": "inferred",
     "note": "第一人称代词，非词表项，结合语境推断", "local_word": null, "agree": null},
    {"raw": "yao", "word": "要", "source": "inferred",
     "note": "能愿动词，非词表项，结合语境推断", "local_word": null, "agree": null},
    {"raw": "buban", "word": "补办", "source": "vocab", "note": null,
     "local_word": "补办", "agree": true},
    {"raw": "shenfenzeng", "word": "身份证", "source": "inferred",
     "note": "标准拼写应为 shenfenzheng，少了一个 h，结合补办语境推断",
     "local_word": "身份证", "agree": true},
    {"raw": "tiyuguan", "word": null, "source": "ignored",
     "note": "体育馆，与政务办事无关", "local_word": null, "agree": null}
  ],
  "translation": "您好，我要补办身份证，请问需要带什么材料、去哪个窗口？",
  "answer": "请携带居民户口簿原件，先到 2 号人像采集室免费拍照，再到 3 号户政窗口受理，工本费 40 元。",
  "confirm_card": {
    "title": "身份证补办",
    "materials": ["居民户口簿原件", "本人到场采集人像"],
    "window": "3 号户政窗口",
    "fee": "40 元",
    "duration": "现场受理，约 15 分钟取证（可邮寄）"
  },
  "clarify": null,
  "tool_calls": [{"name": "query_gov_kb", "args": {"words": ["补办", "身份证"]},
                  "ok": true, "ms": 0, "result": "身份证补办"}],
  "mode": "llm",
  "confidence": 0.82,
  "elapsed_ms": 2388,
  "cached": false,
  "error": null
}
```

**`mapping` 是这一版的核心**：逐 token 说明 LLM 怎么理解的、来源是词表还是推断、
以及和本地查表是否一致。`source` 取值 `vocab` / `inferred` / `ignored` / `local`(规则兜底)。

### POST /api/agent/stream（SSE）

输入同 `/api/agent`，按 `text/event-stream` 逐条推送：

| 事件 | 载荷 | 时机 |
| :-- | :-- | :-- |
| `meta` | 会话 id、本地查表、工具轨迹 | 立即（10~20ms） |
| `delta` | `{"text": "..."}` 模型输出增量 | 边生成边推 |
| `error` | `{"message": "..."}` | 出错时 |
| `final` | 完整响应（结构同非流式） | 结束时 |

前端读法：`fetch` + `ReadableStream`，按 `\n\n` 切帧，取 `data: ` 后的 JSON，收到 `[DONE]` 结束。

实测时间线：`meta 12ms → 首个 delta 1092ms → final 1839ms`。

---

## 三、目录职责

```
backend/
├─ main.py                  # FastAPI 入口，只做挂载
├─ config.py                # 路径与端口
├─ schemas.py               # 请求/响应模型（pydantic v2）
├─ api/
│   ├─ agent_api.py         # POST /api/agent、/api/agent/stream
│   ├─ recognize.py         # POST /api/recognize（当前走 mock_vision）
│   └─ meta.py              # health / vocab / kb / session
├─ services/
│   ├─ agent_service.py     # 编排：本地查表(交叉校验) → 查知识库 → 调 LLM → 组装契约
│   ├─ kb.py                # 政务知识库查询（第一个真实工具）
│   ├─ mock_vision.py       # 假视觉：拼音串 → token
│   └─ session_store.py     # 进程内会话
├─ data/gov_kb.json         # 政务知识库（窗口/材料/费用，带出处说明）
└─ frontend/                # 拼音解译调试台（原生 HTML/CSS/JS）
```

**为什么 `agent_service.py` 是 `import` 而不是把代码搬进来**：05 §2.2 的目标是把引擎迁到 `backend/agent/`，
但那是一次性重构。当前先接线、把接口跑通，迁移留到接口稳定之后，避免"边改结构边调逻辑"。

---

## 四、两件值得单独说的事

### 4.1 LLM 主导解码（不是查表）

详见 [文档/docs/04 §1-§3](../文档/docs/04_Prompt设计与拼音解码说明.md)。

一句话：`pinyin_codec` 的 17 词查表从"闸门"降级为"交叉校验"，原始拼音直接交给 LLM，
模型可以纠错拼、按上下文消歧、推断词表外的常用词，但**每个判断都要标来源（vocab/inferred/ignored）并写理由**。

`strategy=local` 保留了旧的查表行为，用来做对照——界面上同一串拼音切一下策略，
就能看出"只有业务名词"和"带着说话人意图"的差别。

### 4.2 知识库：让"40 元"有出处

之前"2 号室拍照、3 号窗口、40 元"全是 `agent.py` 里的硬编码字符串，评委一问就露。现在：

1. `query_gov_kb` 按解码后的中文词查 `data/gov_kb.json`；
2. 命中则把条目作为「权威办事信息」注入模型；
3. **办事卡片以知识库为准**覆盖模型输出——模型只负责措辞，数字由知识库提供；
4. 调用过程真实回显在 `tool_calls` 里。

`data/gov_kb.json` 目前 4 条（身份证补办 / 社保卡激活 / 医保咨询 / 营业执照），
`_meta.disclaimer` 注明是演示值，正式上线前需按当地办事指南逐条核对。

---

## 五、下一步（按 05 的优先级）

1. **接真实视觉**：把 `services/mock_vision.py` 换成读取 A 组 `best.pt`，
   接口与前端一行都不用改 —— 这就是把 `/api/recognize` 先定死契约的意义。
2. **补工具**：05 要求 ≥3 个真实工具。已有 `query_gov_kb`，候选：
   `lookup_required_docs`（按材料反查可办事项）、`check_queue_status`（叫号，可 mock）。
3. **`tokens` 覆盖度硬校验**：模型偶尔会漏报某个拼音，需要在服务端比对输入数量并补齐。
4. **`demo.html` 接 fetch**：李彤把现有设计稿的 `start()` 改成调 `/api/recognize` + `/api/agent/stream`。
5. **`app.py` 移入 `legacy/`** 冻结，作为中期评审保底版本。
