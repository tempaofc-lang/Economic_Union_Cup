# backend · 手语视界 服务端

> 对应 [05 §2.2 / §3](../文档/docs/05_B组详细分工与项目框架.md) 的目标结构。
> 当前状态：**接口已跑通，视觉侧为模拟实现**，用于在 A 组模型未就绪时完成全链路开发。

---

## 一、启动

```bash
# 项目根目录
pip install -r backend/requirements.txt

cd backend
python -m uvicorn main:app --reload --port 8000
```

或直接双击根目录的 `run_dev.ps1`。

| 地址 | 用途 |
| :-- | :-- |
| <http://127.0.0.1:8000/> | **拼音解译调试台**（手动输入拼音，跑完整链路） |
| <http://127.0.0.1:8000/docs> | FastAPI 自动生成的接口文档，可直接点着测 |

---

## 二、接口

| 方法 | 路径 | 说明 |
| :-- | :-- | :-- |
| GET | `/api/health` | 健康检查：LLM 是否可用、模型名、知识库条数 |
| GET | `/api/vocab` | 当前词表（17 类，拼音 ↔ 中文） |
| GET | `/api/kb` | 政务知识库全部条目（调试用） |
| POST | `/api/recognize` | 拼音串 → YOLO 风格 token（**当前为模拟**） |
| POST | `/api/agent` | 词序列 → 译文 + 导办回答 + 办事卡片 + 反问 + 工具轨迹 |
| GET | `/api/session/{id}` | 读取会话记录 |
| POST | `/api/session/{id}/reset` | 清空会话记录 |

### POST /api/recognize

```json
{"text": "nihao shenfenzheng diushi* buban", "session_id": null}
```

词尾加 `*` 会返回 0.62 的低置信度，用来演示阈值过滤。
响应固定带 `warning` 字段，明确这是**模拟数据**（对应 05 §1 坑 2 的要求）。

### POST /api/agent

请求：

```json
{"words": ["nihao", "shenfenzheng", "buban"], "session_id": "s-001", "staff_reply": null}
```

响应（05 §3.3 契约 + 调试字段）：

```json
{
  "session_id": "s-001",
  "raw_pinyin": ["nihao", "shenfenzheng", "buban"],
  "words": ["你好", "身份证", "补办"],
  "unknown": [],
  "decode": [{"raw": "nihao", "hanzi": "你好", "hit": "exact", "score": 1.0}],
  "translation": "您好，我想补办身份证，请问在哪个窗口办理？",
  "answer": "请携带居民户口簿原件，先到 2 号人像采集室免费拍照，再到 3 号户政窗口受理，工本费 40 元，约 15 分钟取证。",
  "confirm_card": {
    "title": "身份证补办",
    "materials": ["居民户口簿原件", "本人到场采集人像"],
    "window": "3 号户政窗口",
    "fee": "40 元",
    "duration": "现场受理，约 15 分钟取证（可邮寄）"
  },
  "clarify": null,
  "tool_calls": [{"name": "query_gov_kb", "args": {"words": ["你好", "身份证", "补办"]},
                  "ok": true, "ms": 0, "result": "身份证补办"}],
  "mode": "llm",
  "confidence": 0.93,
  "elapsed_ms": 2781,
  "cached": false,
  "error": null
}
```

`mode`：`llm` 表示走了大模型，`rule` 表示断网/失败时回退到本地规则引擎。
`cached`：命中进程内缓存（同样输入第二次调用会接近 0ms，界面会标注"缓存命中"，不假装是真实耗时）。

---

## 三、目录职责

```
backend/
├─ main.py                  # FastAPI 入口，只做挂载
├─ config.py                # 路径与端口
├─ schemas.py               # 请求/响应模型（pydantic v2）
├─ api/
│   ├─ agent_api.py         # POST /api/agent
│   ├─ recognize.py         # POST /api/recognize（当前走 mock_vision）
│   └─ meta.py              # health / vocab / kb / session
├─ services/
│   ├─ agent_service.py     # 编排：解码 → 查知识库 → 调 LLM → 组装契约
│   ├─ kb.py                # 政务知识库查询（第一个真实工具）
│   ├─ mock_vision.py       # 假视觉：拼音串 → token
│   └─ session_store.py     # 进程内会话
├─ data/gov_kb.json         # 政务知识库（窗口/材料/费用，带出处说明）
└─ frontend/                # 拼音解译调试台（原生 HTML/CSS/JS）
```

**为什么 `agent_service.py` 是 `import` 而不是把代码搬进来**：05 §2.2 的目标是把引擎迁到 `backend/agent/`，
但那是一次性重构。当前先接线、把接口跑通，迁移留到接口稳定之后，避免"边改结构边调逻辑"。

---

## 四、知识库：让"40 元"有出处

之前 `agent.py` 里的"2 号室拍照、3 号窗口、40 元"全是硬编码字符串，评委一问就露。
现在：

1. `query_gov_kb` 工具按解码后的中文词查 `data/gov_kb.json`；
2. 命中则把条目渲染成"权威办事信息"注入到大模型的用户消息里；
3. **办事卡片以知识库为准**覆盖模型输出——模型只负责措辞，数字由知识库提供；
4. 调用过程真实回显在 `tool_calls` 里，界面有一个可展开的「Agent 决策轨迹」。

`data/gov_kb.json` 目前 4 条（身份证补办 / 社保卡激活 / 医保咨询 / 营业执照），
`_meta.disclaimer` 已注明是演示值，正式上线前需按当地办事指南逐条核对。

---

## 五、下一步（按 05 的优先级）

1. **接真实视觉**：把 `services/mock_vision.py` 换成读取 A 组 `best.pt`，
   接口与前端一行都不用改 —— 这就是把 `/api/recognize` 先定死契约的意义。
2. **补工具**：05 要求 ≥3 个真实工具。候选：`query_gov_kb`（已有）、
   `lookup_required_docs`（按材料反查可办事项）、`check_queue_status`（叫号排队，可 mock）。
3. **`demo.html` 接 fetch**：李彤把现有设计稿的 `start()` 从 mock 改成调 `/api/recognize` + `/api/agent`。
4. **`app.py` 移入 `legacy/`** 冻结，作为中期评审保底版本。
