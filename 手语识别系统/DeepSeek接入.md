# DeepSeek 接入（Agent 问答的真实 LLM）

## 先说清楚一件事

昨天给你的包（单级YOLO版_v2/v3.zip）**只包含识别链路**：
`sign_real_link.py`、`cam_test.py`、`pinyin_codec.py`、`vocab.json`、`checkpoints/`。
它不含 `app.py` / `agent.py`，也**从来没有包含过 `.env`** —— 你们原工程
`手语识别系统/` 里本来就没有 `.env`（解压核实过），这一点我昨天应该在交付说明里
明确写出来，是我漏了，抱歉。

后果就是截图里那句话：**没有 `.env` → `agent.py` 读不到 `DEEPSEEK_API_KEY`
→ `llm_available()` 返回 False → Agent 问答自动走本地规则兜底**。
识别（手语→拼音→中文）完全不受影响，受影响的只是"AI 问答"那一环。

## 怎么接上（2 分钟）

### 第 1 步：建 `.env`

在 `手语识别系统/` 目录下（与 `app.py`、`agent.py` 同级）新建一个文件，
文件名就是 `.env`（注意 Windows 资源管理器会隐藏扩展名，别存成 `.env.txt`），
内容如下，把 key 换成你们自己的：

```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
USE_REAL_LLM_API=1
```

> ⚠️ **必须写 `DEEPSEEK_MODEL=deepseek-chat`**。
> `agent.py` 里模型名的默认值是 `deepseek-flash`，DeepSeek 官方没有这个模型，
> 不写这行 API 会直接报 `Model Not Exist`。

### 第 2 步：自检（不用起服务就能验证）

```bash
cd /d "C:\Users\Dujia\Desktop\demo-repository-main (4)\Economic_Union_Cup-main\手语识别系统"
python check_llm.py
```

它会依次告诉你：`.env` 找到没有、key 读到没有、模型名是什么、
然后**真实调一次 DeepSeek API**，通了会打印一句回话。

### 第 3 步：重启 app.py

`.env` 是启动时读一次，改完必须重启 `app.py` 才生效。
验证：浏览器开 `/api/health`，看返回里的
`api_key_present: true`、`env_file: <路径>`。

## 常见坑

| 现象 | 原因 |
|---|---|
| health 里 `env_file: null` | `.env` 放错位置。`agent.py` 只找四处：`手语识别系统/.env`、当前目录、`backend/../手语识别系统/.env`、项目根 |
| `api_key_present: false` | key 名写错（必须叫 `DEEPSEEK_API_KEY`），或文件存成了 `.env.txt` |
| API 报 Model Not Exist | 没写 `DEEPSEEK_MODEL=deepseek-chat` |
| 报 401 | key 复制时带了空格，或 key 被停用 |
| 报 402 / 余额不足 | DeepSeek 平台充值，几块钱够演示用很久 |
| 想临时禁用 LLM | `.env` 里 `USE_REAL_LLM_API=0`，走本地规则（离线演示保险） |

## 演示建议

- 现场网络不稳的话，**兜底不是坏事**：`agent.py` 在 LLM 不可用时自动走本地规则，
  界面不会报错，只是回答变"模板化"。答辩可以主动讲这个降级设计，是加分项。
- key 别提交到 git / 别贴进项目书截图。`.env` 加入 `.gitignore`。
