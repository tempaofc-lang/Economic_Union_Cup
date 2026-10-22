# 环境搭建与 Git 协作（零基础版）

> 目标：**从一台什么都没装的电脑，到跑起界面、提交第一次代码。**
> 预计耗时 60 分钟（网速慢的话 90 分钟）。遇到报错先看 §6。

---

## 1. 装 Python

1. 打开 <https://www.python.org/downloads/>，下载 **Python 3.10 或更高版本**（3.11 / 3.12 都行）。
2. 安装时**务必勾选** `Add python.exe to PATH`（这一项在最下面，很容易漏）。
3. 装完打开「命令提示符」或 PowerShell，输入：

```bash
python --version
```

看到 `Python 3.1x.x` 就成功了。

> ⚠️ 如果提示「不是内部或外部命令」，说明 PATH 没勾上。重新运行安装包 → 选 `Modify` → 把 `Add Python to environment variables` 勾上。

---

## 2. 装 Git

1. 打开 <https://git-scm.com/download/win>，下载并一路默认安装。
2. 验证：

```bash
git --version
```

3. 配置身份（只需一次，**名字和邮箱按实际填**）：

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

---

## 3. 装编辑器（推荐 VS Code）

<https://code.visualstudio.com/>

装完建议再装两个扩展：`Python`（微软官方）、`Chinese (Simplified)` 语言包。

---

## 4. 拿到代码并跑起来

### 4.1 克隆仓库

```bash
cd D:\你的工作目录
git clone <组长发你的仓库地址> Economic_Union_Cup
cd Economic_Union_Cup
```

> 如果组长还没建远端仓库，先让他把文件夹压缩包发你，解压后 `cd` 进去也一样，之后再加远端。

### 4.2 建虚拟环境

虚拟环境的作用：**这个项目装的包，不污染你电脑上其他的 Python 项目。**

```bash
cd 手语识别系统
python -m venv .venv
```

激活（Windows PowerShell）：

```bash
.venv\Scripts\activate
```

成功后命令行前面会出现 `(.venv)`。

> ⚠️ PowerShell 可能报「禁止运行脚本」。解决：在 PowerShell 里执行一次
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`，输入 `Y` 确认，然后重新激活。

### 4.3 装依赖

```bash
pip install -r requirements.txt
```

装的是 `gradio`、`opencv-python`、`numpy` 三个包，大约 200MB，耐心等。

### 4.4 启动

```bash
python app.py
```

看到类似 `Running on local URL: http://127.0.0.1:7860` 就成功了，浏览器会自动打开。

点「▶ 开始识别」，右侧应该出现手语词和翻译句子（此刻是**模拟数据**，正常现象）。

### 4.5 配置大模型 Key（只有古宇轩需要）

1. 在 `手语识别系统/` 下新建文件 `.env`（注意前面有个点）。
2. 内容一行：

```
DEEPSEEK_API_KEY=sk-你的真实key
```

3. 装两个包：

```bash
pip install openai python-dotenv
```

4. 在 `agent.py` 开头加：

```python
from dotenv import load_dotenv
load_dotenv()
```

> 🔒 `.env` 已在 `.gitignore` 里，**不会被提交**。永远不要把 Key 直接写在 `.py` 文件里 —— 一旦推到公开仓库，几秒钟就会被爬虫扫走并刷爆额度。

---

## 5. Git 日常操作

### 5.1 五条命令

```bash
git pull                                    # ① 开工前先拉最新代码
git status                                  # ② 看自己改了哪些文件
git add .                                   # ③ 把改动加入暂存区
git commit -m "[模块] 做了什么"                # ④ 提交到本地
git push                                    # ⑤ 推到远端
```

提交信息示例：

```
[前端] 修复静音模式下仍触发语音播报
[视觉] 新增抽帧去抖算法，同一次手势不再重复三遍
[Agent] 接入 DeepSeek JSON 输出并保留规则兜底
[测试] 补充 E2E-03 断网场景用例
[文档] 更新接口契约第 4 节
```

### 5.2 第一次提交

```bash
git status                   # 确认没有 .pt / .env 这类不该提交的文件
git add .
git commit -m "[文档] 初始化个人工作环境"
git push
```

### 5.3 最常遇到的两个问题

**问题 A：`git push` 报错说落后了**

```bash
git pull --rebase
git push
```

**问题 B：提示 `CONFLICT` 冲突**

**立刻停手**，不要乱删代码。截图发群里，让组长处理。冲突处理的成本远低于"乱改导致别人代码消失"。

### 5.4 绝对不要做的事

| ❌ 不要 | 原因 |
| :-- | :-- |
| `git push -f` | 会把别人已提交的代码覆盖掉 |
| 提交 `*.pt` 模型文件 | 几十 MB 起，仓库会被撑爆 |
| 提交 `.env` | Key 泄露，额度可能被刷 |
| 删除 `.gitignore` | 上面两条会立刻发生 |
| 直接改别人的模块文件 | 出问题无法定位，见分工文档 §1 |

---

## 6. 常见报错速查

| 报错关键字 | 原因 | 解决 |
| :-- | :-- | :-- |
| `不是内部或外部命令` | Python 没加 PATH | 重装 Python 并勾选 Add to PATH |
| `禁止运行脚本` / `无法加载文件 ... activate.ps1` | PowerShell 执行策略 | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `ModuleNotFoundError: No module named 'gradio'` | 没装依赖，或虚拟环境没激活 | 重新激活 `.venv` 后 `pip install -r requirements.txt` |
| `No module named 'cv2'` | 同上（opencv 的包名和导入名不同） | `pip install opencv-python` |
| `Address already in use` / 端口占用 | 7860 被上次没关的进程占着 | 关掉旧窗口，或 `app.py` 里 `demo.launch(server_port=7861)` |
| `Error: No such file or directory: 'weights/best.pt'` | A 组权重没放对位置 | 把 `best.pt` 放到 `手语识别系统/weights/` 下 |
| `CUDA out of memory` | 显卡显存不够 | 降 `imgsz`、减小 batch，或改用 CPU 推理（慢但能跑） |
| `openai.AuthenticationError` | Key 错了或额度没了 | 检查 `.env`、去 <https://platform.deepseek.com/> 看余额 |
| `json.decoder.JSONDecodeError` | 大模型没按 JSON 返回 | 检查 prompt 里是否有 "json" 字样，且有输出格式样例 |

**还解决不了？** 把完整报错（**最后 10 行**）+ 你做了什么操作，一起发群里。只发"报错了"这三个字，谁也帮不了你。

---

## 7. 每天的工作循环

```
① git pull                     拉到最新
② 激活虚拟环境                  .venv\Scripts\activate
③ python app.py                先确认能跑（跑不起来先修，别急着写新功能）
④ 改代码                        只改自己负责的文件，一次只改一小块
⑤ 再跑一遍                     确认没改坏
⑥ git add / commit / push      提交
⑦ 群里发三句话                 今天做完 / 明天做 / 卡在哪
```

---

## 8. 给自己的检查清单

- [ ] `python --version` 有输出
- [ ] `git --version` 有输出
- [ ] `git config --global user.name` 有输出
- [ ] 仓库已克隆到本地
- [ ] `.venv` 已创建并能激活（命令行前有 `(.venv)`）
- [ ] `pip install -r requirements.txt` 成功
- [ ] `python app.py` 能打开浏览器界面
- [ ] 点「开始识别」有反应
- [ ] 完成过一次 `git commit` 和 `git push`
- [ ] 确认自己没有提交过 `.env` 和 `*.pt`

十项全勾 = 你的地基打好了，可以开始做任务卡上的事了。
