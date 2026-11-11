# -*- coding: utf-8 -*-
"""
check_llm.py —— DeepSeek 接入自检（不用启动 app.py 就能验证）

用法（在 手语识别系统/ 目录下运行）：
    python check_llm.py

它依次检查：
  1. .env 能不能找到（按 agent.py 的四个候选路径）
  2. DEEPSEEK_API_KEY 读到没有
  3. 模型名是不是 DeepSeek 真实存在的（默认值 deepseek-flash 是错的）
  4. 真实调一次 API（消耗几个 token，验证 key 有效）
"""
import os
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent


def find_env():
    """与 agent.py._load_env_explicit 相同的候选路径。"""
    cands = [
        MODULE_DIR / ".env",
        Path.cwd() / ".env",
        Path.cwd().parent / "手语识别系统" / ".env",
        MODULE_DIR.parent / ".env",
    ]
    for p in cands:
        if p.is_file():
            return p
    return None


def parse_env(path):
    kv = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        kv[k.strip()] = v.strip().strip('"').strip("'")
    return kv


def main():
    print("=" * 60)
    print("DeepSeek 接入自检")
    print("=" * 60)

    env_path = find_env()
    if env_path is None:
        print("[X] 没找到 .env。已按 agent.py 的规则找过这四处：")
        print("     手语识别系统/.env、当前目录/.env、")
        print("     backend同级的 手语识别系统/.env、项目根/.env")
        print("  → 在 app.py 同级目录建一个名为 .env 的文件（别存成 .env.txt）")
        sys.exit(1)
    print(f"[OK] .env 位置: {env_path}")

    kv = parse_env(env_path)
    merged = dict(kv)
    merged.update({k: v for k, v in os.environ.items() if k in kv or k.startswith("DEEPSEEK")})
    key = merged.get("DEEPSEEK_API_KEY", "")
    model = merged.get("DEEPSEEK_MODEL", "deepseek-flash")  # 与 agent.py 默认一致
    base = merged.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    use_llm = merged.get("USE_REAL_LLM_API", "1") == "1"

    if not key:
        print("[X] DEEPSEEK_API_KEY 为空 —— 这就是走本地规则兜底的原因。")
        print("    .env 里写：DEEPSEEK_API_KEY=sk-xxxx")
        sys.exit(1)
    print(f"[OK] key 已读到（{key[:6]}...{key[-4:]}，长度 {len(key)}）")

    if not use_llm:
        print("[!] USE_REAL_LLM_API=0：LLM 被手动关闭，会走本地规则。演示前改回 1。")
        sys.exit(1)

    if model in ("deepseek-flash",):
        print(f"[X] 模型名 '{model}' 是 agent.py 的默认值，DeepSeek 官方没有这个模型！")
        print("    .env 里加一行：DEEPSEEK_MODEL=deepseek-chat")
        sys.exit(1)
    if model not in ("deepseek-chat", "deepseek-reasoner"):
        print(f"[!] 模型名 '{model}' 不是官方标准名（deepseek-chat / deepseek-reasoner），")
        print("    可能报 Model Not Exist，建议改成 deepseek-chat。")

    print(f"[OK] 模型: {model}   BASE_URL: {base}")
    print("-" * 60)
    print("正在真实调用一次 API（约 1~3 秒）...")

    try:
        import requests
    except ImportError:
        print("[!] 本机没装 requests，跳过实测。请在项目环境里跑：")
        print("    python -m pip install requests")
        sys.exit(1)

    try:
        r = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "只回复两个字：通了"}],
                "max_tokens": 8,
                "temperature": 0,
            },
            timeout=20,
        )
    except Exception as e:
        print(f"[X] 网络请求失败: {e}")
        print("    → 检查网络/代理；现场网络不行就先用 USE_REAL_LLM_API=0 兜底演示。")
        sys.exit(1)

    if r.status_code == 200:
        msg = r.json()["choices"][0]["message"]["content"]
        print(f"[OK] API 通了！模型回复：{msg}")
        print("=" * 60)
        print("全部通过。重启 app.py 后 /api/health 应显示")
        print('  "api_key_present": true,  "model": "%s"' % model)
        print("=" * 60)
        return
    if r.status_code == 401:
        print("[X] 401 未授权：key 无效或被停用。去 DeepSeek 平台重新生成。")
    elif r.status_code == 402:
        print("[X] 402 余额不足：去 DeepSeek 平台充值。")
    elif r.status_code == 400:
        print(f"[X] 400 参数错误（多半是模型名不对）：{r.text[:200]}")
        print("    改成 DEEPSEEK_MODEL=deepseek-chat")
    else:
        print(f"[X] HTTP {r.status_code}: {r.text[:200]}")
    sys.exit(1)


if __name__ == "__main__":
    main()
