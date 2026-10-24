"""
================================================================================
模块名称：test_agent.py（Agent 解译层回归测试）
负责人：技术 B 组 - 古宇轩 / 李东瑞（测试）

用途：
    每次改完 system prompt 或词表，跑一遍这个脚本，用同一批用例对比效果。
    没有它，调提示词就是玄学；有了它，"改得更好还是更差"是可验证的。

用法：
    python test_agent.py             # 真实调用 DeepSeek（约 12 次请求，花费几分钱）
    python test_agent.py --offline   # 强制只走本地规则引擎（断网/没额度时）
    python test_agent.py --save      # 结果另存为 tests/agent_cases_result.md

判分规则（每题三条）：
    1. translation 非空，且长度合理
    2. translation 包含 expect_in 里的**至少一个**关键词（保证还原正确）
    3. translation / answer 不包含 forbid 里的词（防止编造业务事项）
================================================================================
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

if "--offline" in sys.argv:
    os.environ["USE_REAL_LLM_API"] = "0"

import agent  # noqa: E402  （必须在设置环境变量之后导入）


# ------------------------------------------------------------------------------
# 用例表：想加场景就往这里加一行
#   tokens     : YOLO 输出的拼音序列（故意混入重复/乱序/误识别）
#   expect_in  : translation 里应该出现的词（命中任意一个即算通过）
#   forbid     : 不该出现的词（编造的业务事项）
#   note       : 这道题在考什么
# ------------------------------------------------------------------------------
CASES = [
    {
        "note": "标准场景",
        "tokens": ["nihao", "shenfenzheng", "diushi", "buban"],
        "expect_in": ["身份证", "补办"],
        "forbid": ["预约", "银行卡"],
    },
    {
        "note": "带「确认」手势（应触发成句且不入语义）",
        "tokens": ["nihao", "shenfenzheng", "diushi", "buban", "queren"],
        "expect_in": ["身份证"],
        "forbid": ["确认"],
    },
    {
        "note": "语序倒置（手语话题优先）",
        "tokens": ["buban", "shenfenzheng"],
        "expect_in": ["身份证", "补办"],
        "forbid": ["社保", "医保"],
    },
    {
        "note": "同一手势重复识别",
        "tokens": ["nihao", "nihao", "diushi", "buban", "buban", "buban"],
        "expect_in": ["丢失", "补办"],
        "forbid": [],
    },
    {
        "note": "社保卡激活问窗口",
        "tokens": ["qingwen", "shebaoka", "jihuo", "chuangkou", "nali"],
        "expect_in": ["社保卡", "窗口"],
        "forbid": ["身份证补办"],
    },
    {
        "note": "混入无关词 tiyuguan",
        "tokens": ["qingwen", "shebaoka", "jihuo", "tiyuguan"],
        "expect_in": ["社保卡"],
        "forbid": ["体育馆"],
    },
    {
        "note": "营业执照材料咨询",
        "tokens": ["banli", "yingyezhizhao", "xuyao", "shenme", "cailiao"],
        "expect_in": ["营业执照", "材料"],
        "forbid": ["身份证补办"],
    },
    {
        "note": "缩写别名 sfz / sbk",
        "tokens": ["sfz", "diushi"],
        "expect_in": ["身份证"],
        "forbid": [],
    },
    {
        "note": "带声调 + 分隔符写法",
        "tokens": ["Shēn Fèn Zhèng", "diu_shi", "buban"],
        "expect_in": ["身份证"],
        "forbid": [],
    },
    {
        "note": "单词（无法成句也要给出合理回应）",
        "tokens": ["xiexie"],
        "expect_in": ["谢谢"],
        "forbid": [],
    },
    {
        "note": "全是词表外拼音（应优雅降级）",
        "tokens": ["abc", "xyz"],
        "expect_in": [],
        "forbid": [],
    },
    {
        "note": "空输入（不崩）",
        "tokens": [],
        "expect_in": [],
        "forbid": [],
    },
]


def judge(case: dict, result: dict) -> tuple[bool, list[str]]:
    """返回 (是否通过, 失败原因列表)"""
    reasons = []
    tr = result["translation"]
    ans = result["answer"]

    if not case["tokens"]:
        # 空输入只要不抛异常就算通过
        return True, []

    if not tr:
        reasons.append("translation 为空")

    if case["expect_in"] and not any(w in tr for w in case["expect_in"]):
        reasons.append(f"未命中关键词 {case['expect_in']}")

    full = tr + ans
    hit_forbid = [w for w in case["forbid"] if w in full]
    if hit_forbid:
        reasons.append(f"出现不该有的词 {hit_forbid}")

    if tr and len(tr) < 4:
        reasons.append("translation 过短")

    return (len(reasons) == 0), reasons


def main() -> int:
    print("=" * 78)
    print(f"Agent 解译层回归测试 | 模型: {agent.DEEPSEEK_MODEL} | "
          f"LLM 可用: {agent.llm_available()}")
    print("=" * 78)

    passed = 0
    total = 0
    lines = ["# Agent 解译层回归测试结果", "",
             f"- 模型：`{agent.DEEPSEEK_MODEL}`",
             f"- LLM 可用：`{agent.llm_available()}`", "",
             "| # | 考察点 | 输入拼音 | 解码结果 | 翻译句 | 来源 | 结果 |",
             "| :-: | :-- | :-- | :-- | :-- | :-: | :-: |"]

    for i, case in enumerate(CASES, 1):
        result = agent.parse_semantics_detail(case["tokens"])
        ok, reasons = judge(case, result)
        if case["tokens"]:
            total += 1
            passed += ok

        mark = "✅" if ok else "❌"
        print(f"\n[{i:02d}] {case['note']}  {mark}")
        print(f"     输入: {' '.join(case['tokens']) if case['tokens'] else '（空）'}")
        print(f"     解码: {result['words']}")
        print(f"     翻译: {result['translation']}")
        print(f"     导办: {result['answer']}")
        print(f"     来源: {result['source']} | 置信度: {result['confidence']:.2f}")
        if result["unknown_pinyin"]:
            print(f"     词表外: {result['unknown_pinyin']}")
        if result["error"]:
            print(f"     ⚠️ 错误: {result['error']}")
        if reasons:
            print(f"     ❌ 原因: {'；'.join(reasons)}")

        lines.append(
            f"| {i} | {case['note']} | `{' '.join(case['tokens']) or '（空）'}` "
            f"| {'、'.join(result['words']) or '—'} "
            f"| {result['translation'] or '—'} | {result['source']} | {mark} |"
        )

    print("\n" + "=" * 78)
    print(f"通过 {passed}/{total}（另有 1 条空输入用例只验证“不崩溃”）")
    print("=" * 78)

    lines.extend(["", f"**通过：{passed}/{total}**", ""])

    if "--save" in sys.argv:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "agent_cases_result.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"结果已保存到 {os.path.normpath(path)}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
