"""
================================================================================
模块名称：agent.py（Agent 智能体语义理解与政务导办问答模块）
负责人：技术 B 组 - 古宇轩（智能体逻辑负责人）
对接说明：
  本文件已被 `app.py` 自动检测并挂载。
  后期接入大模型 API（如 DeepSeek / Qwen / GLM 等）时，只需将 `USE_REAL_LLM_API` 改为 `True`，
  并在 `parse_semantics(word_list)` 中填入 API 调用逻辑，返回 `(中文翻译句, 政务导办回答)` 即可，
  完全不需要修改 `app.py` 的界面代码！
================================================================================
"""

# ------------------------------------------------------------------------------
# 🔧 技术 B 组大模型切换开关：配置好 API Key 后可改为 True
# ------------------------------------------------------------------------------
USE_REAL_LLM_API = False


def parse_semantics(word_list):
    """
    接收前端传来的静态手语词列表，输出连词成句后的中文翻译与政务 AI 导办建议。

    参数:
        word_list (list[str]): 经过置信度过滤后的静态手语词序列，例如 ["你好", "身份证", "丢失", "补办"]

    返回:
        tuple[str, str]: (translation_text, ai_gov_answer)
            - translation_text: 手语原意的大字号中文自然语句翻译（第一人称）
            - ai_gov_answer: 政务服务智能体的导办指引回答
    """
    if not word_list:
        return "", ""

    joined_words = "、".join(word_list)

    if USE_REAL_LLM_API:
        # ======================================================================
        # TODO: 古宇轩在此处接入真实大模型 API (DeepSeek / Qwen / ZhipuAI 等)
        # ======================================================================
        # try:
        #     from openai import OpenAI
        #     client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.deepseek.com/v1")
        #     ...
        # except Exception as e:
        #     print(f"LLM API 调用异常，回退至本地规则引擎: {e}")
        pass

    words_set = set(word_list)

    # 政务场景规则/大模型占位逻辑 1：身份证挂失与补办
    if ("身份证" in words_set or "补办" in words_set or "丢失" in words_set) and len(word_list) >= 3:
        translation = "工作人员您好，我的身份证丢失了，请问怎么办理挂失和补办？"
        ai_answer = (
            "【政务导办回答】您好！补办身份证请携带户口本或驾驶证原件，"
            "先到右侧 2 号人像采集室免费拍照，随后前往 3 号户政窗口办理，工本费 40 元。"
        )
        return translation, ai_answer

    # 政务场景规则/大模型占位逻辑 2：社保卡激活与医保窗口咨询
    if ("社保卡" in words_set or "医保" in words_set or "窗口" in words_set) and len(word_list) >= 3:
        translation = "请问社保卡激活和医保查询业务在哪个窗口办理？"
        ai_answer = (
            "【政务导办回答】请前往大厅东侧 B 区 6～8 号「社保医保综合窗口」，"
            "无需预约，凭身份证在自助取号机取号即可办理。"
        )
        return translation, ai_answer

    # 政务场景规则/大模型占位逻辑 3：个体工商户营业执照材料咨询
    if ("营业执照" in words_set or "材料" in words_set) and len(word_list) >= 3:
        translation = "我想办理个体工商户营业执照，请问需要准备什么材料？"
        ai_answer = (
            "【政务导办回答】请准备：① 经营者身份证原件；② 经营场所租赁合同或产权证明。"
            "材料齐全可在 1 号商事登记窗口当场拿证。"
        )
        return translation, ai_answer

    # 动态词序列兜底合成（当用户点击【⬅️ 撤回上一词】或自定义词组合时动态响应）
    translation = f"您好，我想咨询并办理关于「{joined_words}」的政务服务事项。"
    ai_answer = (
        f"【政务导办回答】已识别到您的核心诉求关键词（{joined_words}），"
        "请携带本人有效身份证件前往一号综合导办台取号办理。"
    )
    return translation, ai_answer
