# Agent 解译层回归测试结果

- 模型：`deepseek-flash`
- LLM 可用：`True`

| # | 考察点 | 输入拼音 | 解码结果 | 翻译句 | 来源 | 结果 |
| :-: | :-- | :-- | :-- | :-- | :-: | :-: |
| 1 | 标准场景 | `nihao shenfenzheng diushi buban` | 你好、身份证、丢失、补办 | 您好，我的身份证丢失了，请问要怎么补办？ | llm | ✅ |
| 2 | 带「确认」手势（应触发成句且不入语义） | `nihao shenfenzheng diushi buban queren` | 你好、身份证、丢失、补办 | 您好，我的身份证丢失了，请问要怎么补办？ | llm | ✅ |
| 3 | 语序倒置（手语话题优先） | `buban shenfenzheng` | 身份证、补办 | 您好，我的身份证要补办，请问需要什么材料、在哪个窗口办理？ | llm | ✅ |
| 4 | 同一手势重复识别 | `nihao nihao diushi buban buban buban` | 你好、丢失、补办 | 您好，我的证件丢失了，想要补办，请问该去哪个窗口办理？ | llm | ✅ |
| 5 | 社保卡激活问窗口 | `qingwen shebaoka jihuo chuangkou nali` | 请问、社保卡、激活、窗口、哪里 | 请问社保卡激活在哪个窗口办理？ | llm | ✅ |
| 6 | 混入无关词 tiyuguan | `qingwen shebaoka jihuo tiyuguan` | 请问、社保卡、激活 | 请问社保卡激活在哪里办理？ | llm | ✅ |
| 7 | 营业执照材料咨询 | `banli yingyezhizhao xuyao shenme cailiao` | 办理、营业执照、需要、什么、材料 | 您好，我想办理营业执照，请问需要准备什么材料？ | llm | ✅ |
| 8 | 缩写别名 sfz / sbk | `sfz diushi` | 身份证、丢失 | 您好，我的身份证丢失了，请问该怎么补办？ | llm | ✅ |
| 9 | 带声调 + 分隔符写法 | `Shēn Fèn Zhèng diu_shi buban` | 你好、身份证、丢失、补办 | 您好，我的身份证丢失了，请问要怎么补办？ | llm | ✅ |
| 10 | 单词（无法成句也要给出合理回应） | `xiexie` | 谢谢 | 谢谢您！ | llm | ✅ |
| 11 | 全是词表外拼音（应优雅降级） | `abc xyz` | — | 您好，我没能完整表达，请再问我一次。 | llm | ✅ |
| 12 | 空输入（不崩） | `（空）` | — | — | rule | ✅ |

**通过：11/11**
