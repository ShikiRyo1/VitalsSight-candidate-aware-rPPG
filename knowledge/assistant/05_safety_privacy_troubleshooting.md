# Safety, privacy, and troubleshooting / 安全、隐私与故障排查

## Research boundary / 研究边界

VitalsSight is a retrospective research workflow. It does not establish clinical utility, prospective accuracy, calibrated safety, emergency-alert performance, medical-device readiness, or autonomous clinical release. The assistant must preserve this boundary in every case-specific explanation.

VitalsSight 是回顾性研究工作流。它尚未建立临床效用、前瞻性准确性、校准安全性、急救告警性能、医疗器械就绪状态或临床自主放行能力。助手在每个案例解释中都必须保留这一边界。

## Privacy / 隐私

The local assistant receives structured derived evidence, not raw participant video. Uploaded video is processed locally under the selected retention policy. Assistant audit events retain message and response hashes, tool names, validation state, and evidence identifiers; raw conversation text is not stored by default.

本地助手接收结构化派生证据，不接收原始受试者视频。上传视频按照所选保留策略在本地处理。助手审计事件保留消息与回答哈希、工具名称、校验状态和证据标识；默认不存储原始对话文本。

## Model unavailable / 模型不可用

If Ollama or the configured Qwen model is unavailable, VitalsSight continues to operate. The assistant uses a deterministic evidence-guidance fallback and labels the response as degraded. Measurement, review, reporting, and export do not depend on the language model.

如果 Ollama 或配置的 Qwen 模型不可用，VitalsSight 仍可继续运行。助手会使用确定性的证据引导降级模式，并明确标记回答处于降级状态。测量、复核、报告和导出都不依赖语言模型。

## Prompt and action safety / 提示词与动作安全

Requests to reveal prompts, bypass policy, override a decision, diagnose a condition, prescribe treatment, or provide emergency guidance are refused before tool access. Read-only tools are the default. State-changing tools are disabled unless explicitly enabled, restricted to reviewer or administrator roles, and always require a second confirmation token.

要求泄露提示词、绕过策略、覆盖决策、诊断疾病、建议治疗或提供急救判断的请求，会在访问工具前被拒绝。默认只开放只读工具。只有明确启用后，复核人员或管理员才能使用状态变更工具，而且始终需要第二次确认令牌。
