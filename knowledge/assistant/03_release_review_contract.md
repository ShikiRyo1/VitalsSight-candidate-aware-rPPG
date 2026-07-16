# Release, review, and evidence contract / 放行、复核与证据契约

## Release / 放行

Release is the only state that may publish a finite `released_hr_bpm`. The value remains linked to the evidence packet, model version, policy version, candidate source identity, and audit trail. Release is not a diagnosis, a calibrated probability of safety, or an autonomous clinical decision.

放行是唯一允许发布有限 `released_hr_bpm` 的状态。该数值必须继续与证据包、模型版本、策略版本、候选来源身份和审计记录保持关联。放行不等于诊断、校准后的安全概率或临床自主决策。

## Review / 人工复核

Review means candidate evidence exists but an unresolved condition prevents automatic publication. Examples include cross-route disagreement, unstable estimates across windows, competing candidate tracks, harmonic ambiguity, or a fallback detector backend. HR remains withheld. The reviewer inspects the evidence, documents the resolution, or requests another recording.

人工复核表示系统已有候选证据，但仍存在未消解条件，不能自动发布结果。例如跨路径分歧、窗口间估计不稳定、竞争候选轨迹、谐波歧义或后备检测后端。心率保持不发布。复核人员检查证据、记录处理结论，或要求重新录制。

## Retake / 重采

Retake means the acquisition gate did not pass. HR remains withheld and the user receives an actionable correction tied to the failed or warning check. The recording should be repeated only after the specified acquisition issue is corrected.

重采表示采集门控未通过。心率保持不发布，用户会收到与真实失败项或警告项关联的纠正建议。只有完成指定采集问题的纠正后，才应重新录制。

## Non-negotiable invariants / 不可突破的规则

- A release must have a finite published HR.
- Review and retake must not publish HR.
- Reference HR and candidate absolute error are evaluation-only and never enter inference.
- Candidate values may be inspected as evidence but cannot be relabelled as a published result.
- The selected value and the proposed output state remain separately traceable.

- 放行必须具有有限的已发布心率。
- 复核和重采不得发布心率。
- 参考心率和候选绝对误差仅用于评估，不得进入推理。
- 候选值可以作为证据检查，但不能被重新描述为正式发布结果。
- 所选数值与建议输出状态必须保持独立可追溯。
