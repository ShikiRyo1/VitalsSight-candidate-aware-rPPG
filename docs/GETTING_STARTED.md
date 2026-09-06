# First 10 minutes with VitalsSight

This guide is for the source-based research console, not the archived native macOS binary. The September 2026 source update does not rebuild that binary.

## 1. Start a local research workspace

Use Python 3.10–3.12. From the repository folder on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.\.venv\Scripts\python.exe scripts/run_vitalssight_local.py
```

These commands do not require activating a PowerShell environment. The launcher starts a local UI and API, selects available loopback ports, and opens the browser. It uses a development identity, not a multi-user login. Do not expose this mode to the internet or use it as a controlled-trial deployment.

An initial demonstration does not require a participant video, model download or an AI provider. For authorized raw-video work, install the pinned face-landmark asset separately:

```powershell
.\.venv\Scripts\python.exe scripts/setup_runtime_assets.py
```

This downloads an official third-party runtime asset and verifies its checksum; it does not download the trained manuscript selector or a participant dataset.

## 2. Learn the workflow using labeled examples

1. Open **Overview** and follow the getting-started actions. Switch between English and Chinese in the sidebar.
2. Open **New assessment** and select a labeled demonstration. Demonstration cases are software-workflow fixtures, not new video measurements or evidence of accuracy.
3. Inspect the resulting **release / review / retake** state and its evidence. A research `release` is not clinical authorization.
4. Use **Cases** or **Review queue** to inspect the reason, policy threshold and recommended next action. Only authorized roles can change review state.
5. Open **Reports** to inspect the linked evidence and available exports. Check the source and status before sharing a report.

Changing the assessment purpose, consent, source or retention setting invalidates the previous session result. Run a new assessment after changing inputs; do not treat the old result as belonging to the new input.

## 3. Prepare an authorized video

- Open the runtime readiness panel before uploading. It explains missing dependencies, model-asset status, storage configuration and upload limits with actionable next steps.
- Readiness is a read-only setup check. It does not execute video inference, grant consent or establish clinical validity.
- Use only recordings you are permitted to process for the stated research purpose. In required-auth mode, select a pseudonymous participant with an active, versioned consent record for that purpose.
- Select **Upload video** and review the file size and supported extensions. An extension alone does not prove a file is a valid, usable video; backend decoding and quality checks remain authoritative.
- Keep the recommended **delete after analysis** setting. Derived evidence and audit records may persist even when the raw upload is deleted.
- If quality requires retake, follow the recorded reason and acquire a new authorized recording. If evidence requires review, keep the HR withheld and inspect the case instead of overriding the gate.

Missing or invalid face-landmark assets can trigger an explicitly labeled static-ROI fallback. That fallback is not release eligible. Passing installation checks does not change this rule.

## 4. Keep three kinds of evidence separate

| What you are running | What it demonstrates | What it does not demonstrate |
|---|---|---|
| Labeled console examples / dataset-free candidate demo | UI, contract routing, reports and review workflow | Measured video accuracy |
| Authorized raw-video console path | The implemented quality-first classical bridge and its output contract | The complete supervised selector's manuscript result |
| Manuscript training / evaluation scripts with authorized artifacts | The specified experiment, if its required data, splits and checkpoints are supplied | Clinical validity or independent prospective utility |

For the current source-update checks, see [release notes](RELEASE_NOTES_20260906.md). For the remaining reproduction requirements, see [reproducibility status](REPRODUCIBILITY.md). For identity and consent configuration, see [controlled-trial operations](CONTROLLED_TRIAL_OPERATIONS.md).

## 中文速览

先启动本地程序，用明确标注的样例熟悉“输入 → 质量检查 → 状态与证据 → 复核 → 报告”。第一次体验不需要上传真人视频，也不需要下载大语言模型。处理自己的研究视频前，先看运行环境检查，再核对用途、授权和删除设置。

“样例能跑通”“环境检查通过”“研究输出为 release”和“临床有效”是不同的事情。当前软件不用于诊断、急救或自动临床放行；不要把样例结果或论文内部评估当作当前上传视频的精度保证。
