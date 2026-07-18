# VitalsSight unified assistant QA - 2026-07-18

## Scope

This acceptance pass covers the single-interface assistant workflow for typed questions, locally transcribed speech, privacy-normalized images, and consented video assessment. The video path invokes the existing deterministic quality, candidate, selector, gate, evidence-store, action-plan, and report services. The language model explains the resulting evidence but cannot estimate or change HR, override the output state, bypass consent, or retain raw media.

## Automated evidence

| Layer | Result | Evidence |
|---|---:|---|
| Python unit and API suite | 129/129 passed | `pytest -q` |
| Full product browser matrix | 171/171 passed | `output/unified_assistant_acceptance_20260718/browser_validation_manifest.json` |
| Real multimodal browser workflow | 36/36 passed | `output/unified_assistant_acceptance_20260718/browser_final` |
| Focused real-video assistant workflow | 10/10 passed | `output/unified_assistant_acceptance_20260718/video_focused_final` |
| Deterministic and API real-video parity | 14/14 passed | `output/unified_assistant_acceptance_20260718/real_video/VALIDATION_REPORT.md` |

The browser workflow uses a real local speech recording, an authorized non-identifying frame, and a real validation video. It checks the direct voice-to-answer path, image-to-answer path, consented video processing, inline release/review/retake output, all report downloads, raw-media cleanup, responsive layout, and browser/API error logs.

## Interaction contract

1. A clear speech transcript enters read-only assistant chat directly. An uncertain transcript remains editable and requires confirmation.
2. Image analysis is privacy-normalized and marked non-authoritative before it reaches the assistant.
3. Video processing requires explicit consent and `delete_after_analysis`; the deterministic pipeline selects the output state.
4. The same workspace returns the state, evidence explanation, next action, governed report, and available exports.
5. A review update can only be prepared by an authorized role and remains inert until a separate explicit confirmation.

## Safety assertions

- Raw audio, image, and video are not retained after processing.
- The LLM has no direct filesystem, database, raw-video, or signal-estimation tool.
- `review` and `retake` answers do not publish a BPM value.
- Media-derived text cannot ground identity, diagnosis, treatment, or output-state claims.
- Reports created by the unified workspace are marked `unversioned` until reviewer approval.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q
npm run validate:browser -- http://127.0.0.1:8502 http://127.0.0.1:8011 runtime\private_validation output\unified_assistant_acceptance_20260718 <git-commit>
npm run validate:multimodal -- http://127.0.0.1:8502 http://127.0.0.1:8011 output\controlled_trial_final_20260717\multimodal\real_authorized_frame.png output\unified_assistant_acceptance_20260718\browser_final output\real_data_full_acceptance_20260717\multimodal\review_question.wav runtime\private_validation\8555_retake_first5s.avi
npm run validate:unified-video -- http://127.0.0.1:8502 runtime\private_validation\8555_retake_first5s.avi output\unified_assistant_acceptance_20260718\video_focused_final
```
