# VitalsSight product console QA report

Verification date: 2026-07-15

## Verified product workflow

- Consent, research purpose, and raw-video retention are explicit before analysis.
- Video qualification precedes the HR pipeline and returns actionable duration, frame-rate, resolution, illumination, motion, and face-visibility guidance.
- The output contract has three states: `release`, `review`, and `retake`. Non-release states cannot publish `released_hr_bpm`.
- The case registry preserves quality components, candidate branches, trends, model and policy versions, recommended action, provenance, and decision evidence.
- The review queue supports filters, priority, assignment, notes, resolution, state transitions, and timestamped audit events.
- Role-based tutorials define the input, action, output, and next destination for capture operators, evidence reviewers, and report/integration users.
- Reports export PDF, JSON, Markdown, and CSV and preserve the recommendation basis, verification criteria, attribution boundary, review record, audit trail, and implementation provenance.
- The REST API shares the SQLite evidence store and output contract with the UI. Multipart uploads are deleted after processing in the tested delete-after-analysis mode.
- Chinese and English interfaces share the same case state and numeric values.

## Automated checks

The full suite contains 46 passing tests. It covers strict non-release HR withholding, finite release values, candidate-track aggregation, evidence attribution, quality-gate semantics, SQLite persistence, bilingual reports, preflight and runtime failures, API endpoints, raw-video deletion, uploader reset, sidebar recovery, navigation behavior, runtime-model installation and provenance, explicit rejection of non-converged ICA routes, and the rule that static-ROI fallback evidence can never enter a release state. Dedicated regressions confirm that a preflight `retake` report does not mislabel an unentered candidate stage or a passing luma check as a failure, that a failed ICA route is not relabelled as an independent method by silently returning GREEN, and that an unpinned Face Landmarker asset fails runtime integrity validation.

```bash
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m compileall -q app src scripts tests
git diff --check
```

Observed result: `46 passed`; compile and whitespace checks passed.

## Real-video implementation conformance

Seven hash-locked MCD-rPPG fixtures form a post hoc curated regression/conformance replay: one expected release, five expected reviews, and one expected retake. The cases were selected during development, while expected states and fixture hashes were fixed in the manifest before the final replay. Each fixture was processed twice through the direct backend and once through the API. The output bundle separates the seven fixture summaries from 21 execution-level rows so every direct repeat and API run can be audited independently. All executions matched the expected state, all non-release outputs withheld HR, and temporary upload directories were empty after processing.

The released fixture produced 75.0628 BPM. A reference-only Lead-II ECG comparison produced 77.1208 BPM, an absolute difference of 2.0581 BPM. ECG/reference HR was not passed to candidate construction, selection, or gating. The fixtures were curated for behavioral conformance, not sampled as an independent accuracy cohort.

Evidence: `output/real_video_product_validation_20260715_iteration83/`. The v2 fixture contract preserves the frozen state expectation and HR-withholding rule while defining the 9825 review invariant at the mechanism level (`cross_window_candidate_track`), because both no-stable-track and competing-track subtypes require the same non-release action.

## Real-browser checks

Playwright exercised the final service at 1440 x 1000 and in a separate fresh 390 x 844 mobile context. The committed browser assertions verified:

- real-video release, review, and retake outcomes against their expected states;
- consent warnings, run feedback, clear/reset behavior, and raw-upload deletion;
- all eight workspaces, the role-based guide content, and its complete start action;
- PDF, JSON, Markdown, CSV, and OpenAPI downloads;
- report detail, evidence-to-action, attribution, review/audit, and structured-data tabs;
- review assignment, note, resolution, save feedback, and `review.updated` persistence;
- integration audit persistence in the shared case store, operator save, and non-destructive demo restoration;
- sidebar restoration on desktop and in a fresh mobile session, plus automatic mobile close after navigation;
- no page-level horizontal overflow at 390 px and zero browser-console errors.

The harness refuses to run against a dirty working tree or a commit argument that differs from the checked-out Git commit. The manifest records both the commit and Git tree, so the browser evidence is source-bound rather than a free-form session label.

The final desktop release case reported 75.1 BPM and an acquisition-gate state of Passed. The final review and retake cases withheld HR. The browser console, page-error stream and unexpected HTTP-response error list were empty. Screenshots, DOM text snapshots, downloads, database and logs are retained under `output/browser_validation_20260715_iteration84/`.

## Defects repaired

1. AVI decoding now falls back when `CAP_PROP_FRAME_COUNT` is zero.
2. Missing face-landmarker assets no longer silently produce zero detections.
3. Direct and API paths use the same model-backed preflight and record provenance.
4. Candidate aggregation uses coherent cross-window tracks and withholds HR for unresolved competing tracks.
5. Unavailable harmonic evidence is neutral rather than treated as a negative signal.
6. Clear and Start new assessment rebuild the upload widget as well as deleting the temporary file.
7. Top-level summaries separate acquisition-gate state from component quality scores.
8. Mobile workspace navigation now closes the sidebar after selection.
9. Preflight-retake reports now use the original acquisition checks, identify candidate construction as not entered, and recommend correction only for failed or warning checks.
10. The default runtime resolves a pinned MediaPipe Face Landmarker asset, enforces its SHA256 before initialization, records backend and integrity provenance, and reports an explicit static-ROI fallback instead of silently changing the candidate pool. Static-ROI evidence is never release eligible.
11. Initial mobile rendering leaves the sidebar interactive; only a completed workspace navigation triggers a one-shot close, so the restore control remains usable.
12. Mixed numeric/text report columns are normalized before DataFrame rendering, eliminating Arrow conversion failures without changing evidence values.
13. The navigation helper uses the current `st.iframe` interface rather than the deprecated `st.components.v1.html` path.
14. FastICA convergence warnings are promoted to route failures; an unconverged or failed ICA output is omitted rather than being counted or replaced by a GREEN signal under the ICA label. Route name, region, window, error class, and omission count remain in runtime evidence.
15. Exported evidence reports expose only detector asset names and hashes, not absolute workstation paths.
16. Browser QA now uses a repository-local pinned Playwright dependency, a clean-tree/commit gate, and a separate fresh mobile browser context.

## Product boundary

This verification establishes finite retrospective research-workflow behavior. It does not establish clinical utility, prospective accuracy, calibrated safety, emergency-alert performance, autonomous clinical release, security certification, EHR certification, end-to-end real-time performance, or production readiness. Those require separate prospective studies, product engineering, governance, and regulatory work.
