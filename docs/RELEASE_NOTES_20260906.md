# September 2026 research-console, presentation and integrity update

This is a source-based research-console, documentation, communication and public-package verification update. It is not a new trained-model release, a rerun of the scientific experiments, or a clinical-validation claim. Previously published native macOS Release assets have not been replaced; push-triggered CI build artifacts are separate.

## Research-console improvements

- Bilingual first-run guidance, a direct research-video entry point and clearer input / output / next-action instructions.
- Separate synthetic-demo and uploaded/other evidence views; overview counts, queues, tables and charts follow the chosen scope. Demonstrations are not presented as participant measurements or paper results.
- A shared, standard-library-only local readiness service and `GET /api/v1/runtime/readiness`. The API retains existing read roles and organization-scoped access logging. Its checks do not initialize inference, contact AI providers, inspect participant data or expose configured private paths and secrets.
- Actionable diagnostics for Python/dependency discovery, advisory storage access, upload settings, identity configuration and the pinned landmark asset. Tasks and legacy MediaPipe backends have distinct asset requirements; an unknown backend is explicitly advisory, not represented as tested.
- Shared UI/API upload metadata checks for supported extensions, empty files and size limits. An invalid byte-limit configuration disables intake with HTTP 503 rather than crashing API initialization. Decoding, consent, role and evidence gates remain authoritative.
- Assessment previews are cleared when source, purpose, retention, organization, participant or consent identity/version/status changes. Saved cases are not deleted by preview invalidation. Session-only raw-upload retention is accurately reflected in the assessment audit event.

Follow the [first 10 minutes guide](GETTING_STARTED.md) to try the source console without participant data.

## Manuscript and status

- The authors report that the manuscript was accepted for publication in *npj Cardiovascular Health*. The exact acceptance date, online publication date and DOI were not independently verified here. The archived author manuscript is not represented as the publisher's version of record.
- The 26-page PDF re-supplied on 6 September is byte-identical to the archived PDF: SHA-256 `B100AE3F126AA50CA37C34486D8F84BCD39DEB45E11C4CBDB263386C1CA129A6`.
- The legacy filename and all reported numerical results are retained. The `NPJ_DM` filename is not a journal-status field.

## Presentation

- The front page now points to a communication schematic adapted from manuscript Figure 1, with separate candidate selection, evidence, research score and proposed output state. It is not a quantitative result.
- The V50 image and editable source remain archived instead of being overwritten.
- The primary mechanism description no longer conflates the per-window set-attention selector with the separate historical V32 causal-path branch.
- The first-run documentation distinguishes a hand-written synthetic contract demo from raw-video inference and supervised-model evaluation.

## Public-package integrity

The historical V31/V32 package originally recorded CRLF hashes for 17 aggregate CSV/JSON files. Git attributes normalize those files to LF, so its original raw-byte verifier failed on a clean checkout. The original manifest and all historical aggregate files are retained unchanged. A separate public-LF checksum manifest now verifies exact committed bytes, and an explicit lineage check confirms that historical aggregate differences are only CRLF-to-LF conversion. Tests reject missing files, byte changes and genuine content changes rather than silently normalizing them away.

Within the historical package, only the verifier and new verification tests/documentation were changed. Scientific training, inference, analysis code, expected metrics and historical contracts were not changed. See [checksum provenance](../reproducibility/v32_submission/CHECKSUM_PROVENANCE.md).

## Current verification (6 September 2026)

- Windows, Python 3.12.14, isolated virtual environment: `python -m pytest -q` — **184 passed** in 24.95 seconds, with two upstream Starlette/httpx/AnyIO deprecation warnings. Includes 25 console helper / Streamlit AppTest cases, readiness and API regression tests, and existing identity/consent/organization checks.
- After installing and verifying the pinned Face Landmarker Tasks asset, the full Windows suite was rerun: **184 passed** in 30.85 seconds, two dependency warnings. The existing synthetic dark-video assistant workflow also passed independently with native Tasks initialization. Synthetic video exercises software paths, not participant accuracy.
- Public historical package verifier — **PASS**: 47 exact public-file hashes, 17 CRLF-to-LF aggregate lineage checks, 14 Python sources compiled, plus the recorded contract and aggregate-metric checks. This verifies artifacts, not experiment recomputation.
- Dataset-free synthetic candidate-release example — **PASS**; no trained selector or participant accuracy claim.
- Existing project-page browser harness — **PASS** at 1440 × 900 desktop and 390 × 844 mobile: image loading, internal anchors, PDF resource, horizontal layout, mobile menu and copy control.
- Actual local console browser checks: separate empty uploaded-evidence view, direct upload entry, consent rejection, stable synthetic release, clearing the old preview on source change, conflict synthetic review with HR withheld, and navigation to linked report exports.
- Local HTTP readiness endpoint responded successfully. The result explicitly reports `runtime_execution_verified=false`, `measurement_release_authorized=false` and `clinical_validity_established=false`; advisory warnings are not a raw-video execution pass.

The console browser emitted no JavaScript errors in these exercised paths. Browser feature-policy and iframe-sandbox warnings from existing components were observed; this is not a zero-warning, full accessibility or hardened-hosting audit. These finite checks do not rerun private real-video fixtures, optional local-AI/multimodal models, macOS packaging, the full trained deep pipeline or prospective clinical evaluation.

## macOS dependency regression and mitigation

The first update's public-contract CI and Pages deployment succeeded. Its [macOS source smoke job](https://github.com/ShikiRyo1/VitalsSight-candidate-aware-rPPG/actions/runs/34046071890/job/101521224399) started the launcher but aborted during an existing synthetic dark-video test inside MediaPipe 1.0.1 native Face Landmarker initialization (exit 134). The exception was not a Python assertion failure. This does not establish an exact native root cause.

A [new upstream report](https://github.com/google-ai-edge/mediapipe/issues/6356) describes a similar macOS arm64 Tasks abort with 1.0.1. It is supporting evidence, not maintainer-confirmed equivalence to this CI failure. The repository's [previously successful macOS source workflow](https://github.com/ShikiRyo1/VitalsSight-candidate-aware-rPPG/actions/runs/29889529218) used 0.10.35. The dependency is therefore narrowly pinned to that version on **macOS arm64 only**; Intel macOS and other platforms keep their existing requirement. This boundary matters because 0.10.35 has no published Intel macOS wheel. The arm64 wheel retains the Tasks Face Landmarker path rather than switching this wrapper to legacy FaceMesh. Inference code, gates and tests are not disabled or changed. A fresh macOS CI run is required to verify this mitigation; the first failure must not be represented as a platform pass.

## Continuing boundaries

- No participant dataset, identifiable source video, checkpoint or private execution ledger is added.
- Full primary-result reproduction still requires authorized data, frozen splits, intermediate predictions and checkpoints.
- Historical validation counts describe their recorded runs. Current checks must be reported with their own environment and outcome.
- The release/review research score is not a calibrated safety probability. Clinical safety, beneficial abstention, clinical utility and end-to-end real-time performance remain unestablished.
