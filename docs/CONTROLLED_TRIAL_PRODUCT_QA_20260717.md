# VitalsSight controlled-trial product acceptance

Verification date: 2026-07-17

Release scope: responsive, single-instance research web product for a supervised controlled trial

Source binding: the final browser manifest records and verifies the exact Git commit, tree, clean-worktree state, service build identity and upload-root fingerprint used during acceptance

## Acceptance result

All finite acceptance suites described below passed in the tested local environment. The result establishes that the implemented research workflow can be operated end to end with the retained fixtures and local model services. It does not establish prospective clinical utility, diagnostic performance, autonomous clinical release, production security certification, EHR certification, regulatory clearance or public-hosting readiness.

| Acceptance surface | Result | Evidence |
|---|---:|---|
| Unit and API regression suite | 124 passed | `python -m pytest -q` |
| Desktop and mobile browser workflow | 171/171 checks | `output/controlled_trial_final_20260717/acceptance_final/browser_validation_manifest.json` |
| Real-video decision conformance | 21/21 executions | `output/controlled_trial_final_20260717/real_video/validation_results.json` |
| Real local-model assistant | 4/4 scenarios, 53/53 checks | `output/controlled_trial_final_20260717/real_assistant_v3/real_assistant_acceptance.json` |
| Real image and audio multimodal intake | 33/33 checks | `output/controlled_trial_final_20260717/multimodal/multimodal_validation.json` |
| SQLite backup integrity | `quick_check: ok` | `output/controlled_trial_final_20260717/backup/vitalssight-20260717T105223Z.json` |

The output directories are intentionally ignored by Git because they include local state, rendered screenshots and controlled research fixtures. The committed validation scripts, tests and this report define how those artifacts were produced. The final browser manifest is the authoritative record of the validated source commit.

## End-to-end product workflow

The accepted workflow covers participant registration, versioned consent, guided capture, acquisition qualification, candidate construction, relation-aware selection, release/review/retake gating, evidence-linked reporting, human review, immutable report versions, approval, FHIR export, longitudinal context and access auditing. Desktop and mobile checks also exercise bilingual navigation, tutorial entry points, sidebar collapse and restoration, visible command feedback, empty states, report tabs and downloads.

The three output states retain the following invariant:

- `release` contains a finite published HR linked to its evidence packet, policy identity and audit trail;
- `review` withholds HR and preserves the candidate conflict or other review reason;
- `retake` withholds HR and names only the acquisition checks that actually triggered corrective action.

The AI layer does not alter this state machine. Deterministic case, policy and report evidence remains authoritative. Model prose is schema constrained and post-validated for evidence identifiers, numerical grounding, state preservation, causal attribution and non-release HR withholding. A failed or invalid model answer falls back to deterministic guidance.

## Real-video validation

Seven hash-locked local research-video cases were exercised: one expected release, five expected reviews and one expected retake. Each case was processed twice through the direct backend and once through the FastAPI upload path, producing 21 passing executions. All observed states matched their frozen expectation and every non-release execution withheld HR.

The release case produced 75.0628 BPM. A reference-only ECG estimate was 77.1208 BPM, an absolute difference of 2.0581 BPM. The reference value was not available to candidate construction, selection or gating. This single case is a behavioral conformance fixture, not an independent accuracy cohort or a clinical-performance claim.

The review fixtures cover no stable cross-window track, competing cross-window tracks and borderline face visibility. The retake fixture covers insufficient usable duration before candidate construction. Raw API uploads were deleted after analysis.

## Local AI and multimodal validation

The real assistant suite used `qwen3.6:35b` through Ollama rather than a mocked response. Four scenarios passed: English release, English review, Chinese review and duration-only retake. All 53 scenario checks passed, including evidence citation, causal-driver restrictions, recorded-state preservation, numerical grounding and HR withholding. Observed response times were 74.033, 165.105, 72.593 and 94.344 seconds on the tested workstation; they are not an end-to-end real-time claim.

The multimodal suite used a frame extracted from an authorized local research video and a real recorded question. Vision ran through `qwen3-vl:4b-instruct`; speech transcription ran through `faster-whisper` `small`. All 33 API, desktop and mobile checks passed. The image was treated as non-authoritative workflow context, no vital-sign value was inferred from it, and raw image/audio bytes were not retained.

## Identity, tenancy and report governance

Automated coverage verifies required-mode OIDC token validation, explicit disabled-auth development mode, role checks, organization scope and participant ownership scope. Organization, user, membership, participant, consent, report-version and access-audit data are persisted using additive SQLite migrations; existing evidence is not deleted or downgraded when a source fixture is absent from a checkout.

Reports are audience specific and content hashed. Approvals are append-only events; a duplicate approval is rejected, and later corrections require a new version or addendum. FHIR output uses an HR `Observation` only for release. Review and retake use a `Task` and keep HR withheld. Patient pseudonym, device, diagnostic report, consent and provenance resources remain linked to the output contract.

## Browser and interaction coverage

The source-bound Playwright harness runs at 1440 x 1000 and 390 x 844. It covers real and built-in release/review/retake paths, participant and consent APIs plus UI, case registry, review assignment and resolution, report drafting and approval, FHIR output, organization membership and access audit, all workspaces and guided entry points. It asserts zero unexpected console errors, page errors, HTTP failures and horizontal overflow, and verifies that the upload directory is empty after processing.

The harness refuses a dirty checkout or a commit mismatch. Both FastAPI and Streamlit publish their build commit/tree, and the manifest compares these identities with the validating checkout before the workflow begins.

## Defects found and repaired during final acceptance

1. Direct execution of the backup utility could not import the project package; repository-root bootstrapping and an integration regression were added.
2. Ollama rejected unsupported JSON-schema numeric bounds; transport-only schema normalization now preserves application validation while remaining compatible with Ollama.
3. Participant rows are virtualized in the browser; acceptance now verifies their API records and the visible UI state instead of relying on off-screen DOM text.
4. The organization-access label differed from the browser assertion; QA now matches the rendered product language.
5. Real review explanations could attribute causality to passing evidence; the composer now hides non-causal raw values from prose generation, performs one validator-guided repair and falls back deterministically if repair fails.
6. Grounded Chinese HR-withholding wording was too narrow for the validator; the accepted lexical forms were expanded without permitting a BPM value in non-release prose.
7. Real research images were rejected by a filename-specific fixture rule; multimodal acceptance now validates a content-grounded real image contract while retaining the no-identification and no-vital-inference boundary.
8. Browser acceptance waited for one English inflection of HR withholding even when the model returned an equivalent validated form; completion now waits for the scoped model provenance marker and the semantic assertion accepts `withhold`, `withholds`, `withholding` and `withheld` without relaxing the no-BPM rule.

## Operational recovery

The backup utility creates a SQLite copy plus a SHA-256 manifest and runs `PRAGMA quick_check`. The retained acceptance backup reported SHA-256 `5c05785b07f8ad8574394c65a1409907358dbb17e8b519bee7ff27ad0e36e5b6`, `integrity_check: ok` and `raw_media_included: false`.

Startup, health verification, backup, restore rehearsal, retention and incident steps are documented in `CONTROLLED_TRIAL_OPERATIONS.md`. Required authentication must be tested again with the trial's real identity provider, issuer, audience, JWKS, role claims and organization claims before any multi-user trial begins.

## Remaining deployment boundary

The accepted product is a responsive web application. A native mobile client, offline capture queue, public cloud hardening, multi-instance database, managed secrets, centralized observability, penetration testing, disaster-recovery exercise, live EHR integration and prospective clinical-workflow validation remain separate deployment phases. The implementation and manuscript must not imply that these capabilities or their associated clinical claims have already been established.
