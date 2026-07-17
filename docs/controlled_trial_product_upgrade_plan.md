# VitalsSight controlled-trial product upgrade plan

Status: implementation and finite local acceptance complete
Baseline commit: `498f1c3`  
Scope: controlled-trial-ready, single-instance web product; not a medical-device, public-hosting, or autonomous-clinical-release claim

## Non-negotiable contracts

- Preserve the validated candidate-aware inference and release/review/retake behavior.
- A review or retake output never exposes a released HR.
- The deterministic evidence payload remains authoritative; an LLM may explain but may not alter facts, thresholds, decisions, or policy identity.
- Raw video, image, and audio are transient by default and are deleted after analysis.
- Existing locked metrics and research evidence are not removed because a source is unavailable in the current checkout.

## Delivery sequence

1. [x] Add organization, user, membership, participant, consent, report-version and access-audit persistence with backward-compatible SQLite migrations.
2. [x] Add OIDC-ready identity resolution, role checks, organization scoping and an explicit disabled-auth local development mode.
3. [x] Add participant pseudonyms, consent versions, longitudinal case lookup and tenant-safe API endpoints.
4. [x] Upgrade evidence reports with audience variants, immutable hashes, approvals, addenda, trend context and FHIR export.
5. [x] Add an evidence-bounded LLM narrative addendum with JSON schema validation, evidence citations, numerical grounding and human approval.
6. [x] Add product UI for identity, organization context, participant/consent capture, longitudinal trends, report versions and approvals.
7. [x] Add provider-neutral OIDC configuration, privacy/retention documentation, integrity-checked backup, health checks and operational audit views. The controlled-trial release intentionally remains a single SQLite service instance; a PostgreSQL/multi-instance adapter is a later deployment phase and is not claimed here.
8. [x] Run the final source-bound unit, API, browser, mobile, security, real-video, AI and multimodal acceptance suites and repair every observed regression. The authoritative acceptance record is `docs/CONTROLLED_TRIAL_PRODUCT_QA_20260717.md`; the generated browser manifest binds the result to the exact validated commit and tree.

## Roles

| Role | Intended capability |
|---|---|
| participant | Complete consented capture and view only owned outputs made available to them |
| operator | Create participants/cases, perform capture and initiate retakes |
| reviewer | Review evidence, record resolution and approve reports |
| researcher | Read de-identified cases and exports within an assigned organization/study |
| auditor | Read version and access history without modifying clinical or research evidence |
| org-admin | Manage memberships, policies, invitations and organization configuration |
| service | Use explicitly scoped API integrations |

## Acceptance gates

- Cross-organization case, report, review and participant reads return no data.
- Production auth mode rejects missing, expired, wrongly issued, wrongly addressed or insufficient-role tokens.
- Local disabled-auth mode remains explicit, clearly labelled and suitable only for development and automated tests.
- Every report version has a payload SHA-256, audience, language, model/policy identity, creator and immutable creation time.
- Approved reports cannot be silently overwritten; corrections create an addendum or superseding version.
- LLM prose contains no unsupported number, no uncited factual claim and no causal attribution to a passing/unavailable check.
- Patient-facing narratives contain no diagnosis or treatment instruction and retain the research claim boundary.
- FHIR exports validate the output contract and keep review/retake HR withheld.
- Existing unit and real-data acceptance suites continue to pass after the upgrade.

## Native application continuation boundary

The controlled-trial release remains a responsive web product. A later Flutter capture application should reuse the same FastAPI contracts after device-compatibility testing, background upload, offline queueing, HealthKit/Health Connect permission review and store-compliance materials are complete.
