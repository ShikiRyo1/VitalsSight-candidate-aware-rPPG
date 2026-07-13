# VitalsSight product benchmark and completion contract

## Scope

This document records the product patterns used to complete the VitalsSight web console. It is a design and implementation audit, not a claim that VitalsSight has the same regulatory status, deployment evidence, or clinical validation as a commercial product.

## Official-product patterns reviewed

| Product | Official pattern retained | VitalsSight implementation |
|---|---|---|
| Binah.ai Health Data Platform | Concrete capture preparation: stable stand, eye-level camera, fully exposed face, no talking or movement, and even front lighting | Role-based capture guide, explicit consent and retention, quality-first preflight, threshold-linked corrective actions, local-processing statement, and raw-video deletion option |
| Neteera System | Active issues, baseline trend and notification history for prioritization; improved patient charts, trend plots and daily/RPM reports | Operations overview, decision and quality distributions, prioritized work list, case trends, review history, and versioned evidence reports |
| LIO digital observations | Smart prompts, dynamic priority lists, upcoming/overdue notifications, time-stamped entry, structured records and compliance auditing | Persistent review status, priority, assignee, note and resolution; timestamped SQLite audit events; visible save feedback; report and queue exports |
| NuraLogix Anura Mobile Core SDK | Approximately 30-second facial-video workflow, integration into standard devices, and an explicit boundary around the measurements and intended use described by the provider | Guided video input, three-state output contract, evidence-linked reporting, integration API, and a persistent research-only claim boundary |

Official sources reviewed on 2026-07-14:

- https://www.binah.ai/
- https://support.binah.ai/en-US/binah/article/5uQeVfY3-best-practices
- https://www.neteera.com/release-of-neteera-system-v2-6-adds-sleep-insights-and-enhanced-user-interface/
- https://www.liohealth.com/digital-observations
- https://www.nuralogix.ai/nuralogix-anura-mobile-core-sdk-receives-fda-510k-clearance-for-contactless-measurement-of-heart-pulse-rate-and-breathing-respiration-rate/

## VitalsSight-specific product advantage

VitalsSight does not treat the largest spectral peak as a publishable result by default. The product surface keeps candidate branches visible, separates the selected candidate from the decision to publish it, withholds HR in review and retake states, and attaches evidence and policy attribution to every output. Reference HR and candidate error remain evaluation fields and must never enter the inference payload.

## Functional completion contract

- Every input passes consent, purpose, retention, and acquisition-quality checks.
- Every role has a guided path that identifies the required input, action, output, and next destination.
- Every command either navigates, exports a file, or returns visible success, information, warning, or error feedback.
- Upload analysis either produces a release with a finite HR or a review/retake state with HR withheld.
- Every review action stores status, priority, assignee, note, resolution, actor, and timestamp.
- Reports link each observed signal to its target, state, operational meaning, corrective action, verification threshold, escalation path, and claim boundary.
- Reports contain result, acquisition quality, candidate branches, evidence attribution, review and audit context, versions, page numbers, and audience-specific PDF/JSON/Markdown/CSV exports.
- API and UI use the same SQLite store and the same release/review contract.
- Multipart video assessment is available through the API; it runs quality qualification before HR inference and deletes the raw upload after analysis.
- UI temporary retention lasts only until the assessment is cleared; stale local uploads are purged automatically.
- Public protocol metrics remain separate from synthetic interface cases.
- Collapsed navigation remains recoverable on desktop and mobile, page changes begin at the first instruction, and tested mobile views have no page-level horizontal overflow.
- The default page exposes no unsupported clinical, emergency-alert, safety-probability, fairness, or universal-generalization claim.

## Deliberately unclaimed commercial capabilities

The benchmark also identified functions that are not yet established in this public research console: continuous live-camera coaching, production identity and role-based access control, EHR certification, patient-specific longitudinal baselines, prospective clinical workflow validation, and regulated medical-device use. Their absence is stated explicitly rather than being disguised by the interface.
