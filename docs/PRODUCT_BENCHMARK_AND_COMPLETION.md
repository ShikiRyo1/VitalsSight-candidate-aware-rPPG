# VitalsSight product benchmark and completion contract

## Scope

This document records the product patterns used to complete the VitalsSight web console. It is a design and implementation audit, not a claim that VitalsSight has the same regulatory status, deployment evidence, or clinical validation as a commercial product.

## Official-product patterns reviewed

| Product | Official pattern retained | VitalsSight implementation |
|---|---|---|
| Binah.ai Health Data Platform | Guided 35-60 second camera check; remain still, avoid speaking, use good lighting; explicit privacy and consent handling; SDK and management integration | Four-step assessment flow, visible capture checklist, video preflight, local-processing statement, raw-video deletion option, OpenAPI export |
| Neteera HealthGate | Centralized population dashboard; current values and trends; smart alert prioritization; baseline and alert-history context; daily/RPM reporting | Overview, decision and quality distributions, searchable cases, prioritized review queue, audit history, versioned reports |
| Oxehealth digital observations | Dynamic work list; assignment and structured observations; timestamps; overdue/missed-action visibility; shift-level reporting and compliance trail | Persistent review status, priority, assignee, note, resolution, timestamped SQLite audit events, report and queue exports |
| NuraLogix Anura | Short camera-based measurement; clear framing as a wellness/research output; longitudinal history and sharable summaries | Guided video acquisition, output-state definitions, history by case, evidence report exports, persistent claim boundary |

Official sources reviewed on 2026-07-14:

- https://www.binah.ai/
- https://www.binah.ai/sdk/
- https://www.neteera.com/product-page/
- https://www.neteera.com/release-of-neteera-system-v2-6-adds-sleep-insights-and-enhanced-user-interface/
- https://www.oxehealth.com/us/digital-observations
- https://nuralogix.ai/anura/
- https://nuralogix.ai/anura-enterprise/

## VitalsSight-specific product advantage

VitalsSight does not treat the largest spectral peak as a publishable result by default. The product surface keeps candidate branches visible, separates the selected candidate from the decision to publish it, withholds HR in review and retake states, and attaches evidence and policy attribution to every output. Reference HR and candidate error remain evaluation fields and must never enter the inference payload.

## Functional completion contract

- Every input passes consent, purpose, retention, and acquisition-quality checks.
- Upload analysis either produces a release with a finite HR or a review/retake state with HR withheld.
- Every review action stores status, priority, assignee, note, resolution, actor, and timestamp.
- Reports contain result, acquisition quality, candidate branches, evidence attribution, audit context, versions, and claim boundary.
- API and UI use the same SQLite store and the same release/review contract.
- Multipart video assessment is available through the API; it runs quality qualification before HR inference and deletes the raw upload after analysis.
- UI temporary retention lasts only until the assessment is cleared; stale local uploads are purged automatically.
- Public protocol metrics remain separate from synthetic interface cases.
- The default page exposes no unsupported clinical, emergency-alert, safety-probability, fairness, or universal-generalization claim.
