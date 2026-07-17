# Report governance and FHIR exchange

## Report lifecycle

1. A case first passes the output-contract validator.
2. The report service applies the selected audience: participant, operator, reviewer, or research.
3. Governance metadata adds organization, pseudonymous participant, consent version, policy/model identity, longitudinal state context, and a canonical SHA-256.
4. The optional local model drafts four evidence-cited fields. The deterministic validator rejects invented numbers, unknown evidence IDs, uncited sentences, non-release candidate HR, unsupported causal claims, and clinical wording.
5. A reviewer or organization administrator inspects the evidence hash, audience, explanation citations and output state before approval.
6. Approval never rewrites content. A correction produces a new version and may identify the superseded report.

The natural-language narrative is an addendum to deterministic evidence, not a new source of truth. When a model is missing, times out, returns malformed JSON, or fails validation, VitalsSight stores a labeled deterministic fallback instead of weakening validation.

## Audience behavior

- **Participant:** simplified action-oriented content; no internal candidate pool, hidden HR, operator notes, or unnecessary implementation details.
- **Operator:** capture quality, state, next action and verification criteria.
- **Reviewer:** full evidence packet, thresholds, attribution, review history and report governance.
- **Research:** de-identified protocol/model/policy context and aggregate-compatible evidence; no direct identifiers.

## FHIR research exchange

The FHIR export is a research interoperability bundle, not a claim of EHR certification. It contains:

- `Patient` with a pseudonymous identifier only;
- `Device` with VitalsSight model/policy identity;
- `DiagnosticReport` for the evidence-report state and report hash;
- `Observation` only when the case is `release`;
- `Task` for `review` or `retake`, with no candidate or withheld HR value;
- `Consent` when a matching active consent record is available;
- `Provenance` linking the generated resources and responsible identity.

Every downstream system must preserve the research-use boundary, report hash, model/policy versions, state, and evidence provenance. It must not reinterpret a review/retake task as an HR observation.
