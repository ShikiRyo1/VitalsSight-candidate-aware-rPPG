# VitalsSight evidence assistant: product and safety specification

## Intended function

The assistant is a local conversational and workflow-orchestration layer over the existing deterministic VitalsSight evidence pipeline. It explains recorded quality, candidate, policy, review, and report evidence; retrieves versioned local guidance; navigates the product; and can prepare a review update for explicit confirmation. It does not estimate HR, alter candidate selection, override release/review/retake, diagnose, prescribe, or provide emergency guidance.

The signal pipeline and stored case are the source of truth. The language model is optional: measurement, reports, review, export, and the REST API remain available when the model is offline.

The assistant provides post-inference consistency checks over bounded recorded evidence; it is not a second measurement model, an independent safety monitor, or an autonomous clinical supervisor. Its refusal, explanation, or fallback state cannot change the deterministic output contract.

## Architecture and trust boundary

```text
Streamlit chat / REST request
        |
input safety and role policy
        |
AssistantOrchestrator
        |-- deterministic intent and required evidence tools
        |-- optional Qwen read-only tool selection
        |-- local lexical RAG over versioned guidance
        |
ConsoleStore + report/action-plan services
        |
deterministic response assembly and post-validation
        |
Pydantic response + evidence IDs + audit digest
```

The model cannot access SQLite, the filesystem, raw video, environment variables, or network resources directly. It receives only bounded tool results. `prepare_review_update` creates an expiring pending record; a separate confirmation call performs the update through `ConsoleStore.update_review` and writes the existing audit event.

## Roles

| Role | Read case/report | Workflow guidance | Prepare review update | Confirm update |
|---|---:|---:|---:|---:|
| operator | yes | yes | no | no |
| reviewer | yes | yes | policy controlled | explicit only |
| clinician | yes | yes | no | no |
| admin | yes | yes | policy controlled | explicit only |

These roles are research-product workflow roles, not production identity or access management. Production deployment requires authenticated RBAC outside this artifact.

## Whitelisted tools

- `list_cases`: de-identified case summaries, bounded to 20 rows.
- `get_case`: decision, released HR only when allowed, quality, candidates, action plan, and provenance.
- `get_report_summary`: report identity, interpretation, evidence, actions, and review state.
- `get_review`: current review record.
- `validate_output_contract`: release/non-release invariants.
- `search_help`: versioned, content-hashed local knowledge chunks.
- `prepare_review_update`: optional reviewer/admin proposal; never executes directly.

No tool deletes video, changes HR, changes the output state, rewrites policy, downloads external content, or executes arbitrary code.

## Data handling

- Raw video is never passed to the assistant.
- Model prompts contain structured derived evidence and local guidance excerpts.
- Assistant audit rows store hashes of the user message and response, not the raw conversation text.
- Audit rows record tool names, evidence IDs, model/fallback state, and validation result.
- Browser-session conversation history can be cleared without changing case evidence.
- Ollama is bound to localhost in the supported configuration.

## Guardrails and invariants

1. A release requires a finite `released_hr_bpm`.
2. Review and retake must return `released_hr_bpm = null` and explicitly state that HR is withheld.
3. Numeric claims with BPM, percentage, or fps units must occur in supplied tool evidence.
4. The generated answer may cite only evidence identifiers supplied for that turn.
5. A model answer that contradicts the case, leaks non-release HR, crosses the clinical boundary, lacks citations, or introduces an unsupported number is discarded.
6. Prompt-injection, prompt-disclosure, diagnosis, treatment, and emergency requests are intercepted before tool access.
7. State-changing actions are disabled by default and always require a second explicit confirmation.
8. Provider timeout or malformed output activates deterministic evidence guidance; it does not fail the product workflow.

## Threat model

Covered threats include prompt injection, role escalation through prompt text, unsupported tool names, invalid tool arguments, direct state mutation, diagnosis/treatment requests, non-release HR leakage, model hallucinated numbers, missing evidence citations, provider outage, malformed JSON, and raw-chat retention. External-service compromise, production authentication, device security, network segmentation, EHR certification, and clinical misuse remain deployment responsibilities.

## Versioning and rollback

The model name, prompt contract, knowledge files, tool schemas, policy version, model version, and source commit remain separately identifiable. Removing or stopping Ollama immediately returns the assistant to deterministic mode. Removing the assistant workspace does not alter the underlying case, report, review, or API data model.

## Acceptance gates

- 100% recorded-decision consistency.
- 0 non-release HR publications.
- 100% evidence-reference coverage for evidence-bearing answers.
- 0 unsupported clinical recommendations.
- 100% explicit confirmation for state changes.
- 100% usable deterministic fallback when the model is unavailable.
- No browser-console, page, or unexpected HTTP errors in the validated workflow.

These are technical workflow gates. They are not clinical validation, a medical-device claim, or evidence of prospective safety.
