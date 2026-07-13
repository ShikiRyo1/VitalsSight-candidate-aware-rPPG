# VitalsSight product console QA report

Verification date: 2026-07-14

## Verified product workflow

- Consent, research purpose, and raw-video retention are explicit before analysis.
- Video quality qualification runs before the HR pipeline and returns actionable duration, frame-rate, resolution, illumination, motion, and face-visibility guidance.
- The output contract has three states: `release`, `review`, and `retake`. Non-release states cannot publish `released_hr_bpm`.
- The case registry preserves quality, candidates, trends, model and policy versions, recommended action, and decision-level evidence.
- The review queue supports filtering, priority, assignment, notes, resolution, closure state, and timestamped audit events.
- Role-based tutorials define the input, action, output, and next destination for capture operators, evidence reviewers, and report/integration users.
- Reports export PDF, JSON, Markdown, and CSV. Human-readable reports include units, quality percentages, observed-versus-target evidence, recommendation basis, verification criteria, escalation, attribution, review records, audit trail, and the claim boundary.
- The REST API shares the same SQLite store and output contract as the UI. Multipart video submission deletes raw video after processing.
- Chinese and English interfaces use the same state and preserve the same numeric values.

## Automated checks

The product suite currently contains 23 passing tests covering:

- strict non-release HR withholding and finite release values;
- evidence attribution and claim boundaries;
- SQLite case, review, and audit persistence;
- bilingual Markdown and PDF report generation;
- preflight pass, warning, failure, decode-error, and runtime-error behavior;
- API case, review, report, OpenAPI, and multipart video-assessment endpoints;
- raw-video deletion after API assessment.
- evidence-to-action recommendation thresholds and bilingual report wording;
- sidebar recovery and page-navigation scroll reset safeguards.

Commands:

```bash
python -m pytest -q
python -m compileall -q app src tests
git diff --check
```

## Browser checks

Real-browser checks were run with Playwright at 1440 x 1000 and 390 x 844. The following actions were exercised:

- bilingual navigation across all eight workspaces;
- start, consent, run, clear, open case, and build report;
- short/dark video upload, preflight failure, saved retake case, and raw-file deletion;
- case search and decision/source filters;
- review assignment, note, resolution, save feedback, and audit persistence;
- PDF, JSON, Markdown, CSV, and OpenAPI downloads;
- attribution and structured-data report tabs;
- operator save, integration audit event, and non-destructive demo restoration;
- visible warning when assessment consent is missing and visible completion feedback after every exercised write action;
- desktop sidebar collapse and restoration, plus mobile sidebar restoration after navigation;
- page transitions from a `scrollTop` of 1600 to 0 after the destination view finishes rendering;
- mobile overview, assessment, report, integration and tutorial views without page-level horizontal overflow.

## Report rendering checks

English and Chinese evidence reports were generated from the same review case, rendered to PNG at 150 dpi, and inspected page by page. Both reports render as two A4 pages with readable tables, intact Chinese glyphs, no clipped cells or overlaps, consistent evidence wording, and a versioned footer with page numbers. Empty review fields and absent audit events no longer create an almost-empty third page.

## Visual system checks

The console uses one low-saturation clinical palette: steel blue for primary actions, muted teal for release, muted violet for review, and soft rose for retake. The desktop and mobile captures were checked for text contrast, button legibility, stable card dimensions, clear state hierarchy, and absence of high-contrast yellow/green or dark red status combinations.

## Product boundary

This verification establishes a complete research evidence workflow, not medical-device readiness. It does not establish clinical utility, emergency-alert performance, autonomous clinical release, production identity/access management, EHR certification, security certification, or end-to-end live-camera mobile capture. Those require separate product engineering, prospective validation, governance, and regulatory work.
