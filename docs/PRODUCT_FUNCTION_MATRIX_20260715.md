# VitalsSight product function matrix

This matrix defines the finite product surface exercised by the browser, API,
unit, and real-video validation suites. It is a software conformance record,
not clinical validation, a usability study, security certification, or a
medical-device claim.

## Global shell

| Function | Expected outcome | Automated coverage |
|---|---|---|
| Chinese/English switch | The complete shell and assessment controls use one language | Browser v4 |
| Sidebar collapse/restore | Navigation can always be reopened | Browser v4, unit |
| Workspace navigation | All eight workspaces open and scroll to their first instruction | Browser v4, unit |
| Quick guide | Shows prepare, assess, act, and export steps | Browser v4 |
| Full guided workflow | Header and sidebar entry points open role guidance | Browser v4 |
| Desktop/mobile layout | No page-level horizontal overflow at tested viewports | Browser v4 |

## Overview

| Function | Expected outcome | Automated coverage |
|---|---|---|
| Operational counters | Cases, releases, open reviews, retakes, and median quality render | Browser v4 |
| Start guided assessment | Opens a clean assessment | Browser v4 |
| Continue review work | Opens the review queue | Browser v4 |
| Learn full workflow | Opens role guidance | Browser v4 |
| Quality and decision charts | Render with the shared status palette | Browser v4 visual capture |

## New assessment

| Function | Expected outcome | Automated coverage |
|---|---|---|
| Consent gate | Missing consent returns an explicit warning and no output | Browser v4 |
| Purpose and retention | Canonical values survive language changes | Browser v4 |
| Stable demo | Produces a released demonstration result | Browser v4 |
| Conflict demo | Produces review and withholds HR | Browser v4 |
| Low-light demo | Produces retake and withholds HR | Browser v4 |
| Real release video | Produces finite released HR and evidence action | Browser v4, real-video suite |
| Real review video | Withholds HR and exposes an actionable reason | Browser v4, real-video suite |
| Real retake video | Withholds HR and exposes acquisition correction | Browser v4, real-video suite |
| Delete-after-analysis | Leaves no raw upload after processing | Browser v4, API tests |
| Session-only retention | Retains one local file until clear/corrected-recording reset | Browser v4 |
| Clear | Rebuilds an empty uploader and clears the session result | Browser v4, unit |
| Open case | Opens the selected evidence packet | Browser v4 |
| Build report | Opens the selected evidence report | Browser v4 |

## Cases and review

| Function | Expected outcome | Automated coverage |
|---|---|---|
| Case registry filters | Search has explicit match and empty states | Browser v4 |
| Case detail | Shows output, trend/candidates, attribution, actions, and audit | Browser v4 |
| Open report | Preserves selected case context | Browser v4 |
| Review save | Status, priority, assignee, note, and resolution persist | Browser v4, unit |
| Review audit | Every save creates a rendered audit event | Browser v4, unit |
| Non-release contract | Review and retake never publish HR | Browser v4, unit, API |

## Reports and evidence

| Function | Expected outcome | Automated coverage |
|---|---|---|
| Report tabs | Detail, action, attribution, review/audit, and structured data open | Browser v4 |
| PDF export | Valid non-empty PDF | Browser v4, unit |
| JSON export | Full redacted evidence contract | Browser v4, unit |
| Markdown export | Review-ready action chain | Browser v4, unit |
| CSV export | Case-level row | Browser v4 |
| Open review workflow | Opens review for a non-release case | Browser v4 |
| Corrected recording | Resets consent/input and clears session-only raw video | Browser v4 |
| Protocol evidence | Metrics, chart, invariants, and claim boundary render | Browser v4 |

## Integration and guidance

| Function | Expected outcome | Automated coverage |
|---|---|---|
| Payload validation | Enforces the release/review output contract | Browser v4, unit, API |
| Integration audit | Writes an event to the shared store | Browser v4, API |
| OpenAPI export | Non-empty schema includes video assessment endpoint | Browser v4, unit |
| API documentation | Interactive documentation loads | Browser v4 |
| Capture guide | Opens a clean assessment | Browser v4 |
| Reviewer guide | Opens review work | Browser v4 |
| Report guide | Opens reports and integrations | Browser v4 |
| Operator setting | Saves actor identity for later audit events | Browser v4 |
| Restore demos | Restores built-in cases without deleting real cases | Browser v4 |
| Troubleshooting help | Expands visible click-feedback guidance | Browser v4 |

## Evidence boundary

Passing this matrix means that the listed finite interface, storage, report,
and API behaviors conformed on the curated fixtures and viewports. It does not
establish clinical accuracy, independent usability, production security,
fairness, continuous monitoring, or autonomous clinical release.
