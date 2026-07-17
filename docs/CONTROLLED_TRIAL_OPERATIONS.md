# Controlled-trial operations

## Deployment boundary

This profile supports a supervised, single-instance research trial. It is not a public internet service, medical device, emergency monitor, or autonomously validated clinical workflow. Use one VitalsSight API/UI pair and one SQLite evidence database per trial environment. A multi-instance deployment requires a separately tested PostgreSQL persistence adapter, distributed locking, centralized secrets, and an infrastructure security review; those capabilities are not claimed here.

## Identity and authorization

VitalsSight does not store passwords. In controlled-trial mode, Streamlit delegates sign-in to an OpenID Connect provider and FastAPI verifies signed bearer tokens against the provider JWKS endpoint. The token must contain:

- `sub`, `iss`, `aud`, and `exp`;
- one organization in `organization`, `organization_id`, `org_id`, or `tenant_id`;
- at least one VitalsSight role: `participant`, `operator`, `reviewer`, `researcher`, `auditor`, `org-admin`, or `service`;
- `participant_id` when a participant login should be restricted to one pseudonymous record.

Roles are administered in the identity provider. The VitalsSight membership table is an audit mirror and is intentionally read-only in the product UI.

1. Copy `.streamlit/secrets.toml.example` to the ignored `.streamlit/secrets.toml` and enter the OIDC client settings.
2. Set the variables in `.env.controlled-trial.example` through the service manager or shell. The example file is documentation and is not loaded automatically.
3. Use an asymmetric algorithm such as `RS256`; `VITALSSIGHT_AUTH_SHARED_SECRET` exists only for isolated automated tests.
4. Keep `VITALSSIGHT_ALLOW_DEV_IDENTITY_HEADERS` unset. Never use development identity headers in required-auth mode.

## Start and preflight

```powershell
$env:VITALSSIGHT_AUTH_MODE = "required"
$env:VITALSSIGHT_AUTH_ISSUER = "https://identity.example.org/realms/vitalssight"
$env:VITALSSIGHT_AUTH_AUDIENCE = "vitalssight-api"
$env:VITALSSIGHT_AUTH_CLIENT_ID = "vitalssight-console"
$env:VITALSSIGHT_AUTH_JWKS_URL = "https://identity.example.org/realms/vitalssight/protocol/openid-connect/certs"
$env:VITALSSIGHT_DB_PATH = "runtime/controlled-trial/vitalssight.db"
$env:VITALSSIGHT_UPLOAD_DIR = "runtime/controlled-trial/uploads"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_vitalssight_with_assistant.ps1 `
  -UiPort 8502 -ApiPort 8011 -RequireModel -RequireMultimodal
```

Place TLS, request-size enforcement, rate limiting, and authenticated network access in a reviewed reverse proxy. Keep Ollama on localhost. Do not expose Streamlit, FastAPI, SQLite, or Ollama directly to the public internet.

Use an access token for protected health checks:

```powershell
.\.venv\Scripts\python.exe scripts\check_controlled_trial_health.py `
  --api-url http://127.0.0.1:8011 `
  --token $env:VITALSSIGHT_HEALTH_TOKEN `
  --require-auth --require-model --require-multimodal
```

The check fails when required authentication is disabled, a protected endpoint rejects the token, or a required model sidecar is unavailable.

## Data lifecycle

- Store only pseudonyms, study identifiers, purpose/version-specific consent records, derived evidence, decisions, report versions, and audit metadata.
- Do not enter names, medical-record numbers, contact details, or other direct identifiers into participants, cases, assistant prompts, or report notes.
- Required-auth video assessments accept only `delete_after_analysis`. The upload is deleted in a `finally` path whether processing succeeds or fails.
- Images and audio used by the assistant are normalized and processed transiently. Raw bytes, OCR text, and absolute temporary paths are not written to assistant audit records.
- Consent withdrawal blocks subsequent processing for the withdrawn purpose; it does not silently erase already generated evidence required for audit. Handle deletion requests through the approved study protocol and document the disposition.

## Backup and restore

Create an online, integrity-checked evidence backup. The command copies only SQLite state, never the raw upload directory:

```powershell
.\.venv\Scripts\python.exe scripts\backup_controlled_trial_state.py `
  --db runtime\controlled-trial\vitalssight.db `
  --output-dir D:\encrypted-vitalssight-backups `
  --keep 14
```

Each backup has a SHA-256 manifest and a successful SQLite `quick_check`. Encrypt the destination volume, restrict it to trial administrators, and test a restore into an isolated directory before the trial. Never overwrite the active database while the service is running; restore to a new path, verify health and report hashes, then change `VITALSSIGHT_DB_PATH` during a controlled restart.

## Monitoring and incident handling

Monitor service health, authentication failures, upload deletion, model availability, report approvals, review updates, and organization-scoped access events. Export access audit CSV from Help & settings for periodic review. Logs must not contain raw media, tokens, client secrets, or direct identifiers.

Stop the trial and preserve evidence when any of these occur:

- a review/retake result exposes a formal HR value;
- an identity reads another organization or participant record;
- an unconfirmed assistant action changes a review;
- report text contains an unsupported number, diagnosis, treatment instruction, or uncited causal claim;
- raw media remains after the configured deletion path;
- report payload/hash and exported PDF/FHIR content disagree.

Use `scripts/stop_vitalssight.ps1`, retain relevant hashed reports and audit events, rotate affected credentials, and document the root cause before resuming.
