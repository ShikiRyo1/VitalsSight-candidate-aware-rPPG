# Native application and scale roadmap

This document starts after the supervised single-instance web trial passes. It is not part of the current validation claim.

## Phase 1: controlled web trial

- Run one authenticated Streamlit/FastAPI pair per environment.
- Use an external OIDC provider for sign-in, role assignment and organization membership.
- Store pseudonyms, purpose/version consent, derived evidence, immutable report versions and access audit events in the local evidence database.
- Delete raw video, image and audio after processing unless a separately approved protocol says otherwise.
- Keep the deterministic gate authoritative; the LLM explains evidence and prepares actions but never changes a state or released HR.

## Phase 2: native capture client

Build a Flutter client only after a device matrix has established camera frame rate, exposure, orientation, thermal and background-upload behavior. Reuse the existing API contracts rather than embedding policy logic in the client.

Required client capabilities:

1. OIDC Authorization Code with PKCE and secure OS credential storage.
2. Guided framing, lighting, duration and motion feedback before upload.
3. Explicit purpose-specific consent and a visible raw-media retention state.
4. Resumable encrypted upload, retry limits and an offline queue with user-controlled deletion.
5. Release/review/retake rendering that never displays HR for a non-release state.
6. Accessible voice, image and text assistant input with the same evidence boundary as the web console.
7. Local notifications only for workflow status, never for emergency or clinical alerts.

## Phase 3: multi-instance service

Before public or multi-site hosting, replace the single-instance persistence layer with a separately tested PostgreSQL adapter and object-storage quarantine for transient uploads. Add distributed job ownership, idempotency keys, centralized secrets, encrypted backups, rate limits, malware scanning, structured monitoring and incident-response runbooks. Run threat modeling, penetration testing, disaster-recovery exercises and independent privacy review.

## Report and LLM continuation

- Keep the deterministic JSON/FHIR evidence packet as the source of truth.
- Allow the LLM to produce audience-specific explanations only through the existing evidence catalog, sentence citations and numerical-grounding validator.
- Require reviewer approval for every governed narrative used outside the immediate operator session.
- Add longitudinal summarization only from released values and non-release state counts; never reveal hidden candidate HR through trends.
- Measure report usefulness in a supervised usability study before changing clinical workflow or claim language.

## Release gates

Do not submit to app stores or expose the service publicly until the native client, identity provider, transport security, persistence adapter, retention jobs, audit export, accessibility checks and device matrix have all passed source-bound acceptance. Clinical or medical-device claims require separate prospective validation, quality management and regulatory work.
