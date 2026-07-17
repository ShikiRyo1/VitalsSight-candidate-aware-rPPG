# Privacy and security model

## Authority boundary

VitalsSight has three deliberately separate layers:

1. The deterministic measurement and policy layer produces quality evidence, candidate evidence, and `release`, `review`, or `retake`.
2. The report layer renders audience-specific evidence and immutable version hashes.
3. The optional LLM layer explains already recorded evidence. It cannot change measurements, thresholds, policy identity, output state, or report approval.

Only `release` may contain `released_hr_bpm`. Review and retake payloads, longitudinal views, participant reports, FHIR exports, and AI explanations must withhold candidate HR values.

## Data classes and controls

| Data | Stored | Control |
|---|---:|---|
| OIDC subject, role and organization | Yes | Signed token verification, organization scoping, access audit |
| Participant pseudonym/study ID | Yes | No direct identifiers; participant-token ownership restriction |
| Purpose/version consent | Yes | Active/withdrawn history; processing blocked after withdrawal |
| Raw video | No by default | Local transient upload, size limit, deletion in success/failure paths |
| Raw image/audio assistant input | No | In-memory/transient normalization; only hashes and bounded context retained |
| Candidate and quality evidence | Yes | Organization/participant scope; report audience redaction |
| Assistant prompt/answer text | No in audit | SHA-256 digests, tool trace, provider status and bounded metadata only |
| Report versions | Yes | Content hash, creator, audience, language, approval and supersession metadata |

## Principal threats and implemented controls

- **Cross-tenant disclosure:** every case, participant, consent, review, report and access query is bound to the verified organization. Participant identities are additionally bound to one `participant_id`.
- **Token misuse:** required mode verifies signature, issuer, audience, expiry and allowed algorithm. Missing or insufficient-role tokens fail before store access.
- **LLM hallucination:** report prose uses a typed JSON schema, inline evidence identifiers, numeric-token grounding, non-release HR leakage checks, causal-evidence checks, prohibited clinical wording checks, and deterministic fallback.
- **Prompt injection through media:** image text is non-authoritative context, bounded and sanitized; it cannot invoke tools or alter the evidence contract.
- **Unauthorized writes:** review changes require reviewer/admin role. Assistant changes are prepared as expiring tokens and execute only after an explicit second confirmation by the same actor.
- **Silent report mutation:** approved versions are immutable; a correction creates another version with its own hash and optional supersession link.
- **Media retention:** required-auth mode accepts only delete-after-analysis. Backup tooling excludes upload directories.

## Operational limits

The current persistence layer is SQLite and is validated for one controlled service instance. The repository does not claim hardened public hosting, high availability, disaster-recovery automation, formal penetration testing, regulatory certification, clinical-workflow validation, or mobile-store compliance. Those items require separate engineering and governance work before broader deployment.
