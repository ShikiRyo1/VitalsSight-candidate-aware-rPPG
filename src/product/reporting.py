from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import base64
import hashlib
import json
import math
from typing import Any, Iterable


FHIR_SYSTEM = "https://vitalssight.local/fhir"
LOINC_HEART_RATE = "8867-4"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def report_content_sha256(payload: dict[str, Any]) -> str:
    stable = deepcopy(payload)
    stable.pop("generated_at", None)
    governance = stable.get("governance")
    if isinstance(governance, dict):
        governance.pop("generated_at", None)
        governance.pop("content_sha256", None)
        governance.pop("approval_status", None)
        governance.pop("narrative_status", None)
    return hashlib.sha256(canonical_json(stable).encode("utf-8")).hexdigest()


def report_version_sha256(
    payload: dict[str, Any],
    narrative: dict[str, Any] | None = None,
) -> str:
    """Hash the exact immutable report version, including validated prose."""

    version = {
        "payload": deepcopy(payload),
        "narrative": deepcopy(narrative or {}),
    }
    return hashlib.sha256(canonical_json(version).encode("utf-8")).hexdigest()


def build_longitudinal_context(
    cases: Iterable[dict[str, Any]],
    *,
    current_case_id: str = "",
) -> dict[str, Any]:
    ordered = sorted(
        (dict(case) for case in cases),
        key=lambda item: (str(item.get("created_at") or ""), str(item.get("case_id") or "")),
    )
    state_counts = {"release": 0, "review": 0, "retake": 0}
    timeline: list[dict[str, Any]] = []
    released_points: list[dict[str, Any]] = []
    for case in ordered:
        decision = str(case.get("decision") or "review")
        if decision not in state_counts:
            decision = "review"
        state_counts[decision] += 1
        item = {
            "case_id": str(case.get("case_id") or ""),
            "display_id": str(case.get("display_id") or ""),
            "created_at": str(case.get("created_at") or ""),
            "decision": decision,
            "quality_score": case.get("quality_score"),
            "is_current": str(case.get("case_id") or "") == current_case_id,
        }
        if decision == "release":
            value = case.get("released_hr_bpm")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                item["released_hr_bpm"] = float(value)
                released_points.append(
                    {
                        "case_id": item["case_id"],
                        "created_at": item["created_at"],
                        "released_hr_bpm": float(value),
                    }
                )
        timeline.append(item)
    return {
        "case_count": len(timeline),
        "state_counts": state_counts,
        "released_measurement_count": len(released_points),
        "released_measurements": released_points,
        "timeline": timeline,
        "boundary": (
            "Only released outputs enter the HR trend. Review and retake records remain visible as states "
            "without exposing candidate or withheld HR values. The trend is not a clinical trajectory."
        ),
    }


def enrich_report_payload(
    payload: dict[str, Any],
    *,
    organization_id: str,
    audience: str,
    language: str,
    participant: dict[str, Any] | None = None,
    consent: dict[str, Any] | None = None,
    longitudinal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    enriched = deepcopy(payload)
    case = enriched.get("case") or {}
    participant = participant or {}
    consent = consent or {}
    enriched["governance"] = {
        "organization_id": organization_id,
        "audience": audience,
        "language": language,
        "participant_id": str(case.get("participant_id") or participant.get("participant_id") or ""),
        "participant_pseudonym": str(participant.get("pseudonym") or ""),
        "study_id": str(case.get("study_id") or participant.get("study_id") or ""),
        "consent_id": str(consent.get("consent_id") or ""),
        "consent_document_version": str(
            consent.get("document_version")
            or (case.get("consent") or {}).get("document_version")
            or ""
        ),
        "consent_status": str(consent.get("status") or ""),
        "raw_media_retained": False,
        "narrative_status": "not_generated",
        "approval_status": "unversioned",
    }
    enriched["longitudinal"] = longitudinal or build_longitudinal_context(
        [case], current_case_id=str(case.get("case_id") or "")
    )
    enriched["governance"]["content_sha256"] = report_content_sha256(enriched)
    return enriched


def report_for_audience(payload: dict[str, Any], *, audience: str) -> dict[str, Any]:
    result = deepcopy(payload)
    if audience != "participant":
        return result
    result["audit_events"] = []
    result["review"] = {
        key: value
        for key, value in (result.get("review") or {}).items()
        if key in {"status", "priority", "resolution", "updated_at"}
    }
    case = result.get("case") or {}
    if case.get("decision") != "release":
        case["selected_candidate_hr_bpm"] = None
        case["candidates"] = []
        case["trend_bpm"] = []
    return result


def _reference(resource_type: str, resource_id: str) -> dict[str, str]:
    return {"reference": f"{resource_type}/{resource_id}"}


def _identifier(system_suffix: str, value: str) -> list[dict[str, str]]:
    return [{"system": f"{FHIR_SYSTEM}/{system_suffix}", "value": value}]


def build_fhir_bundle(payload: dict[str, Any], *, report_id: str = "") -> dict[str, Any]:
    case = payload["case"]
    governance = payload.get("governance") or {}
    decision = str(case.get("decision") or "review")
    case_id = str(case.get("case_id") or "case")
    participant_id = str(governance.get("participant_id") or case.get("participant_id") or f"anonymous-{case_id}")
    diagnostic_id = report_id or f"diagnostic-{case_id}"
    device_id = "vitalssight-research-pipeline"
    generated_at = str(payload.get("generated_at") or datetime.now(UTC).isoformat())
    content_sha256 = str(governance.get("content_sha256") or report_content_sha256(payload))
    try:
        fhir_hash = base64.b64encode(bytes.fromhex(content_sha256)).decode("ascii")
    except ValueError:
        fhir_hash = base64.b64encode(content_sha256.encode("utf-8")).decode("ascii")
    entries: list[dict[str, Any]] = []

    patient = {
        "resourceType": "Patient",
        "id": participant_id,
        "identifier": _identifier("participant", participant_id),
        "active": True,
    }
    pseudonym = str(governance.get("participant_pseudonym") or "")
    if pseudonym:
        patient["name"] = [{"text": pseudonym}]
    entries.append({"fullUrl": f"urn:uuid:{participant_id}", "resource": patient})

    device = {
        "resourceType": "Device",
        "id": device_id,
        "identifier": _identifier("device", device_id),
        "status": "active",
        "deviceName": [{"name": "VitalsSight research pipeline", "type": "model-name"}],
        "version": [
            {"type": {"text": "model"}, "value": str(case.get("model_version") or "unknown")},
            {"type": {"text": "policy"}, "value": str(case.get("policy_version") or "unknown")},
        ],
        "note": [{"text": str(payload.get("claim_boundary") or "")}],
    }
    entries.append({"fullUrl": f"urn:uuid:{device_id}", "resource": device})

    result_references: list[dict[str, str]] = []
    if decision == "release":
        released_hr = case.get("released_hr_bpm")
        if not isinstance(released_hr, (int, float)) or not math.isfinite(float(released_hr)):
            raise ValueError("A release FHIR bundle requires a finite released_hr_bpm")
        observation_id = f"hr-{case_id}"
        observation = {
            "resourceType": "Observation",
            "id": observation_id,
            "identifier": _identifier("case", case_id),
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "vital-signs",
                            "display": "Vital Signs",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": LOINC_HEART_RATE,
                        "display": "Heart rate",
                    }
                ]
            },
            "subject": _reference("Patient", participant_id),
            "effectiveDateTime": str(case.get("created_at") or generated_at),
            "issued": generated_at,
            "valueQuantity": {
                "value": float(released_hr),
                "unit": "beats/minute",
                "system": "http://unitsofmeasure.org",
                "code": "/min",
            },
            "device": _reference("Device", device_id),
            "note": [{"text": "Research workflow output; see DiagnosticReport evidence boundary."}],
        }
        entries.append({"fullUrl": f"urn:uuid:{observation_id}", "resource": observation})
        result_references.append(_reference("Observation", observation_id))

    conclusion = {
        "release": "A heart-rate output was released with its evidence packet.",
        "review": "No heart-rate output was released; human review is required.",
        "retake": "No heart-rate output was released; a new acquisition is required.",
    }.get(decision, "No heart-rate output was released.")
    diagnostic = {
        "resourceType": "DiagnosticReport",
        "id": diagnostic_id,
        "identifier": _identifier("report", report_id or case_id),
        "status": "final" if decision == "release" else "preliminary",
        "code": {
            "coding": [
                {
                    "system": f"{FHIR_SYSTEM}/CodeSystem/report-type",
                    "code": "contactless-hr-evidence",
                    "display": "Contactless HR evidence report",
                }
            ]
        },
        "subject": _reference("Patient", participant_id),
        "effectiveDateTime": str(case.get("created_at") or generated_at),
        "issued": generated_at,
        "result": result_references,
        "conclusion": conclusion,
        "conclusionCode": [
            {
                "coding": [
                    {
                        "system": f"{FHIR_SYSTEM}/CodeSystem/output-state",
                        "code": decision,
                        "display": decision.capitalize(),
                    }
                ]
            }
        ],
        "presentedForm": [
            {
                "contentType": "application/json",
                "title": "VitalsSight deterministic evidence payload",
                "hash": fhir_hash,
            }
        ],
        "note": [{"text": str(payload.get("claim_boundary") or "")}],
    }
    entries.append({"fullUrl": f"urn:uuid:{diagnostic_id}", "resource": diagnostic})

    if decision in {"review", "retake"}:
        task_id = f"task-{case_id}"
        task = {
            "resourceType": "Task",
            "id": task_id,
            "identifier": _identifier("task", task_id),
            "status": "requested",
            "intent": "order",
            "code": {
                "coding": [
                    {
                        "system": f"{FHIR_SYSTEM}/CodeSystem/workflow-action",
                        "code": "human-review" if decision == "review" else "repeat-acquisition",
                        "display": "Human review" if decision == "review" else "Repeat acquisition",
                    }
                ]
            },
            "focus": _reference("DiagnosticReport", diagnostic_id),
            "for": _reference("Patient", participant_id),
            "authoredOn": generated_at,
            "description": str(case.get("recommended_action") or conclusion),
        }
        entries.append({"fullUrl": f"urn:uuid:{task_id}", "resource": task})

    consent_id = str(governance.get("consent_id") or "")
    if consent_id:
        consent = {
            "resourceType": "Consent",
            "id": consent_id,
            "status": "active" if governance.get("consent_status") == "active" else "inactive",
            "scope": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/consentscope",
                        "code": "research",
                        "display": "Research",
                    }
                ]
            },
            "category": [{"text": str((case.get("consent") or {}).get("purpose") or "research processing")}],
            "subject": _reference("Patient", participant_id),
            "date": str(case.get("created_at") or generated_at),
            "policyText": {
                "reference": f"urn:vitalssight:consent:{governance.get('consent_document_version') or 'unknown'}"
            },
        }
        entries.append({"fullUrl": f"urn:uuid:{consent_id}", "resource": consent})

    provenance_id = f"provenance-{case_id}"
    provenance_targets = [_reference("DiagnosticReport", diagnostic_id), *result_references]
    provenance = {
        "resourceType": "Provenance",
        "id": provenance_id,
        "target": provenance_targets,
        "recorded": generated_at,
        "policy": [str(case.get("policy_version") or "unknown")],
        "agent": [
            {
                "type": {"text": "Assembler"},
                "who": _reference("Device", device_id),
            }
        ],
        "entity": [
            {
                "role": "source",
                "what": {
                    "identifier": {
                        "system": f"{FHIR_SYSTEM}/evidence-packet",
                        "value": content_sha256,
                    }
                },
            }
        ],
    }
    entries.append({"fullUrl": f"urn:uuid:{provenance_id}", "resource": provenance})

    return {
        "resourceType": "Bundle",
        "id": f"bundle-{report_id or case_id}",
        "type": "collection",
        "timestamp": generated_at,
        "identifier": {
            "system": f"{FHIR_SYSTEM}/bundle",
            "value": content_sha256,
        },
        "entry": entries,
    }
