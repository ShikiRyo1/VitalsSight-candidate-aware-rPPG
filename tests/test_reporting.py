from __future__ import annotations

from copy import deepcopy
import json

from src.assistant.provider import ProviderReply, ProviderStatus
from src.product.console_service import build_report_payload, make_demo_cases
from src.product.report_narrative import EvidenceBoundedReportNarrator
from src.product.reporting import (
    build_fhir_bundle,
    build_longitudinal_context,
    enrich_report_payload,
    report_content_sha256,
    report_version_sha256,
)


def _governed(case: dict[str, object]) -> dict[str, object]:
    return enrich_report_payload(
        build_report_payload(case),
        organization_id="org-a",
        audience="reviewer",
        language="en",
        participant={"participant_id": "pt-001", "pseudonym": "P-001", "study_id": "trial-a"},
        consent={
            "consent_id": "consent-001",
            "document_version": "consent-v1",
            "status": "active",
        },
    )


def test_report_hash_ignores_generation_and_approval_metadata() -> None:
    payload = _governed(make_demo_cases()[0])
    changed = deepcopy(payload)
    changed["generated_at"] = "2099-01-01T00:00:00+00:00"
    changed["governance"]["approval_status"] = "approved"
    changed["governance"]["narrative_status"] = "approved"

    assert report_content_sha256(payload) == report_content_sha256(changed)


def test_report_version_hash_covers_validated_narrative() -> None:
    payload = _governed(make_demo_cases()[0])
    first = {"direct_summary": "Grounded [E1].", "evidence_ids": ["E1"]}
    changed = {"direct_summary": "Different grounded text [E1].", "evidence_ids": ["E1"]}

    assert report_version_sha256(payload, first) == report_version_sha256(payload, first)
    assert report_version_sha256(payload, first) != report_version_sha256(payload, changed)
    assert report_version_sha256(payload, first) != report_content_sha256(payload)


def test_longitudinal_context_never_places_withheld_hr_in_trend() -> None:
    release, review, retake = make_demo_cases()[0:3]
    review["selected_candidate_hr_bpm"] = 123.0
    retake["selected_candidate_hr_bpm"] = 98.0
    context = build_longitudinal_context([retake, review, release], current_case_id=review["case_id"])

    assert context["state_counts"] == {"release": 1, "review": 1, "retake": 1}
    assert context["released_measurement_count"] == 1
    assert all(
        "released_hr_bpm" not in item
        for item in context["timeline"]
        if item["decision"] != "release"
    )
    assert {item["decision"] for item in context["timeline"]} == {"release", "review", "retake"}


def test_fhir_release_has_observation_and_review_has_task_without_hr() -> None:
    release_bundle = build_fhir_bundle(_governed(make_demo_cases()[0]), report_id="report-release")
    release_resources = [entry["resource"] for entry in release_bundle["entry"]]
    observations = [item for item in release_resources if item["resourceType"] == "Observation"]
    assert len(observations) == 1
    assert observations[0]["code"]["coding"][0]["code"] == "8867-4"
    assert observations[0]["valueQuantity"]["unit"] == "beats/minute"

    review = make_demo_cases()[1]
    review["selected_candidate_hr_bpm"] = 145.0
    review_bundle = build_fhir_bundle(_governed(review), report_id="report-review")
    review_resources = [entry["resource"] for entry in review_bundle["entry"]]
    assert not [item for item in review_resources if item["resourceType"] == "Observation"]
    tasks = [item for item in review_resources if item["resourceType"] == "Task"]
    assert tasks[0]["code"]["coding"][0]["code"] == "human-review"
    encoded = json.dumps(review_bundle, ensure_ascii=False)
    assert "145.0" not in encoded
    assert "No heart-rate output was released" in encoded


def test_fhir_retake_requests_repeat_acquisition() -> None:
    bundle = build_fhir_bundle(_governed(make_demo_cases()[2]), report_id="report-retake")
    resources = [entry["resource"] for entry in bundle["entry"]]
    task = next(item for item in resources if item["resourceType"] == "Task")

    assert task["code"]["coding"][0]["code"] == "repeat-acquisition"
    assert not [item for item in resources if item["resourceType"] == "Observation"]


def test_deterministic_narrative_withholds_review_hr_and_resolves_citations() -> None:
    review = make_demo_cases()[1]
    review["selected_candidate_hr_bpm"] = 145.0
    narrative = EvidenceBoundedReportNarrator().generate(_governed(review), language="en")

    assert narrative["mode"] == "deterministic_fallback"
    assert narrative["validation"]["passed"] is True
    assert "145" not in " ".join(
        narrative[key]
        for key in ("direct_summary", "evidence_explanation", "action_guidance", "limitations")
    )
    assert narrative["evidence_ids"]
    assert all(item.startswith("E") for item in narrative["evidence_ids"])


class UnsafeNarrativeProvider:
    provider_name = "unsafe-test"
    model = "unsafe-test-model"

    def status(self) -> ProviderStatus:
        return ProviderStatus(True, self.provider_name, self.model, "ready")

    def chat(self, messages, *, tools=None, response_schema=None) -> ProviderReply:
        del messages, tools, response_schema
        return ProviderReply(
            content=json.dumps(
                {
                    "direct_summary": "The diagnosis is confirmed at 999 BPM [E1].",
                    "evidence_explanation": "This is safe [E1].",
                    "action_guidance": "Prescribe treatment [E1].",
                    "limitations": "None [E1].",
                    "evidence_ids": ["E1"],
                }
            )
        )


def test_unsafe_model_narrative_is_replaced_by_validated_fallback() -> None:
    review = make_demo_cases()[1]
    narrative = EvidenceBoundedReportNarrator(UnsafeNarrativeProvider()).generate(
        _governed(review), language="en"
    )
    combined = " ".join(
        narrative[key]
        for key in ("direct_summary", "evidence_explanation", "action_guidance", "limitations")
    ).lower()

    assert narrative["mode"] == "deterministic_fallback"
    assert narrative["validation"]["provider_errors"]
    assert "999" not in combined
    assert "prescribe" not in combined
    assert "diagnosis is confirmed" not in combined
