from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import scripts.setup_runtime_assets as runtime_assets
from scripts.run_real_video_product_validation import execution_replay_description
from src.data.video_io import get_video_metadata
from src.product.console_api import create_app
from src.product.console_service import (
    ATTRIBUTION_BOUNDARY,
    aggregate_candidate_tracks,
    aggregate_window_output,
    build_action_plan,
    build_attribution,
    build_report_markdown,
    build_report_payload,
    build_report_pdf,
    case_from_preflight,
    case_from_runtime_failure,
    case_quality_snapshot,
    ensure_output_contract,
    localize_console_text,
    make_demo_cases,
    preflight_from_decode_error,
    video_preflight,
)
from src.product.console_store import ConsoleStore
from src.product.console_store import ScopedConsoleStore
from src.product.identity import local_identity
from src.product.adult_hr_mvp import AdultHRMVPConfig, build_release_windows, detector_is_release_eligible
from src.product.build_identity import path_fingerprint, source_build_identity
from src.vision.face_mesh_roi import resolve_face_landmarker_model_path, validate_face_landmarker_model


def test_real_video_report_describes_direct_and_api_execution_counts() -> None:
    assert execution_replay_description(1) == (
        "each fixture ran 1 time through the direct backend and once through the API"
    )
    assert execution_replay_description(2) == (
        "each fixture ran 2 times through the direct backend and once through the API"
    )


def test_non_release_never_publishes_hr() -> None:
    case = make_demo_cases()[1]
    case["released_hr_bpm"] = 99.0
    normalized = ensure_output_contract(case)
    assert normalized["decision"] == "review"
    assert normalized["released_hr_bpm"] is None


def test_release_requires_finite_hr() -> None:
    case = make_demo_cases()[0]
    case["released_hr_bpm"] = None
    with pytest.raises(ValueError, match="requires a finite"):
        ensure_output_contract(case)


def test_attribution_is_policy_bounded() -> None:
    attribution = build_attribution(make_demo_cases()[1])
    assert attribution["attribution_type"] == "evidence_and_policy_attribution"
    assert attribution["boundary"] == ATTRIBUTION_BOUNDARY
    assert attribution["primary_review_drivers"]
    assert all(item["source_field"] for item in attribution["all_factors"])


def test_store_persists_review_and_audit(tmp_path: Path) -> None:
    store = ConsoleStore(tmp_path / "console.db")
    case = make_demo_cases()[1]
    store.upsert_case(case, actor="test")
    reviews = store.list_reviews()
    assert len(reviews) == 1
    assert reviews[0]["status"] == "open"

    store.update_review(
        case["case_id"],
        status="closed",
        priority="high",
        assignee="reviewer-a",
        note="candidate conflict retained",
        resolution="close_without_release",
        actor="reviewer-a",
    )
    updated = store.list_reviews()[0]
    assert updated["status"] == "closed"
    assert updated["resolution"] == "close_without_release"
    event_types = {event["event_type"] for event in store.audit_events(case["case_id"])}
    assert {"case.created", "review.updated"}.issubset(event_types)


def test_store_isolates_cases_reviews_and_audit_by_organization(tmp_path: Path) -> None:
    base = ConsoleStore(tmp_path / "tenant-console.db")
    alpha = ScopedConsoleStore(
        base,
        local_identity(organization_id="org-alpha", user_id="alpha-reviewer"),
    )
    beta = ScopedConsoleStore(
        base,
        local_identity(organization_id="org-beta", user_id="beta-reviewer"),
    )
    case = make_demo_cases()[1]

    alpha.upsert_case(case)

    assert [item["case_id"] for item in alpha.list_cases()] == [case["case_id"]]
    assert beta.list_cases() == []
    assert beta.get_case(case["case_id"]) is None
    assert len(alpha.list_reviews()) == 1
    assert beta.list_reviews() == []
    assert alpha.audit_events(case["case_id"])
    assert beta.audit_events(case["case_id"]) == []

    with pytest.raises(PermissionError, match="another organization"):
        beta.upsert_case(case)


def test_participant_consent_and_report_versions_are_tenant_scoped(tmp_path: Path) -> None:
    base = ConsoleStore(tmp_path / "governance-console.db")
    alpha = ScopedConsoleStore(
        base,
        local_identity(organization_id="org-alpha", user_id="alpha-operator"),
    )
    beta = ScopedConsoleStore(
        base,
        local_identity(organization_id="org-beta", user_id="beta-operator"),
    )
    participant = alpha.upsert_participant(pseudonym="P-001", study_id="trial-a")
    consent = alpha.record_consent(
        participant_id=participant["participant_id"],
        purpose="workflow_validation",
        document_version="consent-v1",
        details={"raw_video_policy": "delete_after_analysis"},
    )
    assert consent["status"] == "active"
    assert alpha.active_consent(
        participant_id=participant["participant_id"],
        purpose="workflow_validation",
    )["consent_id"] == consent["consent_id"]
    assert beta.get_participant(participant["participant_id"]) is None

    case = make_demo_cases()[0]
    case["participant_id"] = participant["participant_id"]
    case["study_id"] = "trial-a"
    alpha.upsert_case(case)
    report = alpha.save_report_version(
        case_id=case["case_id"],
        report_sha256="a" * 64,
        audience="reviewer",
        language="en",
        payload={"case": {"case_id": case["case_id"]}},
        narrative={},
    )
    assert report["status"] == "draft"
    approved = alpha.approve_report_version(report["report_id"])
    assert approved["status"] == "approved"
    assert beta.list_report_versions(case["case_id"]) == []
    with pytest.raises(ValueError, match="Only a draft"):
        alpha.approve_report_version(report["report_id"])


def test_report_is_valid_pdf(tmp_path: Path) -> None:
    store = ConsoleStore(tmp_path / "console.db")
    case = make_demo_cases()[0]
    store.upsert_case(case, actor="test")
    payload = build_report_payload(case, audit_events=store.audit_events(case["case_id"]))
    pdf = build_report_pdf(payload)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 2000


def test_retake_report_distinguishes_gate_status_from_quality_score() -> None:
    case = make_demo_cases()[2]
    case["decision"] = "retake"
    case["quality_score"] = 1.0
    case["preflight"] = {"overall": "fail"}

    payload = build_report_payload(case)
    markdown = build_report_markdown(payload)

    assert "Acquisition gate: Not passed" in markdown
    assert "Quality score: 100%" in markdown


def test_preflight_retake_report_does_not_invent_candidate_or_illumination_failures() -> None:
    preflight = {
        "file_name": "short_release_fixture.avi",
        "fps": 30.0,
        "frame_count": 151,
        "width": 640,
        "height": 480,
        "duration_sec": 5.033,
        "brightness_mean": 203.801,
        "motion_mean": 1.912,
        "face_detection_rate": 1.0,
        "face_detector_available": True,
        "face_detector_backend": "mediapipe_tasks",
        "overall": "fail",
        "checks": [
            {
                "check": "duration",
                "value": 5.033,
                "unit": "s",
                "status": "fail",
                "action": "Record at least 8 seconds; 20-30 seconds is preferred.",
            },
            {"check": "frame rate", "value": 30.0, "unit": "fps", "status": "pass", "action": "No action required."},
            {"check": "resolution", "value": 480.0, "unit": "px short edge", "status": "pass", "action": "No action required."},
            {"check": "illumination", "value": 203.801, "unit": "luma", "status": "pass", "action": "No action required."},
            {"check": "motion", "value": 1.912, "unit": "mean frame delta", "status": "pass", "action": "No action required."},
            {"check": "face visibility", "value": 1.0, "unit": "fraction", "status": "pass", "action": "No action required."},
        ],
    }
    case = case_from_preflight(
        preflight,
        purpose="workflow_validation",
        retention_policy="delete_after_analysis",
    )
    payload = build_report_payload(case)
    plan = payload["action_plan"]
    markdown = build_report_markdown(payload)
    chinese = build_report_markdown(payload, language="zh")

    assert [item["source_field"] for item in plan["steps"]] == ["preflight.checks.duration"]
    assert "illumination | 203.801 luma | pass" in markdown
    assert "Candidate construction | not entered | not evaluated" in markdown
    assert "candidate count is therefore not an acquisition failure" in markdown
    assert "Illumination score: 41%" not in markdown
    assert "Candidate count | 0" not in markdown
    assert "候选构建 | 未进入 | 未评估" in chinese
    assert build_report_pdf(payload).startswith(b"%PDF")


def test_uploaded_review_uses_preflight_thresholds_and_exposes_landmark_fallback(tmp_path: Path) -> None:
    case = make_demo_cases()[1]
    case["illumination_score"] = 0.40
    case["preflight"] = {
        "overall": "pass",
        "checks": [
            {"check": "illumination", "value": 204.5, "unit": "luma", "status": "pass", "action": "No action required."},
            {"check": "motion", "value": 0.993, "unit": "mean frame delta", "status": "pass", "action": "No action required."},
            {"check": "face visibility", "value": 1.0, "unit": "fraction", "status": "pass", "action": "No action required."},
        ],
    }
    case["runtime_metadata"] = {"detector_backend": "static_roi_fallback"}

    plan = build_action_plan(case)
    attribution = build_attribution(case)

    illumination = next(item for item in plan["evidence"] if item["source_field"] == "preflight.checks.illumination")
    assert illumination["status"] == "within target"
    assert not [item for item in plan["steps"] if item["source_field"] == "illumination_score"]
    assert plan["steps"][0]["source_field"] == "runtime_metadata.detector_backend"
    illumination_factor = next(item for item in attribution["all_factors"] if item["factor"] == "Illumination")
    assert illumination_factor["status"] == "supports release"
    assert illumination_factor["source_field"] == "preflight.checks.illumination"


def test_face_landmarker_model_resolution_prefers_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"model")
    monkeypatch.setenv("MEDIAPIPE_FACE_LANDMARKER_TASK", str(model))

    assert resolve_face_landmarker_model_path() == model.resolve()


def test_face_landmarker_model_rejects_unpinned_asset(tmp_path: Path) -> None:
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"not-the-pinned-runtime-model")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_face_landmarker_model(model)


def test_runtime_asset_installer_accepts_a_verified_offline_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.task"
    source.write_bytes(b"pinned-test-model")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(runtime_assets, "FACE_LANDMARKER_MODEL_SHA256", expected)
    target = tmp_path / "runtime" / "face_landmarker.task"

    result = runtime_assets.install_model(target, source=source)

    assert result["status"] == "installed"
    assert target.read_bytes() == source.read_bytes()
    assert runtime_assets.install_model(target, source=source)["status"] == "already_installed"


def test_static_roi_candidate_evidence_cannot_enter_release_state() -> None:
    candidates = pd.DataFrame(
        [
            {
                "sample_id": "fixture_w000",
                "window_id": 0,
                "start_sec": 0.0,
                "end_sec": 20.0,
                "region": f"region_{index}",
                "method": "GREEN",
                "candidate_bpm": 72.0,
                "power": 1.0,
                "confidence": 0.9,
            }
            for index in range(6)
        ]
    )
    selected = pd.DataFrame(
        [
            {
                "sample_id": "fixture_w000",
                "cluster_bpm": 72.0,
                "roi_evidence_v2_score": 0.9,
                "roi_support": 6,
                "method_support": 1,
                "passes_roi_evidence_v2_gate": 1,
            }
        ]
    )

    windows = build_release_windows(
        candidates,
        selected,
        cfg=AdultHRMVPConfig(min_candidates=6),
        detection_rate=1.0,
        release_eligible_detector=False,
    )

    assert windows.loc[0, "decision"] == "review"
    assert pd.isna(windows.loc[0, "product_hr_bpm"])
    assert windows.loc[0, "refusal_reason"] == "face_landmark_backend_not_release_eligible"


def test_only_verified_task_detector_is_release_eligible() -> None:
    verified = SimpleNamespace(
        available=True,
        backend="mediapipe_face_landmarker_task",
        model_integrity_status="verified_pinned_sha256",
        model_sha256=runtime_assets.FACE_LANDMARKER_MODEL_SHA256,
    )
    legacy = SimpleNamespace(
        available=True,
        backend="mediapipe_face_mesh",
        model_integrity_status="not_applicable_builtin_face_mesh",
        model_sha256=None,
    )
    wrong_hash = SimpleNamespace(
        available=True,
        backend="mediapipe_face_landmarker_task",
        model_integrity_status="verified_pinned_sha256",
        model_sha256="0" * 64,
    )

    assert detector_is_release_eligible(verified)
    assert not detector_is_release_eligible(legacy)
    assert not detector_is_release_eligible(wrong_hash)


def test_report_payload_redacts_legacy_absolute_paths() -> None:
    case = make_demo_cases()[1]
    case["runtime_metadata"] = {
        "detector_model_path": r"G:\\private\\models\\face_landmarker.task",
        "detector_backend": "static_roi_fallback",
    }
    case["preflight"] = {"face_detector_source": "/home/research/private/face_landmarker.task"}
    case["technical_error"] = {
        "windows": r"OpenCV could not open video: G:\\private\\uploads\\clip.avi",
        "unc": r"Decoder failed at \\server\share\private\clip.avi",
        "unc_root": r"Decoder failed at \\server\share",
        "posix": "Decoder failed at /mnt/private/session/clip.avi",
        "posix_root": "Decoder failed at /tmp",
        "url": "Documentation: https://example.org/api/v1",
    }

    payload = build_report_payload(case)
    serialized = json.dumps(payload, ensure_ascii=False)
    markdown = build_report_markdown(payload)
    pdf = build_report_pdf(payload)
    csv = pd.DataFrame([payload["case"]]).to_csv(index=False)

    assert r"G:\\private" not in serialized
    assert "/home/research" not in serialized
    assert r"G:\\private" not in markdown
    assert "/home/research" not in markdown
    assert b"G:\\private" not in pdf
    assert r"G:\\private" not in csv
    assert "/home/research" not in csv
    assert r"G:\\private" not in serialized
    assert r"\server\share" not in serialized
    assert "/mnt/private" not in serialized
    assert "/tmp" not in serialized
    assert "https://example.org/api/v1" in serialized
    assert serialized.count("[local path redacted]") >= 5
    assert payload["case"]["runtime_metadata"]["detector_model_path"] == "face_landmarker.task"


def test_api_outbound_payloads_redact_paths_and_bind_upload_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upload_dir = tmp_path / "browser-uploads"
    monkeypatch.setenv("VITALSSIGHT_UPLOAD_DIR", str(upload_dir))
    app = create_app(tmp_path / "api-redaction.db", seed_demo=False)
    case = make_demo_cases()[1]
    case["technical_error"] = {
        "message": r"OpenCV could not open video: C:\\Users\\research\\private\\clip.avi"
    }
    app.state.store.upsert_case(case, actor="redaction-test")
    client = TestClient(app)

    health = client.get("/health").json()
    assert health["storage"]["upload_dir_fingerprint"] == path_fingerprint(upload_dir)
    assert upload_dir.is_dir()

    list_payload = client.get("/api/v1/cases").json()
    detail_payload = client.get(f"/api/v1/cases/{case['case_id']}").json()
    for payload in (list_payload, detail_payload):
        serialized = json.dumps(payload, ensure_ascii=False)
        assert r"C:\\Users\\research" not in serialized
        assert "[local path redacted]" in serialized


def test_source_build_identity_exposes_commit_and_tree() -> None:
    build = source_build_identity()

    assert len(build["commit"]) == 40
    assert len(build["tree"]) == 40
    assert build["source"] == "git"


def test_demo_quality_snapshots_cover_pass_warn_and_fail() -> None:
    cases = make_demo_cases()
    assert case_quality_snapshot(cases[0])["overall"] == "pass"
    assert case_quality_snapshot(cases[1])["overall"] == "warn"
    assert case_quality_snapshot(cases[2])["overall"] == "fail"


def test_chinese_report_localizes_decision_action_and_boundaries() -> None:
    payload = build_report_payload(make_demo_cases()[1])
    report = build_report_markdown(payload, language="zh")
    assert "决策: 复核" in report
    assert "候选心率: 111.0 BPM" in report
    assert "质量分数: 58%" in report
    assert "请在受试者保持静止时重新采集" in report
    assert "不构成因果解释" in report
    assert "不是诊断" in report


def test_report_includes_review_and_audit_records(tmp_path: Path) -> None:
    store = ConsoleStore(tmp_path / "console.db")
    case = make_demo_cases()[1]
    store.upsert_case(case, actor="operator")
    store.update_review(
        case["case_id"],
        status="in_review",
        priority="high",
        assignee="reviewer",
        note="Candidate conflict verified.",
        resolution="request_retake",
        actor="reviewer",
    )
    review = store.list_reviews(include_closed=True)[0]
    payload = build_report_payload(case, review=review, audit_events=store.audit_events(case["case_id"]))

    markdown = build_report_markdown(payload)
    pdf = build_report_pdf(payload)

    assert "Reviewer note: Candidate conflict verified." in markdown
    assert "review.updated" in markdown
    assert pdf.startswith(b"%PDF")


def test_action_plan_links_motion_trigger_to_threshold_and_verification() -> None:
    plan = build_action_plan(make_demo_cases()[1])
    face = next(item for item in plan["evidence"] if item["source_field"] == "face_coverage")
    motion = next(item for item in plan["evidence"] if item["source_field"] == "motion_score")
    motion_step = next(item for item in plan["steps"] if item["source_field"] == "motion_score")

    assert plan["decision"] == "review"
    assert motion["observed"] == "61%"
    assert motion["target"] == "<= 35%"
    assert motion["status"] == "triggered"
    assert "met the documented target" in face["reason"]
    assert "exceeded the policy limit" in motion["reason"]
    assert "remain still" in motion_step["action"]
    assert "no greater than 35%" in motion_step["verification"]


def test_action_plan_explains_low_light_and_candidate_shortage() -> None:
    plan = build_action_plan(make_demo_cases()[2])
    triggered = {item["source_field"]: item for item in plan["evidence"] if item["status"] == "triggered"}

    assert plan["decision"] == "retake"
    assert triggered["illumination_score"]["target"] == ">= 55%"
    assert triggered["candidate_count"]["target"] == ">= 3"
    assert any(item["source_field"] == "illumination_score" for item in plan["steps"])
    assert "do not force a result" in plan["escalation"]


def test_release_action_plan_preserves_evidence_and_research_boundary() -> None:
    plan = build_action_plan(make_demo_cases()[0])

    assert plan["decision"] == "release"
    assert not [item for item in plan["evidence"] if item["status"] == "triggered"]
    assert plan["steps"][0]["source_field"] == "decision"
    assert "research workflow" in plan["steps"][1]["action"]
    assert "not a clinical recommendation" in plan["boundary"]


def test_report_v2_contains_evidence_to_action_chain() -> None:
    payload = build_report_payload(make_demo_cases()[1])
    report = build_report_markdown(payload, language="en")
    chinese = build_report_markdown(payload, language="zh")

    assert payload["report_version"].endswith(".v2")
    assert payload["action_plan"]["evidence"]
    assert "## Evidence supporting the recommendation" in report
    assert "## Recommended workflow" in report
    assert "## Implementation provenance" in report
    assert "Model version" in report
    assert "Policy version" in report
    assert "| Motion | 61% | <= 35% | triggered |" in report
    assert "## 建议依据" in chinese
    assert "## 建议操作流程" in chinese
    assert "## 实现与运行溯源" in chinese


def test_window_aggregation_releases_only_a_stable_majority() -> None:
    windows = pd.DataFrame(
        [
            {"window_id": 0, "decision": "release", "product_hr_bpm": 75.1, "candidate_hr_bpm": 75.1},
            {"window_id": 1, "decision": "release", "product_hr_bpm": 60.1, "candidate_hr_bpm": 60.1},
            {"window_id": 2, "decision": "release", "product_hr_bpm": 71.8, "candidate_hr_bpm": 71.8},
        ]
    )

    result = aggregate_window_output(windows)

    assert result["decision"] == "release"
    assert result["released_hr_bpm"] == pytest.approx(73.45)
    assert result["stable_window_count"] == 2
    assert result["consistency_fraction"] == pytest.approx(2 / 3)


def test_window_aggregation_withholds_divergent_release_windows() -> None:
    windows = pd.DataFrame(
        [
            {"window_id": 0, "decision": "release", "product_hr_bpm": 112.6, "candidate_hr_bpm": 112.6},
            {"window_id": 1, "decision": "release", "product_hr_bpm": 90.1, "candidate_hr_bpm": 90.1},
            {"window_id": 2, "decision": "release", "product_hr_bpm": 59.8, "candidate_hr_bpm": 59.8},
        ]
    )

    result = aggregate_window_output(windows)

    assert result["decision"] == "review"
    assert result["released_hr_bpm"] is None
    assert result["selected_candidate_hr_bpm"] == pytest.approx(90.1)
    assert result["review_reason"] == "inter_window_hr_disagreement"
    assert result["consistency_fraction"] == pytest.approx(1 / 3)


def test_candidate_track_aggregation_releases_one_dominant_track() -> None:
    windows = pd.DataFrame(
        [
            {"sample_id": "s0", "window_id": 0, "decision": "release", "product_hr_bpm": 75.0, "candidate_hr_bpm": 75.0},
            {"sample_id": "s1", "window_id": 1, "decision": "release", "product_hr_bpm": 76.0, "candidate_hr_bpm": 76.0},
            {"sample_id": "s2", "window_id": 2, "decision": "release", "product_hr_bpm": 74.0, "candidate_hr_bpm": 74.0},
        ]
    )
    clusters = pd.DataFrame(
        [
            {"sample_id": "s0", "cluster_bpm": 75.0, "roi_evidence_v2_score": 0.90, "passes_roi_evidence_v2_gate": 1},
            {"sample_id": "s0", "cluster_bpm": 120.0, "roi_evidence_v2_score": 0.40, "passes_roi_evidence_v2_gate": 0},
            {"sample_id": "s1", "cluster_bpm": 76.0, "roi_evidence_v2_score": 0.80, "passes_roi_evidence_v2_gate": 1},
            {"sample_id": "s1", "cluster_bpm": 119.0, "roi_evidence_v2_score": 0.35, "passes_roi_evidence_v2_gate": 0},
            {"sample_id": "s2", "cluster_bpm": 74.0, "roi_evidence_v2_score": 0.95, "passes_roi_evidence_v2_gate": 1},
            {"sample_id": "s2", "cluster_bpm": 121.0, "roi_evidence_v2_score": 0.30, "passes_roi_evidence_v2_gate": 0},
        ]
    )

    result = aggregate_candidate_tracks(windows, clusters)

    assert result["decision"] == "release"
    assert result["released_hr_bpm"] == pytest.approx(75.0)
    assert result["competing_track_count"] == 0
    assert result["consistency_fraction"] == pytest.approx(1.0)


def test_candidate_track_aggregation_withholds_similarly_supported_competitor() -> None:
    windows = pd.DataFrame(
        [
            {"sample_id": "s0", "window_id": 0, "decision": "release", "product_hr_bpm": 80.0, "candidate_hr_bpm": 80.0},
            {"sample_id": "s1", "window_id": 1, "decision": "release", "product_hr_bpm": 81.0, "candidate_hr_bpm": 81.0},
            {"sample_id": "s2", "window_id": 2, "decision": "release", "product_hr_bpm": 79.0, "candidate_hr_bpm": 79.0},
        ]
    )
    clusters = pd.DataFrame(
        [
            {"sample_id": "s0", "cluster_bpm": 80.0, "roi_evidence_v2_score": 0.90, "passes_roi_evidence_v2_gate": 1},
            {"sample_id": "s0", "cluster_bpm": 102.0, "roi_evidence_v2_score": 0.80, "passes_roi_evidence_v2_gate": 1},
            {"sample_id": "s1", "cluster_bpm": 81.0, "roi_evidence_v2_score": 1.00, "passes_roi_evidence_v2_gate": 1},
            {"sample_id": "s1", "cluster_bpm": 101.0, "roi_evidence_v2_score": 0.85, "passes_roi_evidence_v2_gate": 1},
            {"sample_id": "s2", "cluster_bpm": 79.0, "roi_evidence_v2_score": 0.95, "passes_roi_evidence_v2_gate": 1},
            {"sample_id": "s2", "cluster_bpm": 103.0, "roi_evidence_v2_score": 0.82, "passes_roi_evidence_v2_gate": 1},
        ]
    )

    result = aggregate_candidate_tracks(windows, clusters)

    assert result["decision"] == "review"
    assert result["released_hr_bpm"] is None
    assert result["review_reason"] == "competing_cross_window_candidate_tracks"
    assert result["competing_track_count"] >= 1


def test_candidate_track_aggregation_does_not_promote_lower_support_branch() -> None:
    windows = pd.DataFrame(
        [
            {"sample_id": "s0", "window_id": 0, "decision": "release", "product_hr_bpm": 75.0, "candidate_hr_bpm": 75.0},
            {"sample_id": "s1", "window_id": 1, "decision": "release", "product_hr_bpm": 60.0, "candidate_hr_bpm": 60.0},
            {"sample_id": "s2", "window_id": 2, "decision": "release", "product_hr_bpm": 72.0, "candidate_hr_bpm": 72.0},
        ]
    )
    clusters = pd.DataFrame(
        [
            {"sample_id": "s0", "cluster_bpm": 75.0, "roi_evidence_v2_score": 1.00, "passes_roi_evidence_v2_gate": 1},
            {"sample_id": "s0", "cluster_bpm": 60.0, "roi_evidence_v2_score": 0.80, "passes_roi_evidence_v2_gate": 0},
            {"sample_id": "s1", "cluster_bpm": 76.0, "roi_evidence_v2_score": 0.90, "passes_roi_evidence_v2_gate": 1},
            {"sample_id": "s1", "cluster_bpm": 60.0, "roi_evidence_v2_score": 1.00, "passes_roi_evidence_v2_gate": 1},
            {"sample_id": "s2", "cluster_bpm": 72.0, "roi_evidence_v2_score": 0.95, "passes_roi_evidence_v2_gate": 1},
            {"sample_id": "s2", "cluster_bpm": 61.0, "roi_evidence_v2_score": 0.70, "passes_roi_evidence_v2_gate": 0},
        ]
    )

    result = aggregate_candidate_tracks(windows, clusters)

    assert result["decision"] == "release"
    assert result["released_hr_bpm"] == pytest.approx(75.0)
    assert result["competing_track_count"] == 0


def test_candidate_track_aggregation_counts_majority_coverage_competitor() -> None:
    windows = pd.DataFrame(
        [
            {"sample_id": "s0", "window_id": 0, "decision": "release", "product_hr_bpm": 80.0, "candidate_hr_bpm": 80.0},
            {"sample_id": "s1", "window_id": 1, "decision": "release", "product_hr_bpm": 81.0, "candidate_hr_bpm": 81.0},
            {"sample_id": "s2", "window_id": 2, "decision": "release", "product_hr_bpm": 79.0, "candidate_hr_bpm": 79.0},
        ]
    )
    clusters = pd.DataFrame(
        [
            {"sample_id": "s0", "cluster_bpm": 80.0, "roi_evidence_v2_score": 0.90, "passes_roi_evidence_v2_gate": 1},
            {"sample_id": "s0", "cluster_bpm": 102.0, "roi_evidence_v2_score": 0.88, "passes_roi_evidence_v2_gate": 1},
            {"sample_id": "s1", "cluster_bpm": 81.0, "roi_evidence_v2_score": 0.90, "passes_roi_evidence_v2_gate": 1},
            {"sample_id": "s1", "cluster_bpm": 101.0, "roi_evidence_v2_score": 0.88, "passes_roi_evidence_v2_gate": 1},
            {"sample_id": "s2", "cluster_bpm": 79.0, "roi_evidence_v2_score": 0.90, "passes_roi_evidence_v2_gate": 1},
        ]
    )

    result = aggregate_candidate_tracks(windows, clusters)

    assert result["decision"] == "review"
    assert result["review_reason"] == "competing_cross_window_candidate_tracks"
    assert result["competing_track_count"] >= 1


def test_candidate_track_aggregation_handles_empty_candidates() -> None:
    windows = pd.DataFrame(
        [{"sample_id": "s0", "window_id": 0, "decision": "review", "refusal_reason": "no_selected_cluster"}]
    )

    result = aggregate_candidate_tracks(windows, pd.DataFrame())

    assert result["decision"] == "review"
    assert result["released_hr_bpm"] is None
    assert result["selected_candidate_hr_bpm"] is None
    assert result["review_reason"] == "no_candidate_generated"


def test_window_aggregation_uses_failed_window_reason() -> None:
    windows = pd.DataFrame(
        [
            {"window_id": 0, "decision": "release", "product_hr_bpm": 75.0, "candidate_hr_bpm": 75.0, "refusal_reason": "accepted"},
            {"window_id": 1, "decision": "review", "product_hr_bpm": float("nan"), "candidate_hr_bpm": 76.0, "refusal_reason": "low_face_detection_rate"},
            {"window_id": 2, "decision": "release", "product_hr_bpm": 74.0, "candidate_hr_bpm": 74.0, "refusal_reason": "accepted"},
        ]
    )

    result = aggregate_window_output(windows)

    assert result["decision"] == "review"
    assert result["review_reason"] == "low_face_detection_rate"


def test_candidate_track_aggregation_rejects_excessive_total_spread() -> None:
    windows = pd.DataFrame(
        [
            {"sample_id": "s0", "window_id": 0, "decision": "release", "product_hr_bpm": 70.0, "candidate_hr_bpm": 70.0},
            {"sample_id": "s1", "window_id": 1, "decision": "release", "product_hr_bpm": 76.0, "candidate_hr_bpm": 76.0},
            {"sample_id": "s2", "window_id": 2, "decision": "release", "product_hr_bpm": 82.0, "candidate_hr_bpm": 82.0},
        ]
    )
    clusters = pd.DataFrame(
        [
            {"sample_id": "s0", "cluster_bpm": 70.0, "roi_evidence_v2_score": 0.90, "passes_roi_evidence_v2_gate": 1},
            {"sample_id": "s1", "cluster_bpm": 76.0, "roi_evidence_v2_score": 0.90, "passes_roi_evidence_v2_gate": 1},
            {"sample_id": "s2", "cluster_bpm": 82.0, "roi_evidence_v2_score": 0.90, "passes_roi_evidence_v2_gate": 1},
        ]
    )

    result = aggregate_candidate_tracks(windows, clusters, tolerance_bpm=6.0)

    assert result["decision"] == "review"
    assert result["released_hr_bpm"] is None
    assert result["review_reason"] == "no_stable_cross_window_candidate_track"


def test_unavailable_harmonic_evidence_is_neutral() -> None:
    case = make_demo_cases()[0]
    case["harmonic_risk"] = None

    attribution = build_attribution(case)
    harmonic = next(item for item in attribution["all_factors"] if item["factor"] == "Harmonic ambiguity")

    assert harmonic["status"] == "not available"
    assert harmonic["direction"] == 0


def test_sidebar_restore_control_is_not_hidden_by_product_css() -> None:
    source = (Path(__file__).resolve().parents[1] / "app" / "product_console.py").read_text(encoding="utf-8")

    assert '[data-testid="stSidebarCollapsedControl"]' in source
    assert '[data-testid="stExpandSidebarButton"]' in source
    assert '[data-testid="stToolbar"] { display: flex !important; }' in source
    assert '[data-testid="stToolbar"] { display: none; }' not in source


def test_workspace_navigation_resets_the_main_scroll_position() -> None:
    source = (Path(__file__).resolve().parents[1] / "app" / "product_console.py").read_text(encoding="utf-8")

    assert 'vs_rendered_section' in source
    assert '"vs_rendered_section": "Overview"' in source
    assert 'vs_navigation_nonce' in source
    assert 'const navigationNonce = __NAVIGATION_NONCE__' in source
    assert 'querySelector(\'[data-testid="stMain"]\')' in source
    assert "main.scrollTo({ top: 0, left: 0" in source
    assert "closeMobileSidebar" in source
    assert "window.parent.innerWidth > 900" in source
    assert "sidebar.getAttribute('aria-expanded') ?? sidebar.getAttribute('aria')" in source
    assert "collapse.innerText.includes('keyboard_double_arrow_left')" in source
    assert "let sidebarCloseRequested = false" in source
    assert "window.setInterval" not in source
    assert "}, 150);" in source
    assert "st.iframe(" in source
    assert "streamlit.components.v1" not in source


def test_assessment_reset_rebuilds_the_upload_widget() -> None:
    source = (Path(__file__).resolve().parents[1] / "app" / "product_console.py").read_text(encoding="utf-8")

    assert '"vs_upload_widget_version": 0' in source
    assert 'key=f"vs_video_upload_{st.session_state[\'vs_upload_widget_version\']}"' in source
    assert source.count("_reset_upload_widget()") >= 2


def test_assessment_controls_preserve_canonical_state_across_languages() -> None:
    source = (Path(__file__).resolve().parents[1] / "app" / "product_console.py").read_text(encoding="utf-8")

    assert 'def _sync_language() -> None:' in source
    assert 'def _sync_assessment_control(field: str, widget_key: str) -> None:' in source
    assert 'language_suffix = "zh" if _is_zh() else "en"' in source
    for field in ("vs_purpose", "vs_consent", "vs_retention", "vs_source"):
        assert f'"{field}"' in source
        assert f'f"{field}_control_' in source


def test_assessment_exposes_progress_privacy_and_output_contracts() -> None:
    source = (Path(__file__).resolve().parents[1] / "app" / "product_console.py").read_text(encoding="utf-8")

    assert "vs-processing-contract" in source
    assert "Raw video stays local and is deleted after analysis in the recommended mode." in source
    assert "Review and retake states never publish HR." in source
    assert '"done": _ui("Complete",' in source
    assert '"current": _ui("Current",' in source
    assert '"pending": _ui("Next",' in source
    assert "vs-step-strip div.current" in source
    assert "vs-step-strip div.done" in source


def test_retake_summary_presents_the_acquisition_gate_separately() -> None:
    source = (Path(__file__).resolve().parents[1] / "app" / "product_console.py").read_text(encoding="utf-8")

    assert 'str(case.get("decision")) == "retake" or overall == "fail"' in source
    assert "_ui('Acquisition gate','采集门控')" in source


def test_chinese_localization_handles_joined_preflight_actions() -> None:
    joined = (
        "Record at least 8 seconds; 20-30 seconds is preferred.; "
        "Use a camera or file with at least 15 fps.; "
        "Use at least 320x240 video; 480p or higher is preferred."
    )

    localized = localize_console_text(joined, language="zh")

    assert "至少录制 8 秒" in localized
    assert "至少为 15 fps" in localized
    assert "至少使用 320x240 视频" in localized
    assert "Record at least" not in localized
    assert "。;" not in localized


def test_runtime_failure_is_reviewed_without_publishing_hr() -> None:
    preflight = {
        "file_name": "quality_passed.mp4",
        "overall": "pass",
        "brightness_mean": 120.0,
        "motion_mean": 3.0,
        "face_detection_rate": 0.9,
        "checks": [],
    }
    case = case_from_runtime_failure(
        preflight,
        RuntimeError("backend unavailable"),
        purpose="workflow_validation",
        retention_policy="delete_after_analysis",
    )
    assert case["decision"] == "review"
    assert case["released_hr_bpm"] is None
    assert case["technical_error"]["type"] == "RuntimeError"
    assert "do not report HR" in case["recommended_action"]


def test_decode_error_becomes_a_quality_failure() -> None:
    preflight = preflight_from_decode_error("broken.mp4", ValueError("bad container"))
    assert preflight["overall"] == "fail"
    assert preflight["checks"][0]["check"] == "file readability"
    assert preflight["technical_error"]["type"] == "ValueError"


def test_api_endpoints_share_review_contract(tmp_path: Path) -> None:
    app = create_app(tmp_path / "api.db", seed_demo=True)
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    cases = client.get("/api/v1/cases").json()["items"]
    review_case = next(item for item in cases if item["decision"] == "review")
    response = client.put(
        f"/api/v1/reviews/{review_case['case_id']}",
        json={
            "status": "in_review",
            "priority": "high",
            "assignee": "api-reviewer",
            "note": "inspection started",
            "resolution": "",
            "actor": "api-reviewer",
        },
    )
    assert response.status_code == 200
    assert response.json()["item"]["assignee"] == "api-reviewer"
    report = client.get(f"/api/v1/cases/{review_case['case_id']}/report?format=pdf")
    assert report.status_code == 200
    assert report.content.startswith(b"%PDF")


def test_video_preflight_marks_short_blank_video_for_retake(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "short_blank.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (320, 240))
    assert writer.isOpened()
    for _ in range(30):
        writer.write(np.full((240, 320, 3), 8, dtype=np.uint8))
    writer.release()
    result = video_preflight(path, sample_frames=12)
    assert result["overall"] == "fail"
    failed = {item["check"] for item in result["checks"] if item["status"] == "fail"}
    assert "duration" in failed
    assert "face visibility" in failed


def test_video_preflight_counts_frames_when_container_metadata_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "missing_frame_count.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (320, 240))
    assert writer.isOpened()
    for _ in range(45):
        writer.write(np.full((240, 320, 3), 96, dtype=np.uint8))
    writer.release()

    real_capture = cv2.VideoCapture

    class MissingCountCapture:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._capture = real_capture(*args, **kwargs)

        def get(self, property_id: int) -> float:
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return 0.0
            return float(self._capture.get(property_id))

        def __getattr__(self, name: str) -> object:
            return getattr(self._capture, name)

    monkeypatch.setattr(cv2, "VideoCapture", MissingCountCapture)
    metadata = get_video_metadata(path)
    result = video_preflight(path, sample_frames=8)

    assert metadata.frame_count == 45
    assert metadata.duration_sec == pytest.approx(3.0)
    assert result["frame_count"] == 45
    assert result["duration_sec"] == pytest.approx(3.0)
    assert result["frame_count_source"] == "decoded_fallback"


def test_openapi_schema_is_serializable(tmp_path: Path) -> None:
    schema = create_app(tmp_path / "schema.db", seed_demo=False).openapi()
    encoded = json.dumps(schema, allow_nan=False)
    assert "/api/v1/cases" in encoded
    assert "/api/v1/assessments/video" in encoded
    assert "/api/v1/reviews/{case_id}" in encoded


def test_real_video_validation_manifest_covers_all_output_states() -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "validation" / "real_video_case_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest["cases"]

    assert {case["expected_decision"] for case in cases} == {"release", "review", "retake"}
    assert sum(case["expected_decision"] == "release" for case in cases) >= 1
    assert sum(case["fixture_kind"] == "provider_dataset_original" for case in cases) >= 5
    assert all(len(case["sha256"]) == 64 for case in cases)
    assert all(case["fixture_kind"] for case in cases)
    assert all("provenance" in case for case in cases)
    assert all("reference" in case for case in cases if case["fixture_kind"] == "provider_dataset_original")


def test_video_assessment_api_saves_retake_and_deletes_raw_video(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    video_path = tmp_path / "short_blank.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (320, 240))
    assert writer.isOpened()
    for _ in range(30):
        writer.write(np.full((240, 320, 3), 8, dtype=np.uint8))
    writer.release()

    upload_dir = tmp_path / "api-uploads"
    old_upload_dir = os.environ.get("VITALSSIGHT_UPLOAD_DIR")
    os.environ["VITALSSIGHT_UPLOAD_DIR"] = str(upload_dir)
    try:
        app = create_app(tmp_path / "assessment.db", seed_demo=False)
    finally:
        if old_upload_dir is None:
            os.environ.pop("VITALSSIGHT_UPLOAD_DIR", None)
        else:
            os.environ["VITALSSIGHT_UPLOAD_DIR"] = old_upload_dir
    client = TestClient(app)

    with video_path.open("rb") as handle:
        response = client.post(
            "/api/v1/assessments/video",
            data={
                "consent_recorded": "true",
                "purpose": "workflow_validation",
                "retention_policy": "delete_after_analysis",
                "actor": "api-test",
            },
            files={"file": (video_path.name, handle, "video/x-msvideo")},
        )

    assert response.status_code == 201
    payload = response.json()
    assert payload["item"]["decision"] == "retake"
    assert payload["item"]["released_hr_bpm"] is None
    assert payload["raw_video_retained"] is False
    assert not any(path.is_file() for path in upload_dir.rglob("*"))
    stored = client.get("/api/v1/cases").json()["items"]
    assert stored[0]["case_id"] == payload["item"]["case_id"]
