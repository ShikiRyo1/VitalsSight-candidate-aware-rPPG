from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.product.console_api import create_app
from src.product.console_service import (
    ATTRIBUTION_BOUNDARY,
    build_attribution,
    build_report_markdown,
    build_report_payload,
    build_report_pdf,
    case_from_runtime_failure,
    case_quality_snapshot,
    ensure_output_contract,
    localize_console_text,
    make_demo_cases,
    preflight_from_decode_error,
    video_preflight,
)
from src.product.console_store import ConsoleStore


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


def test_report_is_valid_pdf(tmp_path: Path) -> None:
    store = ConsoleStore(tmp_path / "console.db")
    case = make_demo_cases()[0]
    store.upsert_case(case, actor="test")
    payload = build_report_payload(case, audit_events=store.audit_events(case["case_id"]))
    pdf = build_report_pdf(payload)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 2000


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


def test_openapi_schema_is_serializable(tmp_path: Path) -> None:
    schema = create_app(tmp_path / "schema.db", seed_demo=False).openapi()
    encoded = json.dumps(schema, allow_nan=False)
    assert "/api/v1/cases" in encoded
    assert "/api/v1/assessments/video" in encoded
    assert "/api/v1/reviews/{case_id}" in encoded


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
