from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import jwt
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.assistant.provider import UnavailableProvider
from src.product.auth import AuthSettings
from src.product.console_api import create_app
from src.product.console_service import make_demo_cases
from src.product.console_store import ScopedConsoleStore
from src.product.identity import local_identity


ISSUER = "https://identity.example.test/realms/vitalssight"
AUDIENCE = "vitalssight-api"
SECRET = "controlled-trial-api-test-secret-not-for-production"


def _settings() -> AuthSettings:
    return AuthSettings(
        mode="required",
        issuer=ISSUER,
        audience=AUDIENCE,
        client_id="vitalssight",
        shared_secret=SECRET,
        algorithms=("HS256",),
        leeway_seconds=0,
    )


def _token(
    *,
    subject: str,
    organization: str,
    roles: list[str],
    participant_id: str = "",
) -> str:
    now = datetime.now(UTC)
    claims: dict[str, object] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": subject,
        "email": f"{subject}@example.test",
        "name": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=5)).timestamp()),
        "organization": {organization: {}},
        "realm_access": {"roles": roles},
    }
    if participant_id:
        claims["participant_id"] = participant_id
    return jwt.encode(claims, SECRET, algorithm="HS256")


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_required_auth_api_isolates_cases_and_reviews(tmp_path: Path) -> None:
    app = create_app(
        tmp_path / "tenant-api.db",
        seed_demo=False,
        auth_settings=_settings(),
    )
    alpha_store = ScopedConsoleStore(
        app.state.store,
        local_identity(
            organization_id="org-alpha",
            user_id="alpha-seed",
            roles={"reviewer"},
        ),
    )
    case = make_demo_cases()[1]
    alpha_store.upsert_case(case)
    client = TestClient(app)
    alpha = _token(subject="alpha-reviewer", organization="org-alpha", roles=["reviewer"])
    beta = _token(subject="beta-reviewer", organization="org-beta", roles=["reviewer"])

    assert client.get("/api/v1/cases").status_code == 401
    assert client.get("/api/v1/cases", headers=_headers(alpha)).json()["count"] == 1
    assert client.get("/api/v1/cases", headers=_headers(beta)).json()["count"] == 0
    assert client.get(
        f"/api/v1/cases/{case['case_id']}", headers=_headers(beta)
    ).status_code == 404
    assert client.put(
        f"/api/v1/reviews/{case['case_id']}",
        headers=_headers(beta),
        json={
            "status": "closed",
            "priority": "high",
            "assignee": "beta-reviewer",
            "note": "cross tenant attempt",
            "resolution": "close_without_release",
            "actor": "forged-actor",
        },
    ).status_code == 404

    updated = client.put(
        f"/api/v1/reviews/{case['case_id']}",
        headers=_headers(alpha),
        json={
            "status": "in_review",
            "priority": "high",
            "assignee": "alpha-reviewer",
            "note": "tenant-scoped review",
            "resolution": "",
            "actor": "forged-actor",
        },
    )
    assert updated.status_code == 200
    events = alpha_store.audit_events(case["case_id"])
    assert events[0]["actor"] == "alpha-reviewer"
    assert events[0]["actor"] != "forged-actor"


def test_participant_consent_controls_required_auth_video_assessment(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    app = create_app(
        tmp_path / "consent-api.db",
        seed_demo=False,
        auth_settings=_settings(),
    )
    client = TestClient(app)
    operator = _token(subject="trial-operator", organization="hospital-a", roles=["operator"])

    created = client.post(
        "/api/v1/participants",
        headers=_headers(operator),
        json={"pseudonym": "P-001", "study_id": "trial-a"},
    )
    assert created.status_code == 201
    participant_id = created.json()["item"]["participant_id"]

    video_path = tmp_path / "short_blank.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (320, 240))
    assert writer.isOpened()
    for _ in range(30):
        writer.write(np.full((240, 320, 3), 8, dtype=np.uint8))
    writer.release()

    def upload() -> object:
        with video_path.open("rb") as handle:
            return client.post(
                "/api/v1/assessments/video",
                headers=_headers(operator),
                data={
                    "consent_recorded": "true",
                    "purpose": "workflow_validation",
                    "retention_policy": "delete_after_analysis",
                    "participant_id": participant_id,
                },
                files={"file": (video_path.name, handle, "video/x-msvideo")},
            )

    assert upload().status_code == 422
    consent = client.post(
        f"/api/v1/participants/{participant_id}/consents",
        headers=_headers(operator),
        json={
            "purpose": "workflow_validation",
            "document_version": "consent-v1",
            "details": {"raw_video_policy": "delete_after_analysis"},
        },
    )
    assert consent.status_code == 201
    consent_id = consent.json()["item"]["consent_id"]

    assessed = upload()
    assert assessed.status_code == 201
    case = assessed.json()["item"]
    assert case["decision"] == "retake"
    assert case["participant_id"] == participant_id
    assert case["study_id"] == "trial-a"
    assert case["consent"]["document_version"] == "consent-v1"

    participant = _token(
        subject="participant-p001",
        organization="hospital-a",
        roles=["participant"],
        participant_id=participant_id,
    )
    other_participant = _token(
        subject="participant-p002",
        organization="hospital-a",
        roles=["participant"],
        participant_id="pt-not-p001",
    )
    own_cases = client.get("/api/v1/cases", headers=_headers(participant))
    assert own_cases.status_code == 200
    assert [item["case_id"] for item in own_cases.json()["items"]] == [case["case_id"]]
    assert client.get(
        f"/api/v1/cases/{case['case_id']}", headers=_headers(other_participant)
    ).status_code == 404

    withdrawn = client.post(
        f"/api/v1/participants/{participant_id}/consents/{consent_id}/withdraw",
        headers=_headers(operator),
    )
    assert withdrawn.status_code == 200
    assert upload().status_code == 422


def test_governed_report_versions_fhir_and_approval_are_tenant_scoped(tmp_path: Path) -> None:
    app = create_app(
        tmp_path / "report-api.db",
        seed_demo=False,
        auth_settings=_settings(),
        assistant_provider=UnavailableProvider(),
    )
    alpha_store = ScopedConsoleStore(
        app.state.store,
        local_identity(
            organization_id="org-alpha",
            user_id="alpha-seed",
            roles={"reviewer"},
        ),
    )
    case = make_demo_cases()[0]
    alpha_store.upsert_case(case)
    client = TestClient(app)
    alpha = _token(subject="alpha-reviewer", organization="org-alpha", roles=["reviewer"])
    beta = _token(subject="beta-reviewer", organization="org-beta", roles=["reviewer"])

    fhir = client.get(
        f"/api/v1/cases/{case['case_id']}/report?format=fhir&audience=reviewer",
        headers=_headers(alpha),
    )
    assert fhir.status_code == 200
    assert fhir.headers["content-type"].startswith("application/fhir+json")
    assert fhir.json()["resourceType"] == "Bundle"
    assert any(
        entry["resource"]["resourceType"] == "Observation"
        for entry in fhir.json()["entry"]
    )

    created = client.post(
        f"/api/v1/cases/{case['case_id']}/report-versions",
        headers=_headers(alpha),
        json={"audience": "reviewer", "language": "en"},
    )
    assert created.status_code == 201
    report = created.json()["item"]
    assert report["status"] == "draft"
    assert len(report["report_sha256"]) == 64
    assert report["narrative"]["status"] == "draft"
    assert report["narrative"]["validation"]["passed"] is True
    assert client.get(
        f"/api/v1/cases/{case['case_id']}/report-versions",
        headers=_headers(beta),
    ).status_code == 404
    assert client.post(
        f"/api/v1/report-versions/{report['report_id']}/approve",
        headers=_headers(beta),
    ).status_code == 404

    approved = client.post(
        f"/api/v1/report-versions/{report['report_id']}/approve",
        headers=_headers(alpha),
    )
    assert approved.status_code == 200
    assert approved.json()["item"]["status"] == "approved"
    assert client.post(
        f"/api/v1/report-versions/{report['report_id']}/approve",
        headers=_headers(alpha),
    ).status_code == 409
