from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import jwt
from fastapi.testclient import TestClient

from src.assistant.provider import UnavailableProvider
from src.product.auth import AuthSettings
from src.product.console_api import create_app
from src.product.runtime_readiness import FACE_MODEL_SHA256
from src.vision.face_mesh_roi import FACE_LANDMARKER_MODEL_SHA256


ISSUER = "https://test-identity.invalid"
AUDIENCE = "vitalsight-readiness-test"
SECRET = "readiness-test-only-secret-not-for-production"


def token(role: str, organization: str = "org-alpha") -> dict[str, str]:
    now = datetime.now(UTC)
    claims = {
        "iss": ISSUER, "aud": AUDIENCE, "sub": "readiness-operator", "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=5)).timestamp()),
        "organization": {organization: {}}, "realm_access": {"roles": [role]},
    }
    return {"Authorization": "Bearer " + jwt.encode(claims, SECRET, algorithm="HS256")}


def test_preflight_model_pin_matches_runtime():
    assert FACE_MODEL_SHA256 == FACE_LANDMARKER_MODEL_SHA256


def test_readiness_preserves_auth_and_role_boundaries(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.product.runtime_readiness._loaded_mediapipe_backend", lambda: "tasks")
    monkeypatch.setenv("MEDIAPIPE_FACE_LANDMARKER_TASK", str(tmp_path / "missing-private-model.task"))
    monkeypatch.setenv("VITALSSIGHT_UPLOAD_DIR", str(tmp_path / "private-upload-root"))
    settings = AuthSettings(mode="required", issuer=ISSUER, audience=AUDIENCE, shared_secret=SECRET, algorithms=("HS256",), leeway_seconds=0)
    app = create_app(tmp_path / "state.db", seed_demo=False, assistant_provider=UnavailableProvider(), auth_settings=settings)
    client = TestClient(app)
    assert client.get("/api/v1/runtime/readiness").status_code == 401
    assert client.get("/api/v1/runtime/readiness", headers=token("unrecognized-role")).status_code in {401, 403}
    response = client.get("/api/v1/runtime/readiness", headers=token("operator"))
    assert response.status_code == 200
    body = response.json()
    assert body["auth_mode"] == "required"
    assert body["measurement_release_authorized"] is False
    assert body["clinical_validity_established"] is False
    assert body["participant_data_accessed"] is False
    assert body["status"] == "attention_required"
    rendered = json.dumps(body)
    for value in (SECRET, ISSUER, AUDIENCE, "private-upload-root", "missing-private-model.task", str(tmp_path)):
        assert value not in rendered
    assert client.get("/api/v1/cases", headers=token("operator")).json()["count"] == 0
    alpha_events = client.get("/api/v1/organization/access-events", headers=token("auditor")).json()["items"]
    beta_events = client.get("/api/v1/organization/access-events", headers=token("auditor", "org-beta")).json()["items"]
    assert any(event["action"] == "runtime.readiness" for event in alpha_events)
    assert not any(event["action"] == "runtime.readiness" for event in beta_events)


def test_invalid_limit_blocks_intake_without_startup_crash(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VITALSSIGHT_MAX_UPLOAD_BYTES", "not-a-number")
    monkeypatch.setenv("VITALSSIGHT_UPLOAD_DIR", str(tmp_path / "uploads"))
    app = create_app(tmp_path / "state.db", seed_demo=False, assistant_provider=UnavailableProvider(), auth_settings=AuthSettings())
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/runtime/readiness").json()["upload_policy"]["configuration_valid"] is False
    response = client.post("/api/v1/assessments/video", data={"consent_recorded": "true"}, files={"file": ("sample.mp4", b"not-video", "video/mp4")})
    assert response.status_code == 503
    assert "VITALSSIGHT_MAX_UPLOAD_BYTES" in response.json()["detail"]
    assert not list((tmp_path / "uploads").iterdir())


def test_intake_metadata_rejections_keep_upload_root_empty(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VITALSSIGHT_MAX_UPLOAD_BYTES", "8")
    monkeypatch.setenv("VITALSSIGHT_UPLOAD_DIR", str(tmp_path / "uploads"))
    app = create_app(tmp_path / "state.db", seed_demo=False, assistant_provider=UnavailableProvider(), auth_settings=AuthSettings())
    client = TestClient(app)
    for filename, data, expected in (("empty.mp4", b"", 422), ("large.mp4", b"123456789", 413), ("unsupported.txt", b"x", 415)):
        result = client.post("/api/v1/assessments/video", data={"consent_recorded": "true"}, files={"file": (filename, data, "video/mp4")})
        assert result.status_code == expected
        assert not list((tmp_path / "uploads").iterdir())
    assert client.get("/api/v1/cases").json()["count"] == 0


def test_readiness_reports_the_api_instances_frozen_limit(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VITALSSIGHT_MAX_UPLOAD_BYTES", "1234")
    monkeypatch.setenv("VITALSSIGHT_UPLOAD_DIR", str(tmp_path / "uploads"))
    app = create_app(tmp_path / "state.db", seed_demo=False, assistant_provider=UnavailableProvider(), auth_settings=AuthSettings())
    monkeypatch.setenv("VITALSSIGHT_MAX_UPLOAD_BYTES", "9876")
    assert TestClient(app).get("/api/v1/runtime/readiness").json()["upload_policy"]["max_upload_bytes"] == 1234
