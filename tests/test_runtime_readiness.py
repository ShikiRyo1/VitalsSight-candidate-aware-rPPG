"""Offline setup diagnostics and upload-metadata regression checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import ModuleType

from src.product.runtime_readiness import (
    DEFAULT_MAX_UPLOAD_BYTES,
    build_runtime_readiness,
    get_upload_policy,
    validate_video_upload,
)


class RuntimeReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def build(self, **kwargs):
        kwargs.setdefault("mediapipe_backend", "tasks")
        return build_runtime_readiness(project_root=self.root, environ={}, dependency_probe=lambda name: True, **kwargs)

    def check(self, result, identifier):
        return next(item for item in result["checks"] if item["id"] == identifier)

    def create_asset(self) -> Path:
        model = self.root / "runtime/models/face_landmarker.task"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"synthetic integrity fixture, not a model")
        return model

    def test_missing_asset_has_action_and_never_authorizes_release(self):
        before = list(self.root.rglob("*"))
        result = self.build()
        self.assertEqual(result["status"], "attention_required")
        self.assertEqual(self.check(result, "face_landmarker_asset")["status"], "blocked")
        self.assertIn("setup_runtime_assets.py", self.check(result, "face_landmarker_asset")["action"])
        self.assertFalse(result["measurement_release_authorized"])
        self.assertFalse(result["clinical_validity_established"])
        self.assertFalse(result["runtime_execution_verified"])
        self.assertFalse(result["participant_data_accessed"])
        self.assertEqual(result["outbound_service_requests"], 0)
        self.assertEqual(list(self.root.rglob("*")), before)

    def test_corrupt_asset_is_not_treated_as_installed(self):
        self.create_asset()
        result = self.build()
        self.assertEqual(self.check(result, "face_landmarker_asset")["status"], "blocked")
        self.assertIn("does not match", self.check(result, "face_landmarker_asset")["detail"])

    def test_pass_means_only_local_checks_not_inference(self):
        model = self.create_asset()
        with patch("src.product.runtime_readiness.FACE_MODEL_SHA256", hashlib.sha256(model.read_bytes()).hexdigest()):
            result = self.build()
        self.assertEqual(result["status"], "checks_passed")
        self.assertTrue(result["preflight_passed"])
        self.assertFalse(result["runtime_execution_verified"])
        self.assertFalse(result["measurement_release_authorized"])
        self.assertEqual(self.check(result, "authentication")["status"], "warning")
        self.assertIn("not executed", self.check(result, "face_landmarker_asset")["detail"])

    def test_dependency_probe_reports_missing_modules(self):
        result = build_runtime_readiness(project_root=self.root, environ={}, dependency_probe=lambda name: name != "mediapipe")
        self.assertEqual(self.check(result, "video_dependencies")["status"], "blocked")
        self.assertIn("mediapipe", self.check(result, "video_dependencies")["detail"])

    def test_legacy_backend_does_not_require_external_tasks_asset(self):
        result = self.build(mediapipe_backend="legacy_face_mesh")
        check = self.check(result, "face_landmarker_asset")
        self.assertEqual(check["status"], "warning")
        self.assertFalse(check["required_for_video"])
        self.assertIn("does not require", check["detail"])
        self.assertIn("review-only", check["detail"])
        self.assertFalse(result["measurement_release_authorized"])

    def test_unknown_backend_asset_is_advisory_not_blocking(self):
        result = self.build(mediapipe_backend="unknown")
        check = self.check(result, "face_landmarker_asset")
        self.assertEqual(check["status"], "warning")
        self.assertFalse(check["required_for_video"])
        self.assertIn("backend is unknown", check["detail"])

    def test_loaded_namespace_is_inspected_without_importing_mediapipe(self):
        module = ModuleType("mediapipe")
        module.solutions = object()
        with patch.dict("sys.modules", {"mediapipe": module}):
            result = build_runtime_readiness(project_root=self.root, environ={}, dependency_probe=lambda name: True)
        self.assertEqual(result["mediapipe_backend_hint"], "legacy_face_mesh")
        self.assertEqual(self.check(result, "face_landmarker_asset")["status"], "warning")

    def test_known_tasks_backend_missing_asset_remains_blocked(self):
        result = self.build(mediapipe_backend="tasks")
        check = self.check(result, "face_landmarker_asset")
        self.assertEqual(check["status"], "blocked")
        self.assertTrue(check["required_for_video"])

    def test_required_auth_incomplete_is_blocked(self):
        result = self.build(auth_config={"mode": "required", "issuer_configured": True})
        self.assertEqual(self.check(result, "authentication")["status"], "blocked")

    def test_settings_secrets_and_private_paths_are_not_returned(self):
        secret = "secret-not-for-output-937154"
        private_path = self.root / "private-study-location" / "uploads"
        result = build_runtime_readiness(
            project_root=self.root,
            upload_dir=private_path,
            environ={"VITALSSIGHT_AUTH_MODE": "required", "VITALSSIGHT_AUTH_ISSUER": "https://private-issuer.invalid", "VITALSSIGHT_AUTH_AUDIENCE": "private-audience", "VITALSSIGHT_AUTH_SHARED_SECRET": secret},
            dependency_probe=lambda name: True,
        )
        text = json.dumps(result)
        for sensitive in (secret, "private-issuer.invalid", "private-audience", "private-study-location", str(self.root)):
            self.assertNotIn(sensitive, text)
        self.assertEqual(self.check(result, "authentication")["status"], "warning")

    def test_explicit_missing_model_does_not_use_default_asset(self):
        self.create_asset()
        result = build_runtime_readiness(project_root=self.root, environ={"MEDIAPIPE_FACE_LANDMARKER_TASK": str(self.root / "missing.task")}, dependency_probe=lambda name: True)
        self.assertIn("missing", self.check(result, "face_landmarker_asset")["detail"])

    def test_upload_target_that_is_a_file_is_blocked(self):
        invalid = self.root / "not-directory"
        invalid.write_bytes(b"")
        result = self.build(upload_dir=invalid)
        self.assertEqual(self.check(result, "upload_storage")["status"], "blocked")

    def test_invalid_upload_limit_is_a_diagnostic_not_an_exception(self):
        result = build_runtime_readiness(project_root=self.root, environ={"VITALSSIGHT_MAX_UPLOAD_BYTES": "invalid"}, dependency_probe=lambda name: True)
        self.assertFalse(result["upload_policy"]["configuration_valid"])
        self.assertEqual(self.check(result, "upload_policy")["status"], "blocked")


class UploadMetadataTests(unittest.TestCase):
    def test_default_policy(self):
        self.assertEqual(get_upload_policy({})["max_upload_bytes"], DEFAULT_MAX_UPLOAD_BYTES)

    def test_valid_boundary_size_and_uppercase_extension(self):
        self.assertTrue(validate_video_upload("RESEARCH.MP4", 64, max_upload_bytes=64)["ok"])

    def test_unknown_size_does_not_claim_content_checked(self):
        result = validate_video_upload("study.mp4")
        self.assertTrue(result["ok"])
        self.assertIn("decoding", result["message"])

    def test_rejects_empty_oversized_and_unsupported(self):
        self.assertEqual(validate_video_upload("study.mp4", 0)["status_code"], 422)
        self.assertEqual(validate_video_upload("study.mp4", 65, max_upload_bytes=64)["status_code"], 413)
        self.assertEqual(validate_video_upload("study.txt", 2)["status_code"], 415)

    def test_invalid_policy_fails_closed(self):
        for value in ("invalid", "0", "-1"):
            policy = get_upload_policy({"VITALSSIGHT_MAX_UPLOAD_BYTES": value})
            self.assertFalse(policy["configuration_valid"])
            self.assertEqual(validate_video_upload("study.mp4", 2, max_upload_bytes=policy["max_upload_bytes"])["status_code"], 503)

    def test_invalid_size_not_silently_coerced(self):
        for size in (-1, True, 1.5):
            self.assertEqual(validate_video_upload("study.mp4", size)["status_code"], 422)


if __name__ == "__main__":
    unittest.main()
