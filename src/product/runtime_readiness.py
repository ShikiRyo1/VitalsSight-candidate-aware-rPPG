"""Read-only local configuration diagnostics, not physiological validation.

This module uses only the Python standard library. It does not import or start
inference engines, contact providers, install assets, write files or examine
participant data. Dependency discovery and OS access checks are advisory.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any


DEFAULT_MAX_UPLOAD_BYTES = 200 * 1024 * 1024
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
# Mirrored from src.vision.face_mesh_roi without importing cv2/MediaPipe here.
# An integration test requires these pins to remain identical.
FACE_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MAX_FACE_MODEL_BYTES = 64 * 1024 * 1024
VIDEO_MODULES = ("numpy", "pandas", "scipy", "sklearn", "cv2", "mediapipe")


def get_upload_policy(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    try:
        maximum = int(env.get("VITALSSIGHT_MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES)))
        valid = maximum > 0
    except (TypeError, ValueError):
        maximum, valid = 0, False
    return {
        "max_upload_bytes": maximum if valid else 0,
        "allowed_extensions": list(VIDEO_EXTENSIONS),
        "configuration_valid": valid,
    }


def validate_video_upload(
    filename: str,
    size_bytes: int | None = None,
    *,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> dict[str, Any]:
    """Validate intake metadata only; decodability and consent remain separate."""
    if not isinstance(max_upload_bytes, int) or isinstance(max_upload_bytes, bool) or max_upload_bytes <= 0:
        return {"ok": False, "status_code": 503, "message": "Video uploads are unavailable until a positive VITALSSIGHT_MAX_UPLOAD_BYTES is configured."}
    suffix = Path(str(filename or "")).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        return {"ok": False, "status_code": 415, "message": "Supported video types: mp4, mov, avi, mkv, m4v"}
    if size_bytes is not None:
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            return {"ok": False, "status_code": 422, "message": "Video size must be a non-negative integer."}
        if size_bytes == 0:
            return {"ok": False, "status_code": 422, "message": "Uploaded video is empty"}
        if size_bytes > max_upload_bytes:
            return {"ok": False, "status_code": 413, "message": "Video exceeds the configured upload limit"}
    return {"ok": True, "status_code": 200, "message": "Metadata accepted; consent, decoding, acquisition and evidence gates still apply."}


def _dependency_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _loaded_mediapipe_backend() -> str:
    """Inspect an already-loaded namespace only; never import heavy inference.

    Older MediaPipe exposes solutions.face_mesh and the runtime prefers that
    branch. A discoverable but unloaded package does not establish either API.
    """
    module = sys.modules.get("mediapipe")
    namespace = vars(module) if module is not None else {}
    if "solutions" in namespace:
        return "legacy_face_mesh"
    if "tasks" in namespace and "__getattr__" not in namespace:
        return "tasks"
    return "unknown"


def _network_path(path: str | Path) -> bool:
    return str(path).startswith(("\\\\", "//"))


def _storage_status(path: Path, *, file_target: bool) -> tuple[str, str]:
    """Inspect existing local ancestors without creating a probe file."""
    if _network_path(path):
        return "blocked", "Network filesystem paths are not inspected by this local preflight."
    try:
        if path.exists():
            if file_target and not path.is_file():
                return "blocked", "The configured database target is not a regular file."
            if not file_target and not path.is_dir():
                return "blocked", "The configured upload target is not a directory."
            if file_target and not os.access(path, os.R_OK | os.W_OK):
                return "blocked", "The existing database is not readable and writable by this process."
        ancestor = path.parent if file_target else path
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if not ancestor.is_dir() or not os.access(ancestor, os.R_OK | os.W_OK | os.X_OK):
            return "blocked", "The existing storage ancestor does not have the required process access."
        return "pass", "Existing local ancestor is accessible. This is an advisory OS check; no write, disk-capacity or transaction probe was performed."
    except (OSError, ValueError):
        return "blocked", "The local storage configuration could not be inspected."


def build_runtime_readiness(
    *,
    project_root: str | Path | None = None,
    db_path: str | Path | None = None,
    upload_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    dependency_probe: Callable[[str], bool] | None = None,
    auth_config: Mapping[str, Any] | None = None,
    mediapipe_backend: str | None = None,
) -> dict[str, Any]:
    """Return JSON-safe setup checks without revealing configured paths/secrets.

    ``checks_passed`` means these bounded local checks passed, not that a video,
    model, report, authentication provider or clinical workflow was validated.
    Callers may inject dependency discovery and an already-known backend hint
    (tasks/legacy_face_mesh) for deterministic offline tests. An unknown backend
    makes a missing Tasks asset advisory, not a universal video-intake blocker.
    """
    env = os.environ if environ is None else environ
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    probe = dependency_probe or _dependency_available
    checks: list[dict[str, Any]] = []

    def add(identifier: str, label: str, status: str, detail: str, action: str, required: bool = True) -> None:
        checks.append({"id": identifier, "label": label, "status": status, "detail": detail, "action": action, "required_for_video": required})

    supported_python = (3, 10) <= sys.version_info[:2] < (3, 13)
    add("python_runtime", "Python runtime", "pass" if supported_python else "blocked",
        f"Python {sys.version_info.major}.{sys.version_info.minor}; the declared supported range is 3.10-3.12.",
        "Use a Python 3.10-3.12 environment and install requirements-core.txt." if not supported_python else "No configuration action required.")

    missing = [name for name in VIDEO_MODULES if not probe(name)]
    add("video_dependencies", "Video dependencies", "blocked" if missing else "pass",
        "Missing module declarations: " + ", ".join(missing) if missing else "Declared video modules are discoverable; binary imports and inference were not executed.",
        "Run python -m pip install -r requirements-core.txt in the environment running the application." if missing else "If execution fails, inspect the actual runtime error; presence alone does not verify binary compatibility.")

    backend = mediapipe_backend if mediapipe_backend is not None else _loaded_mediapipe_backend()
    if backend not in {"tasks", "legacy_face_mesh"}:
        backend = "unknown"
    asset_required = backend == "tasks"

    # Match the inference resolver exactly; do not silently repair a path value.
    configured_model = env.get("MEDIAPIPE_FACE_LANDMARKER_TASK", "")
    model = Path(configured_model).expanduser() if configured_model else root / "runtime/models/face_landmarker.task"
    legacy = root / "third_party/mediapipe/face_landmarker.task"
    if not configured_model and not model.is_file() and legacy.is_file():
        model = legacy
    asset_status, asset_detail = "blocked", "The pinned Face Landmarker Tasks asset is missing."
    if _network_path(model):
        asset_detail = "A network model path was not inspected; use an authorized local asset."
    else:
        try:
            if model.is_file():
                if model.stat().st_size > MAX_FACE_MODEL_BYTES:
                    asset_detail = "The model file exceeds the bounded local integrity-check size."
                else:
                    digest = hashlib.sha256()
                    with model.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    if digest.hexdigest() == FACE_MODEL_SHA256:
                        asset_status = "pass"
                        asset_detail = "Pinned SHA-256 verified. MediaPipe initialization and landmark detection were not executed."
                    else:
                        asset_detail = "The local model does not match the pinned SHA-256. It is not accepted by this preflight."
        except (OSError, ValueError):
            asset_detail = "The local model could not be read for integrity verification."
    if backend == "legacy_face_mesh":
        if asset_status != "pass":
            asset_status = "warning"
        asset_detail += " The loaded legacy solutions.face_mesh backend does not require this Tasks file for landmark computation. Existing product policy nevertheless requires the pinned Tasks backend for release; legacy evidence is review-only."
    elif backend == "unknown":
        if asset_status != "pass":
            asset_status = "warning"
        asset_detail += " The active MediaPipe backend is unknown because this check does not import or initialize it. The external asset is required for Tasks, not legacy solutions.face_mesh; confirm the actual assessment backend."
    else:
        asset_detail += " The already-loaded MediaPipe namespace exposes the Tasks branch, which requires the pinned external asset."
    add("face_landmarker_asset", "Pinned landmark asset", asset_status, asset_detail,
        "For the Tasks backend, run python scripts/setup_runtime_assets.py, or provide an authorized local copy with --source. If MEDIAPIPE_FACE_LANDMARKER_TASK is set, ensure it points to that verified copy. Setup may download from the official provider; this check does not." if asset_status != "pass" else "Keep the pinned asset unchanged. Each assessment must still pass its own detector and evidence gates.", asset_required)

    resolved_db = Path(db_path or env.get("VITALSSIGHT_DB_PATH") or root / "runtime/vitalsight_console.db")
    resolved_upload = Path(upload_dir or env.get("VITALSSIGHT_UPLOAD_DIR") or resolved_db.parent / "uploads/api")
    for identifier, label, path, is_file, setting in (
        ("state_storage", "Evidence storage", resolved_db, True, "VITALSSIGHT_DB_PATH"),
        ("upload_storage", "Transient upload storage", resolved_upload, False, "VITALSSIGHT_UPLOAD_DIR"),
    ):
        status, detail = _storage_status(path, file_target=is_file)
        add(identifier, label, status, detail, f"Configure {setting} to an accessible local location. Uploaded videos remain subject to delete-after-analysis policy." if status != "pass" else "Monitor available disk space and backup evidence records according to the research protocol.")

    policy = get_upload_policy(env)
    add("upload_policy", "Upload size limit", "pass" if policy["configuration_valid"] else "blocked",
        f"Maximum video size: {policy['max_upload_bytes']} bytes. Metadata validation does not verify video contents." if policy["configuration_valid"] else "The upload byte limit is missing a valid positive integer configuration.",
        "Set VITALSSIGHT_MAX_UPLOAD_BYTES to a positive integer; the default is 209715200 bytes." if not policy["configuration_valid"] else "Use a supported video type within this limit, with recorded research consent.")

    auth = dict(auth_config) if auth_config is not None else {
        "mode": env.get("VITALSSIGHT_AUTH_MODE", "disabled").strip().lower(),
        "issuer_configured": bool(env.get("VITALSSIGHT_AUTH_ISSUER", "").strip()),
        "audience_configured": bool(env.get("VITALSSIGHT_AUTH_AUDIENCE", "").strip()),
        "jwks_configured": bool(env.get("VITALSSIGHT_AUTH_JWKS_URL", "").strip()),
        "test_shared_secret_configured": bool(env.get("VITALSSIGHT_AUTH_SHARED_SECRET", "").strip()),
        "dev_identity_headers_enabled": env.get("VITALSSIGHT_ALLOW_DEV_IDENTITY_HEADERS", "").lower() in {"1", "true", "yes", "on"},
    }
    mode = str(auth.get("mode", "disabled"))
    complete = bool(auth.get("issuer_configured") and auth.get("audience_configured") and (auth.get("jwks_configured") or auth.get("test_shared_secret_configured")))
    if mode == "required" and complete:
        auth_status = "warning" if auth.get("test_shared_secret_configured") else "pass"
        auth_detail = "Required-auth configuration fields are present; tokens and provider connectivity were not tested."
        if auth.get("test_shared_secret_configured"):
            auth_detail += " A shared-secret test configuration is present; this is not production OIDC readiness."
    elif mode == "disabled":
        auth_status, auth_detail = "warning", "Development identity is enabled. This configuration is for local research only, not public or multi-user hosting."
        if auth.get("dev_identity_headers_enabled"):
            auth_detail += " Development identity headers are enabled."
    else:
        auth_status, auth_detail = "blocked", "Authentication mode is invalid or required-auth configuration is incomplete."
    add("authentication", "Identity configuration", auth_status, auth_detail,
        "For a controlled multi-user study, configure required auth with issuer, audience and JWKS, retain role/organization scoping, and review infrastructure security separately.")
    add("assistant_optional", "Optional local assistant", "warning",
        "No assistant provider was contacted. Deterministic guidance remains available; a local model is optional and cannot authorize measurement release.",
        "Use the assistant health panel for a separate opt-in provider check, or continue with deterministic guidance.", False)

    counts = {status: sum(item["status"] == status for item in checks) for status in ("pass", "warning", "blocked")}
    passed = counts["blocked"] == 0
    return {
        "schema_version": "vitalssight.runtime-readiness.v1",
        "status": "checks_passed" if passed else "attention_required",
        "preflight_passed": passed,
        "scope": "Local dependency presence, configuration and asset integrity only; no inference, provider, storage transaction or clinical validation.",
        "checks": checks,
        "counts": counts,
        "upload_policy": policy,
        "auth_mode": mode if mode in {"disabled", "required"} else "invalid",
        "mediapipe_backend_hint": backend,
        "clinical_validity_established": False,
        "runtime_execution_verified": False,
        "measurement_release_authorized": False,
        "participant_data_accessed": False,
        "outbound_service_requests": 0,
    }
