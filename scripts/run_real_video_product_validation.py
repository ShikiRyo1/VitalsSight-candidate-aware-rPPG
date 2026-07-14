from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
from fastapi.testclient import TestClient
from scipy import signal

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.product.console_api import create_app
from src.product.console_service import (
    build_report_markdown,
    build_report_payload,
    run_uploaded_video,
    video_preflight,
)


REQUIRED_CASE_FIELDS = {
    "schema_version",
    "case_id",
    "decision",
    "released_hr_bpm",
    "selected_candidate_hr_bpm",
    "quality_score",
    "face_coverage",
    "candidate_count",
    "agreement_fraction",
    "review_reason",
    "recommended_action",
    "policy_version",
    "model_version",
    "claim_boundary",
}

INFERENCE_CASE_FIELDS = {"runtime_metadata"}
SNAPSHOT_PATHS = (
    "app/api_server.py",
    "app/product_console.py",
    "src/data/video_io.py",
    "src/product/adult_hr_mvp.py",
    "src/product/console_api.py",
    "src/product/console_service.py",
    "src/vision/face_mesh_roi.py",
    "scripts/run_real_video_product_validation.py",
    "tests/test_console_product.py",
    "validation/real_video_case_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimate_ecg_reference(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sampling_hz = float(payload["frequency"])
    lead_name = str(protocol["lead"])
    lead = next(item for item in payload["data"] if str(item["title"]) == lead_name)
    values = np.asarray(lead["values"], dtype=float)
    window_samples = min(len(values), int(round(float(protocol["window_sec"]) * sampling_hz)))
    values = signal.detrend(values[:window_samples])
    low_hz, high_hz = (float(value) for value in protocol["bandpass_hz"])
    filtered = signal.sosfiltfilt(
        signal.butter(3, [low_hz, high_hz], btype="bandpass", fs=sampling_hz, output="sos"),
        values,
    )
    rr_min, rr_max = (float(value) for value in protocol["rr_interval_sec"])
    candidates: list[dict[str, Any]] = []
    for polarity in (1, -1):
        peaks, properties = signal.find_peaks(
            polarity * filtered,
            distance=max(1, int(0.32 * sampling_hz)),
            prominence=max(float(np.std(filtered)) * 0.8, 1e-8),
        )
        rr = np.diff(peaks) / sampling_hz
        valid_rr = rr[(rr > rr_min) & (rr < rr_max)]
        if len(valid_rr) < 3:
            continue
        candidates.append(
            {
                "polarity": polarity,
                "peak_count": int(len(peaks)),
                "median_prominence": float(np.median(properties["prominences"])),
                "median_rr_sec": float(np.median(valid_rr)),
                "reference_hr_bpm": float(60.0 / np.median(valid_rr)),
            }
        )
    if not candidates:
        raise ValueError(f"Could not derive an ECG reference from {path.name}")
    best = max(candidates, key=lambda item: item["median_prominence"])
    return {
        **best,
        "filename": path.name,
        "sha256": sha256_file(path),
        "lead": lead_name,
        "sampling_hz": sampling_hz,
        "window_sec": window_samples / sampling_hz,
        "method": protocol["estimator"],
        "release_max_abs_error_bpm": float(protocol["release_max_abs_error_bpm"]),
    }


def verify_reference_case(case: dict[str, Any], spec: dict[str, Any], reference: dict[str, Any] | None) -> list[str]:
    if reference is None:
        return []
    failures: list[str] = []
    if spec["expected_decision"] == "release":
        released = case.get("released_hr_bpm")
        if isinstance(released, (int, float)) and math.isfinite(float(released)):
            error = abs(float(released) - float(reference["reference_hr_bpm"]))
            if error > float(reference["release_max_abs_error_bpm"]):
                failures.append(
                    f"released HR error {error:.3f} BPM exceeds ECG audit limit "
                    f"{reference['release_max_abs_error_bpm']:.3f} BPM"
                )
    return failures


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("manifest_version") != "vitalssight.real-video-product-validation.v1":
        raise ValueError("Unsupported validation manifest version")
    if not data.get("cases"):
        raise ValueError("Validation manifest contains no cases")
    return data


def _video_writer(path: Path, fps: float, width: int, height: int, codec: str) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open validation fixture writer: {path}")
    return writer


def prepare_derived_fixture(spec: dict[str, Any], fixture_root: Path) -> None:
    derivation = spec.get("derivation")
    if not derivation:
        return
    source = fixture_root / derivation["source_filename"]
    if not source.is_file():
        raise FileNotFoundError(f"Missing source fixture: {source}")
    if sha256_file(source) != derivation["source_sha256"]:
        raise ValueError(f"Source hash mismatch: {source.name}")

    output = fixture_root / spec["filename"]
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source fixture: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = _video_writer(output, fps, width, height, str(derivation.get("codec", "MJPG")))
    try:
        if derivation["type"] == "frame_slice":
            start = int(derivation["start_frame"])
            end = int(derivation["end_frame_exclusive"])
            frame_index = 0
            while frame_index < end:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index >= start:
                    writer.write(frame)
                frame_index += 1
        elif derivation["type"] == "controlled_visibility":
            total = int(derivation["total_frames"])
            visible = int(derivation["visible_frames"])
            frame_index = 0
            while frame_index < total:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame if frame_index < visible else np.zeros_like(frame))
                frame_index += 1
        else:
            raise ValueError(f"Unsupported derivation type: {derivation['type']}")
    finally:
        capture.release()
        writer.release()


def verify_case_contract(case: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    missing = sorted(REQUIRED_CASE_FIELDS - set(case))
    if missing:
        failures.append(f"missing fields: {', '.join(missing)}")
    expected = spec["expected_decision"]
    if expected in {"release", "review"}:
        missing_inference = sorted(
            field for field in INFERENCE_CASE_FIELDS if not case.get(field)
        )
        if missing_inference:
            failures.append(
                f"missing inference fields: {', '.join(missing_inference)}"
            )
    if case.get("decision") != expected:
        failures.append(f"decision {case.get('decision')} != {expected}")
    released = case.get("released_hr_bpm")
    selected = case.get("selected_candidate_hr_bpm")
    if expected == "release":
        if not isinstance(released, (int, float)) or not math.isfinite(float(released)):
            failures.append("release has no finite released_hr_bpm")
        elif selected is None or abs(float(released) - float(selected)) > 1e-9:
            failures.append("released HR differs from selected candidate HR")
    elif released is not None:
        failures.append("non-release case published released_hr_bpm")
    reason = str(spec.get("expected_reason_contains", ""))
    if reason and reason not in str(case.get("review_reason", "")):
        failures.append(f"review_reason does not contain {reason!r}")
    if case.get("decision") in {"review", "retake"} and not case.get("recommended_action"):
        failures.append("withheld case has no recommended action")
    return failures


def compact_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "source_name": case.get("source_name"),
        "decision": case.get("decision"),
        "released_hr_bpm": case.get("released_hr_bpm"),
        "selected_candidate_hr_bpm": case.get("selected_candidate_hr_bpm"),
        "candidate_count": case.get("candidate_count"),
        "agreement_fraction": case.get("agreement_fraction"),
        "window_consistency_fraction": case.get("window_consistency_fraction"),
        "window_hr_range_bpm": case.get("window_hr_range_bpm"),
        "competing_track_count": case.get("competing_track_count"),
        "review_reason": case.get("review_reason"),
        "recommended_action": case.get("recommended_action"),
        "runtime_metadata": case.get("runtime_metadata"),
        "preflight": case.get("preflight"),
        "window_results": case.get("window_results", []),
        "policy_version": case.get("policy_version"),
        "model_version": case.get("model_version"),
        "claim_boundary": case.get("claim_boundary"),
    }


def run_direct_case(
    path: Path,
    spec: dict[str, Any],
    repeats: int,
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    failures: list[str] = []
    for repeat in range(repeats):
        started = time.perf_counter()
        preflight = video_preflight(path)
        case = run_uploaded_video(
            path,
            purpose="workflow_validation",
            retention_policy="delete_after_analysis",
            preflight=preflight,
        )
        elapsed = time.perf_counter() - started
        run_failures = verify_case_contract(case, spec)
        run_failures.extend(verify_reference_case(case, spec, reference))
        failures.extend(f"repeat {repeat + 1}: {item}" for item in run_failures)
        runs.append(
            {
                "run_id": f"{spec['case_id']}.direct.{repeat + 1}",
                "execution_mode": "direct_backend",
                "repeat": repeat + 1,
                "elapsed_sec": round(elapsed, 6),
                "failures": run_failures,
                "passed": not run_failures,
                "case": compact_case(case),
            }
        )

    decisions = {run["case"]["decision"] for run in runs}
    if len(decisions) != 1:
        failures.append(f"decision changed across repeats: {sorted(decisions)}")
    released_values = [run["case"]["released_hr_bpm"] for run in runs if run["case"]["released_hr_bpm"] is not None]
    if released_values and max(released_values) - min(released_values) > 0.1:
        failures.append(f"released HR changed by more than 0.1 BPM across repeats: {released_values}")
    return {"runs": runs, "reference": reference, "failures": failures, "passed": not failures}


def post_video(client: TestClient, path: Path, *, consent: bool = True) -> Any:
    with path.open("rb") as handle:
        return client.post(
            "/api/v1/assessments/video",
            data={
                "consent_recorded": str(consent).lower(),
                "purpose": "workflow_validation",
                "retention_policy": "delete_after_analysis",
                "actor": "real-video-validation",
            },
            files={"file": (path.name, handle, "video/x-msvideo")},
        )


def run_api_cases(
    manifest: dict[str, Any],
    fixture_root: Path,
    output_dir: Path,
    direct_results: dict[str, Any],
) -> dict[str, Any]:
    db_path = output_dir / "validation_api.db"
    upload_dir = output_dir / "api_uploads"
    if db_path.exists():
        db_path.unlink()
    os.environ["VITALSSIGHT_UPLOAD_DIR"] = str(upload_dir)
    app = create_app(db_path, seed_demo=False)
    client = TestClient(app)
    rows: dict[str, Any] = {}

    for spec in manifest["cases"]:
        path = fixture_root / spec["filename"]
        started = time.perf_counter()
        response = post_video(client, path)
        failures: list[str] = []
        if response.status_code != 201:
            failures.append(f"HTTP status {response.status_code}, expected 201")
            rows[spec["case_id"]] = {"passed": False, "failures": failures}
            continue
        payload = response.json()
        case = payload["item"]
        failures.extend(verify_case_contract(case, spec))
        direct_case = direct_results[spec["case_id"]]["runs"][0]["case"]
        reference = direct_results[spec["case_id"]].get("reference")
        failures.extend(verify_reference_case(case, spec, reference))
        if case.get("decision") != direct_case.get("decision"):
            failures.append("API decision differs from direct backend")
        if case.get("released_hr_bpm") is not None and abs(
            float(case["released_hr_bpm"]) - float(direct_case["released_hr_bpm"])
        ) > 0.1:
            failures.append("API released HR differs from direct backend by more than 0.1 BPM")
        if payload.get("raw_video_retained") is not False:
            failures.append("API did not confirm raw-video deletion")
        if upload_dir.exists() and any(item.is_file() for item in upload_dir.rglob("*")):
            failures.append("temporary raw upload remains on disk")

        report_response = client.get(f"/api/v1/cases/{case['case_id']}/report?format=json")
        pdf_response = client.get(f"/api/v1/cases/{case['case_id']}/report?format=pdf")
        if report_response.status_code != 200 or report_response.json()["case"]["decision"] != case["decision"]:
            failures.append("JSON report does not match API case")
        if pdf_response.status_code != 200 or not pdf_response.content.startswith(b"%PDF"):
            failures.append("PDF report is missing or invalid")

        report_payload = build_report_payload(case)
        markdown = build_report_markdown(report_payload, language="en")
        if str(case["decision"]) not in markdown:
            failures.append("Markdown report does not contain the decision")
        if case["decision"] != "release" and "Released HR: withheld" not in markdown:
            failures.append("Markdown report does not state that released HR is withheld")
        if "## Implementation provenance" not in markdown:
            failures.append("Markdown report does not expose implementation provenance")

        case_dir = output_dir / "reports" / spec["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "report.json").write_text(
            json.dumps(report_response.json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (case_dir / "report.md").write_text(markdown, encoding="utf-8")
        (case_dir / "report.pdf").write_bytes(pdf_response.content)
        rows[spec["case_id"]] = {
            "run_id": f"{spec['case_id']}.api.1",
            "execution_mode": "api",
            "repeat": 1,
            "passed": not failures,
            "failures": failures,
            "elapsed_sec": round(time.perf_counter() - started, 6),
            "status_code": response.status_code,
            "case": compact_case(case),
        }

    smallest = fixture_root / manifest["cases"][-1]["filename"]
    no_consent = post_video(client, smallest, consent=False)
    malformed = output_dir / "malformed_input.avi"
    malformed.write_bytes(b"not-a-video-container")
    malformed_response = post_video(client, malformed)
    unsupported = output_dir / "unsupported_input.txt"
    unsupported.write_text("not a video", encoding="ascii")
    with unsupported.open("rb") as handle:
        unsupported_response = client.post(
            "/api/v1/assessments/video",
            data={
                "consent_recorded": "true",
                "purpose": "workflow_validation",
                "retention_policy": "delete_after_analysis",
            },
            files={"file": (unsupported.name, handle, "text/plain")},
        )
    negative = {
        "no_consent": {
            "status_code": no_consent.status_code,
            "passed": no_consent.status_code == 422,
        },
        "malformed_video": {
            "status_code": malformed_response.status_code,
            "decision": malformed_response.json().get("item", {}).get("decision") if malformed_response.status_code == 201 else None,
            "released_hr_bpm": malformed_response.json().get("item", {}).get("released_hr_bpm") if malformed_response.status_code == 201 else None,
            "passed": (
                malformed_response.status_code == 201
                and malformed_response.json()["item"]["decision"] == "retake"
                and malformed_response.json()["item"]["released_hr_bpm"] is None
            ),
        },
        "unsupported_extension": {
            "status_code": unsupported_response.status_code,
            "passed": unsupported_response.status_code == 415,
        },
    }
    return {"cases": rows, "negative_http": negative}


def write_outputs(
    output_dir: Path,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    direct: dict[str, Any],
    api: dict[str, Any],
) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for spec in manifest["cases"]:
        direct_case = direct[spec["case_id"]]
        api_case = api["cases"].get(spec["case_id"], {})
        first = direct_case["runs"][0]
        reference = direct_case.get("reference") or {}
        released = first["case"].get("released_hr_bpm")
        candidate = first["case"].get("selected_candidate_hr_bpm")
        reference_hr = reference.get("reference_hr_bpm")
        case_rows.append(
            {
                "case_id": spec["case_id"],
                "fixture_kind": spec["fixture_kind"],
                "expected_decision": spec["expected_decision"],
                "direct_decision": first["case"]["decision"],
                "api_decision": api_case.get("case", {}).get("decision"),
                "released_hr_bpm": released,
                "selected_candidate_hr_bpm": candidate,
                "reference_hr_bpm": reference_hr,
                "released_abs_error_bpm": (
                    abs(float(released) - float(reference_hr))
                    if released is not None and reference_hr is not None
                    else None
                ),
                "candidate_abs_error_bpm": (
                    abs(float(candidate) - float(reference_hr))
                    if candidate is not None and reference_hr is not None
                    else None
                ),
                "candidate_count": first["case"].get("candidate_count"),
                "review_reason": first["case"].get("review_reason"),
                "window_consistency_fraction": first["case"].get("window_consistency_fraction"),
                "competing_track_count": first["case"].get("competing_track_count"),
                "policy_version": first["case"].get("policy_version"),
                "direct_elapsed_sec": first["elapsed_sec"],
                "api_elapsed_sec": api_case.get("elapsed_sec"),
                "direct_passed": direct_case["passed"],
                "api_passed": api_case.get("passed", False),
                "failures": " | ".join(direct_case["failures"] + api_case.get("failures", [])),
            }
        )
        for run in direct_case["runs"]:
            case = run["case"]
            released = case.get("released_hr_bpm")
            reference_hr = reference.get("reference_hr_bpm")
            run_rows.append(
                {
                    "run_id": run["run_id"],
                    "case_id": spec["case_id"],
                    "fixture_kind": spec["fixture_kind"],
                    "execution_mode": run["execution_mode"],
                    "repeat": run["repeat"],
                    "expected_decision": spec["expected_decision"],
                    "observed_decision": case.get("decision"),
                    "released_hr_bpm": released,
                    "selected_candidate_hr_bpm": case.get("selected_candidate_hr_bpm"),
                    "reference_hr_bpm": reference_hr,
                    "released_abs_error_bpm": (
                        abs(float(released) - float(reference_hr))
                        if released is not None and reference_hr is not None
                        else None
                    ),
                    "candidate_count": case.get("candidate_count"),
                    "review_reason": case.get("review_reason"),
                    "elapsed_sec": run.get("elapsed_sec"),
                    "passed": run.get("passed", False),
                    "failures": " | ".join(run.get("failures", [])),
                }
            )
        api_case_payload = api_case.get("case", {})
        api_released = api_case_payload.get("released_hr_bpm")
        run_rows.append(
            {
                "run_id": api_case.get("run_id", f"{spec['case_id']}.api.1"),
                "case_id": spec["case_id"],
                "fixture_kind": spec["fixture_kind"],
                "execution_mode": api_case.get("execution_mode", "api"),
                "repeat": api_case.get("repeat", 1),
                "expected_decision": spec["expected_decision"],
                "observed_decision": api_case_payload.get("decision"),
                "released_hr_bpm": api_released,
                "selected_candidate_hr_bpm": api_case_payload.get("selected_candidate_hr_bpm"),
                "reference_hr_bpm": reference_hr,
                "released_abs_error_bpm": (
                    abs(float(api_released) - float(reference_hr))
                    if api_released is not None and reference_hr is not None
                    else None
                ),
                "candidate_count": api_case_payload.get("candidate_count"),
                "review_reason": api_case_payload.get("review_reason"),
                "elapsed_sec": api_case.get("elapsed_sec"),
                "passed": api_case.get("passed", False),
                "failures": " | ".join(api_case.get("failures", [])),
            }
        )
    all_passed = all(row["direct_passed"] and row["api_passed"] for row in case_rows) and all(
        item["passed"] for item in api["negative_http"].values()
    )
    bundle = {
        "validation_version": "vitalssight.real-video-product-validation.result.v1",
        "passed": all_passed,
        "claim_boundary": manifest["claim_boundary"],
        "provenance": provenance,
        "case_results": case_rows,
        "run_level_results": run_rows,
        "direct": direct,
        "api": api,
    }
    (output_dir / "validation_results.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "case_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    with (output_dir / "run_level_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)

    fixture_state_counts = {
        state: sum(row["expected_decision"] == state for row in case_rows)
        for state in ("release", "review", "retake")
    }
    execution_state_counts = {
        state: sum(row["observed_decision"] == state for row in run_rows)
        for state in ("release", "review", "retake")
    }

    lines = [
        "# VitalsSight real-video research implementation validation",
        "",
        f"Overall result: {'PASS' if all_passed else 'FAIL'}",
        "",
        manifest["claim_boundary"],
        "",
        "## Validation design and counts",
        "",
        "- Design: post hoc curated regression/conformance replay. Cases were selected during development; expected states and fixture hashes were recorded in the manifest before this final replay.",
        f"- Fixtures: {len(case_rows)} ({fixture_state_counts['release']} release, {fixture_state_counts['review']} review, {fixture_state_counts['retake']} retake).",
        f"- Executions: {len(run_rows)} ({execution_state_counts['release']} release, {execution_state_counts['review']} review, {execution_state_counts['retake']} retake); each fixture ran twice through the direct backend and once through the API.",
        f"- Execution-level contract matches: {sum(bool(row['passed']) for row in run_rows)}/{len(run_rows)}.",
        "- Per-execution decisions, elapsed times, released values and failures are recorded in `run_level_results.csv`.",
        "",
        "| Case | Fixture | Expected | Direct | API | Released HR | ECG reference | Abs. error | Review reason | Direct/API pass |",
        "|---|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in case_rows:
        lines.append(
            f"| {row['case_id']} | {row['fixture_kind']} | {row['expected_decision']} | "
            f"{row['direct_decision']} | {row['api_decision']} | {row['released_hr_bpm']} | "
            f"{row['reference_hr_bpm']} | {row['released_abs_error_bpm']} | {row['review_reason']} | "
            f"{row['direct_passed']}/{row['api_passed']} |"
        )
    lines.extend(["", "## Negative HTTP checks"])
    for name, result in api["negative_http"].items():
        lines.append(f"- {name}: {'PASS' if result['passed'] else 'FAIL'} (HTTP {result['status_code']})")
    lines.extend(
        [
            "",
            "## Reproducibility snapshot",
            "",
            f"- Git commit: `{provenance['git_commit']}`",
            f"- Dirty working tree recorded: `{provenance['working_tree']['dirty']}`",
            f"- Source snapshot SHA-256: `{provenance['working_tree']['source_snapshot_sha256']}`",
            "- Design boundary: no independent accuracy cohort is established by this suite.",
            "- Reference role: reference-only comparison; ECG/reference HR is not passed to candidate construction, selection or gating.",
            "",
            "## Interpretation boundary",
            "",
            "This suite verifies the listed direct-backend and API paths, state/output withholding, JSON/Markdown/PDF report generation, raw-upload cleanup and decision parity on these fixtures. It does not establish the primary manuscript model end to end, clinical accuracy, usability, safety, security, fairness, real-time readiness, or deployment effectiveness.",
        ]
    )
    (output_dir / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return all_passed


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_working_tree_snapshot() -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", *SNAPSHOT_PATHS],
        check=True,
        capture_output=True,
    ).stdout
    file_hashes = {
        relative: sha256_file(PROJECT / relative)
        for relative in SNAPSHOT_PATHS
        if (PROJECT / relative).is_file()
    }
    file_hash_payload = json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "dirty": bool(status),
        "git_status_porcelain": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_file_sha256": file_hashes,
        "source_snapshot_sha256": hashlib.sha256(file_hash_payload).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate VitalsSight on frozen real-video product cases.")
    parser.add_argument("--manifest", type=Path, default=Path("validation/real_video_case_manifest.json"))
    parser.add_argument("--fixture-root", type=Path, default=Path("runtime/private_validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/real_video_product_validation"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--prepare-derived", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    manifest = load_manifest(args.manifest)
    model_path = args.fixture_root / manifest["model"]["filename"]
    if not model_path.is_file() or sha256_file(model_path) != manifest["model"]["sha256"]:
        raise ValueError("MediaPipe model asset is missing or has the wrong hash")
    os.environ["MEDIAPIPE_FACE_LANDMARKER_TASK"] = str(model_path.resolve())

    if args.prepare_derived:
        for spec in manifest["cases"]:
            prepare_derived_fixture(spec, args.fixture_root)
    for spec in manifest["cases"]:
        path = args.fixture_root / spec["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing validation fixture: {path}")
        if sha256_file(path) != spec["sha256"]:
            raise ValueError(f"Fixture hash mismatch: {path.name}")
        reference_spec = spec.get("reference")
        if reference_spec:
            reference_path = args.fixture_root / reference_spec["filename"]
            if not reference_path.is_file():
                raise FileNotFoundError(f"Missing ECG reference: {reference_path}")
            if sha256_file(reference_path) != reference_spec["sha256"]:
                raise ValueError(f"ECG reference hash mismatch: {reference_path.name}")

    references = {
        spec["case_id"]: (
            estimate_ecg_reference(
                args.fixture_root / spec["reference"]["filename"],
                manifest["reference_protocol"],
            )
            if spec.get("reference")
            else None
        )
        for spec in manifest["cases"]
    }

    provenance = {
        "git_commit": git_commit(),
        "working_tree": git_working_tree_snapshot(),
        "python": sys.version,
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "manifest_sha256": sha256_file(args.manifest),
        "model_sha256": sha256_file(model_path),
        "repeats": args.repeats,
        "validation_design": {
            "case_selection": "post hoc curated regression/conformance fixtures selected during development; expected states and hashes fixed before final replay",
            "ecg_role": "reference-only comparison; ECG/reference HR is unavailable to runtime candidate construction, selection and gating",
            "independent_accuracy_cohort": "not established by this suite",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.manifest, args.output_dir / "fixture_manifest_snapshot.json")
    direct: dict[str, Any] = {}
    for spec in manifest["cases"]:
        direct[spec["case_id"]] = run_direct_case(
            args.fixture_root / spec["filename"],
            spec,
            args.repeats,
            references[spec["case_id"]],
        )
    api = run_api_cases(manifest, args.fixture_root, args.output_dir, direct)
    passed = write_outputs(args.output_dir, manifest, provenance, direct, api)
    print(json.dumps({"passed": passed, "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
