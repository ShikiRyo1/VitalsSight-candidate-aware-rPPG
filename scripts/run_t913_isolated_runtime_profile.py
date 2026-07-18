from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_t905_full_pipeline_runtime_profile import (  # noqa: E402
    aggregate_stage_metrics,
    build_claim_gate,
    build_doc,
    environment_summary,
    json_safe,
)


TASK_ID = "T913"
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"
TMP = ROOT / "runtime" / "t913_runtime_cases"

CASE_CSV = EXP / "t913_isolated_runtime_case_metrics.csv"
STAGE_CSV = EXP / "t913_isolated_runtime_stage_metrics.csv"
ERROR_CSV = EXP / "t913_isolated_runtime_case_errors.csv"
CLAIM_CSV = EXP / "t913_isolated_runtime_claim_gate.csv"
SUMMARY_JSON = EXP / "t913_isolated_runtime_summary.json"
DOC_MD = DOCS / "t913_isolated_runtime_profile.md"
WORKER = ROOT / "scripts" / "run_t913_runtime_case_worker.py"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def markdown_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame.empty:
        return "_No rows._"
    show = frame.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_numeric_dtype(show[col]):
            show[col] = pd.to_numeric(show[col], errors="coerce").map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    lines = [
        "| " + " | ".join(str(c) for c in show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("\n", " ") for col in show.columns) + " |")
    return "\n".join(lines)


def discover_cases(max_mcd: int = 8, include_large: bool = True) -> list[dict[str, str]]:
    roots = []
    if os.environ.get("CONTACTLESS_DATA_ROOT"):
        roots.append(Path(os.environ["CONTACTLESS_DATA_ROOT"]))
    if os.environ.get("ADULT_DATA_ROOT"):
        roots.append(Path(os.environ["ADULT_DATA_ROOT"]).parent)
    roots.append(ROOT / "datasets")
    rows: list[dict[str, str]] = []
    for root in roots:
        mcd_video = root / "adult" / "MCD-rPPG" / "video"
        if mcd_video.exists():
            preferred = [
                "8555_IriunWebcam_before.avi",
                "1181_IriunWebcam_after.avi",
                "1765_IriunWebcam_after.avi",
                "4731_USBVideo_after.avi",
                "5507_USBVideo_after.avi",
                "8785_USBVideo_before.avi",
                "1765_FullHDwebcam_after.avi",
                "4731_FullHDwebcam_after.avi",
            ]
            chosen: list[Path] = []
            for name in preferred:
                p = mcd_video / name
                if p.exists() and p not in chosen:
                    chosen.append(p)
            if len(chosen) < max_mcd:
                for p in sorted(mcd_video.glob("*.avi")):
                    if p not in chosen:
                        chosen.append(p)
                    if len(chosen) >= max_mcd:
                        break
            for p in chosen[:max_mcd]:
                rows.append(
                    {
                        "dataset": "MCD-rPPG",
                        "case_id": p.stem,
                        "video_path": str(p),
                        "note": "MCD raw-video runtime case",
                    }
                )
        if include_large:
            large = [
                (
                    "UBFC-rPPG",
                    "ubfc_subject4",
                    root / "adult" / "UBFC-rPPG" / "kaggle_extracted" / "subject4" / "vid.avi",
                    "UBFC-rPPG isolated stress runtime case",
                ),
                (
                    "UBFC-Phys-S1-S14",
                    "ubfc_phys_s2_t1",
                    root / "adult" / "UBFC-Phys-S1-S14" / "extracted" / "s2" / "s2" / "vid_s2_T1.avi",
                    "UBFC-Phys isolated stress runtime case",
                ),
            ]
            for dataset, case_id, path, note in large:
                if path.exists():
                    rows.append({"dataset": dataset, "case_id": case_id, "video_path": str(path), "note": note})
        if rows:
            break
    return rows


def run_worker(case: dict[str, str], *, timeout: int, python_exe: str, no_mediapipe: bool = False) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    TMP.mkdir(parents=True, exist_ok=True)
    out_json = TMP / f"{case['case_id']}.json"
    if out_json.exists():
        out_json.unlink()
    command = [
        python_exe,
        str(WORKER),
        "--dataset",
        case["dataset"],
        "--case-id",
        case["case_id"],
        "--video-path",
        case["video_path"],
        "--note",
        case.get("note", ""),
        "--out-json",
        str(out_json),
    ]
    if no_mediapipe:
        command.append("--no-mediapipe")
    started = datetime.now().isoformat(timespec="seconds")
    proc = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    ended = datetime.now().isoformat(timespec="seconds")
    if proc.returncode == 0 and out_json.exists():
        row = json.loads(out_json.read_text(encoding="utf-8"))
        row["worker_started_at"] = started
        row["worker_ended_at"] = ended
        row["worker_returncode"] = proc.returncode
        return row, None
    err = {
        "task_id": TASK_ID,
        "dataset": case["dataset"],
        "case_id": case["case_id"],
        "video_path": case["video_path"],
        "worker_started_at": started,
        "worker_ended_at": ended,
        "worker_returncode": proc.returncode,
        "output_tail": proc.stdout[-2000:],
    }
    return None, err


def main() -> int:
    EXP.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    max_mcd = int(os.environ.get("T913_MAX_MCD_CASES", "8"))
    include_large = os.environ.get("T913_INCLUDE_LARGE_CASES", "1") == "1"
    no_mediapipe = os.environ.get("T913_NO_MEDIAPIPE", "0") == "1"
    timeout = int(os.environ.get("T913_CASE_TIMEOUT", "180"))
    python_exe = os.environ.get("T913_PYTHON", sys.executable)
    cases = discover_cases(max_mcd=max_mcd, include_large=include_large)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for case in cases:
        print(f"[{TASK_ID}] isolated runtime case {case['dataset']}/{case['case_id']}", flush=True)
        try:
            row, err = run_worker(case, timeout=timeout, python_exe=python_exe, no_mediapipe=no_mediapipe)
        except subprocess.TimeoutExpired as exc:
            row, err = None, {
                "task_id": TASK_ID,
                "dataset": case["dataset"],
                "case_id": case["case_id"],
                "video_path": case["video_path"],
                "worker_returncode": 998,
                "output_tail": str(exc)[-2000:],
            }
        if row is not None:
            row["task_id"] = TASK_ID
            rows.append(row)
        if err is not None:
            errors.append(err)

    case_df = pd.DataFrame(rows)
    stage_df = aggregate_stage_metrics(case_df) if not case_df.empty else pd.DataFrame()
    gate_df = build_claim_gate(case_df, stage_df)
    case_df.to_csv(CASE_CSV, index=False, encoding="utf-8-sig")
    stage_df.to_csv(STAGE_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(errors).to_csv(ERROR_CSV, index=False, encoding="utf-8-sig")
    gate_df.to_csv(CLAIM_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "task_id": TASK_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_cases_discovered": len(cases),
        "n_cases_completed": int(len(case_df)),
        "n_cases_failed": len(errors),
        "include_large": include_large,
        "no_mediapipe": no_mediapipe,
        "environment": environment_summary(),
        "decision": "isolated_runtime_profile_completed_with_recorded_failures" if len(case_df) else "isolated_runtime_profile_failed_no_completed_cases",
        "claim_boundary": "Isolated raw-video runtime profile. Failed large-video/MediaPipe cases are retained as implementation robustness evidence; deep-backbone inference remains a separate timing boundary unless a deep runtime row is added.",
        "outputs": {
            "case_metrics": str(CASE_CSV.relative_to(ROOT)),
            "stage_metrics": str(STAGE_CSV.relative_to(ROOT)),
            "errors": str(ERROR_CSV.relative_to(ROOT)),
            "claim_gate": str(CLAIM_CSV.relative_to(ROOT)),
            "doc": str(DOC_MD.relative_to(ROOT)),
        },
    }
    if not stage_df.empty:
        total = stage_df[stage_df["stage"].eq("total_measured_pipeline")]
        if not total.empty:
            summary["total_mean_ms"] = float(total["mean_ms"].iloc[0])
            summary["total_p50_ms"] = float(total["p50_ms"].iloc[0])
            summary["total_p95_ms"] = float(total["p95_ms"].iloc[0])
    write_json(SUMMARY_JSON, summary)

    doc = build_doc(case_df, stage_df, gate_df, summary)
    if errors:
        doc += "\n\n## Isolated Case Failures\n\n"
        doc += markdown_table(pd.DataFrame(errors))
        doc += "\n"
    DOC_MD.write_text(doc, encoding="utf-8")
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
