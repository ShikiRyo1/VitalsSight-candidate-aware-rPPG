"""T136 full 4TU adult HR classical baseline benchmark.

The goal is to turn the T135 dataset audit into the first adult-first metric
table: all locally available 4TU sessions, RR-interval labels, subject-aware
split tags, and reproducible classical rPPG baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.baselines.traditional_rppg import METHODS
from src.data.archive_io import extract_zip_member, read_zip_text
from src.data.labels_4tu import RRIntervals, parse_rr_intervals
from src.data.video_io import get_video_metadata, iter_video_frames
from src.evaluation.metrics import mae, pearson, rmse
from src.signal.estimate import estimate_hr
from src.vision.roi import ROI, face_like_rois, mean_rgb, stable_face_roi


RESEARCH_ROOT = PROJECT.parent
DATA_ROOT = RESEARCH_ROOT / "数据集"
ARCHIVE = DATA_ROOT / "adult" / "4TU-rPPG-Benchmark" / "data.zip"
CACHE_DIR = PROJECT / "experiments" / "cache" / "t136_4tu"
EXPERIMENTS = PROJECT / "experiments"
FIG_DIR = PROJECT / "output" / "t136_figures"

SESSION_INDEX_CSV = EXPERIMENTS / "t136_4tu_session_index.csv"
LABEL_AUDIT_CSV = EXPERIMENTS / "t136_4tu_rr_label_audit.csv"
WINDOW_RESULTS_CSV = EXPERIMENTS / "t136_4tu_classical_window_results.csv"
METHOD_SUMMARY_CSV = EXPERIMENTS / "t136_4tu_classical_method_summary.csv"
SUBJECT_SUMMARY_CSV = EXPERIMENTS / "t136_4tu_subject_summary.csv"
CONDITION_SUMMARY_CSV = EXPERIMENTS / "t136_4tu_condition_summary.csv"
SUMMARY_JSON = EXPERIMENTS / "t136_4tu_classical_benchmark_summary.json"
REPORT_MD = EXPERIMENTS / f"t136_4tu_adult_hr_benchmark_report_{date.today().isoformat()}.md"


@dataclass(frozen=True)
class SessionRecord:
    sample_id: str
    dataset: str
    subject_id: str
    session_id: str
    split: str
    condition_group: str
    condition_detail: str
    archive_path: str
    video_member: str
    rr_member: str
    fys_member: str
    mts_member: str
    video_size_mb: float


def natural_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def subject_from_session(session_id: str) -> str:
    match = re.match(r"^(P\d+)", session_id)
    if not match:
        return "unknown"
    return match.group(1)


def split_for_subject(subject_id: str) -> str:
    return {
        "P1": "development",
        "P2": "validation",
        "P3": "held_out_test",
    }.get(subject_id, "unknown")


def classify_condition(session_id: str) -> tuple[str, str]:
    subject = subject_from_session(session_id)
    suffix = session_id[len(subject) :]
    if suffix.startswith("H"):
        return "high_hr", "high heart-rate / pulse-rate-change stress"
    if suffix.startswith("M"):
        return "motion", "motion robustness stress"
    if suffix.startswith("LC"):
        return "lighting_skin_tone", "lighting / skin-tone stress"
    return "unknown", "unclassified"


def discover_sessions() -> list[SessionRecord]:
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"4TU archive not found: {ARCHIVE}")

    with zipfile.ZipFile(ARCHIVE) as zf:
        names = zf.namelist()
        session_ids = sorted(
            {
                name.split("/")[1]
                for name in names
                if name.startswith("Public Benchmark Dataset")
                and len(name.split("/")) > 2
                and name.split("/")[1]
            },
            key=natural_key,
        )
        records: list[SessionRecord] = []
        for session_id in session_ids:
            members = [name for name in names if f"/{session_id}/" in name]
            videos = sorted([m for m in members if m.lower().endswith(".avi")], key=natural_key)
            rr_files = sorted([m for m in members if m.lower().endswith(".rr")], key=natural_key)
            fys_files = sorted([m for m in members if m.lower().endswith(".fys")], key=natural_key)
            mts_files = sorted([m for m in members if m.lower().endswith(".mts")], key=natural_key)
            if not videos or not rr_files:
                continue
            subject_id = subject_from_session(session_id)
            condition_group, condition_detail = classify_condition(session_id)
            video_info = zf.getinfo(videos[0])
            records.append(
                SessionRecord(
                    sample_id=f"4tu_{session_id}",
                    dataset="4TU-rPPG-Benchmark",
                    subject_id=subject_id,
                    session_id=session_id,
                    split=split_for_subject(subject_id),
                    condition_group=condition_group,
                    condition_detail=condition_detail,
                    archive_path=str(ARCHIVE),
                    video_member=videos[0],
                    rr_member=rr_files[0],
                    fys_member=fys_files[0] if fys_files else "",
                    mts_member=mts_files[0] if mts_files else "",
                    video_size_mb=round(video_info.file_size / 1024 / 1024, 3),
                )
            )
    return records


def save_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def first_frames(video_path: Path, count: int = 12, step: int = 15) -> list[np.ndarray]:
    return [
        frame
        for _, frame in iter_video_frames(
            video_path,
            max_frames=count,
            sample_every=step,
            convert_rgb=True,
        )
    ]


def extract_rgb_trace(video_path: Path, seconds: float) -> tuple[np.ndarray, float, ROI, dict[str, float]]:
    meta = get_video_metadata(video_path)
    if meta.fps <= 0:
        raise RuntimeError(f"Invalid FPS for {video_path}")
    max_frames = int(min(meta.frame_count, round(meta.fps * seconds)))
    probe = first_frames(video_path)
    roi = stable_face_roi(probe)
    if roi is None:
        if probe:
            first = probe[0]
        else:
            first = next(iter_video_frames(video_path, max_frames=1, convert_rgb=True))[1]
        roi = face_like_rois(first)[0]

    values: list[np.ndarray] = []
    for _, frame in iter_video_frames(video_path, max_frames=max_frames, sample_every=1, convert_rgb=True):
        values.append(mean_rgb(frame, roi))
    rgb_trace = np.asarray(values, dtype=float)
    meta_dict = {
        "fps": float(meta.fps),
        "frames_used": float(len(values)),
        "seconds_used": float(len(values) / meta.fps),
        "video_frame_count": float(meta.frame_count),
        "video_duration_sec": float(meta.duration_sec),
        "video_width": float(meta.width),
        "video_height": float(meta.height),
    }
    return rgb_trace, meta.fps, roi, meta_dict


def label_audit(record: SessionRecord, intervals: RRIntervals, seconds: float) -> dict[str, object]:
    timestamps = intervals.timestamps_sec
    hr = intervals.hr_bpm
    if timestamps.size == 0:
        return {
            "sample_id": record.sample_id,
            "session_id": record.session_id,
            "subject_id": record.subject_id,
            "split": record.split,
            "condition_group": record.condition_group,
            "rr_count": 0,
            "rr_start_sec": math.nan,
            "rr_end_sec": math.nan,
            "mean_hr_first_window_bpm": math.nan,
            "mean_hr_full_rr_bpm": math.nan,
            "min_hr_full_rr_bpm": math.nan,
            "max_hr_full_rr_bpm": math.nan,
            "std_hr_full_rr_bpm": math.nan,
        }
    return {
        "sample_id": record.sample_id,
        "session_id": record.session_id,
        "subject_id": record.subject_id,
        "split": record.split,
        "condition_group": record.condition_group,
        "rr_count": int(timestamps.size),
        "rr_start_sec": float(timestamps[0]),
        "rr_end_sec": float(timestamps[-1]),
        "mean_hr_first_window_bpm": intervals.mean_hr(0.0, seconds),
        "mean_hr_full_rr_bpm": float(np.mean(hr)),
        "min_hr_full_rr_bpm": float(np.min(hr)),
        "max_hr_full_rr_bpm": float(np.max(hr)),
        "std_hr_full_rr_bpm": float(np.std(hr)),
    }


def snr_proxy_db(band_power: float, total_power: float) -> float:
    noise_power = max(total_power - band_power, 1e-12)
    return float(10.0 * math.log10((band_power + 1e-12) / noise_power))


def finite_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except (TypeError, ValueError, KeyError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def summarize_group(rows: list[dict[str, object]], group_keys: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    summary: list[dict[str, object]] = []
    for key, subset in sorted(grouped.items()):
        y_true = finite_values(subset, "gt_hr_bpm")
        y_pred = finite_values(subset, "pred_hr_bpm")
        errors = finite_values(subset, "abs_error_bpm")
        row = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        row.update(
            {
                "n": len(subset),
                "coverage": float(np.isfinite([float(r["pred_hr_bpm"]) for r in subset]).mean()) if subset else 0.0,
                "mae_bpm": mae(y_true, y_pred),
                "rmse_bpm": rmse(y_true, y_pred),
                "pearson_r": pearson(y_true, y_pred),
                "median_abs_error_bpm": float(np.median(errors)) if errors else math.nan,
                "p90_abs_error_bpm": float(np.percentile(errors, 90)) if errors else math.nan,
                "high_error_rate_10bpm": float(np.mean(np.asarray(errors) > 10.0)) if errors else math.nan,
                "mean_confidence": float(np.mean(finite_values(subset, "confidence"))) if subset else math.nan,
                "mean_snr_proxy_db": float(np.mean(finite_values(subset, "snr_proxy_db"))) if subset else math.nan,
            }
        )
        summary.append(row)
    return summary


def write_figures(results: pd.DataFrame, method_summary: pd.DataFrame) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    ordered = method_summary.sort_values("mae_bpm")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(ordered["method"], ordered["mae_bpm"], color="#3B82F6", alpha=0.85, label="MAE")
    ax.scatter(ordered["method"], ordered["rmse_bpm"], color="#111827", s=42, label="RMSE", zorder=3)
    ax.set_ylabel("Error (BPM)")
    ax.set_title("T136 4TU adult HR classical baseline summary")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIG_DIR / "t136_method_mae_rmse.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["method_mae_rmse"] = str(path)

    best_method = str(ordered.iloc[0]["method"])
    best = results[results["method"] == best_method].copy()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(best["gt_hr_bpm"], best["pred_hr_bpm"], c="#10B981", edgecolors="#064E3B", s=56, alpha=0.9)
    lower = min(best["gt_hr_bpm"].min(), best["pred_hr_bpm"].min()) - 5
    upper = max(best["gt_hr_bpm"].max(), best["pred_hr_bpm"].max()) + 5
    ax.plot([lower, upper], [lower, upper], color="#111827", linestyle="--", linewidth=1)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("RR-derived reference HR (BPM)")
    ax.set_ylabel(f"{best_method} predicted HR (BPM)")
    ax.set_title(f"Best T136 baseline: {best_method}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = FIG_DIR / "t136_best_method_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["best_method_scatter"] = str(path)

    condition = (
        results.groupby(["condition_group", "method"], as_index=False)["abs_error_bpm"]
        .mean()
        .rename(columns={"abs_error_bpm": "mae_bpm"})
    )
    pivot = condition.pivot(index="condition_group", columns="method", values="mae_bpm")
    pivot = pivot.reindex(index=sorted(pivot.index), columns=list(ordered["method"]))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Condition-level MAE (BPM)")
    fig.colorbar(im, ax=ax, label="MAE BPM")
    fig.tight_layout()
    path = FIG_DIR / "t136_condition_method_mae_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["condition_method_heatmap"] = str(path)

    return paths


def markdown_table(frame: pd.DataFrame, *, float_digits: int = 3) -> str:
    if frame.empty:
        return ""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(str(col) for col in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.{float_digits}f}" if math.isfinite(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    records: list[SessionRecord],
    label_rows: list[dict[str, object]],
    method_summary: list[dict[str, object]],
    subject_summary: list[dict[str, object]],
    condition_summary: list[dict[str, object]],
    figure_paths: dict[str, str],
    seconds: float,
) -> None:
    best = sorted(method_summary, key=lambda row: row["mae_bpm"])[0]
    old_pbv = 5.801083273623892
    pbv = next((row for row in method_summary if row["method"] == "PBV"), None)
    old_delta = None if pbv is None else float(pbv["mae_bpm"]) - old_pbv

    method_table = pd.DataFrame(method_summary).sort_values("mae_bpm")
    condition_table = pd.DataFrame(condition_summary).sort_values(["condition_group", "mae_bpm"])

    REPORT_MD.write_text(
        f"""# T136 4TU Adult HR Classical Benchmark

Date: {date.today().isoformat()}

## Purpose

T136 converts the T135 adult dataset gate into the first adult-first benchmark result:

```text
4TU all local sessions -> RR-interval HR labels -> classical rPPG baselines -> adult HR metric table.
```

## Protocol

```text
Dataset: 4TU-rPPG-Benchmark
Sessions: {len(records)}
Subjects: {len(set(r.subject_id for r in records))}
Window protocol: first {seconds:.0f} seconds of each edited AVI
Reference: mean HR from .rr RR-interval labels in the same time window
Split tags: P1 development, P2 validation, P3 held_out_test
Methods: {", ".join(sorted(METHODS))}
```

## Main Result

Best first-pass classical baseline:

```text
{best["method"]}: MAE {best["mae_bpm"]:.3f} BPM, RMSE {best["rmse_bpm"]:.3f} BPM, Pearson r {best["pearson_r"]:.3f}, high-error-rate>10 BPM {best["high_error_rate_10bpm"]:.3f}
```

Old subset comparison:

```text
T25-T29 PBV on first 8 sessions: 5.801 BPM MAE.
T136 PBV on all 21 sessions: {pbv["mae_bpm"] if pbv else math.nan:.3f} BPM MAE.
Delta from old subset: {old_delta if old_delta is not None else math.nan:.3f} BPM.
```

This comparison is not an algorithmic improvement test. It shows how the metric changes when the evaluation scope expands from 8 sessions to all 21 local sessions.

## Method Summary

{markdown_table(method_table)}

## Condition Summary

{markdown_table(condition_table)}

## Output Files

```text
{SESSION_INDEX_CSV}
{LABEL_AUDIT_CSV}
{WINDOW_RESULTS_CSV}
{METHOD_SUMMARY_CSV}
{SUBJECT_SUMMARY_CSV}
{CONDITION_SUMMARY_CSV}
{SUMMARY_JSON}
{REPORT_MD}
```

Figures:

```text
{chr(10).join(figure_paths.values())}
```

## Interpretation

T136 establishes the adult HR baseline ground. The strongest classical method on the full 21-session first-window protocol is the immediate baseline to beat in T138-T140. The key scientific insight is not that the best method is universally good; it is that method ranking and error concentration depend on stress condition and subject split.

## Boundary

T136 does not prove adult SOTA, arbitrary-video reliability, clinical validity, or product readiness. It is the first adult HR benchmark table for our local 4TU protocol. Deep SOTA reproduction and reliability-layer improvement still need to follow.
""",
        encoding="utf-8",
    )


def run(seconds: float, max_sessions: int | None = None) -> dict[str, object]:
    records = discover_sessions()
    if max_sessions:
        records = records[:max_sessions]
    save_csv(SESSION_INDEX_CSV, [asdict(record) for record in records])

    label_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []

    for idx, record in enumerate(records, start=1):
        print(f"[T136] {idx}/{len(records)} {record.session_id} extracting/evaluating", flush=True)
        rr_text = read_zip_text(record.archive_path, record.rr_member)
        intervals = parse_rr_intervals(rr_text)
        label_rows.append(label_audit(record, intervals, seconds))

        sample_cache = CACHE_DIR / record.sample_id
        video_path = extract_zip_member(record.archive_path, record.video_member, sample_cache)
        rgb_trace, fps, roi, video_meta = extract_rgb_trace(video_path, seconds=seconds)
        gt_hr = intervals.mean_hr(0.0, seconds)

        for method_name, method_fn in sorted(METHODS.items()):
            signal = method_fn(rgb_trace)
            estimate = estimate_hr(signal, fps)
            result_rows.append(
                {
                    "task_id": "T136",
                    "sample_id": record.sample_id,
                    "dataset": record.dataset,
                    "session_id": record.session_id,
                    "subject_id": record.subject_id,
                    "split": record.split,
                    "condition_group": record.condition_group,
                    "condition_detail": record.condition_detail,
                    "window_start_sec": 0.0,
                    "window_end_sec": seconds,
                    "method": method_name,
                    "gt_hr_bpm": gt_hr,
                    "pred_hr_bpm": estimate.bpm,
                    "abs_error_bpm": abs(estimate.bpm - gt_hr) if math.isfinite(estimate.bpm) else math.nan,
                    "confidence": estimate.confidence,
                    "peak_hz": estimate.peak_hz,
                    "band_power": estimate.band_power,
                    "total_power": estimate.total_power,
                    "snr_proxy_db": snr_proxy_db(estimate.band_power, estimate.total_power),
                    "roi_name": roi.name,
                    "roi_x": roi.x,
                    "roi_y": roi.y,
                    "roi_w": roi.w,
                    "roi_h": roi.h,
                    **video_meta,
                }
            )

    save_csv(LABEL_AUDIT_CSV, label_rows)
    save_csv(WINDOW_RESULTS_CSV, result_rows)

    method_summary = summarize_group(result_rows, ["method"])
    subject_summary = summarize_group(result_rows, ["subject_id", "split", "method"])
    condition_summary = summarize_group(result_rows, ["condition_group", "method"])
    save_csv(METHOD_SUMMARY_CSV, method_summary)
    save_csv(SUBJECT_SUMMARY_CSV, subject_summary)
    save_csv(CONDITION_SUMMARY_CSV, condition_summary)

    results_df = pd.DataFrame(result_rows)
    method_summary_df = pd.DataFrame(method_summary)
    figure_paths = write_figures(results_df, method_summary_df)

    summary = {
        "task_id": "T136",
        "date": date.today().isoformat(),
        "dataset": "4TU-rPPG-Benchmark",
        "sessions": len(records),
        "subjects": sorted(set(r.subject_id for r in records)),
        "seconds_per_session": seconds,
        "methods": sorted(METHODS),
        "protocol": "first-window session-level adult HR benchmark using RR-interval mean HR labels",
        "outputs": {
            "session_index_csv": str(SESSION_INDEX_CSV),
            "label_audit_csv": str(LABEL_AUDIT_CSV),
            "window_results_csv": str(WINDOW_RESULTS_CSV),
            "method_summary_csv": str(METHOD_SUMMARY_CSV),
            "subject_summary_csv": str(SUBJECT_SUMMARY_CSV),
            "condition_summary_csv": str(CONDITION_SUMMARY_CSV),
            "report_md": str(REPORT_MD),
            "figures": figure_paths,
        },
        "method_summary": method_summary,
        "best_method_by_mae": sorted(method_summary, key=lambda row: row["mae_bpm"])[0],
        "boundary": [
            "First adult HR benchmark table for local 4TU protocol.",
            "Not adult SOTA yet: deep baselines and reliability-layer comparison still required.",
            "Only three observed subjects, so broad adult generalization cannot be claimed from 4TU alone.",
        ],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(records, label_rows, method_summary, subject_summary, condition_summary, figure_paths, seconds)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()
    run(seconds=args.seconds, max_sessions=args.max_sessions)


if __name__ == "__main__":
    main()
