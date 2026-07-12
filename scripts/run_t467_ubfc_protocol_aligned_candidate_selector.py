from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import pickle
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOOLBOX = ROOT / "third_party" / "rPPG-Toolbox"
if str(TOOLBOX) not in sys.path:
    sys.path.insert(0, str(TOOLBOX))

from evaluation.post_process import calculate_metric_per_video  # type: ignore  # noqa: E402
from scripts.run_t140_multi_candidate_hr_inference import top_k_hr_peaks  # noqa: E402
from scripts.run_t157_topk_spectral_candidate_selector import (  # noqa: E402
    TOP_K,
    cluster_sample_peaks,
    score_candidates,
    select_oracle,
    select_top,
)
from scripts.run_t136_4tu_adult_hr_benchmark import snr_proxy_db  # noqa: E402
from src.baselines.traditional_rppg import METHODS  # noqa: E402
from src.evaluation.metrics import mae, pearson, rmse  # noqa: E402
from src.vision.face_mesh_roi import (  # noqa: E402
    MediaPipeFaceLandmarkDetector,
    face_region_masks_from_landmarks,
    masked_mean_rgb,
    mentor_aligned_face_rois,
)


TASK_ID = "T467"
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"
OUTPUT = ROOT / "output" / "t467_figures"

REMOTE_DATA_ROOT = Path("/root/autodl-tmp/datasets/adult/UBFC-rPPG/kaggle_extracted")
LOCAL_DATA_ROOT = ROOT.parent / "数据集" / "adult" / "UBFC-rPPG" / "kaggle_extracted"
CLIP_LEN = 180

PEAKS_CSV = EXP / "t467_ubfc_protocol_window_peak_table.csv"
CANDIDATES_CSV = EXP / "t467_ubfc_protocol_window_candidate_table.csv"
SELECTION_CSV = EXP / "t467_ubfc_protocol_window_selection_table.csv"
PAIRED_CSV = EXP / "t467_ubfc_protocol_window_paired_comparison.csv"
POLICY_SUMMARY_CSV = EXP / "t467_ubfc_protocol_window_policy_summary.csv"
BOOTSTRAP_CSV = EXP / "t467_ubfc_protocol_window_bootstrap.csv"
SUMMARY_JSON = EXP / "t467_ubfc_protocol_aligned_candidate_selector_summary.json"
REPORT_MD = EXP / f"t467_ubfc_protocol_aligned_candidate_selector_report_{date.today().isoformat()}.md"
DOC_MD = DOCS / "t467_ubfc_protocol_aligned_candidate_selector.md"

TASK_REGISTRY = DOCS / "execution_task_registry.md"
LEARNING_JOURNAL = DOCS / "phase_learning_journal.md"
PROJECT_STATUS = DOCS / "project_status.md"
PAPER_CLAIMS = DOCS / "paper_claims_tracker.md"
INNOVATION_LOG = DOCS / "innovation_log.md"
PROBLEM_LOG = DOCS / "problem_and_improvement_log.md"
EVIDENCE_TABLE = EXP / "experiment_evidence_table.csv"


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def append_once(path: Path, marker: str, block: str) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        return
    path.write_text(old.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]] | pd.DataFrame, columns: list[str]) -> str:
    if isinstance(rows, pd.DataFrame):
        data = rows.to_dict("records")
    else:
        data = rows
    if not data:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in data:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = "" if not math.isfinite(value) else f"{value:.4f}"
            vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def tscan_pickle_path() -> Path:
    pattern = ROOT / "output" / "rppg_toolbox_runs" / "t393_ubfc" / "tscan" / "*" / "saved_test_outputs" / "*.pickle"
    matches = [Path(p) for p in glob.glob(str(pattern))]
    if not matches:
        raise FileNotFoundError(f"No TSCAN saved output pickle matched {pattern}")
    return matches[0]


def data_root() -> Path:
    if REMOTE_DATA_ROOT.exists():
        return REMOTE_DATA_ROOT
    if LOCAL_DATA_ROOT.exists():
        return LOCAL_DATA_ROOT
    raise FileNotFoundError(f"UBFC root not found: {REMOTE_DATA_ROOT} or {LOCAL_DATA_ROOT}")


def subject_number(subject_key: str) -> int:
    return int(subject_key.replace("subject", ""))


def video_path_for_subject(subject_key: str) -> Path:
    return data_root() / f"subject{subject_number(subject_key)}" / "vid.avi"


def to_numpy_1d(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float).reshape(-1)


def load_tscan_windows(max_subjects: int | None = None, max_windows: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = tscan_pickle_path()
    with path.open("rb") as f:
        obj = pickle.load(f)
    fs = int(obj.get("fs", 30))
    rows: list[dict[str, Any]] = []
    subjects = sorted(obj["predictions"].keys(), key=lambda s: subject_number(s))
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    for subject in subjects:
        window_ids = sorted(obj["predictions"][subject].keys())
        for window_id in window_ids:
            pred = to_numpy_1d(obj["predictions"][subject][window_id])
            label = to_numpy_1d(obj["labels"][subject][window_id])
            gt_hr, pred_hr, snr, macc = calculate_metric_per_video(
                pred,
                label,
                fs=fs,
                diff_flag=True,
                use_bandpass=True,
                hr_method="FFT",
            )
            rows.append(
                {
                    "sample_id": f"{subject}_w{int(window_id):03d}",
                    "subject_key": subject,
                    "subject_id": f"subject{subject_number(subject):02d}",
                    "window_index": int(window_id),
                    "start_frame": int(window_id) * CLIP_LEN,
                    "end_frame": int(window_id) * CLIP_LEN + CLIP_LEN,
                    "fs": fs,
                    "gt_hr_bpm": float(gt_hr),
                    "tscan_pred_bpm": float(pred_hr),
                    "tscan_abs_error_bpm": abs(float(pred_hr) - float(gt_hr)),
                    "tscan_snr": float(snr),
                    "tscan_macc": float(macc),
                    "pickle_path": str(path),
                }
            )
            if max_windows is not None and len(rows) >= max_windows:
                return rows, {"pickle_path": str(path), "fs": fs}
    return rows, {"pickle_path": str(path), "fs": fs}


def iter_video_windows(video_path: Path, target_windows: set[int]) -> dict[int, list[np.ndarray]]:
    import imageio.v3 as iio

    out: dict[int, list[np.ndarray]] = {}
    if not target_windows:
        return out
    max_window = max(target_windows)
    current: list[np.ndarray] = []
    current_idx = 0
    for frame in iio.imiter(video_path):
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        current.append(np.ascontiguousarray(arr))
        if len(current) == CLIP_LEN:
            if current_idx in target_windows:
                out[current_idx] = current
            current = []
            current_idx += 1
            if current_idx > max_window:
                break
    return out


def window_peak_rows(
    window_meta: dict[str, Any],
    frames: list[np.ndarray],
    detector: MediaPipeFaceLandmarkDetector,
) -> list[dict[str, Any]]:
    if len(frames) < CLIP_LEN:
        return []
    landmarks = detector.detect(frames[0])
    regions = face_region_masks_from_landmarks(frames[0], landmarks) if landmarks is not None else mentor_aligned_face_rois(frames[0])
    traces: dict[str, list[np.ndarray]] = {region.name: [] for region in regions}
    for frame in frames:
        for region in regions:
            traces[region.name].append(masked_mean_rgb(frame, region))

    rows: list[dict[str, Any]] = []
    fs = float(window_meta["fs"])
    for region in regions:
        rgb = np.asarray(traces[region.name], dtype=float)
        if rgb.ndim != 2 or rgb.shape[0] < CLIP_LEN // 2:
            continue
        for method_name, method_fn in sorted(METHODS.items()):
            try:
                signal = method_fn(rgb)
                peaks, band_power, total_power = top_k_hr_peaks(signal, fs, top_k=TOP_K)
            except Exception as exc:
                rows.append(
                    {
                        "task_id": TASK_ID,
                        **window_meta,
                        "roi_name": region.name,
                        "method": method_name,
                        "window_id": "full_0_60",
                        "extract_error": type(exc).__name__,
                    }
                )
                continue
            snr = snr_proxy_db(band_power, total_power)
            for peak in peaks:
                bpm = safe_float(peak.get("peak_bpm"))
                rows.append(
                    {
                        "task_id": TASK_ID,
                        **window_meta,
                        "roi_name": region.name,
                        "method": method_name,
                        "window_id": "full_0_60",
                        "protocol_window_id": "clip_180_frames",
                        "window_start_sec": window_meta["start_frame"] / fs,
                        "window_end_sec": window_meta["end_frame"] / fs,
                        "peak_bpm": bpm,
                        "peak_hz": safe_float(peak.get("peak_hz")),
                        "rank": int(safe_float(peak.get("rank"), 99.0)),
                        "power_fraction": safe_float(peak.get("power_fraction")),
                        "peak_power": safe_float(peak.get("peak_power")),
                        "band_power": band_power,
                        "total_power": total_power,
                        "snr_proxy_db": snr,
                        "abs_error_bpm": abs(bpm - safe_float(window_meta["gt_hr_bpm"])),
                        "roi_area_px": getattr(region, "area_px", ""),
                        "roi_coverage": getattr(region, "coverage", ""),
                        "roi_interpretive_label": getattr(region, "interpretive_label", ""),
                        "extract_error": "",
                    }
                )
    return rows


def extract_protocol_peaks(t_windows: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in t_windows:
        by_subject.setdefault(str(row["subject_key"]), []).append(row)

    for sidx, (subject, subject_rows) in enumerate(sorted(by_subject.items(), key=lambda kv: subject_number(kv[0])), start=1):
        video_path = video_path_for_subject(subject)
        target_windows = {int(row["window_index"]) for row in subject_rows}
        print(f"[T467] subject {sidx}/{len(by_subject)} {subject}: windows={len(target_windows)} video={video_path}", flush=True)
        frame_windows = iter_video_windows(video_path, target_windows)
        with MediaPipeFaceLandmarkDetector() as detector:
            for meta in subject_rows:
                frames = frame_windows.get(int(meta["window_index"]), [])
                if len(frames) < CLIP_LEN:
                    rows.append({**meta, "task_id": TASK_ID, "extract_error": "missing_or_short_video_window"})
                    continue
                rows.extend(window_peak_rows(meta, frames, detector))
    peaks = pd.DataFrame(rows)
    peaks.to_csv(PEAKS_CSV, index=False, encoding="utf-8-sig")
    return peaks


def build_candidates(peaks: pd.DataFrame) -> pd.DataFrame:
    frames = []
    valid = peaks[pd.to_numeric(peaks.get("peak_bpm"), errors="coerce").notna()].copy()
    for _, group in valid.groupby("sample_id", sort=True):
        candidates = cluster_sample_peaks(group)
        if candidates.empty:
            continue
        candidates["t150_selected_bpm"] = np.nan
        candidates["t150_abs_error_bpm"] = np.nan
        candidates["t150_confidence"] = np.nan
        candidates["t150_reason"] = "not_used_protocol_window"
        candidates["dist_to_t150"] = np.nan
        frames.append(candidates)
    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not out.empty:
        out = score_candidates(out)
    out.to_csv(CANDIDATES_CSV, index=False, encoding="utf-8-sig")
    return out


def selection_table(candidates: pd.DataFrame, t_windows: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    tscan = pd.DataFrame(
        [
            {
                "task_id": TASK_ID,
                "dataset": "UBFC-rPPG",
                "sample_id": row["sample_id"],
                "subject_id": row["subject_id"],
                "subject_key": row["subject_key"],
                "window_index": row["window_index"],
                "policy": "TSCAN_rPPGToolbox",
                "gt_hr_bpm": row["gt_hr_bpm"],
                "selected_bpm": row["tscan_pred_bpm"],
                "selected_abs_error_bpm": row["tscan_abs_error_bpm"],
                "released": 1,
                "support_rois": "",
                "support_methods": "",
                "candidate_id": "",
            }
            for row in t_windows
        ]
    )
    rows.append(tscan)
    if not candidates.empty:
        top = select_top(candidates, "T467_mediapipe_multiroi_candidate_selector")
        oracle = select_oracle(candidates)
        oracle["policy"] = "T467_candidate_oracle_same_windows"
        rows.extend([top, oracle])
    selection = pd.concat(rows, ignore_index=True, sort=False)
    selection.to_csv(SELECTION_CSV, index=False, encoding="utf-8-sig")
    return selection


def summarize_selection(selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in selection.groupby("policy", sort=True):
        gt = pd.to_numeric(group["gt_hr_bpm"], errors="coerce").to_numpy(float)
        pred = pd.to_numeric(group["selected_bpm"], errors="coerce").to_numpy(float)
        released = pd.to_numeric(group.get("released", 1), errors="coerce").fillna(1).to_numpy(float) > 0
        finite = np.isfinite(gt) & np.isfinite(pred)
        rel = finite & released
        errors = np.abs(pred[rel] - gt[rel])
        rows.append(
            {
                "task_id": TASK_ID,
                "dataset": "UBFC-rPPG",
                "policy": policy,
                "n_total": int(finite.sum()),
                "released": int(rel.sum()),
                "coverage": float(rel.sum() / finite.sum()) if finite.sum() else 0.0,
                "mae_bpm": mae(gt[rel], pred[rel]) if rel.sum() else math.nan,
                "rmse_bpm": rmse(gt[rel], pred[rel]) if rel.sum() else math.nan,
                "pearson_r": pearson(gt[rel], pred[rel]) if rel.sum() else math.nan,
                "median_abs_error_bpm": float(np.median(errors)) if len(errors) else math.nan,
                "p90_abs_error_bpm": float(np.percentile(errors, 90)) if len(errors) else math.nan,
                "unsafe_10bpm_count": int(np.sum(errors > 10.0)),
                "unsafe_10bpm_rate": float(np.mean(errors > 10.0)) if len(errors) else math.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("mae_bpm", na_position="last")
    out.to_csv(POLICY_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    return out


def paired_comparison(selection: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = selection[selection["policy"] == "TSCAN_rPPGToolbox"].set_index("sample_id")
    policies = [p for p in sorted(selection["policy"].dropna().unique()) if p != "TSCAN_rPPGToolbox"]
    paired_rows = []
    boot_rows = []
    rng = np.random.default_rng(467)
    for policy in policies:
        current = selection[selection["policy"] == policy].set_index("sample_id")
        ids = sorted(set(base.index) & set(current.index))
        if not ids:
            continue
        base_err = pd.to_numeric(base.loc[ids, "selected_abs_error_bpm"], errors="coerce").to_numpy(float)
        cur_err = pd.to_numeric(current.loc[ids, "selected_abs_error_bpm"], errors="coerce").to_numpy(float)
        finite = np.isfinite(base_err) & np.isfinite(cur_err)
        ids = list(np.asarray(ids)[finite])
        base_err = base_err[finite]
        cur_err = cur_err[finite]
        delta = cur_err - base_err
        for sample_id, a, b, d in zip(ids, base_err, cur_err, delta):
            row = current.loc[sample_id]
            paired_rows.append(
                {
                    "task_id": TASK_ID,
                    "sample_id": sample_id,
                    "subject_id": row.get("subject_id", ""),
                    "window_index": row.get("window_index", ""),
                    "policy": policy,
                    "baseline_policy": "TSCAN_rPPGToolbox",
                    "tscan_abs_error_bpm": float(a),
                    "policy_abs_error_bpm": float(b),
                    "delta_abs_error_bpm": float(d),
                    "policy_better": int(d < 0),
                }
            )
        if len(delta):
            boot = []
            for _ in range(5000):
                idx = rng.integers(0, len(delta), size=len(delta))
                boot.append(float(np.mean(delta[idx])))
            ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
            boot_rows.append(
                {
                    "task_id": TASK_ID,
                    "policy": policy,
                    "baseline_policy": "TSCAN_rPPGToolbox",
                    "n_paired": int(len(delta)),
                    "mean_delta_mae_policy_minus_tscan": float(np.mean(delta)),
                    "median_delta_abs_error": float(np.median(delta)),
                    "bootstrap_ci95_low": float(ci_low),
                    "bootstrap_ci95_high": float(ci_high),
                    "policy_better_fraction": float(np.mean(delta < 0)),
                    "claim_direction": "policy_better" if ci_high < 0 else ("policy_worse" if ci_low > 0 else "inconclusive"),
                }
            )
    paired = pd.DataFrame(paired_rows)
    boot = pd.DataFrame(boot_rows)
    paired.to_csv(PAIRED_CSV, index=False, encoding="utf-8-sig")
    boot.to_csv(BOOTSTRAP_CSV, index=False, encoding="utf-8-sig")
    return paired, boot


def update_evidence(summary: dict[str, Any]) -> None:
    rows = read_csv(EVIDENCE_TABLE)
    rows = [r for r in rows if r.get("evidence_id") != "E-0467"]
    rows.append(
        {
            "evidence_id": "E-0467",
            "task_id": TASK_ID,
            "date": date.today().isoformat(),
            "artifact": str(SUMMARY_JSON),
            "metric_or_observation": "UBFC protocol-aligned TSCAN vs MediaPipe multi-candidate selector",
            "result": f"decision={summary['decision']}; n_windows={summary['n_windows']}; selector_mae={summary.get('selector_mae_bpm')}; tscan_mae={summary.get('tscan_mae_bpm')}; paired_claim={summary.get('paired_claim_direction')}",
            "claim_supported": "Supports or blocks direct superiority claim under identical 180-frame UBFC window protocol.",
            "claim_boundary": summary["claim_boundary"],
            "next_action": summary["next_task"],
        }
    )
    write_csv(
        EVIDENCE_TABLE,
        rows,
        [
            "evidence_id",
            "task_id",
            "date",
            "artifact",
            "metric_or_observation",
            "result",
            "claim_supported",
            "claim_boundary",
            "next_action",
        ],
    )


def update_docs(summary: dict[str, Any], policy_summary: pd.DataFrame, boot: pd.DataFrame) -> None:
    report = f"""# T467 UBFC Protocol-Aligned Candidate Selector

Generated: {summary['generated_at']}

Decision: `{summary['decision']}`

## Purpose

T467 compares the strongest reproduced deep baseline, TSCAN, against our MediaPipe multi-ROI multi-candidate selector on the same UBFC subjects and the same 180-frame clip windows stored in the rPPG-Toolbox saved outputs.

This is the first valid bridge from deep baseline reproduction to our paper claim gate.

## Metrics

{markdown_table(policy_summary, ['policy', 'n_total', 'coverage', 'mae_bpm', 'rmse_bpm', 'pearson_r', 'median_abs_error_bpm', 'p90_abs_error_bpm', 'unsafe_10bpm_rate'])}

## Paired Test Against TSCAN

{markdown_table(boot, ['policy', 'n_paired', 'mean_delta_mae_policy_minus_tscan', 'bootstrap_ci95_low', 'bootstrap_ci95_high', 'policy_better_fraction', 'claim_direction'])}

## Insight

If the selector wins, the claim is that physiology-constrained multi-candidate inference can outperform an end-to-end temporal deep baseline on the same protocol. If it loses or is inconclusive, the result is still scientifically useful: it identifies whether our innovation should shift toward reliability, refusal, uncertainty calibration, or a learned candidate selector rather than a direct MAE superiority claim.

## Claim Boundary

{summary['claim_boundary']}
"""
    REPORT_MD.write_text(report, encoding="utf-8")
    DOC_MD.write_text(report, encoding="utf-8")
    marker = "<!-- T467_PROTOCOL_ALIGNED_SELECTOR -->"
    append_once(
        TASK_REGISTRY,
        marker,
        f"""
{marker}
| T467 | UBFC protocol-aligned TSCAN vs MediaPipe multi-candidate selector | `scripts/run_t467_ubfc_protocol_aligned_candidate_selector.py`; `{SUMMARY_JSON}`; `{POLICY_SUMMARY_CSV}`; `{BOOTSTRAP_CSV}` | {summary['decision']} |
""",
    )
    append_once(
        LEARNING_JOURNAL,
        marker,
        f"""
{marker}
# T467 teaching note: identical-protocol comparison

目的：把我们的候选峰选择器放到和 TSCAN 完全相同的 UBFC 180-frame window 上评价，解决之前“subject-level 指标”和“rPPG-Toolbox window-level 指标”不能直接比较的问题。

实现：读取 TSCAN `saved_test_outputs` pickle，重新计算每个 window 的 `gt_hr_bpm` 和 `TSCAN_rPPGToolbox` prediction；然后对同一个 video clip 用 MediaPipe Face Mesh 生成导师对齐 ROI，在每个 ROI 上跑 GREEN/CHROM/POS/PBV/ICA/LGI，保留 top-K spectral peaks，聚类后用 physiology-aware score 选择候选 HR。

结果：`{summary['decision']}`。TSCAN MAE=`{summary.get('tscan_mae_bpm')}`，selector MAE=`{summary.get('selector_mae_bpm')}`，paired claim=`{summary.get('paired_claim_direction')}`。

insight：这一步决定我们能不能把论文主 claim 写成“准确率超过强 deep baseline”。如果 paired CI 不支持，我们必须把下一步转向 learned selector 或 reliability/refusal claim，而不是硬写 SOTA。
""",
    )
    append_once(PROJECT_STATUS, "## T467 protocol-aligned selector", f"\n## T467 protocol-aligned selector\n\nDecision: `{summary['decision']}`. See `{SUMMARY_JSON}`.\n")
    append_once(PAPER_CLAIMS, "## T467 paper-claim update", f"\n## T467 paper-claim update\n\nDecision: `{summary['decision']}`. Direct SOTA claim follows paired bootstrap direction: `{summary.get('paired_claim_direction')}`.\n")
    append_once(INNOVATION_LOG, "## T467 innovation tracking", "\n## T467 innovation tracking\n\nT467 tests whether MediaPipe multi-ROI multi-candidate peak selection is a real algorithmic improvement under the same protocol as TSCAN.\n")
    append_once(PROBLEM_LOG, "## T467 problem/improvement note", "\n## T467 problem/improvement note\n\nProblem: previous UBFC results used subject-level 60 s estimates and could not be compared directly with rPPG-Toolbox deep outputs. Improvement: T467 uses identical 180-frame windows and paired statistics.\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()

    EXP.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    t_windows, meta = load_tscan_windows(max_subjects=args.max_subjects, max_windows=args.max_windows)
    peaks = extract_protocol_peaks(t_windows)
    candidates = build_candidates(peaks)
    selection = selection_table(candidates, t_windows)
    policy_summary = summarize_selection(selection)
    _, boot = paired_comparison(selection)

    def policy_mae(policy: str) -> float | None:
        row = policy_summary[policy_summary["policy"] == policy]
        if row.empty:
            return None
        return safe_float(row.iloc[0].get("mae_bpm"))

    tscan_mae = policy_mae("TSCAN_rPPGToolbox")
    selector_mae = policy_mae("T467_mediapipe_multiroi_candidate_selector")
    selector_boot = boot[boot["policy"] == "T467_mediapipe_multiroi_candidate_selector"]
    paired_direction = selector_boot.iloc[0]["claim_direction"] if not selector_boot.empty else "missing"
    if paired_direction == "policy_better":
        decision = "protocol_aligned_selector_beats_tscan"
        next_task = "T468 cross-dataset/statistical confirmation and ablations"
        boundary = "This supports a direct UBFC same-window superiority claim, but cross-dataset validation and ablations are still required."
    elif paired_direction == "policy_worse":
        decision = "protocol_aligned_selector_underperforms_tscan"
        next_task = "T469 train learned Deep Candidate-ROI Temporal Selector"
        boundary = "Direct SOTA claim is blocked. Use T467 failure cases to train or redesign the learned selector."
    else:
        decision = "protocol_aligned_selector_inconclusive_vs_tscan"
        next_task = "T468/T469 analyze failure taxonomy and train learned selector if needed"
        boundary = "Direct SOTA claim is blocked until paired statistics are conclusive or the method is improved."

    summary = {
        "task_id": TASK_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision": decision,
        "n_windows": len(t_windows),
        "n_peak_rows": int(len(peaks)),
        "n_candidates": int(len(candidates)),
        "tscan_mae_bpm": tscan_mae,
        "selector_mae_bpm": selector_mae,
        "paired_claim_direction": paired_direction,
        "tscan_pickle": meta["pickle_path"],
        "outputs": {
            "peaks": str(PEAKS_CSV),
            "candidates": str(CANDIDATES_CSV),
            "selection": str(SELECTION_CSV),
            "paired": str(PAIRED_CSV),
            "summary": str(SUMMARY_JSON),
            "doc": str(DOC_MD),
        },
        "next_task": next_task,
        "claim_boundary": boundary,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_evidence(summary)
    update_docs(summary, policy_summary, boot)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
