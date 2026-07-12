from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_ID = "T654"
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"

FEATURE_CSV = EXP / "t650_long_trace_window_feature_aggregation.csv"
GRAPH_NPZ = EXP / "t651_data_fused_roi_graph_variants.npz"
OUT_METRICS = EXP / "t654_graph_candidate_risk_selector_metrics.csv"
OUT_PREDS = EXP / "t654_graph_candidate_risk_selector_predictions.csv"
OUT_BOOTSTRAP = EXP / "t654_graph_candidate_risk_selector_bootstrap.csv"
OUT_SUMMARY = EXP / "t654_graph_candidate_risk_selector_summary.json"
OUT_MD = DOCS / "t654_graph_candidate_risk_selector.md"

ROI_ORDER = ["forehead", "nasal_bridge", "nose_tip", "left_cheek", "right_cheek", "chin"]
METHODS = ["green", "pos", "chrom"]
ROI_FEATURE_COLUMNS = [
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "lab_l",
    "lab_a",
    "lab_b",
    "coverage_mean",
    "green_hr_bpm",
    "green_snr_db",
    "green_peak_support",
    "pos_hr_bpm",
    "pos_snr_db",
    "pos_peak_support",
    "chrom_hr_bpm",
    "chrom_snr_db",
    "chrom_peak_support",
]
ALPHA = 10.0
RELEASE_PRED_ERROR_THRESHOLD = 10.0
UNSAFE_ERROR_THRESHOLD = 10.0
BOOTSTRAP_REPS = 1000
SEED = 654


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def graph_variant_names(keys: list[str]) -> list[str]:
    return sorted([key[: -len("_row")] for key in keys if key.endswith("_row")])


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    reg = np.eye(x_aug.shape[1]) * alpha
    reg[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)


def ridge_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    return x_aug @ weights


def mae(errors: np.ndarray) -> float:
    if len(errors) == 0:
        return float("nan")
    return float(np.mean(np.abs(errors)))


def build_clip_tensor(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    feature_cols = [col for col in ROI_FEATURE_COLUMNS if col in df.columns]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median = df[col].median(skipna=True)
        if not np.isfinite(median):
            median = 0.0
        df[col] = df[col].fillna(float(median))

    clips = sorted(df["clip_id"].unique().tolist())
    x = np.zeros((len(clips), len(ROI_ORDER), len(feature_cols)), dtype=float)
    y = np.zeros((len(clips),), dtype=float)
    groups: list[str] = []
    buckets: list[str] = []
    for ci, clip_id in enumerate(clips):
        clip_df = df[df["clip_id"] == clip_id]
        first = clip_df.iloc[0]
        y[ci] = float(first["reference_hr_bpm"])
        groups.append(str(first["source_key"]))
        buckets.append(str(first["preflight_bucket"]))
        for ri, roi in enumerate(ROI_ORDER):
            row = clip_df[clip_df["roi"] == roi]
            if row.empty:
                continue
            x[ci, ri, :] = row.iloc[0][feature_cols].to_numpy(dtype=float)
    return x, y, np.asarray(groups), clips, np.asarray(buckets)


def candidate_rows_for_variant(
    df: pd.DataFrame,
    graph: np.ndarray,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    x, y, groups, clips, buckets = build_clip_tensor(df)
    graph_x = np.einsum("ij,njf->nif", graph, x)

    feature_rows: list[np.ndarray] = []
    targets: list[float] = []
    clip_indices: list[int] = []
    meta: list[dict[str, Any]] = []
    method_eye = np.eye(len(METHODS), dtype=float)
    roi_eye = np.eye(len(ROI_ORDER), dtype=float)

    for ci, clip_id in enumerate(clips):
        clip_df = df[df["clip_id"] == clip_id]
        candidate_values = []
        for _, row in clip_df.iterrows():
            for method in METHODS:
                candidate_values.append(float(row[f"{method}_hr_bpm"]))
        candidate_mean = float(np.mean(candidate_values))
        for ri, roi in enumerate(ROI_ORDER):
            row = clip_df[clip_df["roi"] == roi].iloc[0]
            for mi, method in enumerate(METHODS):
                hr = float(row[f"{method}_hr_bpm"])
                snr = float(row[f"{method}_snr_db"])
                support = float(row[f"{method}_peak_support"])
                target = abs(hr - y[ci])
                features = np.concatenate(
                    [
                        graph_x[ci, ri, :],
                        roi_eye[ri],
                        method_eye[mi],
                        np.asarray([hr, snr, support, abs(hr - candidate_mean), candidate_mean], dtype=float),
                    ]
                )
                feature_rows.append(features)
                targets.append(target)
                clip_indices.append(ci)
                meta.append(
                    {
                        "clip_id": clip_id,
                        "source_key": groups[ci],
                        "bucket": buckets[ci],
                        "roi": roi,
                        "method": method,
                        "truth_hr_bpm": y[ci],
                        "candidate_hr_bpm": hr,
                        "candidate_abs_error": target,
                    }
                )
    return np.asarray(feature_rows, dtype=float), np.asarray(targets, dtype=float), np.asarray(clip_indices), groups, meta


def leave_one_source_candidate_selection(
    x: np.ndarray,
    target_error: np.ndarray,
    clip_indices: np.ndarray,
    groups: np.ndarray,
    meta: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pred_error = np.zeros_like(target_error, dtype=float)
    candidate_groups = np.asarray([meta_i["source_key"] for meta_i in meta])
    for group in np.unique(groups):
        test = candidate_groups == group
        train = ~test
        mean = x[train].mean(axis=0, keepdims=True)
        std = x[train].std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        x_train = (x[train] - mean) / std
        x_test = (x[test] - mean) / std
        weights = ridge_fit(x_train, target_error[train], ALPHA)
        pred_error[test] = ridge_predict(x_test, weights)

    selected: list[dict[str, Any]] = []
    for clip_index in sorted(np.unique(clip_indices).tolist()):
        idx = np.where(clip_indices == clip_index)[0]
        best_local = idx[int(np.argmin(pred_error[idx]))]
        row = dict(meta[best_local])
        row["predicted_abs_error"] = round(float(pred_error[best_local]), 6)
        row["selected_abs_error"] = round(float(row["candidate_abs_error"]), 6)
        row["release_decision"] = "release" if pred_error[best_local] <= RELEASE_PRED_ERROR_THRESHOLD else "review"
        row["unsafe_if_released"] = bool(row["release_decision"] == "release" and row["candidate_abs_error"] > UNSAFE_ERROR_THRESHOLD)
        selected.append(row)
    return selected


def summarize_variant(variant: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(selected)
    errors = df["selected_abs_error"].to_numpy(dtype=float)
    released = df[df["release_decision"] == "release"]
    unsafe_release_rate = float(released["unsafe_if_released"].mean()) if len(released) else float("nan")
    return {
        "task_id": TASK_ID,
        "variant": variant,
        "n_clips": int(len(df)),
        "n_groups": int(df["source_key"].nunique()),
        "alpha": ALPHA,
        "selected_mae": round(float(np.mean(errors)), 6),
        "high_hr_mae": round(float(df[df["bucket"] == "high_hr"]["selected_abs_error"].mean()), 6),
        "low_snr_mae": round(float(df[df["bucket"] == "low_snr"]["selected_abs_error"].mean()), 6),
        "nominal_mae": round(float(df[df["bucket"] == "nominal"]["selected_abs_error"].mean()), 6),
        "release_coverage": round(float(len(released) / max(1, len(df))), 6),
        "unsafe_release_rate": round(unsafe_release_rate, 6) if np.isfinite(unsafe_release_rate) else "",
        "overall_unsafe_rate": round(float((df["selected_abs_error"] > UNSAFE_ERROR_THRESHOLD).mean()), 6),
    }


def bootstrap_group_delta(a_df: pd.DataFrame, b_df: pd.DataFrame) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    groups = sorted(a_df["source_key"].unique().tolist())
    a_by_group = {group: a_df[a_df["source_key"] == group]["selected_abs_error"].to_numpy(dtype=float) for group in groups}
    b_by_group = {group: b_df[b_df["source_key"] == group]["selected_abs_error"].to_numpy(dtype=float) for group in groups}
    deltas: list[float] = []
    for _ in range(BOOTSTRAP_REPS):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        a_vals = np.concatenate([a_by_group[group] for group in sampled])
        b_vals = np.concatenate([b_by_group[group] for group in sampled])
        deltas.append(float(a_vals.mean() - b_vals.mean()))
    arr = np.asarray(deltas, dtype=float)
    return {
        "delta_mean": round(float(arr.mean()), 6),
        "ci_low": round(float(np.quantile(arr, 0.025)), 6),
        "ci_high": round(float(np.quantile(arr, 0.975)), 6),
        "prob_delta_lt_0": round(float(np.mean(arr < 0)), 6),
    }


def main() -> int:
    generated_at = now()
    df = pd.read_csv(FEATURE_CSV)
    graphs = np.load(GRAPH_NPZ)
    feature_cols = [col for col in ROI_FEATURE_COLUMNS if col in df.columns]

    all_predictions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    selected_by_variant: dict[str, pd.DataFrame] = {}

    for variant in graph_variant_names(list(graphs.files)):
        x, target_error, clip_indices, groups, meta = candidate_rows_for_variant(df.copy(), graphs[f"{variant}_row"], feature_cols)
        selected = leave_one_source_candidate_selection(x, target_error, clip_indices, groups, meta)
        for row in selected:
            row["task_id"] = TASK_ID
            row["variant"] = variant
        all_predictions.extend(selected)
        metrics.append(summarize_variant(variant, selected))
        selected_by_variant[variant] = pd.DataFrame(selected)

    metrics_sorted = sorted(metrics, key=lambda row: row["selected_mae"])
    best = metrics_sorted[0] if metrics_sorted else {}
    controls = [name for name in ["identity_no_graph", "shuffled_tcm_prior", "random_prior_seed637", "spatial_face_graph"] if name in selected_by_variant]
    candidates = [name for name in selected_by_variant if name.startswith("fused_tcm_data") or name in {"tcm_inspired_prior", "data_corr_graph", "spatial_face_graph"}]
    bootstrap_rows: list[dict[str, Any]] = []
    for candidate in sorted(candidates):
        for control in controls:
            if candidate == control:
                continue
            stats = bootstrap_group_delta(selected_by_variant[candidate], selected_by_variant[control])
            bootstrap_rows.append(
                {
                    "task_id": TASK_ID,
                    "candidate": candidate,
                    "control": control,
                    "metric": "candidate_selected_mae_minus_control_selected_mae",
                    **stats,
                    "passes_strict_gate": bool(stats["ci_high"] < 0.0),
                }
            )

    strict_rows = [row for row in bootstrap_rows if row["candidate"] == best.get("variant") and row["control"] in {"identity_no_graph", "shuffled_tcm_prior", "random_prior_seed637"}]
    strict_gate = bool(strict_rows) and all(row["passes_strict_gate"] for row in strict_rows)
    decision = "candidate_risk_selector_gate_passed_guarded" if strict_gate else "candidate_risk_selector_gate_not_passed"

    summary = {
        "task_id": TASK_ID,
        "generated_at": generated_at,
        "decision": decision,
        "n_clips": int(len(selected_by_variant[next(iter(selected_by_variant))])) if selected_by_variant else 0,
        "n_candidate_options_per_clip": len(ROI_ORDER) * len(METHODS),
        "best_variant": best,
        "release_pred_error_threshold": RELEASE_PRED_ERROR_THRESHOLD,
        "unsafe_error_threshold": UNSAFE_ERROR_THRESHOLD,
        "strict_controls_passed_for_best_variant": strict_gate,
        "claim_boundary": "T654 evaluates graph-conditioned candidate-risk selection on the T650 MCD subset only. It is not cross-dataset SOTA evidence.",
        "outputs": {
            "metrics": str(OUT_METRICS.relative_to(ROOT)),
            "predictions": str(OUT_PREDS.relative_to(ROOT)),
            "bootstrap": str(OUT_BOOTSTRAP.relative_to(ROOT)),
            "summary": str(OUT_SUMMARY.relative_to(ROOT)),
            "md": str(OUT_MD.relative_to(ROOT)),
        },
    }
    write_csv(OUT_METRICS, metrics_sorted)
    write_csv(OUT_PREDS, all_predictions)
    write_csv(OUT_BOOTSTRAP, bootstrap_rows)
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# T654 Graph Candidate-Risk Selector",
        "",
        f"- Decision: `{decision}`",
        f"- Clips: `{summary['n_clips']}`",
        f"- Candidate options per clip: `{summary['n_candidate_options_per_clip']}`",
        f"- Best variant: `{best.get('variant', 'missing')}`",
        f"- Strict control gate passed: `{strict_gate}`",
        f"- Claim boundary: {summary['claim_boundary']}",
        "",
        "| Variant | Selected MAE | High-HR | Low-SNR | Nominal | Release coverage | Unsafe release rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_sorted:
        lines.append(f"| `{row['variant']}` | {row['selected_mae']} | {row['high_hr_mae']} | {row['low_snr_mae']} | {row['nominal_mae']} | {row['release_coverage']} | {row['unsafe_release_rate']} |")
    lines.extend(["", "## Bootstrap Gate", "", "| Candidate | Control | Delta Mean | 95% CI | P(delta < 0) | Pass |", "|---|---|---:|---:|---:|---|"])
    for row in bootstrap_rows:
        lines.append(f"| `{row['candidate']}` | `{row['control']}` | {row['delta_mean']} | [{row['ci_low']}, {row['ci_high']}] | {row['prob_delta_lt_0']} | `{row['passes_strict_gate']}` |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
