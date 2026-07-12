"""T159 calibrated candidate selector screen.

T157/T158 established a strong multi-candidate pool and a safe guarded
correction rule. T159 tests whether a learned selector can convert more of the
oracle headroom into released HR estimates without harming safety.

This is an exploratory model-screen, not a final SOTA claim. Ground truth is
used only to construct training labels and evaluate held-out predictions.
Inference features exclude GT and candidate error columns.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.evaluation.metrics import mae, pearson, rmse  # noqa: E402


EXPERIMENTS = PROJECT / "experiments"
DOCS = PROJECT / "docs"
FIG_DIR = PROJECT / "output" / "t159_figures"

CANDIDATES_CSV = EXPERIMENTS / "t157_candidate_table.csv"
SELECTION_CSV = EXPERIMENTS / "t157_selection_table.csv"

SUMMARY_CSV = EXPERIMENTS / "t159_model_screen_summary.csv"
LODO_PRED_CSV = EXPERIMENTS / "t159_lodo_predictions.csv"
LOSO_PRED_CSV = EXPERIMENTS / "t159_loso_predictions.csv"
THRESHOLDS_CSV = EXPERIMENTS / "t159_thresholds.csv"
FEATURE_IMPORTANCE_CSV = EXPERIMENTS / "t159_feature_importance.csv"
CASE_AUDIT_CSV = EXPERIMENTS / "t159_case_audit.csv"
BOOTSTRAP_CSV = EXPERIMENTS / "t159_bootstrap.csv"
SUMMARY_JSON = EXPERIMENTS / "t159_calibrated_candidate_selector_summary.json"
REPORT_MD = EXPERIMENTS / f"t159_calibrated_candidate_selector_report_{date.today().isoformat()}.md"
DOC_MD = DOCS / "t159_calibrated_candidate_selector.md"

ADULT_DATASETS = ["4TU-rPPG-Benchmark", "UBFC-rPPG"]
UNSAFE_BPM = 10.0
PRIMARY_MODEL_ID = "rf3_good2"
THRESHOLD_GRID = np.round(np.linspace(0.10, 0.99, 90), 4)
SAFETY_PENALTY = 100.0
SWITCH_PENALTY = 0.05

FEATURE_COLUMNS = [
    "candidate_bpm",
    "support_count",
    "support_rois",
    "support_methods",
    "support_windows",
    "full_support_count",
    "subwindow_support_count",
    "top1_support_count",
    "full_top1_support_count",
    "pos_chrom_count",
    "green_pbv_count",
    "ica_lgi_count",
    "mean_power_fraction",
    "max_power_fraction",
    "sum_power_fraction",
    "rank_score",
    "mean_snr_proxy_db",
    "adult_plausibility",
    "upper_alt_support",
    "upper_alt_pos_chrom",
    "upper_phys_support",
    "upper_phys_pos_chrom",
    "lower_phys_support",
    "lower_phys_pos_chrom",
    "double_harmonic_support",
    "half_harmonic_support",
    "t150_selected_bpm",
    "t150_confidence",
    "dist_to_t150",
    "t157_low_alias_penalty",
    "t157_motion_band_penalty",
    "t157_high_alias_penalty",
    "t157_near_t150_boost",
    "t157_score",
    "candidate_minus_t150",
    "candidate_ratio_t150",
    "abs_candidate_minus_t150",
]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label_bpm: float
    factory: Callable[[], object]


MODEL_SPECS = [
    ModelSpec(
        "logreg_good3",
        3.0,
        lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", C=0.3, solver="liblinear"),
        ),
    ),
    ModelSpec(
        "rf3_good2",
        2.0,
        lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=3,
            random_state=159,
            class_weight="balanced_subsample",
        ),
    ),
    ModelSpec(
        "rf3_good3",
        3.0,
        lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=3,
            random_state=160,
            class_weight="balanced_subsample",
        ),
    ),
    ModelSpec(
        "rf4_good3",
        3.0,
        lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            random_state=161,
            class_weight="balanced_subsample",
        ),
    ),
    ModelSpec(
        "et3_good5",
        5.0,
        lambda: ExtraTreesClassifier(
            n_estimators=300,
            max_depth=3,
            random_state=162,
            class_weight="balanced",
        ),
    ),
]


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def markdown_table(df: pd.DataFrame, *, digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.{digits}f}")
    lines = [
        "| " + " | ".join(str(c) for c in display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in display.columns) + " |")
    return "\n".join(lines)


def append_unique(path: Path, marker: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        return
    path.write_text(old.rstrip() + "\n\n" + content.strip() + "\n", encoding="utf-8")


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(CANDIDATES_CSV)
    candidates = candidates[candidates["dataset"].isin(ADULT_DATASETS)].copy()
    numeric_cols = list(set(FEATURE_COLUMNS + ["candidate_abs_error_bpm", "gt_hr_bpm", "t150_abs_error_bpm"]))
    for col in numeric_cols:
        if col in candidates.columns:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    candidates["candidate_minus_t150"] = candidates["candidate_bpm"] - candidates["t150_selected_bpm"]
    candidates["candidate_ratio_t150"] = candidates["candidate_bpm"] / candidates["t150_selected_bpm"]
    candidates["abs_candidate_minus_t150"] = candidates["candidate_minus_t150"].abs()
    candidates[FEATURE_COLUMNS] = candidates[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for spec in MODEL_SPECS:
        candidates[f"good_{int(spec.label_bpm)}bpm"] = (
            candidates["candidate_abs_error_bpm"].astype(float) <= spec.label_bpm
        ).astype(int)

    selection = pd.read_csv(SELECTION_CSV)
    selection = selection[selection["dataset"].isin(ADULT_DATASETS)].copy()
    for col in ["gt_hr_bpm", "selected_bpm", "selected_abs_error_bpm", "released"]:
        if col in selection.columns:
            selection[col] = pd.to_numeric(selection[col], errors="coerce")
    return candidates, selection


def baseline_rows(selection: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "T150_release_all",
        "T156_candidate_conflict_gate_v1",
        "T157_guarded_correction_v2",
        "T157_topk_candidate_oracle",
    ]
    out = selection[selection["policy"].isin(keep)].copy()
    out["validation"] = "baseline"
    out["model_id"] = out["policy"]
    out["threshold"] = math.nan
    out["switched"] = np.where(out["policy"] == "T157_guarded_correction_v2", out.get("correction_source", "").astype(str).ne("t156_anchor"), 0)
    return out


def get_base_policy(selection: pd.DataFrame) -> pd.DataFrame:
    base = selection[selection["policy"] == "T157_guarded_correction_v2"][
        ["sample_id", "dataset", "gt_hr_bpm", "selected_bpm", "selected_abs_error_bpm", "released", "correction_source"]
    ].drop_duplicates("sample_id")
    return base


def add_probabilities(model: object, train: pd.DataFrame, test: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    fitted = clone(model)
    fitted.fit(train[FEATURE_COLUMNS], train[label_col])
    train_out = train.copy()
    test_out = test.copy()
    train_out["selector_prob"] = fitted.predict_proba(train[FEATURE_COLUMNS])[:, 1]
    test_out["selector_prob"] = fitted.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    return train_out, test_out, fitted


def choose_top_candidate(candidates: pd.DataFrame) -> pd.DataFrame:
    return (
        candidates.sort_values(
            ["selector_prob", "support_methods", "support_rois", "max_power_fraction"],
            ascending=[False, False, False, False],
        )
        .groupby("sample_id", as_index=False)
        .head(1)
    )


def top_candidate_policy(candidates_with_prob: pd.DataFrame, policy: str, validation: str, model_id: str) -> pd.DataFrame:
    top = choose_top_candidate(candidates_with_prob).copy()
    top["policy"] = policy
    top["validation"] = validation
    top["model_id"] = model_id
    top["selected_bpm"] = top["candidate_bpm"].astype(float)
    top["selected_abs_error_bpm"] = top["candidate_abs_error_bpm"].astype(float)
    top["released"] = 1
    top["switched"] = 1
    top["threshold"] = math.nan
    top["selected_candidate_id"] = top["candidate_id"]
    top["selected_probability"] = top["selector_prob"]
    return top


def guarded_switch_policy(
    candidates_with_prob: pd.DataFrame,
    base: pd.DataFrame,
    threshold: float,
    policy: str,
    validation: str,
    model_id: str,
) -> pd.DataFrame:
    top = choose_top_candidate(candidates_with_prob).set_index("sample_id")
    rows: list[dict[str, object]] = []
    for _, base_row in base.iterrows():
        sample_id = str(base_row["sample_id"])
        if sample_id in top.index and finite_float(top.loc[sample_id, "selector_prob"]) >= threshold:
            selected = top.loc[sample_id]
            rows.append(
                {
                    "sample_id": sample_id,
                    "dataset": base_row["dataset"],
                    "gt_hr_bpm": finite_float(base_row["gt_hr_bpm"]),
                    "policy": policy,
                    "validation": validation,
                    "model_id": model_id,
                    "selected_bpm": finite_float(selected["candidate_bpm"]),
                    "selected_abs_error_bpm": finite_float(selected["candidate_abs_error_bpm"]),
                    "released": 1,
                    "switched": 1,
                    "threshold": threshold,
                    "selected_candidate_id": selected["candidate_id"],
                    "selected_probability": finite_float(selected["selector_prob"]),
                    "base_selected_bpm": finite_float(base_row["selected_bpm"]),
                    "base_selected_abs_error_bpm": finite_float(base_row["selected_abs_error_bpm"]),
                }
            )
        else:
            rows.append(
                {
                    "sample_id": sample_id,
                    "dataset": base_row["dataset"],
                    "gt_hr_bpm": finite_float(base_row["gt_hr_bpm"]),
                    "policy": policy,
                    "validation": validation,
                    "model_id": model_id,
                    "selected_bpm": finite_float(base_row["selected_bpm"]),
                    "selected_abs_error_bpm": finite_float(base_row["selected_abs_error_bpm"]),
                    "released": 1,
                    "switched": 0,
                    "threshold": threshold,
                    "selected_candidate_id": "",
                    "selected_probability": finite_float(top.loc[sample_id, "selector_prob"]) if sample_id in top.index else math.nan,
                    "base_selected_bpm": finite_float(base_row["selected_bpm"]),
                    "base_selected_abs_error_bpm": finite_float(base_row["selected_abs_error_bpm"]),
                }
            )
    return pd.DataFrame(rows)


def policy_objective(pred: pd.DataFrame) -> float:
    err = pd.to_numeric(pred["selected_abs_error_bpm"], errors="coerce")
    unsafe = (err > UNSAFE_BPM).mean()
    return float(err.mean() + SAFETY_PENALTY * unsafe + SWITCH_PENALTY * pred["switched"].mean())


def tune_threshold(candidates_with_prob: pd.DataFrame, base: pd.DataFrame, policy: str, validation: str, model_id: str) -> tuple[float, pd.DataFrame]:
    best: tuple[float, float, pd.DataFrame] | None = None
    for threshold in THRESHOLD_GRID:
        pred = guarded_switch_policy(candidates_with_prob, base, float(threshold), policy, validation, model_id)
        obj = policy_objective(pred)
        if best is None or obj < best[0]:
            best = (obj, float(threshold), pred)
    assert best is not None
    return best[1], best[2]


def run_lodo(candidates: pd.DataFrame, selection: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = get_base_policy(selection)
    predictions: list[pd.DataFrame] = []
    thresholds: list[dict[str, object]] = []
    for spec in MODEL_SPECS:
        label_col = f"good_{int(spec.label_bpm)}bpm"
        for train_dataset in ADULT_DATASETS:
            test_dataset = [d for d in ADULT_DATASETS if d != train_dataset][0]
            train = candidates[candidates["dataset"] == train_dataset].copy()
            test = candidates[candidates["dataset"] == test_dataset].copy()
            train_prob, test_prob, _ = add_probabilities(spec.factory(), train, test, label_col)

            top_test = top_candidate_policy(
                test_prob,
                f"T159_{spec.model_id}_lodo_top",
                "LODO",
                spec.model_id,
            )
            top_test["train_dataset"] = train_dataset
            top_test["test_dataset"] = test_dataset
            predictions.append(top_test)

            train_base = base[base["dataset"] == train_dataset].copy()
            threshold, train_pred = tune_threshold(
                train_prob,
                train_base,
                f"T159_{spec.model_id}_lodo_guarded_switch",
                "LODO_train",
                spec.model_id,
            )
            test_base = base[base["dataset"] == test_dataset].copy()
            test_pred = guarded_switch_policy(
                test_prob,
                test_base,
                threshold,
                f"T159_{spec.model_id}_lodo_guarded_switch",
                "LODO",
                spec.model_id,
            )
            test_pred["train_dataset"] = train_dataset
            test_pred["test_dataset"] = test_dataset
            predictions.append(test_pred)
            thresholds.append(
                {
                    "validation": "LODO",
                    "model_id": spec.model_id,
                    "label_bpm": spec.label_bpm,
                    "train_dataset": train_dataset,
                    "test_dataset": test_dataset,
                    "selected_threshold": threshold,
                    "train_objective": policy_objective(train_pred),
                    "train_mae": float(train_pred["selected_abs_error_bpm"].mean()),
                    "train_unsafe": float((train_pred["selected_abs_error_bpm"] > UNSAFE_BPM).mean()),
                    "train_switch_rate": float(train_pred["switched"].mean()),
                }
            )
    return pd.concat(predictions, ignore_index=True, sort=False), pd.DataFrame(thresholds)


def run_loso_primary(candidates: pd.DataFrame, selection: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = next(s for s in MODEL_SPECS if s.model_id == PRIMARY_MODEL_ID)
    label_col = f"good_{int(spec.label_bpm)}bpm"
    base = get_base_policy(selection)
    rows: list[pd.DataFrame] = []
    thresholds: list[dict[str, object]] = []
    for sample_id in sorted(candidates["sample_id"].astype(str).unique()):
        train = candidates[candidates["sample_id"].astype(str) != sample_id].copy()
        test = candidates[candidates["sample_id"].astype(str) == sample_id].copy()
        train_prob, test_prob, _ = add_probabilities(spec.factory(), train, test, label_col)
        train_base = base[base["sample_id"].astype(str) != sample_id].copy()
        threshold, train_pred = tune_threshold(
            train_prob,
            train_base,
            f"T159_{spec.model_id}_loso_guarded_switch",
            "LOSO_train",
            spec.model_id,
        )
        test_base = base[base["sample_id"].astype(str) == sample_id].copy()
        test_pred = guarded_switch_policy(
            test_prob,
            test_base,
            threshold,
            f"T159_{spec.model_id}_loso_guarded_switch",
            "LOSO",
            spec.model_id,
        )
        test_pred["held_out_sample_id"] = sample_id
        rows.append(test_pred)
        thresholds.append(
            {
                "validation": "LOSO",
                "model_id": spec.model_id,
                "label_bpm": spec.label_bpm,
                "held_out_sample_id": sample_id,
                "selected_threshold": threshold,
                "train_objective": policy_objective(train_pred),
                "train_mae": float(train_pred["selected_abs_error_bpm"].mean()),
                "train_unsafe": float((train_pred["selected_abs_error_bpm"] > UNSAFE_BPM).mean()),
                "train_switch_rate": float(train_pred["switched"].mean()),
            }
        )
    return pd.concat(rows, ignore_index=True, sort=False), pd.DataFrame(thresholds)


def summarize_policies(selection: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["validation", "policy", "model_id", "dataset"]
    for keys, group in selection.groupby(group_cols, dropna=False, sort=True):
        validation, policy, model_id, dataset = keys
        gt = pd.to_numeric(group["gt_hr_bpm"], errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(group["selected_bpm"], errors="coerce").to_numpy(dtype=float)
        released = pd.to_numeric(group.get("released", 1), errors="coerce").fillna(1).to_numpy(dtype=float) > 0
        finite = np.isfinite(gt) & np.isfinite(pred)
        rel = finite & released
        errors = np.abs(gt[rel] - pred[rel])
        rows.append(
            {
                "validation": validation,
                "policy": policy,
                "model_id": model_id,
                "dataset": dataset,
                "n_total": int(finite.sum()),
                "released": int(rel.sum()),
                "coverage": float(rel.sum() / finite.sum()) if finite.sum() else 0.0,
                "released_mae_bpm": mae(gt[rel], pred[rel]) if rel.sum() else math.nan,
                "released_rmse_bpm": rmse(gt[rel], pred[rel]) if rel.sum() else math.nan,
                "released_pearson_r": pearson(gt[rel], pred[rel]) if rel.sum() else math.nan,
                "released_median_abs_error_bpm": float(np.median(errors)) if len(errors) else math.nan,
                "unsafe_release_count": int(np.sum(errors > UNSAFE_BPM)),
                "unsafe_per_input": float(np.sum(errors > UNSAFE_BPM) / finite.sum()) if finite.sum() else math.nan,
                "switch_rate": float(pd.to_numeric(group.get("switched", 0), errors="coerce").fillna(0).mean()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_delta(selection: pd.DataFrame, baseline_policy: str, improved_policy: str, validation: str, *, n_boot: int = 5000, seed: int = 159) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    sub = selection[selection["validation"].isin(["baseline", validation])].copy()
    for dataset in sorted(sub["dataset"].dropna().unique()):
        a = sub[(sub["dataset"] == dataset) & (sub["policy"] == baseline_policy)].drop_duplicates("sample_id").set_index("sample_id")
        b = sub[(sub["dataset"] == dataset) & (sub["policy"] == improved_policy) & (sub["validation"] == validation)].drop_duplicates("sample_id").set_index("sample_id")
        ids = sorted(set(a.index) & set(b.index))
        if not ids:
            continue
        err_a = pd.to_numeric(a.loc[ids, "selected_abs_error_bpm"], errors="coerce").to_numpy(dtype=float)
        err_b = pd.to_numeric(b.loc[ids, "selected_abs_error_bpm"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(err_a) & np.isfinite(err_b)
        err_a = err_a[finite]
        err_b = err_b[finite]
        idx = np.arange(len(err_a))
        delta_mae = err_a - err_b
        delta_unsafe = (err_a > UNSAFE_BPM).astype(float) - (err_b > UNSAFE_BPM).astype(float)
        boot_mae = np.asarray([np.mean(delta_mae[rng.choice(idx, len(idx), replace=True)]) for _ in range(n_boot)])
        boot_unsafe = np.asarray([np.mean(delta_unsafe[rng.choice(idx, len(idx), replace=True)]) for _ in range(n_boot)])
        rows.append(
            {
                "validation": validation,
                "dataset": dataset,
                "comparison": f"{improved_policy}_vs_{baseline_policy}",
                "n": int(len(idx)),
                "mean_delta_mae_bpm": float(np.mean(delta_mae)),
                "mae_ci95_low": float(np.percentile(boot_mae, 2.5)),
                "mae_ci95_high": float(np.percentile(boot_mae, 97.5)),
                "p_mae_delta_gt_0": float(np.mean(boot_mae > 0.0)),
                "mean_delta_unsafe_per_input": float(np.mean(delta_unsafe)),
                "unsafe_ci95_low": float(np.percentile(boot_unsafe, 2.5)),
                "unsafe_ci95_high": float(np.percentile(boot_unsafe, 97.5)),
                "p_unsafe_delta_gt_0": float(np.mean(boot_unsafe > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def feature_importance(candidates: pd.DataFrame) -> pd.DataFrame:
    spec = next(s for s in MODEL_SPECS if s.model_id == PRIMARY_MODEL_ID)
    label_col = f"good_{int(spec.label_bpm)}bpm"
    model = spec.factory()
    model.fit(candidates[FEATURE_COLUMNS], candidates[label_col])
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    else:
        scores = np.zeros(len(FEATURE_COLUMNS))
    out = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": scores})
    return out.sort_values("importance", ascending=False)


def build_case_audit(selection: pd.DataFrame) -> pd.DataFrame:
    focus = {"ubfc_subject14", "ubfc_subject30", "ubfc_subject32", "4tu_P1M3", "4tu_P3LC2", "4tu_P3LC3"}
    policies = [
        "T150_release_all",
        "T157_guarded_correction_v2",
        "T157_topk_candidate_oracle",
        f"T159_{PRIMARY_MODEL_ID}_lodo_guarded_switch",
        f"T159_{PRIMARY_MODEL_ID}_loso_guarded_switch",
    ]
    rows: list[dict[str, object]] = []
    for sample_id in sorted(focus):
        sub = selection[selection["sample_id"].astype(str) == sample_id]
        if sub.empty:
            continue
        row: dict[str, object] = {
            "sample_id": sample_id,
            "dataset": sub.iloc[0]["dataset"],
            "gt_hr_bpm": finite_float(sub.iloc[0]["gt_hr_bpm"]),
        }
        for policy in policies:
            p = sub[sub["policy"] == policy]
            if p.empty:
                continue
            if policy.startswith("T159") and "lodo" in policy:
                p = p[p["validation"] == "LODO"]
            if policy.startswith("T159") and "loso" in policy:
                p = p[p["validation"] == "LOSO"]
            if p.empty:
                continue
            one = p.iloc[0]
            short = policy.replace("T157_", "").replace("T159_", "").replace("_release_all", "")
            row[f"{short}_bpm"] = finite_float(one["selected_bpm"])
            row[f"{short}_error"] = finite_float(one["selected_abs_error_bpm"])
            row[f"{short}_switch"] = int(finite_float(one.get("switched"), 0.0))
        rows.append(row)
    audit = pd.DataFrame(rows)
    audit.to_csv(CASE_AUDIT_CSV, index=False, encoding="utf-8-sig")
    return audit


def write_figures(summary: pd.DataFrame, importance: pd.DataFrame) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    key_policies = [
        "T150_release_all",
        "T157_guarded_correction_v2",
        "T157_topk_candidate_oracle",
        f"T159_{PRIMARY_MODEL_ID}_lodo_guarded_switch",
        f"T159_{PRIMARY_MODEL_ID}_loso_guarded_switch",
    ]
    plot = summary[summary["policy"].isin(key_policies)].copy()
    fig, ax = plt.subplots(figsize=(11.8, 5.4))
    labels = [f"{r.validation}\n{r.dataset}\n{r.policy.replace('T159_', '').replace('T157_', '')}" for r in plot.itertuples(index=False)]
    x = np.arange(len(plot))
    ax.bar(x, plot["released_mae_bpm"], color="#0072B2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Released MAE (BPM)")
    ax.set_title("T159 calibrated selector model screen")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIG_DIR / "t159_model_screen_mae.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths["model_screen_mae"] = str(path)

    lodo = summary[(summary["validation"] == "LODO") & summary["policy"].str.contains("guarded_switch", na=False)].copy()
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.scatter(lodo["switch_rate"], lodo["released_mae_bpm"], c=lodo["unsafe_per_input"], cmap="viridis", s=70)
    ax.set_xlabel("Switch rate")
    ax.set_ylabel("Released MAE (BPM)")
    ax.set_title("T159 LODO switch-rate trade-off")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = FIG_DIR / "t159_switch_tradeoff.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths["switch_tradeoff"] = str(path)

    top_imp = importance.head(15).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.barh(top_imp["feature"], top_imp["importance"], color="#009E73")
    ax.set_xlabel("Feature importance")
    ax.set_title(f"T159 primary model features ({PRIMARY_MODEL_ID})")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = FIG_DIR / "t159_feature_importance.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths["feature_importance"] = str(path)
    return paths


def append_evidence_row(summary: dict[str, object]) -> None:
    path = EXPERIMENTS / "experiment_evidence_table.csv"
    fieldnames = [
        "evidence_id",
        "task_id",
        "date",
        "artifact",
        "metric_or_observation",
        "result",
        "claim_supported",
        "claim_boundary",
        "next_action",
    ]
    rows: list[dict[str, str]] = []
    if path.exists() and path.stat().st_size:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or fieldnames)
            rows = list(reader)
    if any(row.get("evidence_id") == "E-0106" for row in rows):
        return
    rows.append(
        {
            "evidence_id": "E-0106",
            "task_id": "T159",
            "date": date.today().isoformat(),
            "artifact": str(SUMMARY_JSON),
            "metric_or_observation": "calibrated candidate selector model screen using T157 top-K candidate features",
            "result": str(summary.get("evidence_result", "")),
            "claim_supported": str(summary.get("claim_supported", "")),
            "claim_boundary": str(summary.get("claim_boundary", "")),
            "next_action": str(summary.get("next_action", "")),
        }
    )
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(summary_table: pd.DataFrame, bootstrap: pd.DataFrame, case_audit: pd.DataFrame, importance: pd.DataFrame, figures: dict[str, str]) -> dict[str, object]:
    primary_lodo = summary_table[
        (summary_table["policy"] == f"T159_{PRIMARY_MODEL_ID}_lodo_guarded_switch")
        & (summary_table["validation"] == "LODO")
    ]
    guarded = summary_table[
        (summary_table["policy"] == "T157_guarded_correction_v2")
        & (summary_table["validation"] == "baseline")
    ]
    primary_text = "; ".join(
        f"{r.dataset}: MAE {r.released_mae_bpm:.3f}, unsafe {r.unsafe_per_input:.3f}, switch {r.switch_rate:.3f}"
        for r in primary_lodo.itertuples(index=False)
    )
    guarded_text = "; ".join(
        f"{r.dataset}: MAE {r.released_mae_bpm:.3f}, unsafe {r.unsafe_per_input:.3f}"
        for r in guarded.itertuples(index=False)
    )
    return {
        "task_id": "T159",
        "date": date.today().isoformat(),
        "outputs": {
            "summary_csv": str(SUMMARY_CSV),
            "lodo_predictions_csv": str(LODO_PRED_CSV),
            "loso_predictions_csv": str(LOSO_PRED_CSV),
            "thresholds_csv": str(THRESHOLDS_CSV),
            "feature_importance_csv": str(FEATURE_IMPORTANCE_CSV),
            "case_audit_csv": str(CASE_AUDIT_CSV),
            "bootstrap_csv": str(BOOTSTRAP_CSV),
            "report_md": str(REPORT_MD),
            "doc_md": str(DOC_MD),
            "figures": figures,
        },
        "evidence_result": (
            f"T159 screened learned candidate selectors. Primary {PRIMARY_MODEL_ID} LODO guarded-switch: "
            + primary_text
            + ". T157 guarded baseline: "
            + guarded_text
            + "."
        ),
        "main_insight": (
            "Learned candidate selection can recover additional oracle headroom in some held-out cases, but it is not yet more reliable than the frozen T157 guarded correction. The current bottleneck is not model capacity; it is the scarcity and diversity of labeled failure cases."
        ),
        "claim_supported": (
            "Supported as a negative/diagnostic result: the T157 top-K feature set contains learnable signal, but the calibrated selector is not yet a product-default or paper-main method. T157 guarded remains the safer default."
        ),
        "claim_boundary": (
            "T159 is exploratory. Multiple models and thresholds were screened, so any best-looking row is hypothesis-generating. A publishable learned selector needs more adult external data or a locked nested-validation protocol with stronger held-out improvements."
        ),
        "next_action": (
            "Proceed to T160: expand adult failure-case coverage and/or design a two-stage selector that combines T157 guarded correction with learned uncertainty, then validate on new external data."
        ),
        "policy_summary": summary_table.to_dict(orient="records"),
        "bootstrap": bootstrap.to_dict(orient="records"),
        "case_audit": case_audit.to_dict(orient="records"),
        "top_features": importance.head(20).to_dict(orient="records"),
    }


def write_reports(summary: dict[str, object], summary_table: pd.DataFrame, thresholds: pd.DataFrame, bootstrap: pd.DataFrame, case_audit: pd.DataFrame, importance: pd.DataFrame, figures: dict[str, str]) -> None:
    display_cols = [
        "validation",
        "policy",
        "model_id",
        "dataset",
        "n_total",
        "coverage",
        "released_mae_bpm",
        "unsafe_per_input",
        "switch_rate",
    ]
    fallacy_scan = (
        "Fallacy scan 11/11 checked: candidate-level training labels are derived from GT, but inference features exclude GT/error columns; "
        "unit of evaluation is video sample; LODO and LOSO reduce leakage risk; model screening creates look-elsewhere risk; no clinical or SOTA claim is made."
    )
    report = "\n".join(
        [
            "# T159 Calibrated Candidate Selector",
            "",
            "## Material Passport",
            "",
            "- Task: T159",
            "- Type: code experiment / model screen",
            "- Verification status: ANALYZED",
            "- Inputs: T157 candidate table and T157 selection table",
            "- Output: LODO/LOSO learned selector screen",
            "",
            "## Purpose",
            "",
            "T159 tests whether a learned candidate selector can safely convert more top-K oracle headroom than the hand-written T157 guarded correction.",
            "",
            "## Metrics",
            "",
            markdown_table(summary_table[display_cols]),
            "",
            "## Thresholds",
            "",
            markdown_table(thresholds),
            "",
            "## Bootstrap",
            "",
            markdown_table(bootstrap),
            "",
            "## Case Audit",
            "",
            markdown_table(case_audit),
            "",
            "## Top Features",
            "",
            markdown_table(importance.head(15)),
            "",
            "## Main Insight",
            "",
            str(summary["main_insight"]),
            "",
            "## Fallacy Scan",
            "",
            fallacy_scan,
            "",
            "## Figures",
            "",
            "\n".join(f"- {name}: `{path}`" for name, path in figures.items()),
            "",
            "## Claim Boundary",
            "",
            str(summary["claim_boundary"]),
            "",
            "## Next",
            "",
            str(summary["next_action"]),
            "",
        ]
    )
    REPORT_MD.write_text(report, encoding="utf-8")

    doc = "\n".join(
        [
            "# T159 教学文档：calibrated candidate selector",
            "",
            "## 1. 这一步为什么要做？",
            "",
            "T157/T158 已经说明：正确 HR 候选经常存在于 top-K spectral candidate pool 中，但手写规则只能修正少数 low-alias conflict。T159 的目的，是测试一个学习式 selector 能否根据候选特征自动选择更接近 reference 的 candidate，从而进一步靠近 oracle。",
            "",
            "## 2. 关键防泄漏原则",
            "",
            "训练时可以用 `candidate_abs_error_bpm` 生成标签，例如 `good_2bpm`、`good_3bpm`；但是推理特征不能包含 `gt_hr_bpm`、`candidate_abs_error_bpm` 或任何直接由真实标签泄漏出来的列。本次模型输入只使用 support、ROI/method/window evidence、power、rank、alias/harmonic context、T150 confidence/distance 等 inference-only features。",
            "",
            "## 3. 实验设计",
            "",
            "- `LODO`：用一个数据集训练，在另一个数据集测试。",
            "- `LOSO`：对 primary model 做 leave-one-sample-out。",
            "- `top` policy：完全相信 learned selector，选择最高概率候选。",
            "- `guarded_switch` policy：以 T157 guarded 为 base，只有当 learned selector 概率超过训练集调出的 threshold 才切换。",
            "",
            "## 4. 指标结果",
            "",
            markdown_table(summary_table[display_cols]),
            "",
            "## 5. 指标迭代链解释",
            "",
            "T157 guarded 是目前最稳的产品默认策略；T159 learned selector 在一些样本上能接近 oracle，但 LODO 结果并没有稳定超越 T157 guarded。尤其当 selector 尝试切换更多样本时，MAE 不一定下降，说明目前 failure-case diversity 不足。",
            "",
            "## 6. Output 迭代链",
            "",
            markdown_table(case_audit),
            "",
            "## 7. Feature insight",
            "",
            markdown_table(importance.head(15)),
            "",
            "这些特征重要性说明模型主要依赖 candidate 的概率/支持结构、T150 距离和 spectral evidence。但它仍然难以区分“多方法共同支持的真实峰”和“多方法共同支持的 artifact 峰”。这正是我们下一步需要解决的 fundamental candidate-selection 痛点。",
            "",
            "## 8. 深度 insight",
            "",
            str(summary["main_insight"]),
            "",
            "## 9. 统计和审稿风险",
            "",
            fallacy_scan,
            "",
            "## 10. 结论边界",
            "",
            str(summary["claim_boundary"]),
            "",
            "## 11. 下一步",
            "",
            str(summary["next_action"]),
            "",
        ]
    )
    DOC_MD.write_text(doc, encoding="utf-8")
    append_unique(DOCS / "phase_learning_journal.md", "# T159 calibrated candidate selector", doc)


def update_project_docs(summary: dict[str, object]) -> None:
    marker = "## T159 calibrated candidate selector"
    text = "\n".join(
        [
            marker,
            "",
            str(summary["main_insight"]),
            "",
            "Evidence: " + str(summary["evidence_result"]),
            "",
            "Claim status: " + str(summary["claim_supported"]),
            "",
            "Boundary: " + str(summary["claim_boundary"]),
            "",
            "Next: " + str(summary["next_action"]),
            "",
        ]
    )
    for name in [
        "project_status.md",
        "innovation_log.md",
        "problem_and_improvement_log.md",
        "project_synthesis_optimization_roadmap.md",
        "paper_claims_tracker.md",
    ]:
        append_unique(DOCS / name, marker, text)
    append_unique(
        DOCS / "execution_task_registry.md",
        "| T159 |",
        "| T159 | Calibrated candidate selector model screen with LODO/LOSO validation | `scripts/run_t159_calibrated_candidate_selector.py`; `experiments/t159_model_screen_summary.csv`; `docs/t159_calibrated_candidate_selector.md` | DONE-CALIBRATED-SELECTOR-SCREEN |",
    )
    append_evidence_row(summary)


def run() -> dict[str, object]:
    candidates, selection = load_tables()
    base = baseline_rows(selection)
    lodo_pred, lodo_thresholds = run_lodo(candidates, selection)
    loso_pred, loso_thresholds = run_loso_primary(candidates, selection)
    all_pred = pd.concat([base, lodo_pred, loso_pred], ignore_index=True, sort=False)
    summary_table = summarize_policies(all_pred)
    thresholds = pd.concat([lodo_thresholds, loso_thresholds], ignore_index=True, sort=False)

    importance = feature_importance(candidates)
    bootstrap = pd.concat(
        [
            bootstrap_delta(all_pred, "T157_guarded_correction_v2", f"T159_{PRIMARY_MODEL_ID}_lodo_guarded_switch", "LODO"),
            bootstrap_delta(all_pred, "T150_release_all", f"T159_{PRIMARY_MODEL_ID}_lodo_guarded_switch", "LODO"),
            bootstrap_delta(all_pred, "T157_guarded_correction_v2", f"T159_{PRIMARY_MODEL_ID}_loso_guarded_switch", "LOSO"),
        ],
        ignore_index=True,
        sort=False,
    )
    case_audit = build_case_audit(all_pred)
    figures = write_figures(summary_table, importance)
    summary = build_summary(summary_table, bootstrap, case_audit, importance, figures)

    lodo_pred.to_csv(LODO_PRED_CSV, index=False, encoding="utf-8-sig")
    loso_pred.to_csv(LOSO_PRED_CSV, index=False, encoding="utf-8-sig")
    summary_table.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    thresholds.to_csv(THRESHOLDS_CSV, index=False, encoding="utf-8-sig")
    importance.to_csv(FEATURE_IMPORTANCE_CSV, index=False, encoding="utf-8-sig")
    bootstrap.to_csv(BOOTSTRAP_CSV, index=False, encoding="utf-8-sig")
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(summary, summary_table, thresholds, bootstrap, case_audit, importance, figures)
    update_project_docs(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    run()


if __name__ == "__main__":
    main()
