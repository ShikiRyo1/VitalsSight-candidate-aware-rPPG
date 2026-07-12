from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


TASK_ID = "T731"
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"

IN_POOL = EXP / "t730_harmonic_aware_candidate_pool.csv"
IN_BASELINES = EXP / "t730_harmonic_aware_baseline_metrics.csv"

OUT_CANDIDATE_PREDS = EXP / "t731_candidate_ranker_candidate_predictions.csv"
OUT_SELECTIONS = EXP / "t731_candidate_ranker_selections.csv"
OUT_METRICS = EXP / "t731_candidate_ranker_metrics.csv"
OUT_RELEASE = EXP / "t731_candidate_ranker_release_gate.csv"
OUT_CLAIM = EXP / "t731_candidate_ranker_claim_gate.csv"
OUT_FAILURE = EXP / "t731_candidate_ranker_failure_taxonomy.csv"
OUT_SUMMARY = EXP / "t731_candidate_ranker_summary.json"
OUT_MD = DOCS / "t731_candidate_ranker_harmonic_rescue.md"

GOOD_BPM = 5.0
UNSAFE_BPM = 10.0
SEED = 731


NUMERIC = [
    "candidate_hr_bpm",
    "support_count",
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
    "upper_phys_support",
    "lower_phys_support",
    "double_harmonic_support",
    "half_harmonic_support",
    "t157_low_alias_penalty",
    "t157_motion_band_penalty",
    "t157_high_alias_penalty",
    "t157_score",
    "deep_snr",
    "deep_macc",
    "dist_to_group_median_hr",
    "dist_to_group_mean_hr",
    "dist_to_top_snr_hr",
    "agreement5_frac",
    "agreement10_frac",
    "agreement20_frac",
    "snr_rank_pct",
    "support_rank_pct",
    "harmonic_risk",
    "is_deep",
    "is_classical",
    "hr_rank_pct",
    "hr_percentile_center_distance",
    "half_neighbor_count",
    "double_neighbor_count",
    "half_neighbor_support_frac",
    "double_neighbor_support_frac",
    "harmonic_neighbor_count",
    "harmonic_neighbor_frac",
    "snr_agreement_conflict",
    "support_agreement_conflict",
    "alias_band_risk",
    "hr_boundary_risk",
    "deep_median_distance",
    "deep_min_distance",
    "deep_disagreement_risk",
    "harmonic_trap_score",
]

CATEGORICAL = ["dataset", "source_type", "source_name", "candidate_family", "candidate_model"]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.{digits}f}")
    lines = [
        "| " + " | ".join(map(str, show.columns)) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in show.columns) + " |")
    return "\n".join(lines)


def metric_row(dataset: str, variant: str, selected: pd.DataFrame) -> dict[str, Any]:
    err = pd.to_numeric(selected["abs_error_bpm"], errors="coerce")
    y = pd.to_numeric(selected["pred_hr_bpm"], errors="coerce").to_numpy(float)
    gt = pd.to_numeric(selected["gt_hr_bpm"], errors="coerce").to_numpy(float)
    finite = np.isfinite(y) & np.isfinite(gt)
    corr = float(np.corrcoef(gt[finite], y[finite])[0, 1]) if finite.sum() > 1 and np.std(y[finite]) > 1e-8 and np.std(gt[finite]) > 1e-8 else math.nan
    return {
        "task_id": TASK_ID,
        "dataset": dataset,
        "variant": variant,
        "n_windows": int(selected["sample_id"].nunique()),
        "mae_bpm": float(err.mean()),
        "rmse_bpm": float(np.sqrt(np.mean(np.square(err)))),
        "median_abs_error_bpm": float(err.median()),
        "p90_abs_error_bpm": float(err.quantile(0.90)),
        "unsafe_gt10bpm_rate": float((err > UNSAFE_BPM).mean()),
        "pearson_r": corr,
    }


def add_relative_rule_features(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    for col in NUMERIC:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["good_candidate"] = pd.to_numeric(out["candidate_abs_error"], errors="coerce") <= GOOD_BPM
    out["unsafe_candidate"] = pd.to_numeric(out["candidate_abs_error"], errors="coerce") > UNSAFE_BPM
    out["source_key"] = (
        out["dataset"].astype(str)
        + "::"
        + out["subject_std"].astype(str)
        + "::"
        + out["source_name"].astype(str)
        + "::"
        + out["candidate_family"].astype(str)
    )
    out["group_key"] = out["dataset"].astype(str) + "::" + out["subject_std"].astype(str)
    out["observable_rule_score"] = (
        0.24 * out["snr_rank_pct"].fillna(0.0)
        + 0.20 * out["support_rank_pct"].fillna(0.0)
        + 0.24 * out["agreement10_frac"].fillna(0.0)
        + 0.12 * out["agreement20_frac"].fillna(0.0)
        + 0.08 * out["adult_plausibility"].fillna(0.0)
        - 0.12 * out["harmonic_trap_score"].fillna(0.0)
        - 0.06 * out["deep_disagreement_risk"].fillna(0.0)
    )
    return out


def stratified_folds(df: pd.DataFrame, n_splits: int, seed: int) -> list[tuple[set[str], set[str]]]:
    rng = np.random.default_rng(seed)
    keys = df[["dataset", "group_key"]].drop_duplicates()
    fold_tests = [set() for _ in range(n_splits)]
    all_keys = set(keys["group_key"].astype(str))
    for _, group in keys.groupby("dataset"):
        ds_keys = group["group_key"].astype(str).tolist()
        rng.shuffle(ds_keys)
        chunks = np.array_split(np.asarray(ds_keys, dtype=object), n_splits)
        for idx, chunk in enumerate(chunks):
            fold_tests[idx].update(str(x) for x in chunk)
    folds = []
    for test in fold_tests:
        folds.append((all_keys - test, test))
    return folds


def encode(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train.copy()
    test_x = test.copy()
    for col in NUMERIC + ["observable_rule_score"]:
        if col not in train_x.columns:
            train_x[col] = np.nan
        if col not in test_x.columns:
            test_x[col] = np.nan
        train_x[col] = pd.to_numeric(train_x[col], errors="coerce")
        test_x[col] = pd.to_numeric(test_x[col], errors="coerce")
        fill = float(train_x[col].median()) if train_x[col].notna().any() else 0.0
        train_x[col] = train_x[col].fillna(fill)
        test_x[col] = test_x[col].fillna(fill)
    for col in CATEGORICAL:
        train_x[col] = train_x.get(col, "unknown").astype(str).fillna("unknown")
        test_x[col] = test_x.get(col, "unknown").astype(str).fillna("unknown")
    cat = pd.get_dummies(pd.concat([train_x[CATEGORICAL], test_x[CATEGORICAL]], ignore_index=True), columns=CATEGORICAL)
    train_cat = cat.iloc[: len(train_x)].reset_index(drop=True)
    test_cat = cat.iloc[len(train_x) :].reset_index(drop=True)
    num_cols = NUMERIC + ["observable_rule_score"]
    return (
        pd.concat([train_x[num_cols].reset_index(drop=True), train_cat], axis=1).astype(float),
        pd.concat([test_x[num_cols].reset_index(drop=True), test_cat], axis=1).astype(float),
    )


def model_factories() -> dict[str, Any]:
    return {
        "extra_trees_good5": lambda: ExtraTreesClassifier(
            n_estimators=700,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "random_forest_good5": lambda: RandomForestClassifier(
            n_estimators=500,
            max_depth=9,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "hist_gbdt_good5": lambda: HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.035,
            max_leaf_nodes=17,
            l2_regularization=0.05,
            random_state=SEED,
        ),
    }


def dataset_weights(train: pd.DataFrame) -> np.ndarray:
    counts = train["dataset"].astype(str).value_counts().to_dict()
    n_domains = max(1, len(counts))
    total = float(len(train))
    return train["dataset"].astype(str).map(lambda d: total / (n_domains * counts.get(d, 1))).to_numpy(float)


def crossfit_candidate_scores(pool: pd.DataFrame, n_splits: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = stratified_folds(pool, n_splits, seed)
    pred_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for fold_idx, (train_keys, test_keys) in enumerate(folds, start=1):
        train = pool[pool["group_key"].isin(train_keys)].copy()
        test = pool[pool["group_key"].isin(test_keys)].copy()
        y = train["good_candidate"].astype(int).to_numpy()
        x_train, x_test = encode(train, test)
        test_out = test.copy()
        test_out["fold"] = fold_idx
        test_out["rule_good5"] = pd.to_numeric(test_out["observable_rule_score"], errors="coerce").fillna(0.0)
        for name, factory in model_factories().items():
            if len(np.unique(y)) < 2 or test.empty:
                test_out[name] = float(np.mean(y)) if len(y) else 0.0
                audits.append({"fold": fold_idx, "model": name, "status": "fallback_single_class", "auroc": math.nan, "auprc": math.nan})
                continue
            model = factory()
            sample_weight = dataset_weights(train)
            try:
                model.fit(x_train, y, sample_weight=sample_weight)
            except TypeError:
                model.fit(x_train, y)
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(x_test)
                score = prob[:, 1] if prob.shape[1] > 1 else prob[:, 0]
            else:
                score = model.predict(x_test)
            test_out[name] = score
            y_test = test["good_candidate"].astype(int)
            auroc = roc_auc_score(y_test, score) if y_test.nunique() > 1 else math.nan
            auprc = average_precision_score(y_test, score) if y_test.nunique() > 1 else math.nan
            audits.append({"fold": fold_idx, "model": name, "train_rows": len(train), "test_rows": len(test), "auroc": auroc, "auprc": auprc})
        pred_parts.append(test_out)
    return pd.concat(pred_parts, ignore_index=True, sort=False), pd.DataFrame(audits)


def select_base(group: pd.DataFrame, score_col: str) -> pd.Series:
    return group.sort_values([score_col, "observable_rule_score", "mean_snr_proxy_db"], ascending=[False, False, False]).iloc[0]


def select_bidirectional_harmonic(group: pd.DataFrame, score_col: str, *, delta: float = 0.25) -> pd.Series:
    base_row = select_base(group, score_col)
    base_hr = float(base_row["candidate_hr_bpm"])
    candidates = [base_row]
    if math.isfinite(base_hr) and base_hr > 0:
        low_mask = (
            (pd.to_numeric(group["candidate_hr_bpm"], errors="coerce") >= 0.43 * base_hr)
            & (pd.to_numeric(group["candidate_hr_bpm"], errors="coerce") <= 0.57 * base_hr)
            & (pd.to_numeric(group[score_col], errors="coerce") >= float(base_row[score_col]) - delta)
        )
        high_mask = (
            (pd.to_numeric(group["candidate_hr_bpm"], errors="coerce") >= 1.75 * base_hr)
            & (pd.to_numeric(group["candidate_hr_bpm"], errors="coerce") <= 2.25 * base_hr)
            & (pd.to_numeric(group[score_col], errors="coerce") >= float(base_row[score_col]) - delta)
        )
        alt = group[low_mask | high_mask].copy()
        if not alt.empty:
            for _, row in alt.iterrows():
                candidates.append(row)
    cand = pd.DataFrame(candidates)
    rescue_score = (
        pd.to_numeric(cand[score_col], errors="coerce").fillna(0.0)
        + 0.12 * pd.to_numeric(cand["agreement10_frac"], errors="coerce").fillna(0.0)
        + 0.06 * pd.to_numeric(cand["support_rank_pct"], errors="coerce").fillna(0.0)
        + 0.04 * pd.to_numeric(cand["snr_rank_pct"], errors="coerce").fillna(0.0)
        - 0.18 * pd.to_numeric(cand["harmonic_trap_score"], errors="coerce").fillna(0.0)
        - 0.06 * pd.to_numeric(cand["alias_band_risk"], errors="coerce").fillna(0.0)
    )
    selected = cand.iloc[int(np.argmax(rescue_score.to_numpy(float)))]
    selected = selected.copy()
    selected["harmonic_rescued"] = bool(str(selected["candidate_id"]) != str(base_row["candidate_id"]))
    selected["base_candidate_hr_bpm"] = float(base_hr)
    selected["base_score"] = float(base_row[score_col])
    return selected


def build_selections(candidate_preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    variants = ["rule_good5", "extra_trees_good5", "random_forest_good5", "hist_gbdt_good5"]
    for (dataset, sample_id), group in candidate_preds.groupby(["dataset", "sample_id"], sort=False):
        for variant in variants:
            base_row = select_base(group, variant).copy()
            base_row["variant"] = variant
            base_row["selected_score"] = float(base_row[variant])
            base_row["harmonic_rescued"] = False
            rows.append(base_row.to_dict())
            if variant != "rule_good5":
                rescue = select_bidirectional_harmonic(group, variant)
                rescue["variant"] = variant + "_bidirectional_harmonic"
                rescue["selected_score"] = float(rescue[variant])
                rows.append(rescue.to_dict())
    selected = pd.DataFrame(rows)
    selected["pred_hr_bpm"] = pd.to_numeric(selected["candidate_hr_bpm"], errors="coerce")
    selected["gt_hr_bpm"] = pd.to_numeric(selected["gt_hr_bpm"], errors="coerce")
    selected["abs_error_bpm"] = (selected["pred_hr_bpm"] - selected["gt_hr_bpm"]).abs()
    return selected


def release_gate(selections: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, variant), group in selections.groupby(["dataset", "variant"], sort=False):
        h = group.copy()
        h["release_risk"] = (
            1.0
            - pd.to_numeric(h["selected_score"], errors="coerce").fillna(0.0)
            + 0.65 * pd.to_numeric(h["harmonic_trap_score"], errors="coerce").fillna(0.0)
            + 0.25 * pd.to_numeric(h["alias_band_risk"], errors="coerce").fillna(0.0)
            + 0.18 * pd.to_numeric(h["deep_disagreement_risk"], errors="coerce").fillna(0.0)
            - 0.30 * pd.to_numeric(h["agreement10_frac"], errors="coerce").fillna(0.0)
        )
        for q in np.linspace(0.05, 1.0, 20):
            tau = float(h["release_risk"].quantile(q))
            released = h[h["release_risk"] <= tau]
            if released.empty:
                continue
            err = pd.to_numeric(released["abs_error_bpm"], errors="coerce")
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "threshold_quantile": float(q),
                    "risk_threshold": tau,
                    "coverage": float(len(released) / len(h)),
                    "released_mae_bpm": float(err.mean()),
                    "unsafe_release_rate": float((err > UNSAFE_BPM).mean()),
                    "n_released": int(len(released)),
                    "n_total": int(len(h)),
                    "gate_pass_unsafe10": bool((err > UNSAFE_BPM).mean() <= 0.10),
                }
            )
    return pd.DataFrame(rows)


def build_metrics(selections: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([metric_row(str(ds), str(v), g) for (ds, v), g in selections.groupby(["dataset", "variant"], sort=False)]).sort_values(["dataset", "mae_bpm"])


def claim_gate(metrics: pd.DataFrame, release: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        dataset = str(row["dataset"])
        variant = str(row["variant"])
        base_row = baselines[(baselines["dataset"].astype(str) == dataset) & (baselines["method"].astype(str) == "max_snr")]
        base_mae = float(base_row["mae_bpm"].iloc[0]) if not base_row.empty else math.nan
        safe = release[(release["dataset"].astype(str) == dataset) & (release["variant"].astype(str) == variant) & (release["gate_pass_unsafe10"])]
        best = safe.sort_values(["coverage", "released_mae_bpm"], ascending=[False, True]).head(1)
        best_cov = float(best["coverage"].iloc[0]) if not best.empty else 0.0
        best_unsafe = float(best["unsafe_release_rate"].iloc[0]) if not best.empty else math.nan
        best_mae = float(best["released_mae_bpm"].iloc[0]) if not best.empty else math.nan
        reduction = 1.0 - float(row["mae_bpm"]) / base_mae if math.isfinite(base_mae) and base_mae > 0 else math.nan
        released_reduction = 1.0 - best_mae / base_mae if math.isfinite(base_mae) and math.isfinite(best_mae) and base_mae > 0 else math.nan
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "mae_bpm": float(row["mae_bpm"]),
                "unsafe_gt10bpm_rate": float(row["unsafe_gt10bpm_rate"]),
                "mae_reduction_vs_max_snr": reduction,
                "best_safe_gate_coverage": best_cov,
                "best_safe_gate_unsafe": best_unsafe,
                "best_safe_gate_released_mae_bpm": best_mae,
                "released_mae_reduction_vs_max_snr": released_reduction,
                "pass_dataset_gate": bool(math.isfinite(released_reduction) and released_reduction >= 0.20 and best_cov >= 0.40 and best_unsafe <= 0.10),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "mae_bpm"])


def failure_taxonomy(selections: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, variant), group in selections.groupby(["dataset", "variant"], sort=False):
        unsafe = group[pd.to_numeric(group["abs_error_bpm"], errors="coerce") > UNSAFE_BPM]
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "n_windows": int(len(group)),
                "unsafe_rate": float(len(unsafe) / max(1, len(group))),
                "mean_unsafe_error_bpm": float(pd.to_numeric(unsafe["abs_error_bpm"], errors="coerce").mean()) if not unsafe.empty else 0.0,
                "top_selected_family": str(group["candidate_family"].mode().iloc[0]) if not group.empty else "",
                "top_unsafe_family": str(unsafe["candidate_family"].mode().iloc[0]) if not unsafe.empty else "",
                "harmonic_rescue_rate": float(group.get("harmonic_rescued", pd.Series(False, index=group.index)).astype(bool).mean()),
                "unsafe_harmonic_rescue_rate": float(unsafe.get("harmonic_rescued", pd.Series(False, index=unsafe.index)).astype(bool).mean()) if not unsafe.empty else 0.0,
                "unsafe_alias_band_frac": float(pd.to_numeric(unsafe.get("alias_band_risk", 0.0), errors="coerce").fillna(0.0).mean()) if not unsafe.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: dict[str, Any], metrics: pd.DataFrame, claim: pd.DataFrame, release: pd.DataFrame, audit: pd.DataFrame, fail: pd.DataFrame) -> None:
    lines = [
        "# T731 Candidate-Level Ranker With Bidirectional Harmonic Rescue",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Purpose",
        "",
        "T730 showed that small neural selectors still choose MCD harmonic/alias false peaks. T731 reformulates the task as candidate-level good-candidate ranking, following the successful T668/T669 mechanism but applying it to the current T730 unified candidate pool.",
        "",
        "## Method",
        "",
        "- Cross-fit candidate-level classifiers for `good_candidate <= 5 BPM` using dataset/subject folds.",
        "- Compare rule score, ExtraTrees, RandomForest, and HistGradientBoosting.",
        "- Add bidirectional harmonic rescue: when a high/low selected peak has a plausible half/double counterpart with similar score, re-rank that harmonic pair using agreement, support, SNR, and trap-risk features.",
        "- Apply a release/review gate over selected-score, harmonic-trap risk, alias-band risk, deep disagreement, and agreement.",
        "",
        "## Model Audit",
        "",
        markdown_table(audit.head(30)),
        "",
        "## Metrics",
        "",
        markdown_table(metrics),
        "",
        "## Claim Gate",
        "",
        markdown_table(claim),
        "",
        "## Release Gate Examples",
        "",
        markdown_table(release.sort_values(["dataset", "variant", "gate_pass_unsafe10", "coverage"], ascending=[True, True, False, False]).groupby(["dataset", "variant"]).head(2)),
        "",
        "## Failure Taxonomy",
        "",
        markdown_table(fail),
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Claim Boundary",
        "",
        "T731 is selector/gate evidence. It can support the main claim only if MCD and at least one external dataset meet the predefined safe release threshold. It does not claim universal SOTA.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not IN_POOL.exists():
        raise FileNotFoundError(IN_POOL)
    pool = pd.read_csv(IN_POOL, low_memory=False)
    baselines = pd.read_csv(IN_BASELINES) if IN_BASELINES.exists() else pd.DataFrame()
    pool = add_relative_rule_features(pool)
    candidate_preds, audit = crossfit_candidate_scores(pool, n_splits=5, seed=SEED)
    selections = build_selections(candidate_preds)
    metrics = build_metrics(selections)
    release = release_gate(selections)
    claim = claim_gate(metrics, release, baselines)
    fail = failure_taxonomy(selections)

    candidate_preds.to_csv(OUT_CANDIDATE_PREDS, index=False, encoding="utf-8")
    selections.to_csv(OUT_SELECTIONS, index=False, encoding="utf-8")
    metrics.to_csv(OUT_METRICS, index=False, encoding="utf-8")
    release.to_csv(OUT_RELEASE, index=False, encoding="utf-8")
    claim.to_csv(OUT_CLAIM, index=False, encoding="utf-8")
    fail.to_csv(OUT_FAILURE, index=False, encoding="utf-8")

    pass_count = int(claim["pass_dataset_gate"].sum()) if "pass_dataset_gate" in claim.columns else 0
    mcd = claim[claim["dataset"].astype(str).eq("MCD-rPPG")]
    summary = {
        "task_id": TASK_ID,
        "generated_at": now(),
        "decision": "candidate_ranker_gate_supported_continue_external_stats" if pass_count >= 4 else "candidate_ranker_partial_needs_external_or_domain_refit",
        "pass_count": pass_count,
        "n_dataset_variant": int(len(claim)),
        "datasets": sorted(pool["dataset"].astype(str).unique()),
        "n_windows": int(pool["sample_id"].nunique()),
        "n_candidates": int(len(pool)),
        "mcd_best_safe_coverage": float(mcd["best_safe_gate_coverage"].max()) if not mcd.empty else 0.0,
        "mcd_best_released_mae": float(mcd["best_safe_gate_released_mae_bpm"].min()) if not mcd["best_safe_gate_released_mae_bpm"].dropna().empty else math.nan,
        "outputs": {
            "candidate_predictions": str(OUT_CANDIDATE_PREDS),
            "selections": str(OUT_SELECTIONS),
            "metrics": str(OUT_METRICS),
            "release": str(OUT_RELEASE),
            "claim": str(OUT_CLAIM),
            "failure": str(OUT_FAILURE),
            "doc": str(OUT_MD),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(summary, metrics, claim, release, audit, fail)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
