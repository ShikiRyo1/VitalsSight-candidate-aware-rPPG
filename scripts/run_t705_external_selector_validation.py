from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


TASK_ID = "T705"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_t704_deep_candidate_physio_gate import (  # noqa: E402
    FEATURE_NUMERIC,
    GraphSelector,
    MLPSelector,
    SetAttentionSelector,
    add_relation_features,
    build_unified_candidate_pool,
    make_groups,
    metric_row,
    predict_model,
    train_one_model,
)


EXP = ROOT / "experiments"
DOCS = ROOT / "docs"

T704_POOL = EXP / "t704_unified_candidate_pool.csv"
MCD_CANDIDATES = EXP / "t667_common_dense_patch_candidate_table.csv"
UBFC_PHYS_CANDIDATES = EXP / "t674_expanded_ubfc_phys_candidate_table.csv"

OUT_EXTERNAL_POOL = EXP / "t705_external_candidate_pool.csv"
OUT_EXTERNAL_PREDS = EXP / "t705_external_selector_predictions.csv"
OUT_EXTERNAL_METRICS = EXP / "t705_external_selector_metrics.csv"
OUT_EXTERNAL_BASELINES = EXP / "t705_external_baseline_metrics.csv"
OUT_EXTERNAL_GATE = EXP / "t705_external_release_gate.csv"
OUT_EXTERNAL_CLAIM = EXP / "t705_external_claim_gate.csv"
OUT_SUMMARY = EXP / "t705_external_selector_validation_summary.json"
OUT_MD = DOCS / "t705_external_selector_validation.md"

UNSAFE_BPM = 10.0
SEED = 705


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def standard_subject(row: pd.Series) -> str:
    for col in ("subject", "subject_id", "source_key"):
        if col in row and pd.notna(row[col]):
            text = str(row[col])
            if text:
                return text
    return str(row.get("clip_id", "")).split("_")[0]


def normalize_external_table(path: Path, dataset_label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for idx, row in raw.iterrows():
        sample_id = str(row.get("clip_id", f"{dataset_label}_row{idx:06d}"))
        gt = float(pd.to_numeric(pd.Series([row.get("reference_hr_bpm")]), errors="coerce").iloc[0])
        cand = float(pd.to_numeric(pd.Series([row.get("candidate_hr_bpm")]), errors="coerce").iloc[0])
        if not np.isfinite(gt) or not np.isfinite(cand):
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "dataset": dataset_label,
                "subject_std": standard_subject(row),
                "gt_hr_bpm": gt,
                "reference_hr_bpm": gt,
                "candidate_id": f"{sample_id}_{row.get('region', 'roi')}_{row.get('method', 'method')}_{idx}",
                "candidate_hr_bpm": cand,
                "candidate_abs_error": abs(cand - gt),
                "source_type": "classical",
                "source_name": "candidate_pool",
                "candidate_family": str(row.get("patch_family", "external_region")),
                "candidate_model": str(row.get("method", "method")),
                "support_count": float(row.get("candidate_peak_support", 0.0)),
                "full_support_count": 0.0,
                "subwindow_support_count": 0.0,
                "top1_support_count": 0.0,
                "full_top1_support_count": 0.0,
                "pos_chrom_count": 1.0 if str(row.get("method", "")).lower() in {"pos", "chrom"} else 0.0,
                "green_pbv_count": 1.0 if str(row.get("method", "")).lower() == "green" else 0.0,
                "ica_lgi_count": 0.0,
                "mean_power_fraction": float(row.get("candidate_peak_support", 0.0)),
                "max_power_fraction": float(row.get("candidate_peak_support", 0.0)),
                "sum_power_fraction": float(row.get("candidate_peak_support", 0.0)),
                "rank_score": float(row.get("candidate_peak_support", 0.0)),
                "mean_snr_proxy_db": float(row.get("candidate_snr_db", 0.0)),
                "adult_plausibility": 1.0,
                "upper_alt_support": 0.0,
                "upper_alt_pos_chrom": 0.0,
                "upper_phys_support": 0.0,
                "upper_phys_pos_chrom": 0.0,
                "lower_phys_support": 0.0,
                "lower_phys_pos_chrom": 0.0,
                "double_harmonic_support": 0.0,
                "half_harmonic_support": 0.0,
                "t150_selected_bpm": np.nan,
                "t150_confidence": 0.0,
                "dist_to_t150": np.nan,
                "t157_low_alias_penalty": 0.0,
                "t157_motion_band_penalty": 0.0,
                "t157_high_alias_penalty": 0.0,
                "t157_near_t150_boost": 0.0,
                "t157_score": 0.0,
                "deep_snr": np.nan,
                "deep_macc": np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["unsafe_candidate"] = out["candidate_abs_error"] > UNSAFE_BPM
    out = add_relation_features(out)
    return out


def feature_frame_with_columns(pool: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    df = pool.copy()
    for col in FEATURE_NUMERIC:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    cat = pd.get_dummies(df[["source_type", "source_name", "candidate_family", "candidate_model"]].astype(str), prefix=["src", "name", "family", "model"])
    features = pd.concat([df[FEATURE_NUMERIC], cat], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    if feature_cols is None:
        feature_cols = list(features.columns)
    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0.0
    features = features[feature_cols]
    return features.astype(float), feature_cols


def baseline_predictions(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for sample_id, g in pool.groupby("sample_id", sort=False):
        gt = float(g["gt_hr_bpm"].iloc[0])
        top = g.loc[pd.to_numeric(g["mean_snr_proxy_db"], errors="coerce").fillna(-999).idxmax()]
        oracle = g.loc[pd.to_numeric(g["candidate_abs_error"], errors="coerce").idxmin()]
        for method, row in [("external_max_snr", top), ("external_oracle", oracle)]:
            pred = float(row["candidate_hr_bpm"])
            rows.append(
                {
                    "method": method,
                    "dataset": row["dataset"],
                    "sample_id": sample_id,
                    "subject_std": row["subject_std"],
                    "gt_hr_bpm": gt,
                    "pred_hr_bpm": pred,
                    "abs_error_bpm": abs(pred - gt),
                    "selected_candidate_id": row["candidate_id"],
                    "selected_source_name": row["source_name"],
                }
            )
    preds = pd.DataFrame(rows)
    metrics = pd.DataFrame(
        [
            {"dataset": ds, **metric_row(method, g)}
            for (ds, method), g in preds.groupby(["dataset", "method"], sort=False)
        ]
    )
    return preds, metrics


def train_final_models(train_pool: pd.DataFrame, *, epochs: int, lr: float) -> tuple[dict[str, torch.nn.Module], list[str], dict[str, np.ndarray]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_features, feature_cols = feature_frame_with_columns(train_pool)
    train_groups, std = make_groups(train_pool.reset_index(drop=True), train_features.reset_index(drop=True), feature_cols, device)
    constructors = {
        "mlp_relation": lambda: MLPSelector(len(feature_cols)),
        "set_attention": lambda: SetAttentionSelector(len(feature_cols)),
        "graph_selector": lambda: GraphSelector(len(feature_cols)),
    }
    models: dict[str, torch.nn.Module] = {}
    for name, ctor in constructors.items():
        model = ctor().to(device)
        print(f"[{TASK_ID}] training final {name} on UBFC protocol pool", flush=True)
        models[name] = train_one_model(name, model, train_groups, epochs=epochs, lr=lr)
    return models, feature_cols, std


def apply_models(models: dict[str, torch.nn.Module], feature_cols: list[str], std: dict[str, np.ndarray], ext_pool: pd.DataFrame) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext_features, _ = feature_frame_with_columns(ext_pool, feature_cols)
    groups, _ = make_groups(ext_pool.reset_index(drop=True), ext_features.reset_index(drop=True), feature_cols, device, standardize=std)
    preds: list[pd.DataFrame] = []
    for name, model in models.items():
        pred = predict_model(name, model, groups)
        sample_to_ds = ext_pool.drop_duplicates("sample_id").set_index("sample_id")["dataset"].to_dict()
        pred["dataset"] = pred["sample_id"].map(sample_to_ds)
        preds.append(pred)
    return pd.concat(preds, ignore_index=True, sort=False)


def release_gate(preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, selector), g in preds.groupby(["dataset", "selector"], sort=False):
        h = g.copy()
        risk = (
            (1.0 - pd.to_numeric(h["max_probability"], errors="coerce").fillna(0.0))
            + pd.to_numeric(h["entropy"], errors="coerce").fillna(0.0) / np.log(pd.to_numeric(h["n_candidates"], errors="coerce").clip(lower=2))
            + 0.15 * pd.to_numeric(h["selected_harmonic_risk"], errors="coerce").fillna(0.0)
            - 0.20 * pd.to_numeric(h["selected_agreement10_frac"], errors="coerce").fillna(0.0)
        )
        h["risk_score"] = risk
        for q in np.linspace(0.05, 1.0, 20):
            tau = float(h["risk_score"].quantile(q))
            released = h[h["risk_score"] <= tau]
            if released.empty:
                continue
            err = pd.to_numeric(released["abs_error_bpm"], errors="coerce")
            rows.append(
                {
                    "dataset": dataset,
                    "selector": selector,
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


def claim_gate(metrics: pd.DataFrame, baseline_metrics: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, m in metrics.iterrows():
        dataset = m["dataset"]
        selector = m["method"]
        base = baseline_metrics[(baseline_metrics["dataset"] == dataset) & (baseline_metrics["method"] == "external_max_snr")]
        base_mae = float(base["mae_bpm"].iloc[0]) if not base.empty else math.nan
        safe = gate[(gate["dataset"] == dataset) & (gate["selector"] == selector) & (gate["gate_pass_unsafe10"])]
        best_safe = safe.sort_values(["coverage", "released_mae_bpm"], ascending=[False, True]).head(1)
        reduction = 1.0 - float(m["mae_bpm"]) / base_mae if math.isfinite(base_mae) and base_mae > 0 else math.nan
        rows.append(
            {
                "dataset": dataset,
                "selector": selector,
                "mae_bpm": float(m["mae_bpm"]),
                "unsafe_gt10bpm_rate": float(m["unsafe_gt10bpm_rate"]),
                "mae_reduction_vs_external_max_snr": reduction,
                "best_safe_gate_coverage": float(best_safe["coverage"].iloc[0]) if not best_safe.empty else 0.0,
                "best_safe_gate_unsafe": float(best_safe["unsafe_release_rate"].iloc[0]) if not best_safe.empty else math.nan,
                "external_gate_pass": bool(reduction >= 0.20 and not best_safe.empty and float(best_safe["coverage"].iloc[0]) >= 0.40),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "mae_bpm"])


def write_report(summary: dict[str, Any], metrics: pd.DataFrame, baselines: pd.DataFrame, claim: pd.DataFrame) -> None:
    lines = [
        "# T705 External Selector Validation",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Purpose",
        "",
        "Train the three T704 selectors on the protocol-aligned UBFC-rPPG candidate pool and test them on external dense candidate tables without retraining on external ground truth.",
        "",
        "## External Baselines",
        "",
        markdown_table(baselines),
        "",
        "## External Selector Metrics",
        "",
        markdown_table(metrics),
        "",
        "## External Claim Gate",
        "",
        markdown_table(claim),
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Boundary",
        "",
        "External validation here tests selector transfer on existing external candidate tables. It does not include deep candidates on external datasets unless per-window deep predictions are generated later.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    set_seed(args.seed)

    if T704_POOL.exists():
        train_pool = pd.read_csv(T704_POOL)
    else:
        train_pool, _ = build_unified_candidate_pool()

    external_parts = [
        normalize_external_table(MCD_CANDIDATES, "MCD-rPPG"),
        normalize_external_table(UBFC_PHYS_CANDIDATES, "UBFC-Phys-S1-S14"),
    ]
    external_pool = pd.concat(external_parts, ignore_index=True, sort=False)
    external_pool.to_csv(OUT_EXTERNAL_POOL, index=False, encoding="utf-8")

    baseline_preds, baseline_metrics = baseline_predictions(external_pool)
    models, feature_cols, std = train_final_models(train_pool, epochs=args.epochs, lr=args.lr)
    preds = apply_models(models, feature_cols, std, external_pool)
    metrics = pd.DataFrame(
        [
            {"dataset": ds, **metric_row(selector, g.rename(columns={"selector": "method"}))}
            for (ds, selector), g in preds.groupby(["dataset", "selector"], sort=False)
        ]
    ).sort_values(["dataset", "mae_bpm"])
    gate = release_gate(preds)
    claim = claim_gate(metrics, baseline_metrics, gate)

    baseline_metrics.to_csv(OUT_EXTERNAL_BASELINES, index=False, encoding="utf-8")
    preds.to_csv(OUT_EXTERNAL_PREDS, index=False, encoding="utf-8")
    metrics.to_csv(OUT_EXTERNAL_METRICS, index=False, encoding="utf-8")
    gate.to_csv(OUT_EXTERNAL_GATE, index=False, encoding="utf-8")
    claim.to_csv(OUT_EXTERNAL_CLAIM, index=False, encoding="utf-8")

    pass_count = int(claim["external_gate_pass"].sum())
    summary = {
        "task_id": TASK_ID,
        "generated_at": now(),
        "decision": "external_selector_transfer_has_support" if pass_count >= 2 else "external_selector_transfer_partial_or_failed_requires_analysis",
        "external_gate_pass_count": pass_count,
        "n_external_windows": int(external_pool["sample_id"].nunique()),
        "datasets": sorted(external_pool["dataset"].astype(str).unique()),
        "outputs": {
            "external_pool": str(OUT_EXTERNAL_POOL),
            "predictions": str(OUT_EXTERNAL_PREDS),
            "metrics": str(OUT_EXTERNAL_METRICS),
            "baselines": str(OUT_EXTERNAL_BASELINES),
            "gate": str(OUT_EXTERNAL_GATE),
            "claim": str(OUT_EXTERNAL_CLAIM),
            "doc": str(OUT_MD),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, metrics, baseline_metrics, claim)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
