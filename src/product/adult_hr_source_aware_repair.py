from __future__ import annotations

import math
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.product.adult_hr_artifact_scorer import class_one_probability
from src.product.adult_hr_feature_harmonizer import FEATURE_COLUMNS, harmonize_candidate_features


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_PATH = PROJECT / "models" / "t345_source_aware_candidate_repair.pkl"

SOURCE_CATEGORIES = {
    "mcd_phase": ["before", "after", "unknown"],
    "mcd_camera": ["FullHDwebcam", "IriunWebcam", "USBVideo", "unknown"],
    "mcd_view": ["front", "left", "right", "unknown"],
}

EXTRA_NUMERIC_COLUMNS = [
    "bpm_rank_pct",
    "support_rank_pct",
    "top1_rank_pct",
    "power_rank_pct",
    "snr_rank_pct",
    "bpm_rank_first",
    "support_rank_first",
    "top1_rank_first",
    "power_rank_first",
    "snr_rank_first",
    "sequence_bin5_presence_frac",
    "sequence_bin5_mean_power",
    "sequence_bin5_mean_support",
]

T345_FEATURE_COLUMNS = [
    *FEATURE_COLUMNS,
    *EXTRA_NUMERIC_COLUMNS,
    *[f"{column}_{value}" for column, values in SOURCE_CATEGORIES.items() for value in values],
]


def parse_sequence_window(sample_id: object, window_idx: object | None = None) -> tuple[str, int]:
    text = str(sample_id)
    if window_idx is not None:
        try:
            value = float(window_idx)
        except (TypeError, ValueError):
            value = math.nan
        if math.isfinite(value):
            return re.sub(r"_w\d+.*$", "", text) or text, int(value)
    match = re.search(r"^(?P<prefix>.+?)_w(?P<idx>\d+)", text)
    if match:
        return match.group("prefix"), int(match.group("idx"))
    return text, 0


def add_sequence_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    sequence_ids: list[str] = []
    window_indices: list[int] = []
    for _, row in out.iterrows():
        sequence_id, window_idx = parse_sequence_window(row.get("sample_id", "sample"), row.get("window_idx", row.get("window_id")))
        sequence_ids.append(sequence_id)
        window_indices.append(window_idx)
    out["sequence_id"] = out.get("sequence_id", pd.Series(sequence_ids, index=out.index))
    out["artifact_window_idx"] = out.get("artifact_window_idx", pd.Series(window_indices, index=out.index))
    return out


def parse_mcd_source(sample_id: object) -> tuple[str, str, str]:
    match = re.match(r"^mcd_([^_]+)_(before|after)_(.+)_(front|left|right)_w\d+$", str(sample_id))
    if not match:
        return ("unknown", "unknown", "unknown")
    return (match.group(2), match.group(3), match.group(4))


def add_source_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    meta = pd.DataFrame([parse_mcd_source(value) for value in out["sample_id"]], columns=["mcd_phase", "mcd_camera", "mcd_view"])
    meta.index = out.index
    out = pd.concat([out, meta], axis=1)
    for column, values in SOURCE_CATEGORIES.items():
        out[column] = out[column].where(out[column].isin(values), "unknown")
    return out


def add_source_aware_repair_features(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the frozen T345 feature matrix without labels or ground truth."""

    if candidates.empty:
        empty = candidates.copy()
        return empty, pd.DataFrame(columns=T345_FEATURE_COLUMNS)

    out = harmonize_candidate_features(candidates)
    out = add_sequence_columns(out)
    out = add_source_metadata(out)

    group = out.groupby(["sample_id"], sort=False, dropna=False)
    rank_specs = [
        ("candidate_bpm", "bpm", True),
        ("support_methods", "support", False),
        ("top1_support_count", "top1", False),
        ("sum_power_fraction", "power", False),
        ("mean_snr_proxy_db", "snr", False),
    ]
    for column, name, ascending in rank_specs:
        out[f"{name}_rank_pct"] = group[column].rank(method="average", pct=True, ascending=ascending)
        out[f"{name}_rank_first"] = group[column].rank(method="first", ascending=ascending)

    out["bpm_bin5"] = (pd.to_numeric(out["candidate_bpm"], errors="coerce") / 5.0).round() * 5.0
    seq_key = ["sequence_id"]
    seq_bin_key = ["sequence_id", "bpm_bin5"]
    seq_total = out.groupby(seq_key)["sample_id"].transform("nunique").replace(0, np.nan)
    out["sequence_bin5_presence_frac"] = out.groupby(seq_bin_key)["sample_id"].transform("nunique") / seq_total
    out["sequence_bin5_mean_power"] = out.groupby(seq_bin_key)["sum_power_fraction"].transform("mean")
    out["sequence_bin5_mean_support"] = out.groupby(seq_bin_key)["support_methods"].transform("mean")

    feature_parts = [out[FEATURE_COLUMNS + EXTRA_NUMERIC_COLUMNS].reset_index(drop=True)]
    for column, values in SOURCE_CATEGORIES.items():
        for value in values:
            feature_parts.append(pd.Series((out[column].astype(str) == value).astype(float).to_numpy(), name=f"{column}_{value}"))
    features = pd.concat(feature_parts, axis=1)
    features = features.reindex(columns=T345_FEATURE_COLUMNS)
    features = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, features


def load_t345_repair_artifact(path: str | Path = DEFAULT_ARTIFACT_PATH) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        artifact = pickle.load(f)
    feature_columns = list(artifact.get("feature_columns", []))
    if feature_columns != T345_FEATURE_COLUMNS:
        raise ValueError("T345 artifact feature columns do not match product repair feature columns.")
    return artifact


def score_live_candidates_with_t345_repair_artifact(
    candidates: pd.DataFrame,
    *,
    artifact: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    model_artifact = artifact or load_t345_repair_artifact()
    scored, features = add_source_aware_repair_features(candidates)
    scored["t345_repair_probability"] = class_one_probability(model_artifact["model"], features[model_artifact["feature_columns"]])
    return scored


def select_t345_repaired_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    return (
        scored.sort_values(
            ["sample_id", "t345_repair_probability", "support_methods", "sum_power_fraction"],
            ascending=[True, False, False, False],
        )
        .groupby("sample_id", sort=False, dropna=False)
        .head(1)
        .copy()
    )


def build_t345_repair_router_product(
    candidates: pd.DataFrame,
    t330_router_predictions: pd.DataFrame,
    *,
    artifact: dict[str, Any] | None = None,
    high_t345_min: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the T345 repaired branch only to high-candidate windows.

    The caller supplies the existing T339/T330 router output for non-high
    branches. This keeps T345 conditional and prevents the repaired model from
    being used globally on UBFC/rPPG-10-like inputs.
    """

    scored = score_live_candidates_with_t345_repair_artifact(candidates, artifact=artifact)
    selected = select_t345_repaired_candidates(scored)
    repair = selected[["sample_id", "candidate_bpm", "t345_repair_probability", "window_candidate_count"]].rename(
        columns={
            "candidate_bpm": "t345_repair_hr_bpm",
            "window_candidate_count": "t345_window_candidate_count",
        }
    )
    product = t330_router_predictions.copy()
    product = product.merge(repair, on="sample_id", how="left")
    counts = pd.to_numeric(product.get("window_candidate_count", product.get("t345_window_candidate_count")), errors="coerce")
    high_mask = counts >= high_t345_min
    has_repair = pd.to_numeric(product["t345_repair_hr_bpm"], errors="coerce").notna()
    apply_mask = high_mask & has_repair
    product.loc[apply_mask, "selected_hr_bpm"] = product.loc[apply_mask, "t345_repair_hr_bpm"]
    if "product_hr_bpm" in product.columns:
        product.loc[apply_mask, "product_hr_bpm"] = product.loc[apply_mask, "t345_repair_hr_bpm"]
    product.loc[apply_mask, "decision"] = "release"
    product.loc[apply_mask, "released"] = 1
    product.loc[apply_mask, "source"] = "t345_source_aware_candidate_repair"
    product.loc[apply_mask, "policy"] = "t345_source_aware_repair_router"
    product.loc[apply_mask, "repair_probability"] = product.loc[apply_mask, "t345_repair_probability"]
    if "gt_hr_bpm" in product.columns and "abs_error_bpm" in product.columns:
        gt = pd.to_numeric(product.loc[apply_mask, "gt_hr_bpm"], errors="coerce")
        hr = pd.to_numeric(product.loc[apply_mask, "selected_hr_bpm"], errors="coerce")
        product.loc[apply_mask, "abs_error_bpm"] = (hr - gt).abs()
    return scored, selected, product
