from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT / "models" / "t236_live_minimal_hgb_guard.joblib"
DEFAULT_CONFIG_PATH = PROJECT / "configs" / "t236_live_minimal_hgb_guard.json"

LIVE_MINIMAL_FEATURES = [
    "selected_bpm",
    "max_power_candidate_bpm",
    "gap",
    "roi_support",
    "method_support",
]

T240_CONTEXT_CANDIDATE_POLICY = "t240_context_aware_high_hr_candidate_v1"


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def parse_evidence(raw: object) -> dict[str, Any]:
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def numeric_series(frame: pd.DataFrame, column: str, default: float = math.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def first_finite_series(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(math.nan, index=frame.index, dtype=float)
    for column in columns:
        values = numeric_series(frame, column)
        out = out.where(out.notna(), values)
    return out


def prepare_live_reliability_features(product: pd.DataFrame) -> pd.DataFrame:
    """Build T236 live-compatible reliability features from a product table.

    T236 intentionally uses the minimal product-safe feature contract instead of
    T234's research-only `t214_score`. The returned columns are label-free and
    can be computed from an uploaded video without ground truth.
    """

    if product.empty:
        return pd.DataFrame(columns=LIVE_MINIMAL_FEATURES, index=product.index)

    selected_bpm = first_finite_series(
        product,
        [
            "pre_conflict_guard_product_hr_bpm",
            "pre_upstream_review_product_hr_bpm",
            "pre_temporal_product_hr_bpm",
            "product_hr_bpm",
            "candidate_hr_bpm",
        ],
    )
    max_power = numeric_series(product, "max_power_candidate_bpm")
    roi_support = first_finite_series(product, ["selected_support_rois", "roi_support"])
    method_support = first_finite_series(product, ["selected_support_methods", "method_support"])
    gap = (selected_bpm - max_power).abs()

    return pd.DataFrame(
        {
            "selected_bpm": selected_bpm,
            "max_power_candidate_bpm": max_power,
            "gap": gap,
            "roi_support": roi_support,
            "method_support": method_support,
        },
        index=product.index,
    )


def load_reliability_guard(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> tuple[Any | None, dict[str, Any]]:
    """Load a frozen T236 reliability guard.

    Returns `(None, config)` if the artifact is unavailable. This keeps the UI
    and product path usable before T236 artifacts are generated.
    """

    model_path = Path(model_path)
    config_path = Path(config_path)
    config: dict[str, Any] = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config = {}
    if not model_path.exists():
        return None, config
    try:
        import joblib

        artifact = joblib.load(model_path)
    except Exception:
        return None, config
    if isinstance(artifact, dict) and "model" in artifact:
        config = {**artifact.get("config", {}), **config}
        return artifact["model"], config
    return artifact, config


def score_product_table_with_reliability_guard(
    product: pd.DataFrame,
    *,
    model: Any | None = None,
    config: dict[str, Any] | None = None,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    apply_review: bool = False,
) -> pd.DataFrame:
    """Annotate or gate a product table with the frozen T236 guard.

    By default this is display-only. If `apply_review=True`, released rows with
    risk above threshold are routed to review and their pre-guard HR is retained
    in traceability columns.
    """

    if product.empty:
        return product
    if model is None:
        model, loaded_config = load_reliability_guard(model_path=model_path, config_path=config_path)
        config = {**loaded_config, **(config or {})}
    else:
        config = dict(config or {})

    out = product.copy()
    out["support_guard_available"] = int(model is not None)
    out["support_guard_policy"] = str(config.get("policy_name", "t236_live_minimal_hgb_guard"))
    out["support_guard_feature_contract"] = str(config.get("feature_contract", "live_minimal_v1"))
    out["support_guard_apply_review"] = int(bool(apply_review))
    out["support_guard_risk_score"] = math.nan
    out["support_guard_threshold"] = finite_float(config.get("threshold"), math.nan)
    out["support_guard_passed"] = 0
    out["support_guard_reason"] = "artifact_unavailable" if model is None else "missing_features"
    out["support_guard_context_status"] = "artifact_unavailable" if model is None else "missing_features"
    out["support_guard_context_release_candidate"] = 0
    out["support_guard_context_candidate_policy"] = T240_CONTEXT_CANDIDATE_POLICY

    if model is None:
        return out

    features = prepare_live_reliability_features(out)
    feature_columns = list(config.get("feature_columns", LIVE_MINIMAL_FEATURES))
    missing_columns = [col for col in feature_columns if col not in features.columns]
    if missing_columns:
        out["support_guard_reason"] = "missing_feature_columns:" + ",".join(missing_columns)
        return out

    valid = features[feature_columns].notna().all(axis=1)
    if valid.any():
        scores = np.full(len(out), np.nan, dtype=float)
        scores[np.where(valid.to_numpy())[0]] = model.predict_proba(features.loc[valid, feature_columns])[:, 1]
        out["support_guard_risk_score"] = scores

    threshold = finite_float(config.get("threshold"), 0.5)
    released = pd.to_numeric(out.get("released", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int) > 0
    passed = valid & (pd.to_numeric(out["support_guard_risk_score"], errors="coerce") < threshold)
    selected_bpm = pd.to_numeric(features.get("selected_bpm", pd.Series(dtype=float)), errors="coerce")
    gap = pd.to_numeric(features.get("gap", pd.Series(dtype=float)), errors="coerce")
    roi_support = pd.to_numeric(features.get("roi_support", pd.Series(dtype=float)), errors="coerce")
    method_support = pd.to_numeric(features.get("method_support", pd.Series(dtype=float)), errors="coerce")
    high_hr_context_required = (
        valid
        & released
        & ~passed
        & selected_bpm.ge(float(config.get("high_hr_context_min_bpm", 95.0)))
        & gap.le(float(config.get("high_hr_context_max_gap_bpm", 15.0)))
        & roi_support.ge(float(config.get("high_hr_context_min_roi_support", 3.0)))
        & method_support.ge(float(config.get("high_hr_context_min_method_support", 5.0)))
    )
    context_release_candidate = (
        high_hr_context_required
        & selected_bpm.ge(float(config.get("t240_context_candidate_min_bpm", 100.0)))
        & gap.le(float(config.get("t240_context_candidate_max_gap_bpm", 1.0)))
        & roi_support.ge(float(config.get("t240_context_candidate_min_roi_support", 3.0)))
        & method_support.ge(float(config.get("t240_context_candidate_min_method_support", 5.0)))
        & (roi_support + method_support).ge(float(config.get("t240_context_candidate_min_combined_support", 10.0)))
    )
    out.loc[valid & released & passed, "support_guard_reason"] = "released_low_risk"
    out.loc[valid & released & ~passed, "support_guard_reason"] = "high_risk_review_recommended"
    out.loc[high_hr_context_required, "support_guard_reason"] = "high_hr_context_required"
    out.loc[valid & ~released, "support_guard_reason"] = "not_released_upstream"
    out.loc[valid & passed, "support_guard_passed"] = 1
    out.loc[valid & released & passed, "support_guard_context_status"] = "low_risk_release"
    out.loc[valid & released & ~passed, "support_guard_context_status"] = "high_risk_review"
    out.loc[high_hr_context_required, "support_guard_context_status"] = "high_hr_context_required"
    out.loc[valid & ~released, "support_guard_context_status"] = "not_released_upstream"
    out.loc[context_release_candidate, "support_guard_context_release_candidate"] = 1

    if apply_review:
        block = valid & released & ~passed
        out["pre_support_guard_product_hr_bpm"] = pd.to_numeric(out.get("product_hr_bpm", pd.Series(dtype=float)), errors="coerce")
        out["pre_support_guard_bridge_source"] = out.get("bridge_source", pd.Series(dtype=str)).astype(str)
        out.loc[block, "product_hr_bpm"] = math.nan
        out.loc[block, "decision"] = "review"
        out.loc[block, "released"] = 0
        out.loc[block, "bridge_source"] = "support_guard_review"
        for idx in out.index[block]:
            row = out.loc[idx]
            evidence = parse_evidence(row.get("evidence_json", "{}"))
            evidence["support_reliability_guard"] = {
                "policy": out.loc[idx, "support_guard_policy"],
                "feature_contract": out.loc[idx, "support_guard_feature_contract"],
                "passed": False,
                "reason": "high_risk_review_recommended",
                "risk_score": finite_float(row.get("support_guard_risk_score")),
                "threshold": threshold,
                "features": {
                    col: finite_float(features.loc[idx, col])
                    for col in feature_columns
                },
                "pre_gate_product_hr_bpm": finite_float(row.get("pre_support_guard_product_hr_bpm")),
            }
            out.loc[idx, "evidence_json"] = json.dumps(evidence, ensure_ascii=False)

    return out
