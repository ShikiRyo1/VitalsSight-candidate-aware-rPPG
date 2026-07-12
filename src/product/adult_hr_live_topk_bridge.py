from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.traditional_rppg import METHODS
from src.product.adult_hr_mvp import AdultHRMVPResult
from src.product.adult_hr_topk_bridge import build_adult_hr_bridge_product_table
from src.selection.topk_bridge import TopKBridgeConfig, select_topk_bridge
from src.signal.estimate import top_k_rate_fft


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def adult_plausibility(bpm: float) -> float:
    if 55.0 <= bpm <= 105.0:
        return 1.0
    if 105.0 < bpm <= 135.0:
        return 0.85
    if 135.0 < bpm <= 170.0:
        return 0.75
    if 45.0 <= bpm < 55.0 or 170.0 < bpm <= 180.0:
        return 0.45
    return 0.0


def snr_proxy_db(band_power: float, total_power: float) -> float:
    noise_power = max(total_power - band_power, 1e-12)
    return float(10.0 * math.log10((band_power + 1e-12) / noise_power))


def build_live_peak_table(
    roi_ts: pd.DataFrame,
    *,
    fps: float,
    top_k: int = 5,
) -> pd.DataFrame:
    if roi_ts.empty or fps <= 0:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for sample_id, window in roi_ts.groupby("sample_id", sort=True):
        for region, group in window.groupby("region", sort=True):
            rgb = group.sort_values("frame_index")[["mean_r", "mean_g", "mean_b"]].to_numpy(dtype=float)
            if len(rgb) < max(32, int(8.0 * fps)):
                continue
            for method_name, method_fn in sorted(METHODS.items()):
                try:
                    signal_values = method_fn(rgb)
                    peaks, band_power, total_power = top_k_rate_fft(signal_values, fps, min_bpm=45.0, max_bpm=180.0, top_k=top_k)
                except Exception:
                    continue
                snr = snr_proxy_db(band_power, total_power)
                first = group.iloc[0]
                for peak in peaks:
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "region": region,
                            "method": method_name,
                            "window_id": first.get("window_id"),
                            "start_sec": first.get("start_sec"),
                            "end_sec": first.get("end_sec"),
                            "candidate_bpm": finite_float(peak.get("peak_bpm")),
                            "peak_hz": finite_float(peak.get("peak_hz")),
                            "rank": int(finite_float(peak.get("rank"), 99.0)),
                            "power_fraction": finite_float(peak.get("power_fraction")),
                            "peak_power": finite_float(peak.get("peak_power")),
                            "band_power": band_power,
                            "total_power": total_power,
                            "snr_proxy_db": snr,
                        }
                    )
    return pd.DataFrame(rows)


def roi_timeseries_with_window_ids(result: AdultHRMVPResult) -> pd.DataFrame:
    roi_ts = result.roi_timeseries.copy()
    if roi_ts.empty or result.windows.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for _, win in result.windows.iterrows():
        start = finite_float(win.get("start_sec"))
        end = finite_float(win.get("end_sec"))
        sample_id = str(win.get("sample_id"))
        if not math.isfinite(start) or not math.isfinite(end) or not sample_id:
            continue
        sub = roi_ts[(pd.to_numeric(roi_ts["timestamp_s"], errors="coerce") >= start) & (pd.to_numeric(roi_ts["timestamp_s"], errors="coerce") < end)].copy()
        if sub.empty:
            continue
        sub["sample_id"] = sample_id
        sub["window_id"] = win.get("window_id")
        sub["start_sec"] = start
        sub["end_sec"] = end
        rows.append(sub)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def cluster_live_peaks(peaks: pd.DataFrame, *, tolerance_bpm: float = 4.0) -> pd.DataFrame:
    if peaks.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for sample_id, group in peaks.groupby("sample_id", sort=True):
        valid = group[np.isfinite(pd.to_numeric(group["candidate_bpm"], errors="coerce"))].copy()
        if valid.empty:
            continue
        valid["candidate_bpm"] = pd.to_numeric(valid["candidate_bpm"], errors="coerce")
        valid = valid.sort_values("candidate_bpm")
        clusters: list[list[int]] = []
        center = math.nan
        current: list[int] = []
        for idx, row in valid.iterrows():
            bpm = float(row["candidate_bpm"])
            if not current or abs(bpm - center) <= tolerance_bpm:
                current.append(idx)
                center = float(valid.loc[current, "candidate_bpm"].median())
            else:
                clusters.append(current)
                current = [idx]
                center = bpm
        if current:
            clusters.append(current)
        for cluster_id, idxs in enumerate(clusters):
            cluster = valid.loc[idxs].copy()
            power = pd.to_numeric(cluster["peak_power"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            bpm = pd.to_numeric(cluster["candidate_bpm"], errors="coerce").to_numpy(dtype=float)
            weights = power + 1e-6
            candidate_bpm = float(np.average(bpm, weights=weights))
            methods = sorted(cluster["method"].astype(str).dropna().unique())
            rois = sorted(cluster["region"].astype(str).dropna().unique())
            rows.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": f"{sample_id}_tk{cluster_id:02d}",
                    "candidate_bpm": candidate_bpm,
                    "support_count": int(len(cluster)),
                    "support_rois": int(len(rois)),
                    "support_methods": int(len(methods)),
                    "support_windows": 1,
                    "full_support_count": int(len(cluster)),
                    "subwindow_support_count": 0,
                    "top1_support_count": int((pd.to_numeric(cluster["rank"], errors="coerce") == 1).sum()),
                    "full_top1_support_count": int((pd.to_numeric(cluster["rank"], errors="coerce") == 1).sum()),
                    "pos_chrom_count": int(cluster["method"].astype(str).isin(["POS", "CHROM"]).sum()),
                    "green_pbv_count": int(cluster["method"].astype(str).isin(["GREEN", "PBV"]).sum()),
                    "ica_lgi_count": int(cluster["method"].astype(str).isin(["ICA", "LGI"]).sum()),
                    "mean_power_fraction": float(pd.to_numeric(cluster["power_fraction"], errors="coerce").mean()),
                    "max_power_fraction": float(pd.to_numeric(cluster["power_fraction"], errors="coerce").max()),
                    "sum_power_fraction": float(pd.to_numeric(cluster["power_fraction"], errors="coerce").sum()),
                    "rank_score": float((1.0 / pd.to_numeric(cluster["rank"], errors="coerce").replace(0, np.nan)).sum()),
                    "mean_snr_proxy_db": float(pd.to_numeric(cluster["snr_proxy_db"], errors="coerce").mean()),
                    "adult_plausibility": adult_plausibility(candidate_bpm),
                    "member_summary": ",".join(f"{r.region}:{r.method}:r{int(finite_float(r.rank, 99.0))}" for r in cluster.itertuples(index=False)),
                    "regions": ",".join(rois),
                    "methods": ",".join(methods),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Context features needed by the bridge and evidence UI.
    context_rows: list[dict[str, Any]] = []
    for sample_id, group in out.groupby("sample_id", sort=False):
        bpms = pd.to_numeric(group["candidate_bpm"], errors="coerce").to_numpy(dtype=float)
        for idx, row in group.iterrows():
            bpm = finite_float(row["candidate_bpm"])
            upper = (bpms > bpm + 8.0) & (bpms <= min(180.0, bpm + 60.0))
            lower = (bpms < bpm - 8.0) & (bpms >= max(45.0, bpm - 60.0))
            half = np.abs(bpms - bpm / 2.0) <= tolerance_bpm
            double = np.abs(bpms - bpm * 2.0) <= tolerance_bpm
            context_rows.append(
                {
                    "_idx": idx,
                    "upper_phys_support": float(np.sum(upper)),
                    "upper_phys_pos_chrom": 0.0,
                    "lower_phys_support": float(np.sum(lower)),
                    "lower_phys_pos_chrom": 0.0,
                    "double_harmonic_support": float(np.sum(double)),
                    "half_harmonic_support": float(np.sum(half)),
                    "upper_alt_support": float(np.sum(upper)),
                    "upper_alt_pos_chrom": 0.0,
                    "lower_alt_support": float(np.sum(lower)),
                }
            )
    return out.join(pd.DataFrame(context_rows).set_index("_idx"), how="left")


def score_live_candidates(candidates: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    out = candidates.copy()
    anchor_map = anchors.drop_duplicates("sample_id").set_index("sample_id") if not anchors.empty else pd.DataFrame()
    anchor_values: list[float] = []
    for sample_id in out["sample_id"].astype(str):
        if sample_id in anchor_map.index:
            anchor_values.append(finite_float(anchor_map.loc[sample_id].get("selected_bpm")))
        else:
            anchor_values.append(math.nan)
    out["t150_selected_bpm"] = anchor_values
    out["dist_to_t150"] = (pd.to_numeric(out["candidate_bpm"], errors="coerce") - pd.to_numeric(out["t150_selected_bpm"], errors="coerce")).abs()
    out["t150_confidence"] = 1.0
    out["t150_reason"] = "live_anchor"
    out["t157_score"] = (
        1.45 * pd.to_numeric(out["support_rois"], errors="coerce")
        + 1.15 * pd.to_numeric(out["support_methods"], errors="coerce")
        + 0.12 * pd.to_numeric(out["support_count"], errors="coerce")
        + 0.18 * pd.to_numeric(out["rank_score"], errors="coerce")
        + 4.50 * pd.to_numeric(out["max_power_fraction"], errors="coerce")
        + 2.00 * pd.to_numeric(out["mean_power_fraction"], errors="coerce")
        + 1.10 * pd.to_numeric(out["adult_plausibility"], errors="coerce")
        + 0.30 * pd.to_numeric(out["pos_chrom_count"], errors="coerce")
        - 0.04 * pd.to_numeric(out["dist_to_t150"], errors="coerce").fillna(0.0)
    )
    return out


def build_live_anchor_table(result: AdultHRMVPResult) -> pd.DataFrame:
    if result.windows.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in result.windows.iterrows():
        candidate = finite_float(row.get("candidate_hr_bpm"), finite_float(row.get("max_power_candidate_bpm")))
        rows.append(
            {
                "sample_id": str(row.get("sample_id")),
                "selected_bpm": candidate,
                "released": int(bool(row.get("accepted", False))),
                "review_reason": str(row.get("refusal_reason", "")),
            }
        )
    return pd.DataFrame(rows)


def attach_live_window_context(product: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    if product.empty or windows.empty or "sample_id" not in windows.columns:
        return product
    context_cols = [
        "sample_id",
        "window_id",
        "start_sec",
        "end_sec",
        "max_power_candidate_bpm",
        "candidate_hr_bpm",
        "accepted",
        "refusal_reason",
    ]
    available = [col for col in context_cols if col in windows.columns]
    if "sample_id" not in available:
        return product
    context = windows[available].copy()
    context["sample_id"] = context["sample_id"].astype(str)
    rename = {
        "candidate_hr_bpm": "anchor_candidate_hr_bpm",
        "accepted": "anchor_accepted",
        "refusal_reason": "anchor_refusal_reason",
    }
    context = context.rename(columns={key: value for key, value in rename.items() if key in context.columns})
    out = product.copy()
    out["sample_id"] = out["sample_id"].astype(str)
    return out.merge(context.drop_duplicates("sample_id"), on="sample_id", how="left")


def build_live_adult_hr_topk_bridge(
    result: AdultHRMVPResult,
    *,
    top_k: int = 5,
    bridge_config: TopKBridgeConfig | None = None,
    policy_name: str = "live_topk_bridge_v1",
    temporal_gate_policy: str = "none",
    temporal_neighbor_tolerance_bpm: float = 12.0,
    temporal_low_anchor_bpm: float = 82.0,
    respect_anchor_review: bool = False,
    conflict_guard_policy: str = "none",
    conflict_low_pred_bpm: float = 75.0,
    conflict_max_gap_bpm: float = 15.0,
    conflict_max_power_upper_bpm: float = 120.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fps = finite_float(result.metadata.get("analysis_fps"))
    windowed = roi_timeseries_with_window_ids(result)
    anchors = build_live_anchor_table(result)
    peaks = build_live_peak_table(windowed, fps=fps, top_k=top_k)
    clusters = cluster_live_peaks(peaks)
    candidates = score_live_candidates(clusters, anchors)
    bridge = select_topk_bridge(candidates, anchors, config=bridge_config, policy_name=policy_name) if not candidates.empty else pd.DataFrame()
    product = build_adult_hr_bridge_product_table(bridge, candidates) if not bridge.empty else pd.DataFrame()
    product = attach_live_window_context(product, result.windows)
    product = apply_temporal_support_gate(
        product,
        policy=temporal_gate_policy,
        neighbor_tolerance_bpm=temporal_neighbor_tolerance_bpm,
        low_anchor_bpm=temporal_low_anchor_bpm,
    )
    if respect_anchor_review:
        product = apply_upstream_review_gate(product)
    product = apply_low_prediction_conflict_guard(
        product,
        policy=conflict_guard_policy,
        low_pred_bpm=conflict_low_pred_bpm,
        max_gap_bpm=conflict_max_gap_bpm,
        max_power_upper_bpm=conflict_max_power_upper_bpm,
    )
    return candidates, bridge, product


def sample_sort_key(sample_id: object) -> tuple[str, int, str]:
    text = str(sample_id)
    prefix = text
    number = 0
    if "_w" in text:
        prefix, tail = text.rsplit("_w", 1)
        digits = "".join(ch for ch in tail if ch.isdigit())
        number = int(digits) if digits else 0
    return prefix, number, text


def parse_evidence(raw: object) -> dict[str, Any]:
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def apply_temporal_support_gate(
    product: pd.DataFrame,
    *,
    policy: str = "none",
    neighbor_tolerance_bpm: float = 12.0,
    low_anchor_bpm: float = 82.0,
) -> pd.DataFrame:
    """Gate upper-band rescues using neighboring anchor support.

    Policy values:
    - ``none``: leave the product table unchanged.
    - ``veto_to_anchor``: unsupported upper-band rescues fall back to anchor.
    - ``review_unsupported``: unsupported upper-band rescues become review rows.

    The gate is label-free. It uses the neighboring product anchors from the
    same uploaded video, so an implementation can either delay release by one
    window or update provisional readings after the next window arrives.
    """

    normalized = str(policy or "none").strip().lower()
    if normalized in {"", "none", "off", "disabled"} or product.empty:
        return product
    if normalized not in {"veto_to_anchor", "review_unsupported"}:
        raise ValueError(f"Unknown temporal gate policy: {policy}")

    out = product.copy()
    for col in [
        "pre_temporal_product_hr_bpm",
        "pre_temporal_bridge_source",
        "temporal_support_count",
        "temporal_gate_passed",
        "temporal_gate_policy",
        "temporal_gate_reason",
    ]:
        if col not in out.columns:
            out[col] = "" if col.endswith(("source", "policy", "reason")) else math.nan
    out["pre_temporal_product_hr_bpm"] = pd.to_numeric(out.get("product_hr_bpm", pd.Series(dtype=float)), errors="coerce")
    out["pre_temporal_bridge_source"] = out.get("bridge_source", pd.Series(dtype=str)).astype(str)
    out["temporal_gate_policy"] = normalized
    out["temporal_gate_reason"] = "not_upper_band_rescue"
    out["temporal_gate_passed"] = 1
    out["temporal_support_count"] = 0

    sorted_index = sorted(out.index.tolist(), key=lambda idx: sample_sort_key(out.loc[idx].get("sample_id", idx)))
    anchor_values = pd.to_numeric(out.get("bridge_anchor_bpm", pd.Series(dtype=float)), errors="coerce")
    for idx in sorted_index:
        row = out.loc[idx]
        if str(row.get("bridge_source", "")) != "upper_band_rescue_v1":
            continue
        rescue_bpm = finite_float(row.get("product_hr_bpm"))
        anchor_bpm = finite_float(row.get("bridge_anchor_bpm"))
        if not math.isfinite(rescue_bpm) or not math.isfinite(anchor_bpm):
            out.loc[idx, "temporal_gate_passed"] = 0
            out.loc[idx, "temporal_gate_reason"] = "missing_rescue_or_anchor"
            continue
        neighbor_values = anchor_values.drop(index=idx, errors="ignore").dropna()
        support_count = int(((neighbor_values - rescue_bpm).abs() <= neighbor_tolerance_bpm).sum())
        anchor_low = anchor_bpm < low_anchor_bpm
        gate_passed = support_count >= 1 or anchor_low
        out.loc[idx, "temporal_support_count"] = support_count
        out.loc[idx, "temporal_gate_passed"] = int(gate_passed)
        out.loc[idx, "temporal_gate_reason"] = "supported_by_neighbor_or_low_anchor" if gate_passed else "unsupported_rescue"
        if gate_passed:
            continue
        if normalized == "veto_to_anchor":
            out.loc[idx, "product_hr_bpm"] = anchor_bpm
            out.loc[idx, "candidate_hr_bpm"] = anchor_bpm
            out.loc[idx, "bridge_source"] = "temporal_veto_to_anchor"
            out.loc[idx, "decision"] = "release"
            out.loc[idx, "released"] = 1
        else:
            out.loc[idx, "product_hr_bpm"] = math.nan
            out.loc[idx, "bridge_source"] = "temporal_review_unsupported_rescue"
            out.loc[idx, "decision"] = "review"
            out.loc[idx, "released"] = 0
        evidence = parse_evidence(row.get("evidence_json", "{}"))
        evidence["temporal_gate"] = {
            "policy": normalized,
            "passed": bool(gate_passed),
            "reason": "unsupported_rescue",
            "neighbor_tolerance_bpm": neighbor_tolerance_bpm,
            "support_count": support_count,
            "low_anchor_bpm": low_anchor_bpm,
            "anchor_bpm": anchor_bpm,
            "pre_temporal_product_hr_bpm": rescue_bpm,
            "final_product_hr_bpm": finite_float(out.loc[idx].get("product_hr_bpm")),
        }
        out.loc[idx, "evidence_json"] = json.dumps(evidence, ensure_ascii=False)
    return out


def apply_upstream_review_gate(product: pd.DataFrame) -> pd.DataFrame:
    """Respect the upstream review/refusal decision in the final product output.

    T226/T227 showed that forcing every reviewed anchor into a released bridge
    output improves coverage but increases unsafe releases. This label-free gate
    keeps traceability fields while routing upstream-reviewed windows back to
    review unless a future policy explicitly validates a stronger rescue path.
    """

    if product.empty:
        return product
    out = product.copy()
    out["pre_upstream_review_product_hr_bpm"] = pd.to_numeric(out.get("product_hr_bpm", pd.Series(dtype=float)), errors="coerce")
    out["pre_upstream_review_bridge_source"] = out.get("bridge_source", pd.Series(dtype=str)).astype(str)
    out["upstream_review_gate_passed"] = 1
    out["upstream_review_gate_reason"] = "anchor_released"
    for idx, row in out.iterrows():
        evidence = parse_evidence(row.get("evidence_json", "{}"))
        anchor_released = int(finite_float(evidence.get("anchor_released"), 1.0))
        if anchor_released > 0:
            continue
        out.loc[idx, "upstream_review_gate_passed"] = 0
        out.loc[idx, "upstream_review_gate_reason"] = "anchor_review_respected"
        out.loc[idx, "product_hr_bpm"] = math.nan
        out.loc[idx, "decision"] = "review"
        out.loc[idx, "released"] = 0
        out.loc[idx, "bridge_source"] = "upstream_review_respected"
        evidence["upstream_review_gate"] = {
            "passed": False,
            "reason": "anchor_review_respected",
            "pre_gate_product_hr_bpm": finite_float(row.get("product_hr_bpm")),
            "pre_gate_bridge_source": str(row.get("bridge_source", "")),
        }
        out.loc[idx, "evidence_json"] = json.dumps(evidence, ensure_ascii=False)
    return out


def apply_low_prediction_conflict_guard(
    product: pd.DataFrame,
    *,
    policy: str = "none",
    low_pred_bpm: float = 75.0,
    max_gap_bpm: float = 15.0,
    max_power_upper_bpm: float = 120.0,
) -> pd.DataFrame:
    """Route low-HR outputs with strong max-peak conflict to review.

    T228 found that some T227 residual unsafe releases are low-frequency alias
    outputs where the released HR is low while the strongest raw peak is far
    away. T232 calibrates this into a transfer-safer v2. The guard is
    intentionally conservative: it does not correct to the alternative peak; it
    withholds the reading for review.
    """

    normalized = str(policy or "none").strip().lower()
    if product.empty or normalized in {"", "none", "off", "disabled"}:
        return product
    if normalized not in {"low_pred_conflict_v1", "low_pred_conflict_v2"}:
        raise ValueError(f"Unknown conflict guard policy: {policy}")

    out = product.copy()
    out["pre_conflict_guard_product_hr_bpm"] = pd.to_numeric(
        out.get("product_hr_bpm", pd.Series(dtype=float)), errors="coerce"
    )
    out["pre_conflict_guard_bridge_source"] = out.get("bridge_source", pd.Series(dtype=str)).astype(str)
    out["conflict_guard_policy"] = normalized
    out["conflict_guard_passed"] = 1
    out["conflict_guard_reason"] = "passed_or_not_released"
    product_hr = pd.to_numeric(out.get("product_hr_bpm", pd.Series(dtype=float)), errors="coerce")
    max_power = pd.to_numeric(out.get("max_power_candidate_bpm", pd.Series(dtype=float)), errors="coerce")
    released = pd.to_numeric(out.get("released", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int) > 0
    out["conflict_guard_max_gap_bpm"] = (product_hr - max_power).abs()
    if normalized == "low_pred_conflict_v2":
        low_pred_bpm = 80.0
        max_gap_bpm = 25.0
        block = (
            released
            & product_hr.lt(low_pred_bpm)
            & out["conflict_guard_max_gap_bpm"].gt(max_gap_bpm)
            & max_power.lt(max_power_upper_bpm)
        )
    else:
        block = released & product_hr.lt(low_pred_bpm) & out["conflict_guard_max_gap_bpm"].gt(max_gap_bpm)
    for idx in out.index[block]:
        row = out.loc[idx]
        out.loc[idx, "conflict_guard_passed"] = 0
        out.loc[idx, "conflict_guard_reason"] = "low_pred_with_strong_max_power_conflict"
        out.loc[idx, "product_hr_bpm"] = math.nan
        out.loc[idx, "decision"] = "review"
        out.loc[idx, "released"] = 0
        out.loc[idx, "bridge_source"] = "low_pred_conflict_review"
        evidence = parse_evidence(row.get("evidence_json", "{}"))
        evidence["conflict_guard"] = {
            "policy": normalized,
            "passed": False,
            "reason": "low_pred_with_strong_max_power_conflict",
            "low_pred_bpm": low_pred_bpm,
            "max_gap_bpm": max_gap_bpm,
            "max_power_upper_bpm": max_power_upper_bpm if normalized == "low_pred_conflict_v2" else None,
            "pre_gate_product_hr_bpm": finite_float(row.get("product_hr_bpm")),
            "max_power_candidate_bpm": finite_float(row.get("max_power_candidate_bpm")),
            "observed_gap_bpm": finite_float(row.get("conflict_guard_max_gap_bpm")),
        }
        out.loc[idx, "evidence_json"] = json.dumps(evidence, ensure_ascii=False)
    return out
