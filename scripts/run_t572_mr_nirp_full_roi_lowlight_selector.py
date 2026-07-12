from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_t493_selected_domain_roi_trace_cache as t493  # noqa: E402
import scripts.run_t494_roi_candidate_evaluation as t494  # noqa: E402
import scripts.run_t495_method_aware_roi_selector as t495  # noqa: E402
import scripts.run_t551_mr_nirp_lowlight_pilot as t551  # noqa: E402


TASK_ID = "T572"
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"
RUNTIME = ROOT / "runtime" / "t572_mr_nirp_full_roi_lowlight_selector"

CONDITION_INDEX = EXP / "t486_mr_nirp_condition_index.csv"
ZIP_AUDIT = EXP / "t486_mr_nirp_zip_audit.csv"
T552_METRICS = EXP / "t552_mr_nirp_lowlight_full_metrics.csv"

TRACE_INDEX_CSV = EXP / "t572_mr_nirp_full_roi_trace_index.csv"
QUALITY_CSV = EXP / "t572_mr_nirp_full_roi_quality.csv"
REFERENCE_CSV = EXP / "t572_mr_nirp_full_reference_hr.csv"
CANDIDATE_CSV = EXP / "t572_mr_nirp_full_roi_candidate_table.csv"
CLUSTER_CSV = EXP / "t572_mr_nirp_full_roi_cluster_table.csv"
DECISION_CSV = EXP / "t572_mr_nirp_full_roi_policy_decisions.csv"
SUBGROUP_CSV = EXP / "t572_mr_nirp_full_roi_subgroup_metrics.csv"
SWEEP_CSV = EXP / "t572_mr_nirp_full_roi_threshold_sweep.csv"
GATE_CSV = EXP / "t572_mr_nirp_full_roi_claim_gate.csv"
SUMMARY_JSON = EXP / "t572_mr_nirp_full_roi_lowlight_selector_summary.json"
DOC_MD = DOCS / "t572_mr_nirp_full_roi_lowlight_selector.md"
LIVE_INSIGHTS = DOCS / "t548_live_training_insights.md"

WINDOW_SECONDS = float(t551.WINDOW_SECONDS)
UNSAFE_BPM_ERROR = 10.0
CLUSTER_TOL_BPM = 8.0


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row})
    preferred = [
        "condition_id",
        "subject",
        "location",
        "motion",
        "wavelength_nm",
        "modality",
        "roi",
        "method",
        "candidate_bpm",
        "reference_bpm",
        "absolute_error_bpm",
        "artifact_flag",
        "relative_power",
        "support_modalities",
        "support_rois",
        "support_methods",
        "policy",
        "released_bpm",
        "released",
        "reason",
    ]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def append_live_insight(block: str) -> None:
    LIVE_INSIGHTS.parent.mkdir(parents=True, exist_ok=True)
    old = LIVE_INSIGHTS.read_text(encoding="utf-8") if LIVE_INSIGHTS.exists() else ""
    marker = "<!-- T572: MR-NIRP full ROI low-light selector -->"
    block = marker + "\n" + block.strip() + "\n"
    if marker in old:
        start = old.index(marker)
        after = start + len(marker)
        stops = [idx for idx in [old.find("\n<!-- ", after), old.find("\n## ", after)] if idx != -1]
        end = min(stops) if stops else len(old)
        new = old[:start].rstrip() + "\n\n" + block + "\n" + old[end:].lstrip()
    else:
        new = old.rstrip() + "\n\n" + block
    LIVE_INSIGHTS.write_text(new, encoding="utf-8")


def zip_match(zips: pd.DataFrame, condition: str, modality: str) -> Path | None:
    sub = zips[
        zips["condition_id"].astype(str).eq(condition)
        & zips["modality"].astype(str).str.lower().eq(modality.lower())
        & zips.get("exists", True).astype(bool)
        & zips.get("zip_readable", True).astype(bool)
    ].copy()
    if sub.empty:
        return None
    return Path(str(sub.iloc[0]["zip_path"]))


def reference_hr(condition: str, zips: pd.DataFrame) -> dict[str, Any]:
    pulse = zip_match(zips, condition, "PulseOx")
    if pulse is None:
        pulse = zip_match(zips, condition, "PulseOX")
    if pulse is None:
        return {"condition_id": condition, "reference_bpm": math.nan, "reference_status": "missing_pulseox"}
    try:
        trace, fs, meta = t551.read_pulseox(pulse, WINDOW_SECONDS)
        bpm, _, conf = t551.estimate_hr(trace, fs)
        return {
            "condition_id": condition,
            "reference_bpm": bpm,
            "reference_confidence": conf,
            "pulse_fs": fs,
            "pulse_zip": pulse.as_posix(),
            "reference_status": "ok",
            **meta,
        }
    except Exception as exc:
        return {
            "condition_id": condition,
            "reference_bpm": math.nan,
            "reference_status": f"{type(exc).__name__}: {exc}",
            "pulse_zip": pulse.as_posix(),
        }


def condition_meta(index: pd.DataFrame, condition: str) -> dict[str, Any]:
    row = index[index["condition_id"].astype(str).eq(condition)].iloc[0].to_dict()
    return {
        "condition_id": condition,
        "subject": str(row.get("subject", "")),
        "location": str(row.get("location", "")),
        "motion": str(row.get("motion", "")),
        "wavelength_nm": str(row.get("wavelength_nm", "")),
        "scenario": str(row.get("scenario", "")),
    }


def artifact_features(values: np.ndarray, candidate_bpm: float, relative_power: float, nyquist_bpm: float) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    mean = float(np.nanmean(x)) if x.size else math.nan
    std = float(np.nanstd(x)) if x.size else math.nan
    ac_ratio = float(std / (abs(mean) + 1e-9)) if math.isfinite(mean) else math.nan
    alt = t494.alternating_score(x)
    near_boundary = bool(math.isfinite(candidate_bpm) and math.isfinite(nyquist_bpm) and candidate_bpm >= 0.95 * nyquist_bpm)
    low_power = bool(math.isfinite(relative_power) and relative_power < 0.025)
    unstable = bool(math.isfinite(ac_ratio) and ac_ratio > 0.45)
    alternating = bool(math.isfinite(alt) and alt >= 0.75)
    return {
        "mean_signal": mean,
        "std_signal": std,
        "ac_ratio": ac_ratio,
        "alternating_score": alt,
        "near_nyquist_boundary": near_boundary,
        "low_power": low_power,
        "unstable_ac_ratio": unstable,
        "alternating_artifact": alternating,
        "artifact_flag": bool(near_boundary or low_power or unstable or alternating),
    }


def candidates_for_trace(
    trace_path: Path,
    meta: dict[str, Any],
    reference: float,
) -> list[dict[str, Any]]:
    trace = pd.read_csv(trace_path)
    rows: list[dict[str, Any]] = []
    for roi, group in trace.groupby("roi"):
        group = group.sort_values("timestamp_s")
        times = group["timestamp_s"].to_numpy(dtype=float)
        signals = {
            "intensity": group["mean_intensity"].to_numpy(dtype=float),
            "green": group["mean_g"].to_numpy(dtype=float),
            "red": group["mean_r"].to_numpy(dtype=float),
            "blue": group["mean_b"].to_numpy(dtype=float),
        }
        for method, values in signals.items():
            peaks, fs = t494.spectral_peaks(values, times, k=5)
            for peak in peaks:
                candidate = float(peak["candidate_bpm"])
                rel_power = float(peak["relative_power"])
                nyquist = float(peak.get("nyquist_bpm", math.nan))
                err = abs(candidate - reference) if math.isfinite(candidate) and math.isfinite(reference) else math.nan
                art = artifact_features(values, candidate, rel_power, nyquist)
                rows.append(
                    {
                        **meta,
                        "roi": roi,
                        "method": method,
                        "rank": int(peak["rank"]),
                        "sample_fs": fs,
                        "candidate_bpm": candidate,
                        "reference_bpm": reference,
                        "absolute_error_bpm": err,
                        "unsafe_error_gt10": bool(math.isfinite(err) and err > UNSAFE_BPM_ERROR),
                        "relative_power": rel_power,
                        "nyquist_bpm": nyquist,
                        **art,
                    }
                )
    return rows


def cluster_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    valid = candidates[np.isfinite(pd.to_numeric(candidates["candidate_bpm"], errors="coerce"))].copy()
    for condition, sub in valid.groupby("condition_id"):
        clean = sub[~sub["artifact_flag"].astype(bool)].copy()
        ref = float(sub["reference_bpm"].iloc[0])
        meta_cols = ["subject", "location", "motion", "wavelength_nm", "scenario"]
        meta = {c: str(sub[c].iloc[0]) if c in sub.columns else "" for c in meta_cols}
        for cluster_id, cluster in enumerate(t495.clusters(clean, CLUSTER_TOL_BPM)):
            bpm = float(np.nanmedian(cluster["candidate_bpm"].to_numpy(dtype=float)))
            err = abs(bpm - ref) if math.isfinite(ref) else math.nan
            out.append(
                {
                    "condition_id": condition,
                    **meta,
                    "cluster_id": cluster_id,
                    "cluster_bpm": bpm,
                    "reference_bpm": ref,
                    "absolute_error_bpm": err,
                    "unsafe_error_gt10": bool(math.isfinite(err) and err > UNSAFE_BPM_ERROR),
                    "support_rows": int(len(cluster)),
                    "support_modalities": int(cluster["modality"].nunique()),
                    "support_rois": int(cluster["roi"].nunique()),
                    "support_methods": int(cluster["method"].nunique()),
                    "mean_relative_power": float(cluster["relative_power"].fillna(0).mean()),
                    "max_relative_power": float(cluster["relative_power"].fillna(0).max()),
                    "modalities": ";".join(sorted(set(cluster["modality"].astype(str)))),
                    "rois": ";".join(sorted(set(cluster["roi"].astype(str)))),
                    "methods": ";".join(sorted(set(cluster["method"].astype(str)))),
                }
            )
    return pd.DataFrame(out)


def select_decisions(clusters: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition, sub in clusters.groupby("condition_id"):
        ref = float(sub["reference_bpm"].iloc[0])
        base = {
            "condition_id": condition,
            "subject": str(sub["subject"].iloc[0]),
            "location": str(sub["location"].iloc[0]),
            "motion": str(sub["motion"].iloc[0]),
            "wavelength_nm": str(sub["wavelength_nm"].iloc[0]),
            "scenario": str(sub["scenario"].iloc[0]),
            "reference_bpm": ref,
        }
        cross = sub[
            (sub["support_modalities"] >= 2)
            & (sub["support_rois"] >= 2)
            & (sub["support_rows"] >= 4)
            & (sub["mean_relative_power"] >= 0.015)
        ].copy()
        if cross.empty:
            single = sub[
                (sub["support_rois"] >= 3)
                & (sub["support_rows"] >= 4)
                & (sub["mean_relative_power"] >= 0.03)
            ].copy()
            pool = single
            reason = "single_modality_high_support_cluster" if not single.empty else "no_release_review_lowlight"
        else:
            pool = cross
            reason = "cross_modal_roi_cluster"

        if pool.empty:
            rows.append({**base, "policy": "review", "released": False, "released_bpm": math.nan, "absolute_error_bpm": math.nan, "unsafe_release_gt10": False, "reason": reason})
            continue
        pool = pool.sort_values(
            ["support_modalities", "support_rois", "support_rows", "mean_relative_power"],
            ascending=[False, False, False, False],
        )
        best = pool.iloc[0].to_dict()
        bpm = float(best["cluster_bpm"])
        err = abs(bpm - ref) if math.isfinite(ref) else math.nan
        rows.append(
            {
                **base,
                "policy": "release",
                "released": True,
                "released_bpm": bpm,
                "absolute_error_bpm": err,
                "unsafe_release_gt10": bool(math.isfinite(err) and err > UNSAFE_BPM_ERROR),
                "reason": reason,
                "support_rows": int(best["support_rows"]),
                "support_modalities": int(best["support_modalities"]),
                "support_rois": int(best["support_rois"]),
                "support_methods": int(best["support_methods"]),
                "mean_relative_power": float(best["mean_relative_power"]),
                "modalities": str(best["modalities"]),
                "rois": str(best["rois"]),
                "methods": str(best["methods"]),
            }
        )
    return pd.DataFrame(rows)


def metric_block(frame: pd.DataFrame, policy: str, total: int | None = None) -> dict[str, Any]:
    n_total = int(total if total is not None else len(frame))
    released = frame[frame["released"].astype(bool)].copy()
    err = pd.to_numeric(released["absolute_error_bpm"], errors="coerce").dropna()
    unsafe = pd.to_numeric(released["absolute_error_bpm"], errors="coerce") > UNSAFE_BPM_ERROR
    return {
        "policy": policy,
        "n_total": n_total,
        "n_released": int(len(released)),
        "coverage": float(len(released) / n_total) if n_total else 0.0,
        "mae_bpm": float(err.mean()) if len(err) else math.nan,
        "median_abs_error_bpm": float(err.median()) if len(err) else math.nan,
        "p90_abs_error_bpm": float(err.quantile(0.9)) if len(err) else math.nan,
        "unsafe_gt10_rate": float(unsafe.mean()) if len(released) else math.nan,
        "unsafe_gt10_per_input": float(unsafe.sum() / n_total) if n_total else 0.0,
    }


def subgroup_metrics(decisions: pd.DataFrame) -> list[dict[str, Any]]:
    rows = [metric_block(decisions, "overall")]
    for key in ["location", "motion", "wavelength_nm", "subject"]:
        for value, sub in decisions.groupby(key, dropna=False):
            row = metric_block(sub, f"{key}={value}")
            row[key] = value
            rows.append(row)
    return rows


def threshold_sweep(clusters: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for min_modalities in [1, 2]:
        for min_rois in [1, 2, 3]:
            for min_rows in [2, 3, 4, 5, 6]:
                for min_power in [0.0, 0.01, 0.015, 0.025, 0.035, 0.05]:
                    selected = []
                    for condition, sub in clusters.groupby("condition_id"):
                        pool = sub[
                            (sub["support_modalities"] >= min_modalities)
                            & (sub["support_rois"] >= min_rois)
                            & (sub["support_rows"] >= min_rows)
                            & (sub["mean_relative_power"] >= min_power)
                        ].copy()
                        if pool.empty:
                            selected.append({"condition_id": condition, "released": False, "absolute_error_bpm": math.nan})
                            continue
                        pool = pool.sort_values(["support_modalities", "support_rois", "support_rows", "mean_relative_power"], ascending=[False, False, False, False])
                        best = pool.iloc[0]
                        selected.append(
                            {
                                "condition_id": condition,
                                "released": True,
                                "absolute_error_bpm": float(best["absolute_error_bpm"]),
                            }
                        )
                    m = metric_block(pd.DataFrame(selected), "sweep")
                    rows.append(
                        {
                            "min_modalities": min_modalities,
                            "min_rois": min_rois,
                            "min_rows": min_rows,
                            "min_power": min_power,
                            **m,
                        }
                    )
    return pd.DataFrame(rows).sort_values(["unsafe_gt10_per_input", "unsafe_gt10_rate", "mae_bpm", "coverage"], ascending=[True, True, True, False])


def load_t552_best_mae() -> float:
    if not T552_METRICS.exists():
        return math.nan
    df = pd.read_csv(T552_METRICS)
    mod = df[df["level"].astype(str).eq("modality")]
    vals = pd.to_numeric(mod["mae_bpm"], errors="coerce").dropna()
    return float(vals.min()) if len(vals) else math.nan


def main() -> int:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(CONDITION_INDEX)
    zips = pd.read_csv(ZIP_AUDIT)
    ready = index[index.get("ready_rgb_nir_pulseox", True).astype(bool)].copy()

    trace_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for _, row in ready.iterrows():
        condition = str(row["condition_id"])
        meta = condition_meta(index, condition)
        ref = reference_hr(condition, zips)
        reference_rows.append({**meta, **ref})
        reference = float(ref.get("reference_bpm", math.nan))
        if not math.isfinite(reference):
            failures.append({**meta, "stage": "reference", "error": ref.get("reference_status", "missing_reference")})
            continue
        for modality in ["RGB", "NIR"]:
            zip_path = zip_match(zips, condition, modality)
            if zip_path is None:
                failures.append({**meta, "modality": modality, "stage": "zip_match", "error": "missing_zip"})
                continue
            try:
                trace, trace_meta = t493.process_mr_zip(zip_path, condition, modality)
                out_path = RUNTIME / condition / f"{modality.lower()}_roi_trace.csv"
                t493.write_trace(trace, out_path)
                meta_path = out_path.with_name(f"{modality.lower()}_roi_trace_meta.json")
                write_json(meta_path, trace_meta)
                trace_row = {
                    **meta,
                    "dataset": "MR-NIRP",
                    "modality": modality,
                    "trace_path": rel(out_path),
                    "meta_path": rel(meta_path),
                    "n_rows": int(len(trace)),
                    "n_rois": int(trace["roi"].nunique()) if not trace.empty else 0,
                    "accepted_frames": int(trace_meta.get("accepted_frames", 0)),
                    "roi_method": trace_meta.get("roi_method", ""),
                    "zip_path": zip_path.as_posix(),
                }
                trace_rows.append(trace_row)
                quality_rows.extend(t493.trace_quality_rows("MR-NIRP", condition, modality, out_path, trace, trace_meta))
                candidate_rows.extend(candidates_for_trace(out_path, trace_row, reference))
            except Exception as exc:
                failures.append({**meta, "modality": modality, "stage": "trace_or_candidate", "error": f"{type(exc).__name__}: {exc}", "zip_path": zip_path.as_posix()})

    write_csv(TRACE_INDEX_CSV, trace_rows)
    write_csv(QUALITY_CSV, quality_rows)
    write_csv(REFERENCE_CSV, reference_rows)
    write_csv(CANDIDATE_CSV, candidate_rows)

    candidates = pd.DataFrame(candidate_rows)
    clusters = cluster_rows(candidates) if not candidates.empty else pd.DataFrame()
    clusters.to_csv(CLUSTER_CSV, index=False, encoding="utf-8-sig")
    decisions = select_decisions(clusters) if not clusters.empty else pd.DataFrame()
    decisions.to_csv(DECISION_CSV, index=False, encoding="utf-8-sig")
    subgroup = subgroup_metrics(decisions) if not decisions.empty else []
    write_csv(SUBGROUP_CSV, subgroup)
    sweep = threshold_sweep(clusters) if not clusters.empty else pd.DataFrame()
    sweep.to_csv(SWEEP_CSV, index=False, encoding="utf-8-sig")

    overall = metric_block(decisions, "t572_deployable_roi_policy") if not decisions.empty else {}
    t552_best = load_t552_best_mae()
    best_20 = sweep[sweep["coverage"] >= 0.20].head(1).to_dict(orient="records") if not sweep.empty else []
    best_20_row = best_20[0] if best_20 else {}
    t572_mae = float(overall.get("mae_bpm", math.nan)) if overall else math.nan
    gates = [
        {
            "gate": "mr_nirp_full_roi_candidates_available",
            "passed": bool(len(trace_rows) >= 2 * max(1, ready["condition_id"].nunique() * 0.8) and len(candidate_rows) > 0),
            "evidence": f"conditions_ready={ready['condition_id'].nunique()} trace_files={len(trace_rows)} candidate_rows={len(candidate_rows)} failures={len(failures)}",
            "claim_allowed": "MR-NIRP can be audited at ROI/candidate level instead of only full-frame mean trace.",
            "claim_not_allowed": "Low-light robustness solved.",
        },
        {
            "gate": "deployable_roi_policy_20pct_safe",
            "passed": bool(
                overall
                and overall.get("coverage", 0.0) >= 0.20
                and (overall.get("unsafe_gt10_rate") or 1.0) <= 0.10
                and (overall.get("mae_bpm") or 999.0) <= 10.0
            ),
            "evidence": overall,
            "claim_allowed": "A label-free ROI/cross-modal policy supports bounded low-light release on MR-NIRP.",
            "claim_not_allowed": "Low-light/RGB-NIR solved for all camera or lighting settings.",
        },
        {
            "gate": "roi_policy_beats_full_frame_t552",
            "passed": bool(math.isfinite(t572_mae) and math.isfinite(t552_best) and t572_mae < t552_best),
            "evidence": f"t572_mae={t572_mae}; t552_best_full_frame_mae={t552_best}",
            "claim_allowed": "ROI/candidate routing improves over full-frame MR-NIRP trace extraction.",
            "claim_not_allowed": "SOTA low-light performance.",
        },
        {
            "gate": "audit_sweep_has_20pct_headroom",
            "passed": bool(best_20_row and (best_20_row.get("unsafe_gt10_rate") or 1.0) <= 0.10 and (best_20_row.get("mae_bpm") or 999.0) <= 10.0),
            "evidence": best_20_row,
            "claim_allowed": "There is ROI/candidate headroom for further low-light selector training.",
            "claim_not_allowed": "The sweep threshold is a deployable trained policy; it is same-data audit evidence.",
        },
    ]
    write_csv(GATE_CSV, gates)

    summary = {
        "task_id": TASK_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "mr_nirp_full_roi_lowlight_selector_completed_review_gates",
        "conditions_ready": int(ready["condition_id"].nunique()),
        "trace_files": len(trace_rows),
        "candidate_rows": len(candidate_rows),
        "cluster_rows": int(len(clusters)),
        "failures": failures,
        "overall_policy": overall,
        "t552_best_full_frame_mae": t552_best,
        "best_20pct_audit_sweep": best_20_row,
        "claim_gate_passed": int(sum(bool(g["passed"]) for g in gates)),
        "claim_gate_total": len(gates),
        "gates": gates,
        "claim_boundary": "T572 tests all available MR-NIRP conditions with ROI/candidate evidence. Passed gates can support bounded low-light release language; failed gates remain blockers for solved low-light or SOTA claims.",
        "outputs": {
            "trace_index": TRACE_INDEX_CSV,
            "candidate_table": CANDIDATE_CSV,
            "cluster_table": CLUSTER_CSV,
            "decisions": DECISION_CSV,
            "subgroup_metrics": SUBGROUP_CSV,
            "sweep": SWEEP_CSV,
            "claim_gate": GATE_CSV,
            "doc": DOC_MD,
        },
    }
    write_json(SUMMARY_JSON, summary)

    doc_lines = [
        "# T572 MR-NIRP Full ROI Low-Light Selector",
        "",
        "## Purpose",
        "",
        "T552 showed that full-frame RGB/NIR mean traces cannot support a low-light superiority claim. T572 tests the next technically meaningful route: stream all available MR-NIRP RGB/NIR zip videos, extract ROI-level traces, preserve multiple spectral candidates, and release only when candidate evidence has cross-modal or high-support ROI agreement.",
        "",
        "## Result",
        "",
        f"- Conditions attempted: {ready['condition_id'].nunique()}",
        f"- Trace files: {len(trace_rows)}",
        f"- Candidate rows: {len(candidate_rows)}",
        f"- Claim gates passed: {summary['claim_gate_passed']}/{summary['claim_gate_total']}",
        f"- Deployable policy: coverage={overall.get('coverage') if overall else None}, MAE={overall.get('mae_bpm') if overall else None}, unsafe/released={overall.get('unsafe_gt10_rate') if overall else None}",
        f"- Best 20% same-data audit sweep: {best_20_row}",
        "",
        "## Interpretation",
        "",
        "This task separates deployable no-label release evidence from same-data audit headroom. If the deployable gate fails but the audit sweep passes, the correct next step is learned low-light selector training or additional low-light data, not stronger manuscript wording.",
        "",
        "## Claim Boundary",
        "",
        summary["claim_boundary"],
    ]
    DOC_MD.write_text("\n".join(doc_lines) + "\n", encoding="utf-8")
    append_live_insight(
        "\n".join(
            [
                f"## T572 MR-NIRP full ROI low-light selector ({datetime.now().isoformat(timespec='seconds')})",
                "",
                f"- Conditions: {ready['condition_id'].nunique()}, traces: {len(trace_rows)}, candidates: {len(candidate_rows)}.",
                f"- Deployable policy evidence: {overall}.",
                f"- 20% audit headroom: {best_20_row}.",
                "- If MR-NIRP still fails the deployable 20% gate, low-light must remain a review-boundary or future-work claim until ROI/deep low-light training closes it.",
            ]
        )
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
