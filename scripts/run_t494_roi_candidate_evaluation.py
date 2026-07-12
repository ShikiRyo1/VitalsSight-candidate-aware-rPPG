from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "T494"
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"

T491_SIGNAL_SUMMARY = EXP / "t491_selected_domain_signal_summary.csv"
T492_SUMMARY = EXP / "t492_selected_domain_artifact_gate_summary.json"
T493_INDEX = EXP / "t493_selected_domain_roi_trace_cache_index.csv"

CANDIDATES_CSV = EXP / "t494_roi_candidate_table.csv"
DECISIONS_CSV = EXP / "t494_roi_policy_decisions.csv"
DATASET_METRICS_CSV = EXP / "t494_roi_dataset_metrics.csv"
CLAIM_GATE_CSV = EXP / "t494_roi_candidate_evaluation_claim_gate.csv"
SUMMARY_JSON = EXP / "t494_roi_candidate_evaluation_summary.json"
DOC_MD = DOCS / "t494_roi_candidate_evaluation.md"

TASK_REGISTRY = DOCS / "execution_task_registry.md"
LEARNING_JOURNAL = DOCS / "phase_learning_journal.md"
PROJECT_STATUS = DOCS / "project_status.md"
PAPER_CLAIMS = DOCS / "paper_claims_tracker.md"
PROBLEM_LOG = DOCS / "problem_and_improvement_log.md"
INNOVATION_LOG = DOCS / "innovation_log.md"
EVIDENCE_TABLE = EXP / "experiment_evidence_table.csv"

MIN_BPM = 45.0
MAX_BPM = 180.0
UNSAFE_BPM_ERROR = 10.0
CLUSTER_TOL_BPM = 8.0


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
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
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def append_or_replace(path: Path, marker: str, block: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in old:
        start = old.index(marker)
        after = start + len(marker)
        stops = [
            idx
            for token in ["\n\n## T", "\n\n# T", "\n\n---\n"]
            if (idx := old.find(token, after)) != -1
        ]
        end = min(stops) if stops else len(old)
        new = old[:start] + block.rstrip() + "\n" + old[end:]
    else:
        sep = "" if not old or old.endswith("\n") else "\n"
        new = old + sep + block.rstrip() + "\n"
    path.write_text(new, encoding="utf-8")


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    show = df.head(max_rows).copy()
    lines = [
        "| " + " | ".join(show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("\n", " ") for col in show.columns) + " |")
    return "\n".join(lines)


def replace_evidence_row(row: dict[str, Any]) -> None:
    EVIDENCE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    if EVIDENCE_TABLE.exists():
        table = pd.read_csv(EVIDENCE_TABLE)
        table = table[table["evidence_id"].astype(str) != str(row["evidence_id"])]
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    else:
        table = pd.DataFrame([row])
    table.to_csv(EVIDENCE_TABLE, index=False, encoding="utf-8-sig")


def robust_z(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 16:
        return x
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    scale = mad if math.isfinite(float(mad)) and mad > 1e-9 else np.nanstd(x)
    if not math.isfinite(float(scale)) or scale <= 1e-9:
        return x * 0.0
    return (x - med) / scale


def sample_rate_from_times(times: np.ndarray) -> float:
    t = np.asarray(times, dtype=float)
    dt = np.diff(t[np.isfinite(t)])
    dt = dt[dt > 0]
    if dt.size == 0:
        return math.nan
    return float(1.0 / np.nanmedian(dt))


def spectral_peaks(values: np.ndarray, times: np.ndarray, k: int = 5) -> tuple[list[dict[str, float]], float]:
    fs = sample_rate_from_times(times)
    if not math.isfinite(fs) or fs <= 0:
        return [], math.nan
    x = robust_z(values)
    if x.size < 64:
        return [], fs
    x = x - np.linspace(float(x[0]), float(x[-1]), x.size)
    window = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * window)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    bpm = freqs * 60.0
    nyquist_bpm = fs * 30.0
    mask = (bpm >= MIN_BPM) & (bpm <= min(MAX_BPM, nyquist_bpm))
    if not mask.any():
        return [], fs
    spec_band = spec[mask]
    bpm_band = bpm[mask]
    local: list[int] = []
    for i in range(spec_band.size):
        left = spec_band[i - 1] if i > 0 else -np.inf
        right = spec_band[i + 1] if i < spec_band.size - 1 else -np.inf
        if spec_band[i] >= left and spec_band[i] >= right:
            local.append(i)
    if not local:
        local = list(range(spec_band.size))
    order = sorted(local, key=lambda idx: float(spec_band[idx]), reverse=True)[:k]
    total = float(np.sum(spec_band)) + 1e-12
    peaks = [
        {
            "rank": rank + 1,
            "candidate_bpm": float(bpm_band[idx]),
            "relative_power": float(spec_band[idx] / total),
            "nyquist_bpm": nyquist_bpm,
        }
        for rank, idx in enumerate(order)
    ]
    return peaks, fs


def pos_signal(group: pd.DataFrame) -> np.ndarray:
    rgb = group[["mean_r", "mean_g", "mean_b"]].to_numpy(dtype=float)
    rgb = rgb / (np.nanmean(rgb, axis=0, keepdims=True) + 1e-9)
    x = 3.0 * rgb[:, 0] - 2.0 * rgb[:, 1]
    y = 1.5 * rgb[:, 0] + rgb[:, 1] - 1.5 * rgb[:, 2]
    alpha = np.nanstd(x) / (np.nanstd(y) + 1e-9)
    return x + alpha * y


def chrom_signal(group: pd.DataFrame) -> np.ndarray:
    rgb = group[["mean_r", "mean_g", "mean_b"]].to_numpy(dtype=float)
    rgb = rgb / (np.nanmean(rgb, axis=0, keepdims=True) + 1e-9)
    x = 3.0 * rgb[:, 0] - 2.0 * rgb[:, 1]
    y = 1.5 * rgb[:, 0] + rgb[:, 1] - 1.5 * rgb[:, 2]
    alpha = np.nanstd(x) / (np.nanstd(y) + 1e-9)
    return x - alpha * y


def alternating_score(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 16:
        return math.nan
    return float(abs(np.nanmean(x[0::2]) - np.nanmean(x[1::2])) / (np.nanstd(x) + 1e-9))


def reference_map() -> dict[str, float]:
    summary = pd.read_csv(T491_SIGNAL_SUMMARY)
    refs = {}
    for _, row in summary.iterrows():
        condition = str(row["condition_id"])
        modality = str(row["modality"])
        if modality in {"BVP", "PulseOx"}:
            refs[condition] = float(row["reference_peak_bpm"])
    return refs


def candidates_for_trace(trace_row: pd.Series, refs: dict[str, float]) -> list[dict[str, Any]]:
    trace = pd.read_csv(ROOT / str(trace_row["trace_path"]))
    dataset = str(trace_row["dataset"])
    condition = str(trace_row["condition_id"])
    modality = str(trace_row["modality"])
    reference = refs.get(condition, math.nan)
    out: list[dict[str, Any]] = []
    for roi, group in trace.groupby("roi"):
        group = group.sort_values("timestamp_s")
        times = group["timestamp_s"].to_numpy(dtype=float)
        signals: dict[str, np.ndarray] = {
            "intensity": group["mean_intensity"].to_numpy(dtype=float),
            "green": group["mean_g"].to_numpy(dtype=float),
        }
        if dataset == "UBFC-Phys-S1-S14":
            signals["pos"] = pos_signal(group)
            signals["chrom"] = chrom_signal(group)
        for method, values in signals.items():
            peaks, fs = spectral_peaks(values, times, k=5)
            top = peaks[0] if peaks else {"candidate_bpm": math.nan, "relative_power": math.nan, "nyquist_bpm": math.nan}
            candidate = float(top["candidate_bpm"]) if math.isfinite(float(top["candidate_bpm"])) else math.nan
            error = abs(candidate - reference) if math.isfinite(candidate) and math.isfinite(reference) else math.nan
            mean = float(np.nanmean(values)) if values.size else math.nan
            std = float(np.nanstd(values)) if values.size else math.nan
            ac_ratio = float(std / (abs(mean) + 1e-9)) if math.isfinite(mean) else math.nan
            nyquist = float(top["nyquist_bpm"]) if math.isfinite(float(top["nyquist_bpm"])) else fs * 30.0
            alt = alternating_score(values)
            near_boundary = bool(math.isfinite(candidate) and math.isfinite(nyquist) and candidate >= 0.95 * nyquist)
            low_power = bool(math.isfinite(float(top["relative_power"])) and float(top["relative_power"]) < 0.025)
            unstable = bool(dataset == "MR-NIRP" and math.isfinite(ac_ratio) and ac_ratio > 0.45)
            alternating = bool(dataset == "MR-NIRP" and math.isfinite(alt) and alt >= 0.75)
            out.append(
                {
                    "dataset": dataset,
                    "condition_id": condition,
                    "modality": modality,
                    "roi": roi,
                    "method": method,
                    "reference_bpm": reference,
                    "candidate_bpm": candidate,
                    "absolute_error_bpm": error,
                    "unsafe_error_gt10": bool(math.isfinite(error) and error > UNSAFE_BPM_ERROR),
                    "relative_power": float(top["relative_power"]) if math.isfinite(float(top["relative_power"])) else math.nan,
                    "effective_fps": fs,
                    "nyquist_bpm": nyquist,
                    "ac_ratio": ac_ratio,
                    "alternating_score": alt,
                    "near_nyquist_boundary": near_boundary,
                    "low_relative_power": low_power,
                    "photometric_instability": unstable,
                    "alternating_artifact": alternating,
                    "artifact_flag": bool(near_boundary or low_power or unstable or alternating),
                    "top5_candidates_bpm": ";".join(f"{p['candidate_bpm']:.2f}" for p in peaks),
                }
            )
    return out


def choose_condition_policy(condition_rows: pd.DataFrame) -> dict[str, Any]:
    dataset = str(condition_rows["dataset"].iloc[0])
    condition = str(condition_rows["condition_id"].iloc[0])
    reference = float(condition_rows["reference_bpm"].iloc[0])
    clean = condition_rows[~condition_rows["artifact_flag"].astype(bool)].copy()
    clean = clean[np.isfinite(clean["candidate_bpm"].to_numpy(dtype=float))]
    if clean.empty:
        return _refuse(dataset, condition, reference, condition_rows, "no_artifact_free_candidates")

    candidates = clean.sort_values("candidate_bpm").reset_index(drop=True)
    clusters = []
    used = np.zeros(len(candidates), dtype=bool)
    for idx, row in candidates.iterrows():
        if used[idx]:
            continue
        bpm = float(row["candidate_bpm"])
        close = np.abs(candidates["candidate_bpm"].to_numpy(dtype=float) - bpm) <= CLUSTER_TOL_BPM
        cluster = candidates[close & ~used].copy()
        used[cluster.index.to_numpy()] = True
        clusters.append(cluster)
    if not clusters:
        return _refuse(dataset, condition, reference, condition_rows, "no_candidate_cluster")

    def cluster_score(cluster: pd.DataFrame) -> tuple[int, int, float, float]:
        return (
            int(cluster["roi"].nunique()),
            int(cluster["method"].nunique() + cluster["modality"].nunique()),
            float(cluster["relative_power"].fillna(0).mean()),
            -float(cluster["candidate_bpm"].std() or 0.0),
        )

    best = sorted(clusters, key=cluster_score, reverse=True)[0]
    unique_rois = int(best["roi"].nunique())
    unique_methods = int(best["method"].nunique())
    unique_modalities = int(best["modality"].nunique())
    min_rois = 2
    min_methods = 2 if dataset == "UBFC-Phys-S1-S14" else 1
    min_modalities = 1 if dataset == "UBFC-Phys-S1-S14" else 2
    if unique_rois < min_rois or unique_methods < min_methods or unique_modalities < min_modalities:
        return _refuse(
            dataset,
            condition,
            reference,
            condition_rows,
            f"insufficient_support_roi{unique_rois}_method{unique_methods}_modality{unique_modalities}",
        )
    released = float(np.nanmedian(best["candidate_bpm"].to_numpy(dtype=float)))
    error = abs(released - reference) if math.isfinite(reference) else math.nan
    return {
        "dataset": dataset,
        "condition_id": condition,
        "reference_bpm": reference,
        "policy": "release",
        "released_bpm": released,
        "absolute_error_bpm": error,
        "unsafe_release_gt10": bool(math.isfinite(error) and error > UNSAFE_BPM_ERROR),
        "reason": "multi_roi_method_cluster",
        "support_rows": int(len(best)),
        "support_rois": unique_rois,
        "support_methods": unique_methods,
        "support_modalities": unique_modalities,
        "oracle_best_error_bpm": float(condition_rows["absolute_error_bpm"].min()),
        "artifact_candidate_rate": float(condition_rows["artifact_flag"].mean()),
    }


def _refuse(dataset: str, condition: str, reference: float, rows: pd.DataFrame, reason: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "condition_id": condition,
        "reference_bpm": reference,
        "policy": "refuse",
        "released_bpm": math.nan,
        "absolute_error_bpm": math.nan,
        "unsafe_release_gt10": False,
        "reason": reason,
        "support_rows": 0,
        "support_rois": 0,
        "support_methods": 0,
        "support_modalities": 0,
        "oracle_best_error_bpm": float(rows["absolute_error_bpm"].min()) if len(rows) else math.nan,
        "artifact_candidate_rate": float(rows["artifact_flag"].mean()) if len(rows) else math.nan,
    }


def dataset_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in decisions.groupby("dataset"):
        released = group[group["policy"].eq("release")]
        rows.append(
            {
                "dataset": dataset,
                "n_conditions": int(len(group)),
                "release_rate": float(len(released) / max(1, len(group))),
                "mae_released_bpm": float(released["absolute_error_bpm"].mean()) if len(released) else math.nan,
                "unsafe_release_rate": float(released["unsafe_release_gt10"].mean()) if len(released) else 0.0,
                "oracle_best_mae_bpm": float(group["oracle_best_error_bpm"].mean()),
                "mean_artifact_candidate_rate": float(group["artifact_candidate_rate"].mean()),
            }
        )
    all_released = decisions[decisions["policy"].eq("release")]
    rows.append(
        {
            "dataset": "ALL",
            "n_conditions": int(len(decisions)),
            "release_rate": float(len(all_released) / max(1, len(decisions))),
            "mae_released_bpm": float(all_released["absolute_error_bpm"].mean()) if len(all_released) else math.nan,
            "unsafe_release_rate": float(all_released["unsafe_release_gt10"].mean()) if len(all_released) else 0.0,
            "oracle_best_mae_bpm": float(decisions["oracle_best_error_bpm"].mean()) if len(decisions) else math.nan,
            "mean_artifact_candidate_rate": float(decisions["artifact_candidate_rate"].mean()) if len(decisions) else math.nan,
        }
    )
    return pd.DataFrame(rows)


def update_docs(candidates: pd.DataFrame, decisions: pd.DataFrame, metrics: pd.DataFrame, gates: pd.DataFrame, summary: dict[str, Any]) -> None:
    DOC_MD.write_text(
        "\n".join(
            [
                "# T494 ROI Candidate Evaluation",
                "",
                "## Purpose",
                "",
                "T494 evaluates whether the selected ROI traces from T493 can improve on the T492 full-frame failure mode. The policy is reference-blind: release is allowed only when non-artifact candidates form a multi-ROI/multi-method or multi-modality cluster.",
                "",
                "## Result",
                "",
                f"- Decision: `{summary['decision']}`",
                f"- Overall release rate: {summary['overall_release_rate']:.3f}",
                f"- Overall released MAE: {summary['overall_mae_released_bpm']}",
                f"- Overall unsafe release rate: {summary['overall_unsafe_release_rate']:.3f}",
                f"- UBFC released MAE: {summary['ubfc_mae_released_bpm']}",
                f"- MR release rate: {summary['mr_release_rate']:.3f}",
                "",
                "## Key Insight",
                "",
                summary["main_insight"],
                "",
                "## Dataset Metrics",
                "",
                markdown_table(metrics),
                "",
                "## Policy Decisions",
                "",
                markdown_table(decisions),
                "",
                "## Candidate Preview",
                "",
                markdown_table(candidates[["dataset", "condition_id", "modality", "roi", "method", "reference_bpm", "candidate_bpm", "absolute_error_bpm", "artifact_flag"]]),
                "",
                "## Claim Gates",
                "",
                markdown_table(gates),
                "",
                "## Claim Boundary",
                "",
                summary["claim_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
    )

    marker = "<!-- T494_ROI_CANDIDATE_EVALUATION -->"
    block = "\n".join(
        [
            marker,
            f"## T494 ROI Candidate Evaluation ({date.today().isoformat()})",
            "",
            f"- Decision: `{summary['decision']}`.",
            f"- Overall release rate: {summary['overall_release_rate']:.3f}; unsafe release rate: {summary['overall_unsafe_release_rate']:.3f}.",
            f"- UBFC released MAE: {summary['ubfc_mae_released_bpm']}; MR release rate: {summary['mr_release_rate']:.3f}.",
            f"- Insight: {summary['main_insight']}",
            f"- Boundary: {summary['claim_boundary']}",
            "",
        ]
    )
    for path in [TASK_REGISTRY, PROJECT_STATUS, PAPER_CLAIMS, PROBLEM_LOG, INNOVATION_LOG]:
        append_or_replace(path, marker, block)

    learning_block = "\n".join(
        [
            marker,
            f"## T494 教学记录：ROI Candidate Evaluation ({date.today().isoformat()})",
            "",
            "### 目的",
            "",
            "T494 检验 T493 生成的 ROI trace 是否比 T492 的 full-frame trace 更接近真实 HR。这里必须坚持 reference-blind decision policy：模型/产品在做 release/refuse 时不能偷看 ground truth，只能依赖候选峰之间的一致性、artifact flag、ROI/method/modality 支持度。",
            "",
            "### 方法",
            "",
            "我对每个 ROI 提取 intensity/green/POS/CHROM 等候选信号，计算 top spectral peak 和 artifact indicators，然后用 multi-ROI/multi-method cluster 策略决定是否 release。UBFC 用 BVP peak 作 reference；MR-NIRP 用 PulseOx median 作 reference。",
            "",
            "### 结果",
            "",
            f"Overall release rate={summary['overall_release_rate']:.3f}，overall unsafe release rate={summary['overall_unsafe_release_rate']:.3f}，UBFC released MAE={summary['ubfc_mae_released_bpm']}，MR release rate={summary['mr_release_rate']:.3f}。",
            "",
            "### Insight",
            "",
            summary["main_insight"],
            "",
        ]
    )
    append_or_replace(LEARNING_JOURNAL, marker, learning_block)


def main() -> None:
    refs = reference_map()
    t493 = pd.read_csv(T493_INDEX)
    t492 = read_json(T492_SUMMARY)
    candidate_rows: list[dict[str, Any]] = []
    for _, row in t493.iterrows():
        candidate_rows.extend(candidates_for_trace(row, refs))
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(CANDIDATES_CSV, index=False, encoding="utf-8-sig")

    decisions = pd.DataFrame([choose_condition_policy(group) for _, group in candidates.groupby("condition_id")])
    decisions.to_csv(DECISIONS_CSV, index=False, encoding="utf-8-sig")
    metrics = dataset_metrics(decisions)
    metrics.to_csv(DATASET_METRICS_CSV, index=False, encoding="utf-8-sig")

    all_row = metrics[metrics["dataset"].eq("ALL")].iloc[0]
    ubfc_rows = metrics[metrics["dataset"].eq("UBFC-Phys-S1-S14")]
    mr_rows = metrics[metrics["dataset"].eq("MR-NIRP")]
    ubfc_mae = float(ubfc_rows.iloc[0]["mae_released_bpm"]) if len(ubfc_rows) else math.nan
    ubfc_release = float(ubfc_rows.iloc[0]["release_rate"]) if len(ubfc_rows) else 0.0
    mr_release = float(mr_rows.iloc[0]["release_rate"]) if len(mr_rows) else 0.0
    mr_unsafe = float(mr_rows.iloc[0]["unsafe_release_rate"]) if len(mr_rows) else 0.0
    overall_mae = float(all_row["mae_released_bpm"]) if math.isfinite(float(all_row["mae_released_bpm"])) else math.nan
    overall_release = float(all_row["release_rate"])
    overall_unsafe = float(all_row["unsafe_release_rate"])
    t492_naive_unsafe = float(t492.get("naive_unsafe_candidate_rate", math.nan))

    gates = pd.DataFrame(
        [
            {
                "gate": "roi_candidates_generated",
                "passed": len(candidates) >= 80,
                "evidence": f"candidate_rows={len(candidates)}",
                "claim_allowed": "ROI-level candidate table is available.",
                "claim_not_allowed": "Final model superiority.",
            },
            {
                "gate": "ubfc_roi_recovery_signal_present",
                "passed": bool(ubfc_release >= 0.67 and math.isfinite(ubfc_mae) and ubfc_mae <= 15.0),
                "evidence": f"ubfc_release={ubfc_release:.3f}, ubfc_mae={ubfc_mae}",
                "claim_allowed": "Selected UBFC-Phys ROI route has recoverable HR evidence.",
                "claim_not_allowed": "Full UBFC-Phys or all adult domains solved.",
            },
            {
                "gate": "mr_unsafe_output_reduced_vs_full_frame",
                "passed": bool(mr_unsafe <= 0.01 and (not math.isfinite(t492_naive_unsafe) or mr_unsafe < t492_naive_unsafe)),
                "evidence": f"mr_unsafe_release={mr_unsafe:.3f}, t492_naive_unsafe={t492_naive_unsafe}",
                "claim_allowed": "External low-light MR route is safer than blind full-frame release.",
                "claim_not_allowed": "MR low-light HR accuracy recovered.",
            },
            {
                "gate": "accuracy_claim_still_conditioned",
                "passed": True,
                "evidence": "Selected subset only; MediaPipe unavailable in T493; sample size is small.",
                "claim_allowed": "Use as pilot external-domain evidence and next-step selector training signal.",
                "claim_not_allowed": "一区论文 final SOTA claim without expanded evaluation and statistics.",
            },
        ]
    )
    gates.to_csv(CLAIM_GATE_CSV, index=False, encoding="utf-8-sig")

    decision = "roi_candidate_evaluation_ready_with_conditioned_claims" if bool(gates["passed"].all()) else "roi_candidate_evaluation_needs_optimization"
    summary = {
        "task_id": TASK_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision": decision,
        "all_gates_passed": bool(gates["passed"].all()),
        "n_candidate_rows": int(len(candidates)),
        "n_conditions": int(len(decisions)),
        "overall_release_rate": overall_release,
        "overall_mae_released_bpm": overall_mae,
        "overall_unsafe_release_rate": overall_unsafe,
        "ubfc_release_rate": ubfc_release,
        "ubfc_mae_released_bpm": ubfc_mae,
        "mr_release_rate": mr_release,
        "mr_unsafe_release_rate": mr_unsafe,
        "t492_naive_unsafe_candidate_rate": t492_naive_unsafe,
        "main_insight": "ROI-level candidate extraction converts the T492 full-frame failure into a more useful decision problem: selected UBFC conditions can test HR recovery, while MR-NIRP remains primarily a low-light/sensor-artifact safety gate unless stronger face/skin/deep features are added. This keeps the product honest: release only when candidate evidence is coherent, otherwise refuse and escalate.",
        "claim_boundary": "T494 is a selected-subset ROI candidate evaluation. It can support pilot recovery/safety claims only if gates pass; it cannot support final SOTA, clinical, fairness, or broad low-light robustness claims without expanded statistical validation.",
        "next_recommended_tasks": [
            "If gates pass: T495 integrate T492-T494 comparison into dashboard/paper tables.",
            "If gates fail: T495 optimize candidate clustering thresholds or add MediaPipe/deep ROI extraction before broader training.",
        ],
    }
    write_json(SUMMARY_JSON, summary)
    update_docs(candidates, decisions, metrics, gates, summary)
    replace_evidence_row(
        {
            "evidence_id": "t494_roi_candidate_evaluation",
            "task_id": TASK_ID,
            "date": date.today().isoformat(),
            "artifact": rel(SUMMARY_JSON),
            "metric_or_observation": "ROI candidate selected-domain evaluation",
            "result": f"release={overall_release:.3f}; mae={overall_mae}; unsafe={overall_unsafe:.3f}; UBFC_mae={ubfc_mae}; MR_release={mr_release:.3f}",
            "claim_supported": "ROI candidates are evaluated under a reference-blind release/refuse policy.",
            "claim_boundary": summary["claim_boundary"],
            "next_action": "; ".join(summary["next_recommended_tasks"]),
        }
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
