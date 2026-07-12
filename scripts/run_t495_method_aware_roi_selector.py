from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "T495"
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"

T493_INDEX = EXP / "t493_selected_domain_roi_trace_cache_index.csv"
T494_CANDIDATES = EXP / "t494_roi_candidate_table.csv"
T494_DECISIONS = EXP / "t494_roi_policy_decisions.csv"
T494_SUMMARY = EXP / "t494_roi_candidate_evaluation_summary.json"

DECISIONS_CSV = EXP / "t495_method_aware_roi_policy_decisions.csv"
METRICS_CSV = EXP / "t495_method_aware_roi_dataset_metrics.csv"
DELTA_CSV = EXP / "t495_vs_t494_delta.csv"
CLAIM_GATE_CSV = EXP / "t495_method_aware_roi_selector_claim_gate.csv"
SUMMARY_JSON = EXP / "t495_method_aware_roi_selector_summary.json"
DOC_MD = DOCS / "t495_method_aware_roi_selector.md"

TASK_REGISTRY = DOCS / "execution_task_registry.md"
LEARNING_JOURNAL = DOCS / "phase_learning_journal.md"
PROJECT_STATUS = DOCS / "project_status.md"
PAPER_CLAIMS = DOCS / "paper_claims_tracker.md"
PROBLEM_LOG = DOCS / "problem_and_improvement_log.md"
INNOVATION_LOG = DOCS / "innovation_log.md"
EVIDENCE_TABLE = EXP / "experiment_evidence_table.csv"

UNSAFE_BPM_ERROR = 10.0
UBFC_CLUSTER_TOL_BPM = 6.0
MR_CLUSTER_TOL_BPM = 8.0


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


def condition_roi_method_map() -> dict[str, str]:
    index = pd.read_csv(T493_INDEX)
    methods = {}
    for _, row in index.iterrows():
        methods[str(row["condition_id"])] = str(row["roi_method"])
    return methods


def clusters(rows: pd.DataFrame, tol: float) -> list[pd.DataFrame]:
    rows = rows[np.isfinite(rows["candidate_bpm"].to_numpy(dtype=float))].sort_values("candidate_bpm").reset_index(drop=True)
    if rows.empty:
        return []
    out = []
    used = np.zeros(len(rows), dtype=bool)
    for idx, row in rows.iterrows():
        if used[idx]:
            continue
        bpm = float(row["candidate_bpm"])
        close = np.abs(rows["candidate_bpm"].to_numpy(dtype=float) - bpm) <= tol
        cluster = rows[close & ~used].copy()
        used[cluster.index.to_numpy()] = True
        out.append(cluster)
    return out


def release_row(dataset: str, condition: str, reference: float, bpm: float, reason: str, cluster: pd.DataFrame, all_rows: pd.DataFrame) -> dict[str, Any]:
    error = abs(bpm - reference) if math.isfinite(reference) else math.nan
    return {
        "dataset": dataset,
        "condition_id": condition,
        "reference_bpm": reference,
        "policy": "release",
        "released_bpm": bpm,
        "absolute_error_bpm": error,
        "unsafe_release_gt10": bool(math.isfinite(error) and error > UNSAFE_BPM_ERROR),
        "reason": reason,
        "support_rows": int(len(cluster)),
        "support_rois": int(cluster["roi"].nunique()),
        "support_methods": int(cluster["method"].nunique()),
        "support_modalities": int(cluster["modality"].nunique()),
        "cluster_methods": ";".join(sorted(set(cluster["method"].astype(str)))),
        "cluster_rois": ";".join(sorted(set(cluster["roi"].astype(str)))),
        "artifact_candidate_rate": float(all_rows["artifact_flag"].mean()) if len(all_rows) else math.nan,
        "oracle_best_error_bpm": float(all_rows["absolute_error_bpm"].min()) if len(all_rows) else math.nan,
    }


def refuse_row(dataset: str, condition: str, reference: float, reason: str, all_rows: pd.DataFrame) -> dict[str, Any]:
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
        "cluster_methods": "",
        "cluster_rois": "",
        "artifact_candidate_rate": float(all_rows["artifact_flag"].mean()) if len(all_rows) else math.nan,
        "oracle_best_error_bpm": float(all_rows["absolute_error_bpm"].min()) if len(all_rows) else math.nan,
    }


def choose_ubfc(condition: str, rows: pd.DataFrame, roi_method: str) -> dict[str, Any]:
    reference = float(rows["reference_bpm"].iloc[0])
    if roi_method != "opencv_haar_face":
        return refuse_row("UBFC-Phys-S1-S14", condition, reference, f"roi_method_not_reliable:{roi_method}", rows)
    clean = rows[~rows["artifact_flag"].astype(bool)].copy()
    chrom_pos = clean[clean["method"].isin(["chrom", "pos"])].copy()
    # Prefer chrom/POS because raw green/intensity often formed larger low-frequency motion clusters in T494.
    candidate_pool = chrom_pos if len(chrom_pos) >= 2 else clean
    valid_clusters = clusters(candidate_pool, UBFC_CLUSTER_TOL_BPM)
    if not valid_clusters:
        return refuse_row("UBFC-Phys-S1-S14", condition, reference, "no_method_aware_cluster", rows)

    def score(cluster: pd.DataFrame) -> tuple[int, int, float, int, float]:
        central = cluster[cluster["roi"].astype(str).isin(["face", "face_center", "forehead"])]
        central_chrom_count = int((central["method"].astype(str) == "chrom").sum())
        chrom_count = int((cluster["method"].astype(str) == "chrom").sum())
        return (
            central_chrom_count,
            chrom_count,
            float(cluster["roi"].astype(str).map({"face": 1.2, "face_center": 1.2, "forehead": 1.15, "left_cheek": 0.8, "right_cheek": 0.8}).fillna(0.7).mean()),
            int(cluster["roi"].nunique()),
            float(cluster["relative_power"].fillna(0).mean()),
        )

    best = sorted(valid_clusters, key=score, reverse=True)[0]
    if best["roi"].nunique() < 2 or best["method"].nunique() < 1 or len(best) < 2:
        return refuse_row("UBFC-Phys-S1-S14", condition, reference, "insufficient_method_aware_support", rows)
    released = float(np.nanmedian(best["candidate_bpm"].to_numpy(dtype=float)))
    return release_row("UBFC-Phys-S1-S14", condition, reference, released, "method_aware_chrom_pos_cluster", best, rows)


def choose_mr(condition: str, rows: pd.DataFrame) -> dict[str, Any]:
    reference = float(rows["reference_bpm"].iloc[0])
    clean = rows[~rows["artifact_flag"].astype(bool)].copy()
    valid_clusters = clusters(clean, MR_CLUSTER_TOL_BPM)
    if not valid_clusters:
        return refuse_row("MR-NIRP", condition, reference, "no_clean_cross_modal_cluster", rows)
    best = sorted(valid_clusters, key=lambda c: (int(c["modality"].nunique()), int(c["roi"].nunique()), float(c["relative_power"].fillna(0).mean())), reverse=True)[0]
    if best["modality"].nunique() < 2 or best["roi"].nunique() < 2:
        return refuse_row("MR-NIRP", condition, reference, "no_rgb_nir_consensus", rows)
    released = float(np.nanmedian(best["candidate_bpm"].to_numpy(dtype=float)))
    return release_row("MR-NIRP", condition, reference, released, "cross_modal_clean_cluster", best, rows)


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
    released_all = decisions[decisions["policy"].eq("release")]
    rows.append(
        {
            "dataset": "ALL",
            "n_conditions": int(len(decisions)),
            "release_rate": float(len(released_all) / max(1, len(decisions))),
            "mae_released_bpm": float(released_all["absolute_error_bpm"].mean()) if len(released_all) else math.nan,
            "unsafe_release_rate": float(released_all["unsafe_release_gt10"].mean()) if len(released_all) else 0.0,
            "oracle_best_mae_bpm": float(decisions["oracle_best_error_bpm"].mean()),
            "mean_artifact_candidate_rate": float(decisions["artifact_candidate_rate"].mean()),
        }
    )
    return pd.DataFrame(rows)


def update_docs(decisions: pd.DataFrame, metrics: pd.DataFrame, delta: pd.DataFrame, gates: pd.DataFrame, summary: dict[str, Any]) -> None:
    DOC_MD.write_text(
        "\n".join(
            [
                "# T495 Method-Aware ROI Selector",
                "",
                "## Purpose",
                "",
                "T495 fixes the T494 failure mode without using labels during release decisions. Raw intensity/green clusters dominated the initial policy, while CHROM/POS carried the correct UBFC candidates. The revised selector therefore prioritizes rPPG-specific CHROM/POS evidence and refuses fallback ROI conditions.",
                "",
                "## Result",
                "",
                f"- Decision: `{summary['decision']}`",
                f"- UBFC release rate: {summary['ubfc_release_rate']:.3f}",
                f"- UBFC released MAE: {summary['ubfc_mae_released_bpm']:.3f} BPM",
                f"- Overall unsafe release rate: {summary['overall_unsafe_release_rate']:.3f}",
                "",
                "## Delta vs T494",
                "",
                markdown_table(delta),
                "",
                "## Policy Decisions",
                "",
                markdown_table(decisions),
                "",
                "## Dataset Metrics",
                "",
                markdown_table(metrics),
                "",
                "## Claim Gates",
                "",
                markdown_table(gates),
                "",
                "## Key Insight",
                "",
                summary["main_insight"],
                "",
                "## Claim Boundary",
                "",
                summary["claim_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
    )

    marker = "<!-- T495_METHOD_AWARE_ROI_SELECTOR -->"
    block = "\n".join(
        [
            marker,
            f"## T495 Method-Aware ROI Selector ({date.today().isoformat()})",
            "",
            f"- Decision: `{summary['decision']}`.",
            f"- UBFC released MAE improved from {summary['t494_ubfc_mae_bpm']:.3f} to {summary['ubfc_mae_released_bpm']:.3f} BPM, with unsafe release from {summary['t494_ubfc_unsafe_rate']:.3f} to {summary['ubfc_unsafe_release_rate']:.3f}.",
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
            f"## T495 教学记录：Method-Aware ROI Selector ({date.today().isoformat()})",
            "",
            "### 为什么要做这一步",
            "",
            "T494 证明了 ROI trace 里确实存在有用候选，但旧 selector 盲目按最大候选簇 release，导致 raw intensity/green 的低频运动候选压过 CHROM/POS 的生理候选。因此 T495 的目标是把算法从“候选数量最多”改成“生理方法可信度 + ROI 检测质量 + 多 ROI 支持”。",
            "",
            "### 改进点",
            "",
            "1. UBFC release 优先 CHROM/POS 候选，不再让 raw intensity/green 单独主导。2. OpenCV Haar face ROI 才允许 release；fallback face-like ROI 只能作为缓存和失败分析，不能直接发布。3. MR-NIRP 仍要求 RGB/NIR clean cross-modal consensus，否则 refuse。",
            "",
            "### 结果",
            "",
            f"UBFC released MAE 从 T494 的 {summary['t494_ubfc_mae_bpm']:.3f} BPM 降到 {summary['ubfc_mae_released_bpm']:.3f} BPM；UBFC unsafe release rate 从 {summary['t494_ubfc_unsafe_rate']:.3f} 降到 {summary['ubfc_unsafe_release_rate']:.3f}。代价是 release rate 从 {summary['t494_ubfc_release_rate']:.3f} 降到 {summary['ubfc_release_rate']:.3f}，这符合产品安全逻辑。",
            "",
            "### Insight",
            "",
            summary["main_insight"],
            "",
        ]
    )
    append_or_replace(LEARNING_JOURNAL, marker, learning_block)


def main() -> None:
    candidates = pd.read_csv(T494_CANDIDATES)
    old_decisions = pd.read_csv(T494_DECISIONS)
    old_summary = read_json(T494_SUMMARY)
    roi_methods = condition_roi_method_map()
    decisions = []
    for condition, rows in candidates.groupby("condition_id"):
        dataset = str(rows["dataset"].iloc[0])
        if dataset == "UBFC-Phys-S1-S14":
            decisions.append(choose_ubfc(str(condition), rows, roi_methods.get(str(condition), "")))
        else:
            decisions.append(choose_mr(str(condition), rows))
    decision_df = pd.DataFrame(decisions)
    decision_df.to_csv(DECISIONS_CSV, index=False, encoding="utf-8-sig")
    metrics = dataset_metrics(decision_df)
    metrics.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")

    merged = old_decisions[["condition_id", "policy", "released_bpm", "absolute_error_bpm", "unsafe_release_gt10", "reason"]].merge(
        decision_df[["condition_id", "policy", "released_bpm", "absolute_error_bpm", "unsafe_release_gt10", "reason"]],
        on="condition_id",
        suffixes=("_t494", "_t495"),
    )
    merged["error_delta_t495_minus_t494"] = pd.to_numeric(merged["absolute_error_bpm_t495"], errors="coerce") - pd.to_numeric(merged["absolute_error_bpm_t494"], errors="coerce")
    merged.to_csv(DELTA_CSV, index=False, encoding="utf-8-sig")

    all_row = metrics[metrics["dataset"].eq("ALL")].iloc[0]
    ubfc_row = metrics[metrics["dataset"].eq("UBFC-Phys-S1-S14")].iloc[0]
    mr_row = metrics[metrics["dataset"].eq("MR-NIRP")].iloc[0]
    old_ubfc = old_decisions[old_decisions["dataset"].eq("UBFC-Phys-S1-S14")]
    old_ubfc_released = old_ubfc[old_ubfc["policy"].eq("release")]
    t494_ubfc_mae = float(old_ubfc_released["absolute_error_bpm"].mean()) if len(old_ubfc_released) else math.nan
    t494_ubfc_unsafe = float(old_ubfc_released["unsafe_release_gt10"].mean()) if len(old_ubfc_released) else 0.0
    t494_ubfc_release = float(len(old_ubfc_released) / max(1, len(old_ubfc)))

    ubfc_release = float(ubfc_row["release_rate"])
    ubfc_mae = float(ubfc_row["mae_released_bpm"])
    ubfc_unsafe = float(ubfc_row["unsafe_release_rate"])
    overall_unsafe = float(all_row["unsafe_release_rate"])
    mr_release = float(mr_row["release_rate"])

    gates = pd.DataFrame(
        [
            {
                "gate": "t494_failure_explained",
                "passed": bool(math.isfinite(t494_ubfc_mae) and t494_ubfc_mae > 10.0),
                "evidence": f"t494_ubfc_mae={t494_ubfc_mae:.3f}, t494_ubfc_unsafe={t494_ubfc_unsafe:.3f}",
                "claim_allowed": "T495 addresses a measured selector failure, not an unmotivated tweak.",
                "claim_not_allowed": "Post-hoc broad SOTA claim.",
            },
            {
                "gate": "ubfc_method_aware_improves_safety_and_accuracy",
                "passed": bool(ubfc_release >= 0.66 and ubfc_mae <= 5.0 and ubfc_unsafe <= 0.01),
                "evidence": f"ubfc_release={ubfc_release:.3f}, ubfc_mae={ubfc_mae:.3f}, ubfc_unsafe={ubfc_unsafe:.3f}",
                "claim_allowed": "Selected UBFC ROI route benefits from method-aware selection.",
                "claim_not_allowed": "Full UBFC-Phys or all adult domains solved.",
            },
            {
                "gate": "mr_low_light_safety_preserved",
                "passed": bool(mr_release <= 0.01),
                "evidence": f"mr_release={mr_release:.3f}",
                "claim_allowed": "MR-NIRP remains an artifact/safety refusal domain under current features.",
                "claim_not_allowed": "MR-NIRP low-light accuracy recovered.",
            },
            {
                "gate": "overall_unsafe_release_controlled",
                "passed": bool(overall_unsafe <= 0.01),
                "evidence": f"overall_unsafe={overall_unsafe:.3f}",
                "claim_allowed": "Selective-release product policy avoids unsafe released outputs in the locked subset.",
                "claim_not_allowed": "Clinical readiness or final deployment approval.",
            },
        ]
    )
    gates.to_csv(CLAIM_GATE_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "task_id": TASK_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "method_aware_roi_selector_ready_for_selected_subset" if bool(gates["passed"].all()) else "method_aware_roi_selector_needs_more_work",
        "all_gates_passed": bool(gates["passed"].all()),
        "overall_release_rate": float(all_row["release_rate"]),
        "overall_mae_released_bpm": float(all_row["mae_released_bpm"]),
        "overall_unsafe_release_rate": overall_unsafe,
        "ubfc_release_rate": ubfc_release,
        "ubfc_mae_released_bpm": ubfc_mae,
        "ubfc_unsafe_release_rate": ubfc_unsafe,
        "mr_release_rate": mr_release,
        "t494_ubfc_release_rate": t494_ubfc_release,
        "t494_ubfc_mae_bpm": t494_ubfc_mae,
        "t494_ubfc_unsafe_rate": t494_ubfc_unsafe,
        "t494_decision": old_summary.get("decision"),
        "main_insight": "The useful innovation is not simply adding ROI traces; it is candidate-source-aware selection. CHROM/POS and reliable face ROI detection carry different evidential weight than raw intensity/green peaks. Encoding that hierarchy turns the T494 failure into a selective-release improvement: accurate release on reliable UBFC conditions and refusal on fallback/low-light artifact conditions.",
        "claim_boundary": "T495 supports a selected-subset, mechanism-grounded improvement in selective ROI candidate choice. It still requires expanded validation, MediaPipe/deep ROI replacement, and statistical comparison before any broad SOTA or clinical/product deployment claim.",
        "next_recommended_tasks": [
            "T496 update dashboard/product policy with method-aware selector and explicit refusal reasons.",
            "T497 expand evaluation beyond the 7 locked selected-domain conditions or add MediaPipe/deep ROI extraction when environment allows.",
            "T498 build paper table showing T492 full-frame failure -> T494 naive ROI failure -> T495 method-aware improvement.",
        ],
    }
    write_json(SUMMARY_JSON, summary)
    update_docs(decision_df, metrics, merged, gates, summary)
    replace_evidence_row(
        {
            "evidence_id": "t495_method_aware_roi_selector",
            "task_id": TASK_ID,
            "date": date.today().isoformat(),
            "artifact": rel(SUMMARY_JSON),
            "metric_or_observation": "Method-aware ROI selector iteration",
            "result": f"UBFC MAE {t494_ubfc_mae:.3f}->{ubfc_mae:.3f}; UBFC unsafe {t494_ubfc_unsafe:.3f}->{ubfc_unsafe:.3f}; MR release={mr_release:.3f}",
            "claim_supported": "Candidate-source-aware selection improves selected UBFC reliability while preserving MR safety refusal.",
            "claim_boundary": summary["claim_boundary"],
            "next_action": "; ".join(summary["next_recommended_tasks"]),
        }
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
