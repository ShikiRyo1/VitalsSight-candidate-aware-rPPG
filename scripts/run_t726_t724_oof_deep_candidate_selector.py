from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


TASK_ID = "T726"
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_t716_risk_aware_deep_candidate_selector as t716  # noqa: E402


base = t716.base

base.TASK_ID = TASK_ID
base.T618_MCD_DEEP_PREDS = EXP / "t724_mcd_packed_oof_predictions.csv"

base.OUT_POOL = EXP / "t726_t724_oof_deep_candidate_pool.csv"
base.OUT_PREDS = EXP / "t726_t724_oof_selector_predictions.csv"
base.OUT_METRICS = EXP / "t726_t724_oof_selector_metrics.csv"
base.OUT_BASELINES = EXP / "t726_t724_oof_baseline_metrics.csv"
base.OUT_RELEASE = EXP / "t726_t724_oof_release_gate.csv"
base.OUT_CLAIM = EXP / "t726_t724_oof_claim_gate.csv"
base.OUT_FAILURE = EXP / "t726_t724_oof_failure_taxonomy.csv"
base.OUT_SUMMARY = EXP / "t726_t724_oof_selector_gate_summary.json"
base.OUT_MD = DOCS / "t726_t724_oof_deep_candidate_selector.md"


_original_mcd_deep_candidates = base.mcd_deep_candidates


def t724_mcd_deep_candidates(existing_pool: pd.DataFrame) -> pd.DataFrame:
    rows = _original_mcd_deep_candidates(existing_pool)
    if rows.empty:
        return rows
    rows["candidate_id"] = rows["sample_id"].astype(str) + "_deep_T724_MCD_OOF"
    rows["source_name"] = "T724_MCD_OOF"
    rows["candidate_family"] = "fold_safe_deep_backbone"
    rows["candidate_model"] = "t724_packed_memmap_small_video_regressor_oof"
    rows["claim_use"] = "fold_safe_oof_deep_candidate"
    return rows


base.mcd_deep_candidates = t724_mcd_deep_candidates


def write_report(summary, base_metrics, metrics, claim, fail) -> None:
    lines = [
        "# T726 T724 OOF Deep-Candidate Risk-Aware Selector",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Purpose",
        "",
        "T726 repeats the T716 risk-aware candidate-selection experiment, but replaces the diagnostic MCD deep candidate with T724 fold-safe out-of-fold deep predictions. This is the paper-critical test: the selector/gate improvement must survive without using in-fold diagnostic deep predictions.",
        "",
        "## Selector Objective",
        "",
        "`L = KL(p || q_adjusted) + E_p[|HR_i-y|]/20 + 0.55 E_p[sigmoid((|HR_i-y|-10)/2)] + margin + harmonic_penalty`",
        "",
        "The objective keeps the core claim fixed: learn to choose among candidate physiological estimates while penalizing unsafe (>10 BPM) releases and harmonic/alias artifacts.",
        "",
        "## Baselines",
        "",
        base.markdown_table(base_metrics),
        "",
        "## Selector Metrics",
        "",
        base.markdown_table(metrics),
        "",
        "## Claim Gate",
        "",
        base.markdown_table(claim),
        "",
        "## Failure Taxonomy",
        "",
        base.markdown_table(fail),
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Claim Boundary",
        "",
        "T726 can support the final paper claim only if the downstream T727 release/review gate preserves adequate coverage while keeping unsafe release <=10%, and if at least MCD plus one external dataset pass the locked gate.",
    ]
    base.OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


base.write_report = write_report


def main() -> int:
    if not base.T618_MCD_DEEP_PREDS.exists():
        raise FileNotFoundError(f"Missing T724 OOF deep predictions: {base.T618_MCD_DEEP_PREDS}")
    preds = pd.read_csv(base.T618_MCD_DEEP_PREDS)
    if preds.empty:
        raise ValueError("T724 OOF deep predictions are empty.")
    required = {"clip_id", "hr_true", "hr_pred", "fold"}
    missing = sorted(required - set(preds.columns))
    if missing:
        raise ValueError(f"T724 predictions are missing required columns: {missing}")
    rc = base.main()
    if base.OUT_SUMMARY.exists():
        summary = json.loads(base.OUT_SUMMARY.read_text(encoding="utf-8"))
        summary["t724_prediction_rows"] = int(len(preds))
        summary["t724_prediction_folds"] = sorted(int(x) for x in pd.to_numeric(preds["fold"], errors="coerce").dropna().unique())
        summary["claim_boundary"] = (
            "T726 uses T724 fold-safe MCD OOF deep predictions as candidate inputs. "
            "It remains a selector-stage result until T727 calibrated gate and statistical tests are run."
        )
        base.OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(base.OUT_SUMMARY.read_text(encoding="utf-8"), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
