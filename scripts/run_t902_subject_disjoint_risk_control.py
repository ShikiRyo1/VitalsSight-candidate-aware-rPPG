from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_ID = "T902"
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"

IN_T901 = EXP / "t901_subject_cluster_input_audit.csv"

OUT_SPLITS = EXP / "t902_subject_disjoint_splits.csv"
OUT_THRESHOLDS = EXP / "t902_risk_control_thresholds.csv"
OUT_TEST = EXP / "t902_risk_control_test_metrics.csv"
OUT_SUMMARY = EXP / "t902_risk_control_summary.csv"
OUT_JSON = EXP / "t902_risk_control_summary.json"
OUT_MD = DOCS / "t902_subject_disjoint_risk_control.md"

SEEDS = [2024, 2025, 2026]
UNSAFE_THRESHOLDS = [5.0, 8.0, 10.0, 15.0]
ALPHA = 0.10
DELTA = 0.05


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def upper_confidence_bound(k: int, n: int, delta: float = DELTA) -> float:
    if n <= 0:
        return 1.0
    try:
        from scipy.stats import beta  # type: ignore

        if k >= n:
            return 1.0
        return float(beta.ppf(1.0 - delta, k + 1, n - k))
    except Exception:
        phat = k / n
        return float(min(1.0, phat + math.sqrt(math.log(1.0 / delta) / (2.0 * n))))


def split_subjects(subjects: list[str], seed: int) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    n_cal = max(1, int(round(len(shuffled) * 0.5)))
    if len(shuffled) > 1:
        n_cal = min(n_cal, len(shuffled) - 1)
    cal = set(shuffled[:n_cal])
    test = set(shuffled[n_cal:])
    return cal, test


def evaluate_frame(frame: pd.DataFrame, tau: float, unsafe_bpm: float) -> dict[str, Any]:
    selected = frame[num(frame, "release_risk") <= tau].copy()
    err = num(selected, "selected_abs_error_bpm")
    n_total = int(len(frame))
    n_release = int(len(selected))
    k_unsafe = int((err > unsafe_bpm).sum()) if n_release else 0
    unsafe = float(k_unsafe / n_release) if n_release else math.nan
    subject_rows: list[dict[str, float]] = []
    for _, g in frame.groupby("subject_std", sort=False):
        rel = g[num(g, "release_risk") <= tau].copy()
        rel_err = num(rel, "selected_abs_error_bpm")
        subject_rows.append(
            {
                "coverage": len(rel) / len(g) if len(g) else math.nan,
                "released_mae": float(rel_err.mean()) if len(rel) else math.nan,
                "unsafe": float((rel_err > unsafe_bpm).mean()) if len(rel) else math.nan,
            }
        )
    subj = pd.DataFrame(subject_rows)
    subj_released = subj[subj["coverage"] > 0]
    return {
        "n_total": n_total,
        "n_released": n_release,
        "coverage": float(n_release / n_total) if n_total else math.nan,
        "released_mae_bpm": float(err.mean()) if n_release else math.nan,
        "unsafe_release_rate": unsafe,
        "unsafe_count": k_unsafe,
        "unsafe_upper95": upper_confidence_bound(k_unsafe, n_release, DELTA) if n_release else 1.0,
        "subject_mean_coverage": float(subj["coverage"].mean()) if len(subj) else math.nan,
        "subject_mean_released_mae_bpm": float(subj_released["released_mae"].mean()) if len(subj_released) else math.nan,
        "subject_mean_unsafe_release_rate": float(subj_released["unsafe"].mean()) if len(subj_released) else math.nan,
        "n_release_subjects": int((subj["coverage"] > 0).sum()) if len(subj) else 0,
    }


def select_threshold(cal: pd.DataFrame, unsafe_bpm: float, alpha: float = ALPHA) -> dict[str, Any]:
    risks = np.unique(num(cal, "release_risk").dropna().to_numpy(float))
    if len(risks) == 0:
        return {
            "risk_threshold": math.nan,
            "calibration_pass": False,
            "reason": "no finite risk scores",
        }
    candidates: list[dict[str, Any]] = []
    for tau in np.sort(risks):
        m = evaluate_frame(cal, float(tau), unsafe_bpm)
        if m["n_released"] <= 0:
            continue
        row = {"risk_threshold": float(tau)}
        row.update({f"calibration_{k}": v for k, v in m.items()})
        row["calibration_pass"] = bool(m["unsafe_upper95"] <= alpha)
        candidates.append(row)
    passing = [r for r in candidates if r["calibration_pass"]]
    if not passing:
        if candidates:
            best = min(candidates, key=lambda r: (r["calibration_unsafe_upper95"], -r["calibration_coverage"]))
            best["calibration_pass"] = False
            best["reason"] = "no threshold meets the one-sided unsafe-release upper bound"
            return best
        return {
            "risk_threshold": math.nan,
            "calibration_pass": False,
            "reason": "no release candidates",
        }
    best = sorted(passing, key=lambda r: (r["calibration_coverage"], -r["calibration_released_mae_bpm"]), reverse=True)[0]
    best["reason"] = "largest calibration coverage with unsafe upper bound within alpha"
    return best


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    show = df[cols].copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    lines = [
        "| " + " | ".join(show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in show.columns) + " |")
    return "\n".join(lines)


def summarize(test_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, unsafe_bpm), g in test_metrics.groupby(["dataset", "unsafe_bpm"], sort=False):
        cal_pass_rate = float(g["calibration_pass"].mean())
        empirical_pass_rate = float(g["test_empirical_pass"].mean())
        upper_pass_rate = float(g["test_upper_bound_pass"].mean())
        rows.append(
            {
                "task_id": TASK_ID,
                "dataset": dataset,
                "unsafe_bpm": float(unsafe_bpm),
                "n_splits": int(len(g)),
                "calibration_pass_rate": cal_pass_rate,
                "test_empirical_pass_rate": empirical_pass_rate,
                "test_upper_bound_pass_rate": upper_pass_rate,
                "mean_test_coverage": float(num(g, "test_coverage").mean()),
                "mean_test_released_mae_bpm": float(num(g, "test_released_mae_bpm").mean()),
                "mean_test_unsafe_release_rate": float(num(g, "test_unsafe_release_rate").mean()),
                "mean_subject_test_coverage": float(num(g, "test_subject_mean_coverage").mean()),
                "mean_subject_test_unsafe_release_rate": float(num(g, "test_subject_mean_unsafe_release_rate").mean()),
            }
        )
    out = pd.DataFrame(rows)
    uses: list[str] = []
    reasons: list[str] = []
    for _, r in out.iterrows():
        if float(r["unsafe_bpm"]) != 10.0:
            uses.append("threshold_sensitivity")
            reasons.append("non-primary unsafe threshold; use as sensitivity evidence")
        elif r["calibration_pass_rate"] == 1.0 and r["test_empirical_pass_rate"] == 1.0 and r["test_upper_bound_pass_rate"] == 1.0:
            uses.append("main_text_risk_control")
            reasons.append("all subject-disjoint splits pass calibration and held-out upper-bound checks")
        elif r["calibration_pass_rate"] > 0.0 and r["test_empirical_pass_rate"] > 0.0:
            uses.append("appendix_calibration_evidence")
            reasons.append("calibrated thresholds exist, but held-out support is not strong enough for a guarantee claim")
        else:
            uses.append("boundary_only")
            reasons.append("risk-control threshold is not stable enough under subject-disjoint validation")
    out["manuscript_use"] = uses
    out["reason"] = reasons
    return out


def write_report(generated_at: str, summary: pd.DataFrame) -> None:
    main = summary[summary["unsafe_bpm"].eq(10.0)].copy()
    lines = [
        "# T902 Subject-Disjoint Risk-Control Calibration",
        "",
        f"Generated: `{generated_at}`",
        "",
        "## Purpose",
        "",
        "T902 tests whether the current release/review gate can be described as a split-calibrated selective-release rule. Subjects are separated into calibration and test folds; the release-risk threshold is selected using only calibration subjects, then evaluated on held-out subjects.",
        "",
        "## Main 10 BPM Risk-Control Summary",
        "",
        md_table(
            main,
            [
                "dataset",
                "unsafe_bpm",
                "calibration_pass_rate",
                "test_empirical_pass_rate",
                "test_upper_bound_pass_rate",
                "mean_test_coverage",
                "mean_test_released_mae_bpm",
                "mean_test_unsafe_release_rate",
                "manuscript_use",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "This experiment is stricter than the previous heuristic gate. A dataset can show useful selective-release behavior even when it does not support a formal guarantee. In that case, the manuscript should use risk-aware or split-calibrated language only in a bounded way and move the detailed threshold table to the Appendix.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    generated_at = now()
    df = read_csv(IN_T901)
    required = ["dataset", "sample_id", "subject_std", "selected_abs_error_bpm", "release_risk"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    df = df.copy()
    df["subject_std"] = df["subject_std"].astype(str)
    df["release_risk"] = num(df, "release_risk")
    df["selected_abs_error_bpm"] = num(df, "selected_abs_error_bpm")

    split_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for dataset, dsg in df.groupby("dataset", sort=False):
        subjects = sorted(dsg["subject_std"].astype(str).dropna().unique().tolist())
        if len(subjects) < 2:
            continue
        for seed in SEEDS:
            cal_subj, test_subj = split_subjects(subjects, seed)
            split_rows.append(
                {
                    "task_id": TASK_ID,
                    "dataset": dataset,
                    "seed": seed,
                    "n_calibration_subjects": len(cal_subj),
                    "n_test_subjects": len(test_subj),
                    "calibration_subjects": ";".join(sorted(cal_subj)),
                    "test_subjects": ";".join(sorted(test_subj)),
                }
            )
            cal = dsg[dsg["subject_std"].isin(cal_subj)].copy()
            test = dsg[dsg["subject_std"].isin(test_subj)].copy()
            for unsafe_bpm in UNSAFE_THRESHOLDS:
                selected = select_threshold(cal, unsafe_bpm, ALPHA)
                thr_row = {
                    "task_id": TASK_ID,
                    "dataset": dataset,
                    "seed": seed,
                    "unsafe_bpm": unsafe_bpm,
                }
                thr_row.update(selected)
                threshold_rows.append(thr_row)
                tau = float(selected.get("risk_threshold", math.nan))
                if not np.isfinite(tau):
                    test_eval = {
                        "n_total": int(len(test)),
                        "n_released": 0,
                        "coverage": 0.0,
                        "released_mae_bpm": math.nan,
                        "unsafe_release_rate": math.nan,
                        "unsafe_count": 0,
                        "unsafe_upper95": 1.0,
                        "subject_mean_coverage": 0.0,
                        "subject_mean_released_mae_bpm": math.nan,
                        "subject_mean_unsafe_release_rate": math.nan,
                        "n_release_subjects": 0,
                    }
                else:
                    test_eval = evaluate_frame(test, tau, unsafe_bpm)
                row = {
                    "task_id": TASK_ID,
                    "dataset": dataset,
                    "seed": seed,
                    "unsafe_bpm": unsafe_bpm,
                    "risk_threshold": tau,
                    "calibration_pass": bool(selected.get("calibration_pass", False)),
                    "calibration_reason": selected.get("reason", ""),
                }
                row.update({f"test_{k}": v for k, v in test_eval.items()})
                row["test_empirical_pass"] = bool(
                    row["test_n_released"] > 0
                    and np.isfinite(row["test_unsafe_release_rate"])
                    and row["test_unsafe_release_rate"] <= ALPHA
                )
                row["test_upper_bound_pass"] = bool(
                    row["test_n_released"] > 0
                    and np.isfinite(row["test_unsafe_upper95"])
                    and row["test_unsafe_upper95"] <= ALPHA
                )
                test_rows.append(row)

    splits = pd.DataFrame(split_rows)
    thresholds = pd.DataFrame(threshold_rows)
    tests = pd.DataFrame(test_rows)
    summary = summarize(tests)

    splits.to_csv(OUT_SPLITS, index=False, encoding="utf-8-sig")
    thresholds.to_csv(OUT_THRESHOLDS, index=False, encoding="utf-8-sig")
    tests.to_csv(OUT_TEST, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    payload = {
        "task_id": TASK_ID,
        "generated_at": generated_at,
        "alpha": ALPHA,
        "delta": DELTA,
        "seeds": SEEDS,
        "unsafe_thresholds": UNSAFE_THRESHOLDS,
        "outputs": {
            "splits": str(OUT_SPLITS.relative_to(ROOT)),
            "thresholds": str(OUT_THRESHOLDS.relative_to(ROOT)),
            "test_metrics": str(OUT_TEST.relative_to(ROOT)),
            "summary": str(OUT_SUMMARY.relative_to(ROOT)),
            "doc": str(OUT_MD.relative_to(ROOT)),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(generated_at, summary)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
