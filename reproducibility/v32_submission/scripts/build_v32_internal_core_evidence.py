#!/usr/bin/env python3
"""Build V32 internal source data, publication table and Figure 2 in Python."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


METHOD_ORDER = [
    "TSCAN",
    "matched_ridge_stacker",
    "matched_extra_trees_stacker",
    "VitalsSight_candidate_aware",
    "VitalsSight_v32_independent_emission",
    "VitalsSight_v32_causal_candidate_path",
]
LABELS = {
    "TSCAN": "TS-CAN",
    "matched_ridge_stacker": "Matched ridge",
    "matched_extra_trees_stacker": "Matched ExtraTrees",
    "VitalsSight_candidate_aware": "Prior candidate-aware",
    "VitalsSight_v32_independent_emission": "V32 independent emission",
    "VitalsSight_v32_causal_candidate_path": "VitalsSight V32 (ours)",
}
COLORS = {
    "TSCAN": "#A9B2BA",
    "matched_ridge_stacker": "#909BA5",
    "matched_extra_trees_stacker": "#71808D",
    "VitalsSight_candidate_aware": "#7396B8",
    "VitalsSight_v32_independent_emission": "#8DB5AA",
    "VitalsSight_v32_causal_candidate_path": "#4F8F85",
}
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 720719


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def participant_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, subject), group in predictions.groupby(["method", "subject_std"], sort=True):
        error = pd.to_numeric(group["abs_error_bpm"], errors="raise").to_numpy(float)
        rows.append(
            {
                "method": str(method),
                "subject_std": str(subject),
                "windows": int(len(group)),
                "mae_bpm": float(np.mean(error)),
                "rmse_bpm": float(np.sqrt(np.mean(np.square(error)))),
                "within5": float(np.mean(error <= 5.0)),
                "within10": float(np.mean(error <= 10.0)),
                "unsafe_gt10": float(np.mean(error > 10.0)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate_table(participants: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, method in enumerate(METHOD_ORDER):
        frame = participants[participants["method"].eq(method)]
        require(len(frame) == 42, f"participant count mismatch: {method}")
        mae_values = frame["mae_bpm"].to_numpy(float)
        low, high = bootstrap_mean_interval(mae_values, BOOTSTRAP_SEED + index)
        rows.append(
            {
                "Method": LABELS[method],
                "Protocol role": (
                    "single learned route" if method == "TSCAN" else
                    "matched direct stacker" if "stacker" in method else
                    "source-preserving comparator" if method == "VitalsSight_candidate_aware" else
                    "candidate-path ablation" if method == "VitalsSight_v32_independent_emission" else
                    "proposed joint candidate path"
                ),
                "Participants": 42,
                "Windows": 439,
                "Participant-equal MAE (BPM)": float(frame["mae_bpm"].mean()),
                "MAE 95% CI low": low,
                "MAE 95% CI high": high,
                "Participant-equal RMSE (BPM)": float(frame["rmse_bpm"].mean()),
                "Within 5 BPM": float(frame["within5"].mean()),
                "Within 10 BPM": float(frame["within10"].mean()),
                "Error >10 BPM": float(frame["unsafe_gt10"].mean()),
            }
        )
    return pd.DataFrame(rows)


def effect_table(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for label, key, p_role in (
        ("Prior candidate-aware", "causal_path_minus_primary_comparator", "primary repaired-selector contrast"),
        ("Independent emission", "causal_path_minus_independent", "exploratory temporal increment"),
    ):
        value = summary[key]
        rows.append(
            {
                "Comparator": label,
                "Mean delta (V32 minus comparator), BPM": float(value["mean_delta_a_minus_b_bpm"]),
                "CI95 low": float(value["ci95_low"]),
                "CI95 high": float(value["ci95_high"]),
                "Paired sign-flip p": float(value["paired_sign_flip_p_plus_one"]),
                "Statistical role": p_role,
            }
        )
    return pd.DataFrame(rows)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.13, 1.06, label, transform=axis.transAxes, fontsize=9, fontweight="bold", va="top")


def build_figure(table: pd.DataFrame, effects: pd.DataFrame, output_base: Path) -> None:
    configure_matplotlib()
    fig = plt.figure(figsize=(7.25, 3.75), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.25, 1.0, 1.25],
        left=0.075,
        right=0.985,
        bottom=0.27,
        top=0.84,
        wspace=0.42,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    fig.suptitle("Participant-disjoint evidence for the repaired candidate path", x=0.075, y=0.965, ha="left", fontsize=10.5, fontweight="bold", color="#273746")
    fig.text(0.075, 0.90, "All summaries use 42 participants and 439 windows; error bars are participant-bootstrap 95% confidence intervals.", ha="left", fontsize=6.8, color="#617181")

    display = table.copy()
    y = np.arange(len(display))
    means = display["Participant-equal MAE (BPM)"].to_numpy(float)
    lows = display["MAE 95% CI low"].to_numpy(float)
    highs = display["MAE 95% CI high"].to_numpy(float)
    for idx, row in display.iterrows():
        method = METHOD_ORDER[idx]
        ax_a.plot([lows[idx], highs[idx]], [idx, idx], color=COLORS[method], lw=1.4, solid_capstyle="round")
        ax_a.scatter(means[idx], idx, s=28 if idx == len(display) - 1 else 22, color=COLORS[method], edgecolor="white", linewidth=0.6, zorder=3)
    ax_a.set_yticks(y, display["Method"])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Participant-equal MAE (BPM)")
    ax_a.set_title("Accuracy across matched controls", loc="left", pad=7, fontweight="bold")
    ax_a.grid(axis="x", color="#E8EDF0", lw=0.6)
    ax_a.set_axisbelow(True)
    panel_label(ax_a, "a")

    effect_y = np.arange(len(effects))
    ax_b.axvline(0, color="#83909A", lw=0.8)
    for idx, row in effects.iterrows():
        color = "#4F8F85" if idx == 0 else "#7396B8"
        ax_b.plot([row["CI95 low"], row["CI95 high"]], [idx, idx], color=color, lw=1.6, solid_capstyle="round")
        ax_b.scatter(row["Mean delta (V32 minus comparator), BPM"], idx, s=30, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        p = row["Paired sign-flip p"]
        p_text = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax_b.text(0.98, idx, p_text, transform=ax_b.get_yaxis_transform(), ha="right", va="bottom", fontsize=6.2, color="#4E5D69")
    ax_b.set_yticks(effect_y, effects["Comparator"])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("MAE difference (BPM)\nV32 - comparator")
    ax_b.set_title("Paired participant effects", loc="left", pad=7, fontweight="bold")
    ax_b.grid(axis="x", color="#E8EDF0", lw=0.6)
    ax_b.set_axisbelow(True)
    panel_label(ax_b, "b")

    metric_methods = [
        "VitalsSight_candidate_aware",
        "VitalsSight_v32_independent_emission",
        "VitalsSight_v32_causal_candidate_path",
    ]
    metrics = ["Within 5 BPM", "Within 10 BPM", "Error >10 BPM"]
    marker_map = ["o", "s", "D"]
    x = np.arange(len(metrics))
    for method_index, method in enumerate(metric_methods):
        row = table.iloc[METHOD_ORDER.index(method)]
        values = np.array([row[metric] for metric in metrics], dtype=float) * 100.0
        offset = (method_index - 1) * 0.16
        ax_c.scatter(x + offset, values, s=25, marker=marker_map[method_index], color=COLORS[method], edgecolor="white", linewidth=0.5, label=LABELS[method], zorder=3)
        for xx, value in zip(x + offset, values):
            ax_c.vlines(xx, 0, value, color=COLORS[method], alpha=0.35, lw=0.8)
    ax_c.set_xticks(x, ["Within\n5 BPM", "Within\n10 BPM", "Error\n>10 BPM"])
    ax_c.set_ylabel("Participant-equal proportion (%)")
    ax_c.set_ylim(0, 104)
    ax_c.set_title("Threshold behaviour", loc="left", pad=7, fontweight="bold")
    ax_c.grid(axis="y", color="#E8EDF0", lw=0.6)
    ax_c.set_axisbelow(True)
    ax_c.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=1,
        fontsize=5.8,
        handletextpad=0.4,
        labelspacing=0.25,
        borderaxespad=0,
    )
    panel_label(ax_c, "c")

    fig.text(
        0.075,
        0.035,
        "The temporal-path increment is exploratory (paired sign-flip p=0.099); the supported primary contrast is V32 versus the prior source-preserving selector.",
        ha="left",
        fontsize=6.2,
        color="#617181",
    )
    for suffix, kwargs in (
        ("png", {"dpi": 300}),
        ("tiff", {"dpi": 600}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(output_base.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-ledger", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(not args.output_dir.exists(), f"output directory exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    predictions = pd.read_csv(args.prediction_ledger, low_memory=False)
    require(set(METHOD_ORDER).issubset(set(predictions["method"])), "required methods are missing")
    require(not predictions.duplicated(["method", "sample_id"]).any(), "duplicate method-window rows")
    participants = participant_frame(predictions[predictions["method"].isin(METHOD_ORDER)])
    table = aggregate_table(participants)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    effects = effect_table(summary)

    participants.to_csv(args.output_dir / "figure2_participant_metrics.csv", index=False)
    table.to_csv(args.output_dir / "table2_internal_v32_metrics.csv", index=False)
    effects.to_csv(args.output_dir / "figure2_paired_effects.csv", index=False)
    build_figure(table, effects, args.output_dir / "figure2_v32_internal_core_evidence_python")

    tiff = Image.open(args.output_dir / "figure2_v32_internal_core_evidence_python.tiff")
    contract = {
        "task_id": "V32_INTERNAL_CORE_EVIDENCE_FIGURE_AND_TABLE",
        "core_conclusion": "The repaired joint candidate path lowers participant-equal error while preserving one observed candidate and improving threshold behaviour.",
        "archetype": "quantitative grid with one primary accuracy panel and two subordinate mechanism panels",
        "backend": "Python/matplotlib only",
        "participants": 42,
        "windows": 439,
        "split": "3 outer participant folds x 3 inner participant folds; seeds 704, 1704 and 2704",
        "primary_contrast": "V32 causal candidate path versus prior source-preserving candidate-aware selector",
        "exploratory_contrast": "V32 causal candidate path versus independent emission",
        "tiff_pixels": list(tiff.size),
        "tiff_dpi": [float(value) for value in tiff.info.get("dpi", ())],
        "source_prediction_sha256": sha256(args.prediction_ledger),
        "source_summary_sha256": sha256(args.summary),
        "claim_boundary": "The joint V32 improvement is supported; the temporal transition is not independently significant.",
    }
    (args.output_dir / "FIGURE_CONTRACT.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in args.output_dir.iterdir() if path.is_file())
    manifest = {"files": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **contract}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
