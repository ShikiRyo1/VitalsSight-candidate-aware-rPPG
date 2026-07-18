from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "T551"
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"
DATA_ROOT = Path(
    os.environ.get(
        "ADULT_DATA_ROOT",
        Path(os.environ.get("CONTACTLESS_DATA_ROOT", ROOT / "datasets")) / "adult",
    )
)
MR_ROOT = Path(os.environ.get("MR_NIRP_ROOT", DATA_ROOT / "MR-NIRP"))

MAX_CONDITIONS = int(os.environ.get("MR_NIRP_MAX_CONDITIONS", "8"))
WINDOW_SECONDS = float(os.environ.get("MR_NIRP_WINDOW_SECONDS", "60"))

PRED_CSV = EXP / "t551_mr_nirp_lowlight_pilot_predictions.csv"
METRICS_CSV = EXP / "t551_mr_nirp_lowlight_pilot_metrics.csv"
CLAIM_GATE_CSV = EXP / "t551_mr_nirp_lowlight_pilot_claim_gate.csv"
SUMMARY_JSON = EXP / "t551_mr_nirp_lowlight_pilot_summary.json"
DOC_MD = DOCS / "t551_mr_nirp_lowlight_pilot.md"


def json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, list):
        return [json_safe(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, Path):
        return v.as_posix()
    return v


def estimate_hr(trace: np.ndarray, fs: float) -> tuple[float, float, float]:
    trace = np.asarray(trace, dtype=float)
    trace = trace[np.isfinite(trace)]
    if len(trace) < max(32, int(fs * 10)):
        return math.nan, math.nan, math.nan
    trace = signal.detrend(trace)
    lo, hi = 0.75, 4.0
    try:
        b, a = signal.butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        trace = signal.filtfilt(b, a, trace)
    except Exception:
        pass
    nperseg = min(len(trace), max(64, int(fs * 16)))
    freqs, power = signal.welch(trace, fs=fs, nperseg=nperseg)
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return math.nan, math.nan, math.nan
    bf = freqs[mask]
    bp = power[mask]
    idx = int(np.argmax(bp))
    conf = float(bp[idx] / (np.sum(bp) + 1e-12))
    return float(bf[idx] * 60.0), float(bf[idx]), conf


def read_pulseox(pulse_zip: Path, window_seconds: float) -> tuple[np.ndarray, float, dict[str, Any]]:
    with zipfile.ZipFile(pulse_zip) as z:
        mats = [n for n in z.namelist() if n.lower().endswith("pulseox.mat")]
        if not mats:
            raise FileNotFoundError(f"pulseOx.mat missing in {pulse_zip}")
        mat = sio.loadmat(io.BytesIO(z.read(mats[0])))
    rec = np.asarray(mat["pulseOxRecord"]).ravel().astype(float)
    t = np.asarray(mat["pulseOxTime"]).ravel().astype(float)
    duration = float(t[-1] - t[0]) if len(t) > 1 else math.nan
    fs = float((len(t) - 1) / duration) if duration and duration > 0 else 60.0
    n = min(len(rec), int(fs * window_seconds))
    meta = {"pulse_samples": int(len(rec)), "pulse_duration_sec": duration, "pulse_fs": fs}
    return rec[:n], fs, meta


def sorted_pgm_names(z: zipfile.ZipFile) -> list[str]:
    names = [n for n in z.namelist() if n.lower().endswith(".pgm")]
    def key(n: str) -> int:
        m = re.search(r"(\d+)(?=\.pgm$)", n, flags=re.I)
        return int(m.group(1)) if m else 0
    return sorted(names, key=key)


def read_zip_mean_trace(zip_path: Path, window_seconds: float, duration_hint: float) -> tuple[np.ndarray, float, dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as z:
        names = sorted_pgm_names(z)
        total = len(names)
        if total == 0:
            raise ValueError(f"no pgm frames in {zip_path}")
        fs = float(total / duration_hint) if duration_hint and duration_hint > 0 else 30.0
        n = min(total, int(fs * window_seconds))
        means: list[float] = []
        for name in names[:n]:
            data = np.frombuffer(z.read(name), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            means.append(float(np.mean(img)))
    meta = {"frame_count": int(total), "frames_used": int(len(means)), "video_fs": fs}
    return np.asarray(means, dtype=float), fs, meta


def condition_metadata(cond: Path) -> dict[str, str]:
    name = cond.name
    parts = name.split("_")
    subject = parts[0] if parts else ""
    scenario = parts[1] if len(parts) > 1 else ""
    motion = "large_motion" if "large_motion" in name else "small_motion" if "small_motion" in name else "motion" if "motion" in name else "still" if "still" in name else ""
    wavelength = "975" if "975" in name else "940" if "940" in name else ""
    location = "garage" if "garage" in name else "driving" if "driving" in name else "indoor" if "indoor" in name else ""
    return {"condition_id": name, "subject": subject, "scenario": scenario, "location": location, "motion": motion, "wavelength_nm": wavelength}


def find_conditions() -> list[Path]:
    pulse_zips = sorted(MR_ROOT.rglob("PulseOx.zip")) + sorted(MR_ROOT.rglob("PulseOX.zip"))
    out = []
    for p in pulse_zips:
        cond = p.parent
        if (cond / "RGB.zip").exists() and (cond / "NIR.zip").exists():
            out.append(cond)
    return out[:MAX_CONDITIONS]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    preferred = [
        "condition_id", "subject", "scenario", "location", "motion", "wavelength_nm", "modality",
        "gt_hr_bpm", "pred_hr_bpm", "abs_error_bpm", "unsafe_error_gt10", "confidence",
        "signal_fs", "samples_used", "frame_count", "frames_used",
    ]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(pred: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for keys in [["modality"], ["motion", "modality"], ["location", "modality"]]:
        for group_key, sub in pred.groupby(keys, dropna=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            err = pd.to_numeric(sub["abs_error_bpm"], errors="coerce").dropna()
            unsafe = sub["unsafe_error_gt10"].astype(bool)
            row = {k: v for k, v in zip(keys, group_key)}
            row.update(
                {
                    "level": "+".join(keys),
                    "n": int(len(err)),
                    "mae_bpm": float(err.mean()) if len(err) else math.nan,
                    "median_abs_error_bpm": float(err.median()) if len(err) else math.nan,
                    "unsafe_gt10_rate": float(unsafe.mean()) if len(sub) else math.nan,
                    "mean_confidence": float(pd.to_numeric(sub["confidence"], errors="coerce").mean()),
                }
            )
            rows.append(row)
    return rows


def main() -> int:
    conditions = find_conditions()
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for cond in conditions:
        meta = condition_metadata(cond)
        pulse_zip = cond / "PulseOx.zip" if (cond / "PulseOx.zip").exists() else cond / "PulseOX.zip"
        try:
            pulse, pulse_fs, pulse_meta = read_pulseox(pulse_zip, WINDOW_SECONDS)
            gt_hr, _, gt_conf = estimate_hr(pulse, pulse_fs)
            duration_hint = pulse_meta["pulse_duration_sec"]
            for modality, zp in [("RGB", cond / "RGB.zip"), ("NIR", cond / "NIR.zip")]:
                trace, fs, vmeta = read_zip_mean_trace(zp, WINDOW_SECONDS, duration_hint)
                pred_hr, _, conf = estimate_hr(trace, fs)
                err = abs(pred_hr - gt_hr) if math.isfinite(pred_hr) and math.isfinite(gt_hr) else math.nan
                predictions.append(
                    {
                        **meta,
                        "modality": modality,
                        "gt_hr_bpm": gt_hr,
                        "pred_hr_bpm": pred_hr,
                        "abs_error_bpm": err,
                        "unsafe_error_gt10": bool(math.isfinite(err) and err > 10.0),
                        "confidence": conf,
                        "gt_confidence": gt_conf,
                        "signal_fs": fs,
                        "samples_used": len(trace),
                        **vmeta,
                        **pulse_meta,
                    }
                )
        except Exception as exc:
            failures.append({**meta, "error": f"{type(exc).__name__}: {exc}"})

    write_csv(PRED_CSV, predictions)
    metrics = summarize(pd.DataFrame(predictions)) if predictions else []
    write_csv(METRICS_CSV, metrics)
    by_mod = {r.get("modality"): r for r in metrics if r.get("level") == "modality"}
    rgb = by_mod.get("RGB", {})
    nir = by_mod.get("NIR", {})
    rgb_mae = float(rgb.get("mae_bpm", math.nan)) if rgb else math.nan
    nir_mae = float(nir.get("mae_bpm", math.nan)) if nir else math.nan
    best_mae = min([x for x in [rgb_mae, nir_mae] if math.isfinite(x)], default=math.nan)
    best_modality = "NIR" if math.isfinite(nir_mae) and (not math.isfinite(rgb_mae) or nir_mae <= rgb_mae) else "RGB"
    gates = [
        {
            "gate": "mr_nirp_metric_pilot_available",
            "passed": bool(predictions),
            "evidence": f"conditions={len(conditions)} predictions={len(predictions)} failures={len(failures)}",
            "claim_allowed": "MR-NIRP can move beyond archive preflight into pilot HR metric analysis.",
            "claim_not_allowed": "MR-NIRP full low-light robustness is solved.",
        },
        {
            "gate": "nir_or_rgb_low_error_release",
            "passed": bool(math.isfinite(best_mae) and best_mae <= 5.0),
            "evidence": f"best_modality={best_modality} best_mae={best_mae}",
            "claim_allowed": "If true, report low-light pilot feasibility for the best modality.",
            "claim_not_allowed": "If false, keep MR-NIRP as low-light review/preflight boundary.",
        },
        {
            "gate": "nir_beats_rgb",
            "passed": bool(math.isfinite(rgb_mae) and math.isfinite(nir_mae) and nir_mae < rgb_mae),
            "evidence": f"RGB_MAE={rgb_mae}; NIR_MAE={nir_mae}",
            "claim_allowed": "NIR can be discussed as a promising multimodal route if true.",
            "claim_not_allowed": "NIR superiority across low-light domains without full validation.",
        },
    ]
    write_csv(CLAIM_GATE_CSV, gates)
    summary = {
        "task_id": TASK_ID,
        "mr_root": MR_ROOT,
        "max_conditions": MAX_CONDITIONS,
        "window_seconds": WINDOW_SECONDS,
        "conditions_attempted": len(conditions),
        "predictions": len(predictions),
        "failures": failures,
        "modality_metrics": by_mod,
        "gates": gates,
        "decision": "mr_nirp_lowlight_pilot_complete",
        "claim_boundary": "This is a streamed zip pilot, not full MR-NIRP low-light solved evidence.",
    }
    SUMMARY_JSON.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# T551 MR-NIRP Low-Light / RGB-NIR Pilot",
        "",
        "## Purpose",
        "",
        "Move MR-NIRP from archive/preflight evidence toward real HR metric evidence by streaming RGB/NIR zip frames and comparing them with PulseOx reference signals.",
        "",
        "## Main Result",
        "",
        f"- Conditions attempted: `{len(conditions)}`.",
        f"- Predictions produced: `{len(predictions)}`.",
        f"- RGB MAE: `{rgb_mae if math.isfinite(rgb_mae) else 'NA'}`.",
        f"- NIR MAE: `{nir_mae if math.isfinite(nir_mae) else 'NA'}`.",
        "",
        "## Claim Gate",
        "",
        "| Gate | Passed | Evidence | Allowed claim | Not allowed |",
        "|---|---:|---|---|---|",
    ]
    for g in gates:
        lines.append(f"| {g['gate']} | {g['passed']} | {g['evidence']} | {g['claim_allowed']} | {g['claim_not_allowed']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "This pilot tells us whether MR-NIRP can support low-light/multimodal HR metric claims now. If the low-error gate fails, the correct manuscript wording is review/preflight boundary rather than solved low-light robustness.",
    ]
    DOC_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
