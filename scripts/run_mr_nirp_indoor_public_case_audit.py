from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image
from scipy import io as scipy_io


PROTOCOL_KEY = "MRNIRP-INDOOR-CASE-AUDIT-v1"
SOURCE_FPS = 30.0
FRAME_STRIDE = 5
WINDOW_SECONDS = 60.0
SAMPLED_FRAMES = int(SOURCE_FPS * WINDOW_SECONDS / FRAME_STRIDE)
UNSAFE_BPM = 10.0

BLUE = "#4C78A8"
PALE_BLUE = "#DCE8F4"
CORAL = "#D97A6C"
PALE_CORAL = "#F2D7D3"
GOLD = "#B58A4A"
PALE_GOLD = "#F2E7CF"
TEAL = "#75A99B"
PALE_TEAL = "#DCEBE6"
INK = "#252A31"
MID = "#6B7178"
GRID = "#E7E9EC"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_number(member: str) -> int:
    match = re.search(r"Frame(\d+)\.pgm$", member, flags=re.I)
    return int(match.group(1)) if match else -1


def modality_members(zf: zipfile.ZipFile, modality: str) -> list[str]:
    token = f"/{modality.upper()}/"
    members = [
        name
        for name in zf.namelist()
        if token in name and name.lower().endswith(".pgm")
    ]
    # Subject2_still uses the camera directory name instead of the NIR alias
    # used by the other released archives. The README identifies camera 1 as
    # the narrow-band stream; preserve this naming exception explicitly.
    if modality.upper() == "NIR" and not members:
        members = [
            name
            for name in zf.namelist()
            if "/cam_flea3_1/" in name and name.lower().endswith(".pgm")
        ]
    return sorted(members, key=frame_number)


def normalize_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    lo, hi = np.nanpercentile(arr, [1.0, 99.5])
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
    scaled = np.clip((arr - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def decode_member(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    payload = np.frombuffer(zf.read(member), dtype=np.uint8)
    frame = cv2.imdecode(payload, cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise ValueError(f"Could not decode {member}")
    return frame


def display_rgb(frame: np.ndarray, subject: int, modality: str) -> np.ndarray:
    u8 = normalize_uint8(frame)
    if modality.upper() == "NIR":
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)
    # The public README records BGGR for subject 6 and RGGB for all other
    # released subjects. Demosaicing is for display and RGB trace extraction.
    # OpenCV names the conversion by the second-row pattern; the apparent
    # mapping is therefore the inverse of the sensor label in the README.
    code = cv2.COLOR_BAYER_RG2RGB if subject == 6 else cv2.COLOR_BAYER_BG2RGB
    return cv2.cvtColor(u8, code)


def pulse_reference(zf: zipfile.ZipFile) -> tuple[float, float, dict[str, Any]]:
    from scripts import run_t551_mr_nirp_lowlight_pilot as t551

    mats = [name for name in zf.namelist() if name.lower().endswith("pulseox.mat")]
    if not mats:
        raise FileNotFoundError("pulseOx.mat missing")
    mat = scipy_io.loadmat(io.BytesIO(zf.read(mats[0])))
    raw_values = np.asarray(mat["pulseOxRecord"])
    if raw_values.dtype == object:
        values = np.asarray(
            [float(np.asarray(item).reshape(-1)[0]) for item in raw_values.ravel()],
            dtype=float,
        )
    else:
        values = raw_values.ravel().astype(float)
    times = np.asarray(mat["pulseOxTime"]).ravel().astype(float)
    duration = float(times[-1] - times[0]) if len(times) > 1 else math.nan
    fs = float((len(times) - 1) / duration) if duration > 0 else 60.0
    n = min(len(values), int(round(fs * WINDOW_SECONDS)))
    bpm, _, confidence = t551.estimate_hr(values[:n], fs)
    return float(bpm), float(confidence), {
        "pulse_member": mats[0],
        "pulse_samples_total": int(len(values)),
        "pulse_samples_used": int(n),
        "pulse_fs": fs,
        "pulse_duration_sec": duration,
    }


def roi_rows(
    archive: Path,
    condition_id: str,
    subject: int,
    modality: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from scripts import run_t493_selected_domain_roi_trace_cache as t493
    from src.vision.roi import crop

    with zipfile.ZipFile(archive) as zf:
        members = modality_members(zf, modality)
        selected = members[: int(SOURCE_FPS * WINDOW_SECONDS) : FRAME_STRIDE]
        selected = selected[:SAMPLED_FRAMES]
        if not selected:
            raise ValueError(f"No {modality} frames in {archive}")

        first = display_rgb(decode_member(zf, selected[0]), subject, modality)
        rois, roi_method = t493.rois_from_face_or_fallback(first)
        rows: list[dict[str, Any]] = []
        for sample_index, member in enumerate(selected):
            rgb = display_rgb(decode_member(zf, member), subject, modality)
            for item in rois:
                region = crop(rgb, item.roi)
                if region.size == 0:
                    continue
                mean_rgb = region.reshape(-1, 3).mean(axis=0)
                rows.append(
                    {
                        "protocol_key": PROTOCOL_KEY,
                        "condition_id": condition_id,
                        "subject": f"Subject{subject}",
                        "modality": modality.upper(),
                        "frame_index": frame_number(member),
                        "timestamp_s": frame_number(member) / SOURCE_FPS,
                        "roi": item.name,
                        "mean_r": float(mean_rgb[0]),
                        "mean_g": float(mean_rgb[1]),
                        "mean_b": float(mean_rgb[2]),
                        "member": member,
                    }
                )

    return pd.DataFrame(rows), {
        "protocol_key": PROTOCOL_KEY,
        "condition_id": condition_id,
        "subject": f"Subject{subject}",
        "modality": modality.upper(),
        "roi_method": roi_method,
        "n_rois": int(len(rois)),
        "source_fps": SOURCE_FPS,
        "frame_stride": FRAME_STRIDE,
        "effective_fps": SOURCE_FPS / FRAME_STRIDE,
        "window_seconds": WINDOW_SECONDS,
        "frames_available": int(len(members)),
        "frames_used": int(len(selected)),
        "first_member": selected[0],
        "last_member": selected[-1],
    }


def candidate_rows(
    trace: pd.DataFrame,
    condition_id: str,
    subject: int,
    reference_bpm: float,
) -> list[dict[str, Any]]:
    from scripts import run_t494_roi_candidate_evaluation as t494
    from scripts import run_t572_mr_nirp_full_roi_lowlight_selector as t572
    from src.baselines.traditional_rppg import METHODS

    rows: list[dict[str, Any]] = []
    for (modality, roi), group in trace.groupby(["modality", "roi"], sort=True):
        group = group.sort_values("timestamp_s")
        times = group["timestamp_s"].to_numpy(dtype=float)
        rgb = group[["mean_r", "mean_g", "mean_b"]].to_numpy(dtype=float)
        if modality == "RGB":
            signals = {
                "GREEN": METHODS["GREEN"](rgb),
                "CHROM": METHODS["CHROM"](rgb),
                "POS": METHODS["POS"](rgb),
            }
        else:
            signals = {"NIR": rgb[:, 1]}

        for method, values in signals.items():
            peaks, fs = t494.spectral_peaks(np.asarray(values, dtype=float), times, k=5)
            for peak in peaks:
                bpm = float(peak["candidate_bpm"])
                power = float(peak["relative_power"])
                nyquist = float(peak.get("nyquist_bpm", math.nan))
                error = abs(bpm - reference_bpm)
                artifact = t572.artifact_features(values, bpm, power, nyquist)
                rows.append(
                    {
                        "protocol_key": PROTOCOL_KEY,
                        "condition_id": condition_id,
                        "subject": f"Subject{subject}",
                        "modality": modality,
                        "roi": roi,
                        "method": method,
                        "rank": int(peak["rank"]),
                        "sample_fs": float(fs),
                        "candidate_bpm": bpm,
                        "reference_bpm": reference_bpm,
                        "absolute_error_bpm": error,
                        "relative_power": power,
                        "unsafe_error_gt10": bool(error > UNSAFE_BPM),
                        **artifact,
                    }
                )
    return rows


def parse_condition(path: Path) -> tuple[str, int, str]:
    match = re.fullmatch(r"Subject(\d+)_(still|motion)_940", path.stem, flags=re.I)
    if not match:
        raise ValueError(f"Unexpected archive name: {path.name}")
    subject = int(match.group(1))
    state = match.group(2).lower()
    return path.stem.lower(), subject, state


def classify_decision(row: pd.Series) -> str:
    released = str(row.get("policy", "")).lower() == "release"
    error = pd.to_numeric(pd.Series([row.get("absolute_error_bpm")]), errors="coerce").iloc[0]
    oracle = pd.to_numeric(pd.Series([row.get("oracle_best_error_bpm")]), errors="coerce").iloc[0]
    if released and math.isfinite(float(error)) and float(error) <= UNSAFE_BPM:
        return "reliable_release"
    if released:
        return "unsafe_release_exposed"
    if math.isfinite(float(oracle)) and float(oracle) <= UNSAFE_BPM:
        return "review_with_safe_candidate"
    return "review_without_safe_candidate"


def run_audit(project_root: Path, dataset_root: Path, output_dir: Path) -> None:
    from scripts import run_t495_method_aware_roi_selector as t495

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(exist_ok=True)
    archives = sorted(dataset_root.glob("Subject*_940.zip"), key=lambda p: parse_condition(p)[:2])
    if len(archives) != 15:
        raise RuntimeError(f"Expected 15 released MR-NIRP Indoor archives, found {len(archives)}")

    candidates_all: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, archive in enumerate(archives, start=1):
        condition_id, subject, state = parse_condition(archive)
        print(f"[{index:02d}/{len(archives):02d}] {condition_id}", flush=True)
        try:
            with zipfile.ZipFile(archive) as zf:
                reference_bpm, reference_confidence, pulse_meta = pulse_reference(zf)
            traces: list[pd.DataFrame] = []
            modality_meta: list[dict[str, Any]] = []
            for modality in ["RGB", "NIR"]:
                trace, meta = roi_rows(archive, condition_id, subject, modality)
                traces.append(trace)
                modality_meta.append(meta)
            joined = pd.concat(traces, ignore_index=True)
            trace_path = trace_dir / f"{condition_id}_roi_trace.csv"
            joined.to_csv(trace_path, index=False, encoding="utf-8-sig")
            candidates_all.extend(candidate_rows(joined, condition_id, subject, reference_bpm))

            with zipfile.ZipFile(archive) as zf:
                selected_members = []
                for modality in ["RGB", "NIR"]:
                    members = modality_members(zf, modality)
                    selected = members[: int(SOURCE_FPS * WINDOW_SECONDS) : FRAME_STRIDE][:SAMPLED_FRAMES]
                    selected_members.extend(
                        {
                            "modality": modality,
                            "member": name,
                            "crc32": f"{zf.getinfo(name).CRC:08x}",
                            "file_size": int(zf.getinfo(name).file_size),
                        }
                        for name in (selected[:1] + selected[-1:])
                    )
            sources.append(
                {
                    "protocol_key": PROTOCOL_KEY,
                    "condition_id": condition_id,
                    "subject": f"Subject{subject}",
                    "state": state,
                    "archive": archive.as_posix(),
                    "archive_bytes": int(archive.stat().st_size),
                    "reference_bpm": reference_bpm,
                    "reference_confidence": reference_confidence,
                    "trace_path": trace_path.as_posix(),
                    "trace_sha256": digest(trace_path),
                    "pulse": pulse_meta,
                    "modalities": modality_meta,
                    "selected_member_checks": selected_members,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "condition_id": condition_id,
                    "archive": archive.as_posix(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)

    candidates = pd.DataFrame(candidates_all)
    candidates_path = output_dir / "mr_nirp_indoor_candidates.csv"
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    decisions: list[dict[str, Any]] = []
    for condition_id, rows in candidates.groupby("condition_id", sort=True):
        decision = t495.choose_mr(condition_id, rows.copy())
        decision["protocol_key"] = PROTOCOL_KEY
        decision["subject"] = str(rows["subject"].iloc[0])
        source = next(item for item in sources if item["condition_id"] == condition_id)
        decision["state"] = source["state"]
        decisions.append(decision)
    decision_frame = pd.DataFrame(decisions)
    decision_frame["case_class"] = decision_frame.apply(classify_decision, axis=1)
    decision_path = output_dir / "mr_nirp_indoor_decisions.csv"
    decision_frame.to_csv(decision_path, index=False, encoding="utf-8-sig")

    released = decision_frame[decision_frame["policy"].eq("release")]
    summary = {
        "protocol_key": PROTOCOL_KEY,
        "generated_from": "MR-NIRP Indoor released subset",
        "window_seconds": WINDOW_SECONDS,
        "source_fps": SOURCE_FPS,
        "frame_stride": FRAME_STRIDE,
        "effective_fps": SOURCE_FPS / FRAME_STRIDE,
        "archives_expected": 15,
        "archives_completed": int(decision_frame.shape[0]),
        "subjects": int(decision_frame["subject"].nunique()) if not decision_frame.empty else 0,
        "failures": failures,
        "class_counts": decision_frame["case_class"].value_counts().to_dict(),
        "n_released": int(len(released)),
        "coverage": float(len(released) / len(decision_frame)) if len(decision_frame) else 0.0,
        "released_mae_bpm": float(released["absolute_error_bpm"].mean()) if len(released) else None,
        "unsafe_release_rate": float(released["unsafe_release_gt10"].mean()) if len(released) else None,
        "claim_boundary": (
            "Descriptive, same-window MR-NIRP Indoor case audit. It is a separate protocol "
            "from retained MR-NIRP manuscript rows and must not replace or merge with them."
        ),
        "permission_basis": (
            "The public MR-NIRP Indoor README states that two non-consenting subjects were "
            "excluded from the released dataset; only the eight consenting subjects are used."
        ),
        "files": {
            "candidates": candidates_path.as_posix(),
            "decisions": decision_path.as_posix(),
            "sources": (output_dir / "mr_nirp_indoor_sources.json").as_posix(),
        },
    }
    write_json(output_dir / "mr_nirp_indoor_sources.json", sources)
    write_json(output_dir / "mr_nirp_indoor_summary.json", summary)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), flush=True)


def crop_face_context(rgb: np.ndarray) -> np.ndarray:
    from src.vision.roi import detect_face_roi

    detected = detect_face_roi(rgb)
    h, w = rgb.shape[:2]
    if detected is None:
        side = min(h, w)
        x0 = max(0, (w - side) // 2)
        y0 = max(0, (h - side) // 2)
        crop = rgb[y0 : y0 + side, x0 : x0 + side]
    else:
        cx = detected.x + detected.w / 2
        cy = detected.y + detected.h / 2
        side = int(max(detected.w, detected.h) * 1.48)
        x0 = max(0, min(w - side, int(cx - side / 2)))
        y0 = max(0, min(h - side, int(cy - side / 2)))
        crop = rgb[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(crop, (170, 170), interpolation=cv2.INTER_AREA)


def frame_strip(
    archive: Path,
    subject: int,
    output_dir: Path,
    condition_id: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    with zipfile.ZipFile(archive) as zf:
        members = modality_members(zf, "RGB")
        last = min(len(members) - 1, int(SOURCE_FPS * WINDOW_SECONDS) - 1)
        indices = np.linspace(0, last, 8).round().astype(int)
        frames: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        frame_dir = output_dir / "selected_frames" / condition_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        for index in indices:
            member = members[int(index)]
            rgb = display_rgb(decode_member(zf, member), subject, "RGB")
            crop = crop_face_context(rgb)
            path = frame_dir / f"frame_{frame_number(member):05d}.png"
            Image.fromarray(crop).save(path)
            frames.append(crop)
            records.append(
                {
                    "condition_id": condition_id,
                    "member": member,
                    "frame_index": frame_number(member),
                    "timestamp_s": frame_number(member) / SOURCE_FPS,
                    "crc32": f"{zf.getinfo(member).CRC:08x}",
                    "png": path.as_posix(),
                    "png_sha256": digest(path),
                }
            )
    return np.concatenate(frames, axis=1), records


def route_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    method: str,
    reference: float,
    released_bpm: float,
    released: bool,
    show_ylabel: bool,
) -> None:
    route = rows[rows["method"].eq(method)].copy()
    route["relative_power"] = pd.to_numeric(route["relative_power"], errors="coerce").fillna(0.0)
    scale = float(route["relative_power"].max()) if len(route) else 1.0
    route["plot_power"] = route["relative_power"] / max(scale, 1e-12)
    safe = (route["candidate_bpm"] - reference).abs() <= UNSAFE_BPM
    ax.axvspan(reference - UNSAFE_BPM, reference + UNSAFE_BPM, color=PALE_BLUE, alpha=0.55, zorder=0)
    ax.scatter(
        route.loc[~safe, "candidate_bpm"],
        route.loc[~safe, "plot_power"],
        s=11,
        facecolors=PALE_CORAL,
        edgecolors=CORAL,
        linewidths=0.55,
        zorder=2,
    )
    ax.scatter(
        route.loc[safe, "candidate_bpm"],
        route.loc[safe, "plot_power"],
        s=13,
        facecolors=PALE_TEAL,
        edgecolors=TEAL,
        linewidths=0.65,
        zorder=3,
    )
    ax.axvline(reference, color=INK, linewidth=0.85, zorder=4)
    if released and math.isfinite(released_bpm):
        ax.axvline(released_bpm, color=CORAL, linewidth=1.05, zorder=5)
    ax.set_xlim(40, 140)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xticks([50, 80, 110, 140])
    ax.set_yticks([0.0, 0.5, 1.0] if show_ylabel else [])
    if show_ylabel:
        ax.set_ylabel("relative\nevidence", fontsize=6.4)
    ax.tick_params(labelsize=6.6, length=2.2, pad=1.6)
    ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax.set_title(method, fontsize=7.2, weight="bold", loc="left", pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def run_figure(
    project_root: Path,
    dataset_root: Path,
    output_dir: Path,
    case_spec: Path,
) -> None:
    candidates = pd.read_csv(output_dir / "mr_nirp_indoor_candidates.csv")
    decisions = pd.read_csv(output_dir / "mr_nirp_indoor_decisions.csv")
    sources = json.loads((output_dir / "mr_nirp_indoor_sources.json").read_text(encoding="utf-8"))
    source_map = {item["condition_id"]: item for item in sources}
    spec = json.loads(case_spec.read_text(encoding="utf-8"))
    groups = spec["groups"]
    selected = [case for group in groups for case in group["cases"]]
    if len(groups) != 4 or any(len(group["cases"]) != 2 for group in groups):
        raise ValueError("Case spec must contain four groups with two cases each")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.edgecolor": INK,
            "axes.linewidth": 0.65,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
        }
    )
    # Sized for a 6.65-inch portrait manuscript insertion; this avoids the
    # unreadable tick-label shrinkage caused by scaling down a poster canvas.
    fig = plt.figure(figsize=(8.4, 10.4), facecolor="white")
    outer = fig.add_gridspec(4, 2, left=0.055, right=0.985, top=0.925, bottom=0.055, hspace=0.42, wspace=0.19)
    frame_records: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        for case_index, case in enumerate(group["cases"]):
            condition_id = case["condition_id"]
            decision_row = decisions[decisions["condition_id"].eq(condition_id)]
            if len(decision_row) != 1:
                raise ValueError(f"Decision not unique for {condition_id}")
            decision = decision_row.iloc[0]
            source = source_map[condition_id]
            subject = int(re.search(r"(\d+)", str(decision["subject"])).group(1))
            archive = Path(source["archive"])
            strip, records = frame_strip(archive, subject, output_dir, condition_id)
            frame_records.extend(records)
            sub = outer[group_index, case_index].subgridspec(3, 4, height_ratios=[0.64, 0.86, 0.86], hspace=0.50, wspace=0.18)

            ax_image = fig.add_subplot(sub[0, :])
            ax_image.imshow(strip)
            ax_image.axis("off")
            ref = float(decision["reference_bpm"])
            released = str(decision["policy"]).lower() == "release"
            selected_bpm = pd.to_numeric(pd.Series([decision.get("released_bpm")]), errors="coerce").iloc[0]
            selected_value = float(selected_bpm) if math.isfinite(float(selected_bpm)) else math.nan
            state = "RELEASE" if released else "REVIEW"
            if released:
                summary = f"reference {ref:.1f} | selected {selected_value:.1f} BPM | error {abs(selected_value-ref):.1f}"
            else:
                oracle = float(decision["oracle_best_error_bpm"])
                summary = f"reference {ref:.1f} BPM | review | nearest candidate error {oracle:.1f}"
            title = f"{case['label']}  {decision['subject']} {decision['state']} | {state}"
            ax_image.text(
                0.0,
                1.23,
                title,
                transform=ax_image.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.4,
                weight="bold",
                color=INK,
                clip_on=False,
            )
            ax_image.text(
                0.0,
                1.06,
                summary,
                transform=ax_image.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.0,
                color=MID,
                clip_on=False,
            )

            rows = candidates[candidates["condition_id"].eq(condition_id)]
            for method_index, method in enumerate(["POS", "CHROM", "GREEN", "NIR"]):
                row = 1 + method_index // 2
                col0 = (method_index % 2) * 2
                ax = fig.add_subplot(sub[row, col0 : col0 + 2])
                route_panel(ax, rows, method, ref, selected_value, released, method_index % 2 == 0)
                if row == 2:
                    ax.set_xlabel("HR candidate (BPM)", fontsize=6.4, labelpad=1.5)

    legend = [
        Line2D([0], [0], color=INK, lw=1.1, label="retrospective reference HR"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALE_TEAL, markeredgecolor=TEAL, markersize=5, label="candidate within 10 BPM"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALE_CORAL, markeredgecolor=CORAL, markersize=5, label="candidate outside 10 BPM"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.69, 0.965), frameon=False, ncol=3, fontsize=7.2, handlelength=2.2)
    fig.suptitle(
        "MR-NIRP Indoor: candidate evidence under conservative review",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=10.2,
        weight="bold",
        color=INK,
    )
    stem = output_dir / "figure_mr_nirp_indoor_eight_case_public_release_python"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    write_json(output_dir / "mr_nirp_indoor_selected_frame_manifest.json", frame_records)
    write_json(
        output_dir / "mr_nirp_indoor_figure_manifest.json",
        {
            "protocol_key": PROTOCOL_KEY,
            "generator": "Python / matplotlib",
            "script": Path(__file__).resolve().as_posix(),
            "case_spec": case_spec.resolve().as_posix(),
            "selected_conditions": selected,
            "figure_png": stem.with_suffix(".png").as_posix(),
            "figure_png_sha256": digest(stem.with_suffix(".png")),
            "permission_basis": (
                "MR-NIRP Indoor public README: two subjects without public-release consent "
                "were excluded; the released eight-subject dataset is used here."
            ),
            "display_note": "RGB Bayer frames were demosaiced for display; no facial masking was used.",
            "claim_boundary": (
                "Eight descriptive cases from a separate same-window protocol; not independent "
                "replications, not a clinical safety claim, and not merged with retained MR-NIRP rows."
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["audit", "figure"], default="audit")
    parser.add_argument("--case-spec", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project_root.resolve()))
    if args.mode == "audit":
        run_audit(args.project_root, args.dataset_root, args.output_dir)
    else:
        if args.case_spec is None:
            raise SystemExit("--case-spec is required for figure mode")
        run_figure(args.project_root, args.dataset_root, args.output_dir, args.case_spec)


if __name__ == "__main__":
    main()
