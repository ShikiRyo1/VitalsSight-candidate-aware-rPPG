from __future__ import annotations

import json
import math
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision.roi import ROI, crop, detect_face_roi, face_like_rois  # noqa: E402


TASK_ID = "T493"
EXP = ROOT / "experiments"
DOCS = ROOT / "docs"
RUNTIME = ROOT / "runtime" / "t493_selected_domain_roi_trace_cache"
TEMP = ROOT / "runtime" / "t493_temp_extract"

SUBSET = EXP / "t488_external_domain_locked_subset_manifest.csv"
MR_ZIP_AUDIT = EXP / "t486_mr_nirp_zip_audit.csv"
T485_SUMMARY = EXP / "t485_ubfc_phys_selective_index_summary.json"
T491_SUMMARY = EXP / "t491_selected_domain_compact_trace_cache_summary.json"
T492_SUMMARY = EXP / "t492_selected_domain_artifact_gate_summary.json"

TRACE_INDEX_CSV = EXP / "t493_selected_domain_roi_trace_cache_index.csv"
QUALITY_SUMMARY_CSV = EXP / "t493_selected_domain_roi_trace_quality_summary.csv"
CLAIM_GATE_CSV = EXP / "t493_selected_domain_roi_trace_cache_claim_gate.csv"
SUMMARY_JSON = EXP / "t493_selected_domain_roi_trace_cache_summary.json"
DOC_MD = DOCS / "t493_selected_domain_roi_trace_cache.md"

TASK_REGISTRY = DOCS / "execution_task_registry.md"
LEARNING_JOURNAL = DOCS / "phase_learning_journal.md"
PROJECT_STATUS = DOCS / "project_status.md"
PAPER_CLAIMS = DOCS / "paper_claims_tracker.md"
PROBLEM_LOG = DOCS / "problem_and_improvement_log.md"
INNOVATION_LOG = DOCS / "innovation_log.md"
EVIDENCE_TABLE = EXP / "experiment_evidence_table.csv"

UBFC_SOURCE_FPS_DEFAULT = float(os.environ.get("T493_UBFC_FPS", "30"))
UBFC_FRAME_STRIDE = int(os.environ.get("T493_UBFC_FRAME_STRIDE", "3"))
UBFC_MAX_ACCEPTED_FRAMES = int(os.environ.get("T493_UBFC_MAX_ACCEPTED_FRAMES", "1800"))
MR_FRAME_STRIDE = int(os.environ.get("T493_MR_FRAME_STRIDE", "5"))
MR_MAX_ACCEPTED_FRAMES = int(os.environ.get("T493_MR_MAX_ACCEPTED_FRAMES", "900"))
MR_SOURCE_FPS = float(os.environ.get("T493_MR_SOURCE_FPS", "30"))


@dataclass(frozen=True)
class SimpleROI:
    name: str
    roi: ROI


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


def rois_from_face_or_fallback(frame_rgb: np.ndarray) -> tuple[list[SimpleROI], str]:
    detected = detect_face_roi(frame_rgb)
    if detected is None:
        return [SimpleROI(r.name, r) for r in face_like_rois(frame_rgb)], "fallback_face_like"

    face = detected.clamp(frame_rgb.shape[1], frame_rgb.shape[0])
    regions = [
        ROI("face", face.x, face.y, face.w, face.h),
        ROI("forehead", face.x + int(face.w * 0.20), face.y + int(face.h * 0.02), int(face.w * 0.60), int(face.h * 0.20)),
        ROI("left_cheek", face.x + int(face.w * 0.10), face.y + int(face.h * 0.42), int(face.w * 0.28), int(face.h * 0.28)),
        ROI("right_cheek", face.x + int(face.w * 0.62), face.y + int(face.h * 0.42), int(face.w * 0.28), int(face.h * 0.28)),
    ]
    return [SimpleROI(r.name, r.clamp(frame_rgb.shape[1], frame_rgb.shape[0])) for r in regions], "opencv_haar_face"


def frame_to_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] == 3:
        if frame.dtype == np.uint8:
            return frame
        arr = frame.astype(np.float32)
        lo, hi = np.nanpercentile(arr, [1, 99])
        out = np.clip((arr - lo) / max(1e-6, hi - lo), 0, 1)
        return (out * 255).astype(np.uint8)
    arr = frame.astype(np.float32)
    lo, hi = np.nanpercentile(arr, [1, 99])
    out = np.clip((arr - lo) / max(1e-6, hi - lo), 0, 1)
    gray = (out * 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def roi_feature_row(frame_rgb: np.ndarray, frame_index: int, timestamp_s: float, rois: list[SimpleROI]) -> list[dict[str, Any]]:
    rows = []
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    for item in rois:
        rgb_region = crop(frame_rgb, item.roi)
        gray_region = crop(gray, item.roi)
        if rgb_region.size == 0 or gray_region.size == 0:
            continue
        mean_rgb = rgb_region.reshape(-1, 3).mean(axis=0)
        rows.append(
            {
                "frame_index": frame_index,
                "timestamp_s": timestamp_s,
                "roi": item.name,
                "mean_r": float(mean_rgb[0]),
                "mean_g": float(mean_rgb[1]),
                "mean_b": float(mean_rgb[2]),
                "mean_intensity": float(gray_region.reshape(-1).mean()),
                "std_intensity": float(gray_region.reshape(-1).std()),
                "x": int(item.roi.x),
                "y": int(item.roi.y),
                "w": int(item.roi.w),
                "h": int(item.roi.h),
            }
        )
    return rows


def extract_ubfc_video(archive: Path, member: str, condition: str) -> tuple[Path, dict[str, Any]]:
    TEMP.mkdir(parents=True, exist_ok=True)
    extract_root = TEMP / condition
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        extracted = Path(zf.extract(member, extract_root))
    return extracted, {"temp_video": extracted.as_posix(), "member": member}


def process_ubfc_video(video_path: Path, condition: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or UBFC_SOURCE_FPS_DEFAULT)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    first_rgb = None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        first_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        break
    if first_rgb is None:
        cap.release()
        raise RuntimeError(f"No frames in video {video_path}")
    rois, roi_method = rois_from_face_or_fallback(first_rgb)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    rows: list[dict[str, Any]] = []
    frame_idx = 0
    accepted = 0
    while accepted < UBFC_MAX_ACCEPTED_FRAMES:
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_idx % UBFC_FRAME_STRIDE == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rows.extend(roi_feature_row(rgb, frame_idx, frame_idx / source_fps, rois))
            accepted += 1
        frame_idx += 1
    cap.release()
    df = pd.DataFrame(rows)
    meta = {
        "condition_id": condition,
        "source_fps": source_fps,
        "effective_fps": source_fps / max(1, UBFC_FRAME_STRIDE),
        "total_frames_reported": total_frames,
        "read_until_frame": frame_idx,
        "accepted_frames": accepted,
        "roi_method": roi_method,
        "n_rois": len(rois),
        "frame_stride": UBFC_FRAME_STRIDE,
        "max_accepted_frames": UBFC_MAX_ACCEPTED_FRAMES,
    }
    return df, meta


def zip_pgm_members(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return sorted([m for m in zf.namelist() if m.lower().endswith(".pgm")])


def process_mr_zip(zip_path: Path, condition: str, modality: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    members = zip_pgm_members(zip_path)
    selected = members[::MR_FRAME_STRIDE][:MR_MAX_ACCEPTED_FRAMES]
    rows: list[dict[str, Any]] = []
    roi_method = "not_initialized"
    rois: list[SimpleROI] | None = None
    with zipfile.ZipFile(zip_path) as zf:
        for sample_idx, member in enumerate(selected):
            payload = np.frombuffer(zf.read(member), dtype=np.uint8)
            frame = cv2.imdecode(payload, cv2.IMREAD_UNCHANGED)
            if frame is None:
                continue
            rgb = frame_to_rgb_uint8(frame)
            if rois is None:
                rois, roi_method = rois_from_face_or_fallback(rgb)
            frame_idx = sample_idx * MR_FRAME_STRIDE
            rows.extend(roi_feature_row(rgb, frame_idx, frame_idx / MR_SOURCE_FPS, rois))
    df = pd.DataFrame(rows)
    meta = {
        "condition_id": condition,
        "modality": modality,
        "source_fps": MR_SOURCE_FPS,
        "effective_fps": MR_SOURCE_FPS / max(1, MR_FRAME_STRIDE),
        "n_members_total": len(members),
        "accepted_frames": len(selected),
        "roi_method": roi_method,
        "n_rois": len(rois or []),
        "frame_stride": MR_FRAME_STRIDE,
        "max_accepted_frames": MR_MAX_ACCEPTED_FRAMES,
    }
    return df, meta


def write_trace(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def trace_quality_rows(dataset: str, condition: str, modality: str, trace_path: Path, trace_df: pd.DataFrame, meta: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for roi, group in trace_df.groupby("roi"):
        duration = float(group["timestamp_s"].max() - group["timestamp_s"].min()) if len(group) > 1 else 0.0
        for signal_col in ["mean_intensity", "mean_r", "mean_g", "mean_b"]:
            values = group[signal_col].to_numpy(dtype=float)
            mean = float(np.nanmean(values)) if values.size else math.nan
            std = float(np.nanstd(values)) if values.size else math.nan
            rows.append(
                {
                    "dataset": dataset,
                    "condition_id": condition,
                    "modality": modality,
                    "roi": roi,
                    "signal": signal_col,
                    "n_samples": int(values.size),
                    "duration_sec": duration,
                    "mean": mean,
                    "std": std,
                    "ac_ratio": float(std / (abs(mean) + 1e-9)) if math.isfinite(mean) else math.nan,
                    "roi_method": meta.get("roi_method", ""),
                    "effective_fps": meta.get("effective_fps", math.nan),
                    "trace_path": rel(trace_path),
                }
            )
    return rows


def update_docs(index_df: pd.DataFrame, quality_df: pd.DataFrame, gates: pd.DataFrame, summary: dict[str, Any]) -> None:
    DOC_MD.write_text(
        "\n".join(
            [
                "# T493 Selected-Domain ROI Trace Cache",
                "",
                "## Purpose",
                "",
                "T493 implements the escalation route required by T492: when full-frame RGB/NIR traces are artifact-prone, extract selected-condition ROI traces from raw video/PGM streams without full dataset extraction.",
                "",
                "## Result",
                "",
                f"- Decision: `{summary['decision']}`",
                f"- ROI trace files: {summary['n_trace_files']}",
                f"- UBFC conditions cached: {summary['n_ubfc_conditions_cached']}",
                f"- MR conditions cached: {summary['n_mr_conditions_cached']}",
                f"- Disk after task: {summary['disk_free_gib_after']:.3f} GiB free",
                "",
                "## Key Insight",
                "",
                summary["main_insight"],
                "",
                "## Trace Index",
                "",
                markdown_table(index_df),
                "",
                "## Quality Summary Preview",
                "",
                markdown_table(quality_df[["dataset", "condition_id", "modality", "roi", "signal", "n_samples", "duration_sec", "ac_ratio", "roi_method"]]),
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

    marker = "<!-- T493_SELECTED_DOMAIN_ROI_TRACE_CACHE -->"
    block = "\n".join(
        [
            marker,
            f"## T493 Selected-Domain ROI Trace Cache ({date.today().isoformat()})",
            "",
            f"- Decision: `{summary['decision']}`.",
            f"- Cached {summary['n_trace_files']} ROI trace files: UBFC={summary['n_ubfc_conditions_cached']} conditions, MR={summary['n_mr_conditions_cached']} conditions.",
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
            f"## T493 教学记录：Selected-Domain ROI Trace Cache ({date.today().isoformat()})",
            "",
            "### 目的",
            "",
            "T492 发现 full-frame trace 在 MR-NIRP selected subset 上会被光照、背景和 Nyquist/alternating artifact 严重误导。因此 T493 的目的不是直接训练大模型，而是先把真实视频/PGM 中的 ROI-level time-series 提取出来，形成后续 T494 可评估的候选输入。",
            "",
            "### 实现",
            "",
            "UBFC-Phys 的 AVI 被从 zip 中逐个临时抽出，处理完成后删除，避免一次性占满磁盘。MR-NIRP 的 RGB/NIR PGM 直接从 zip 流式读取，不全量解压。ROI route 使用 OpenCV Haar face detection；若检测失败，则退回 deterministic face-like ROI。MediaPipe 在当前远端 Python 环境暂不可用，因此本任务明确记录为环境缺口，不把 fallback 等同于最终 Face Mesh。",
            "",
            "### 得到的结果",
            "",
            f"生成 {summary['n_trace_files']} 个 ROI trace 文件，覆盖 UBFC-Phys {summary['n_ubfc_conditions_cached']} 个 condition 和 MR-NIRP {summary['n_mr_conditions_cached']} 个 condition。后续 T494 会用这些 ROI trace 和 T492 full-frame failure 做直接对比。",
            "",
            "### Insight",
            "",
            summary["main_insight"],
            "",
        ]
    )
    append_or_replace(LEARNING_JOURNAL, marker, learning_block)


def disk_free_gib(path: Path) -> float:
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)


def main() -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    subset = pd.read_csv(SUBSET)
    mr_zip = pd.read_csv(MR_ZIP_AUDIT)
    t485 = read_json(T485_SUMMARY)
    t491 = read_json(T491_SUMMARY)
    t492 = read_json(T492_SUMMARY)
    ubfc_archive = Path(str(t485["archive"]))

    index_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    temp_removed = True

    selected_ubfc = subset[subset["source_dataset"].astype(str).eq("UBFC-Phys-S1-S14")]
    for _, row in selected_ubfc.iterrows():
        condition = str(row["condition_id"])
        member = str(row["video_member"])
        extracted, extract_meta = extract_ubfc_video(ubfc_archive, member, condition)
        try:
            trace_df, meta = process_ubfc_video(extracted, condition)
            meta.update(extract_meta)
            out_path = RUNTIME / "ubfc_phys" / condition / "roi_trace.csv"
            write_trace(trace_df, out_path)
            meta_path = out_path.with_name("roi_trace_meta.json")
            write_json(meta_path, meta)
            index_rows.append(
                {
                    "dataset": "UBFC-Phys-S1-S14",
                    "condition_id": condition,
                    "modality": "RGB_video",
                    "trace_path": rel(out_path),
                    "meta_path": rel(meta_path),
                    "n_rows": int(len(trace_df)),
                    "n_rois": int(trace_df["roi"].nunique()) if not trace_df.empty else 0,
                    "accepted_frames": int(meta["accepted_frames"]),
                    "roi_method": meta["roi_method"],
                    "source_access": "zip_member_temp_extract_then_delete",
                }
            )
            quality_rows.extend(trace_quality_rows("UBFC-Phys-S1-S14", condition, "RGB_video", out_path, trace_df, meta))
        finally:
            condition_temp = TEMP / condition
            if condition_temp.exists():
                shutil.rmtree(condition_temp, ignore_errors=True)
            temp_removed = temp_removed and not condition_temp.exists()

    selected_mr = subset[subset["source_dataset"].astype(str).eq("MR-NIRP")]
    for _, row in selected_mr.iterrows():
        condition = str(row["condition_id"])
        condition_zips = mr_zip[mr_zip["condition_id"].astype(str).eq(condition)]
        for modality in ["RGB", "NIR"]:
            match = condition_zips[condition_zips["modality"].astype(str).eq(modality)]
            if match.empty:
                continue
            zip_path = Path(str(match.iloc[0]["zip_path"]))
            trace_df, meta = process_mr_zip(zip_path, condition, modality)
            out_path = RUNTIME / "mr_nirp" / condition / f"{modality.lower()}_roi_trace.csv"
            write_trace(trace_df, out_path)
            meta_path = out_path.with_name(f"{modality.lower()}_roi_trace_meta.json")
            write_json(meta_path, meta)
            index_rows.append(
                {
                    "dataset": "MR-NIRP",
                    "condition_id": condition,
                    "modality": modality,
                    "trace_path": rel(out_path),
                    "meta_path": rel(meta_path),
                    "n_rows": int(len(trace_df)),
                    "n_rois": int(trace_df["roi"].nunique()) if not trace_df.empty else 0,
                    "accepted_frames": int(meta["accepted_frames"]),
                    "roi_method": meta["roi_method"],
                    "source_access": "zip_stream_pgm_no_bulk_extract",
                }
            )
            quality_rows.extend(trace_quality_rows("MR-NIRP", condition, modality, out_path, trace_df, meta))

    index_df = pd.DataFrame(index_rows)
    quality_df = pd.DataFrame(quality_rows)
    index_df.to_csv(TRACE_INDEX_CSV, index=False, encoding="utf-8-sig")
    quality_df.to_csv(QUALITY_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    n_ubfc = int(index_df[index_df["dataset"].eq("UBFC-Phys-S1-S14")]["condition_id"].nunique()) if not index_df.empty else 0
    n_mr = int(index_df[index_df["dataset"].eq("MR-NIRP")]["condition_id"].nunique()) if not index_df.empty else 0
    disk_after = disk_free_gib(ROOT)
    mediapipe_available = False
    try:
        import mediapipe  # type: ignore  # noqa: F401

        mediapipe_available = True
    except Exception:
        mediapipe_available = False

    gates = pd.DataFrame(
        [
            {
                "gate": "ubfc_selected_video_roi_cached",
                "passed": n_ubfc >= 3,
                "evidence": f"ubfc_conditions={n_ubfc}",
                "claim_allowed": "Selected UBFC-Phys video ROI traces are ready for T494.",
                "claim_not_allowed": "Full UBFC-Phys S1-S14 video evaluation complete.",
            },
            {
                "gate": "mr_selected_rgb_nir_roi_cached",
                "passed": n_mr >= 4 and int((index_df["dataset"].eq("MR-NIRP")).sum()) >= 8,
                "evidence": f"mr_conditions={n_mr}, mr_trace_files={int((index_df['dataset'].eq('MR-NIRP')).sum())}",
                "claim_allowed": "Selected MR-NIRP RGB/NIR ROI traces are ready for T494.",
                "claim_not_allowed": "Low-light robustness proven.",
            },
            {
                "gate": "bulk_video_temp_removed",
                "passed": temp_removed,
                "evidence": f"temp_removed={temp_removed}",
                "claim_allowed": "Disk-safe selective extraction route.",
                "claim_not_allowed": "All raw videos retained locally.",
            },
            {
                "gate": "mediapipe_environment_recorded",
                "passed": True,
                "evidence": f"mediapipe_available={mediapipe_available}",
                "claim_allowed": "Current T493 ROI route is explicitly labeled as Haar/fallback unless MediaPipe is available.",
                "claim_not_allowed": "Final MediaPipe Face Mesh route validated on AutoDL.",
            },
        ]
    )
    gates.to_csv(CLAIM_GATE_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "task_id": TASK_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "selected_domain_roi_trace_cache_ready" if bool(gates["passed"].all()) else "selected_domain_roi_trace_cache_blocked",
        "all_gates_passed": bool(gates["passed"].all()),
        "n_trace_files": int(len(index_df)),
        "n_quality_rows": int(len(quality_df)),
        "n_ubfc_conditions_cached": n_ubfc,
        "n_mr_conditions_cached": n_mr,
        "ubfc_frame_stride": UBFC_FRAME_STRIDE,
        "ubfc_max_accepted_frames": UBFC_MAX_ACCEPTED_FRAMES,
        "mr_frame_stride": MR_FRAME_STRIDE,
        "mr_max_accepted_frames": MR_MAX_ACCEPTED_FRAMES,
        "mediapipe_available": mediapipe_available,
        "disk_free_gib_after": disk_after,
        "t491_decision": t491.get("decision"),
        "t492_decision": t492.get("decision"),
        "main_insight": "The pipeline has now moved from full-frame artifact detection into selected ROI evidence extraction. This preserves the T492 safety insight while creating the concrete ROI-level inputs needed to test whether ROI/deep candidate selection can recover useful signals instead of merely refusing output.",
        "claim_boundary": "T493 prepares ROI trace inputs only. It does not yet prove HR accuracy, SOTA performance, MediaPipe Face Mesh validity, low-light robustness, or clinical readiness.",
        "next_recommended_tasks": [
            "T494 evaluate ROI-level candidates against UBFC BVP and MR PulseOx references.",
            "T495 compare ROI/deep route against T492 full-frame failure and update product/paper claim gates.",
        ],
    }
    write_json(SUMMARY_JSON, summary)
    update_docs(index_df, quality_df, gates, summary)
    replace_evidence_row(
        {
            "evidence_id": "t493_selected_domain_roi_trace_cache",
            "task_id": TASK_ID,
            "date": date.today().isoformat(),
            "artifact": rel(SUMMARY_JSON),
            "metric_or_observation": "Selected-domain ROI trace-cache readiness",
            "result": f"trace_files={len(index_df)}; UBFC={n_ubfc}; MR={n_mr}; mediapipe={mediapipe_available}",
            "claim_supported": "External-domain selected conditions now have ROI-level traces for candidate evaluation.",
            "claim_boundary": summary["claim_boundary"],
            "next_action": "; ".join(summary["next_recommended_tasks"]),
        }
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
