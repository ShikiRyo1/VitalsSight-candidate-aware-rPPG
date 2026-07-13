from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import pandas as pd


SCHEMA_VERSION = "vitalssight.console.case.v1"
REPORT_VERSION = "vitalssight.evidence-report.v1"
POLICY_VERSION = "public_candidate_release_gate.v1"
MODEL_VERSION = "public_research_artifact.2026-07"
CLAIM_BOUNDARY = (
    "Retrospective research workflow only. The output is not a diagnosis, emergency alert, "
    "medical-device decision, or validated autonomous clinical release."
)
ATTRIBUTION_BOUNDARY = (
    "This panel explains observed evidence and policy rules. It is not a causal explanation, "
    "a calibrated probability of safety, or a clinical rationale."
)

CONSOLE_TEXT_ZH = {
    CLAIM_BOUNDARY: "仅用于回顾性研究流程。本输出不是诊断、急救告警、医疗器械决定，也不是经验证的临床自主放行。",
    ATTRIBUTION_BOUNDARY: "本面板说明观测证据和策略规则，不构成因果解释、安全概率或临床理由。",
    "Synthetic candidate-release demonstration": "合成候选-放行流程演示",
    "Synthetic workflow demonstration; values are not manuscript metrics.": "合成工作流演示；其中数值不是论文实验指标。",
    "Retrospective protocol-bound research evidence; no clinical, fairness, universal-generalisation or participant-level risk guarantee.": "仅代表回顾性协议限定的研究证据；不提供临床、公平性、普遍泛化或参与者个体风险保证。",
    "Retain the evidence packet with the reported estimate.": "保留已发布估计及其证据包。",
    "Motion and cross-route disagreement prevent automatic reporting.": "运动和跨路径分歧阻止自动发布。",
    "Repeat the recording while the participant remains still.": "请在受试者保持静止时重新采集。",
    "Illumination and candidate count are below the acquisition gate.": "光照和候选数量未达到采集门槛。",
    "Move to even front lighting and record at least 20 seconds again.": "调整为均匀正面光照，并重新录制至少 20 秒。",
    "A plausible half-rate branch remains unresolved in the candidate pool.": "候选池中仍存在未消解的合理半频分支。",
    "Inspect the competing 72 and 144 BPM branches or repeat the recording.": "检查相互竞争的 72 与 144 BPM 分支，或重新采集。",
    "The face was not visible for enough of the recording.": "录制期间人脸可见时长不足。",
    "Center the full face inside the frame and record again.": "将完整人脸置于画面中央后重新录制。",
    "Correct the listed acquisition issue and run a new recording.": "修正列出的采集问题后重新录制。",
    "Inspect candidate evidence or repeat the recording.": "检查候选证据，或重新采集。",
    "Inspect the evidence packet before proceeding.": "继续前请检查证据包。",
    "No action required.": "无需操作。",
    "Record at least 8 seconds; 20-30 seconds is preferred.": "至少录制 8 秒，建议 20-30 秒。",
    "Use a camera or file with at least 15 fps.": "使用帧率至少为 15 fps 的相机或文件。",
    "Use at least 320x240 video; 480p or higher is preferred.": "至少使用 320x240 视频，建议 480p 或更高。",
    "Use even front lighting and avoid deep shadow or saturation.": "使用均匀正面光照，避免深阴影或过曝。",
    "Keep the head and camera steady during capture.": "采集期间保持头部和相机稳定。",
    "Center the full face and remove major occlusion.": "将完整人脸置于中央并移除明显遮挡。",
    "Face-quality inspection is unavailable; reinstall the detector asset before analysis.": "人脸质量检查不可用；分析前请恢复检测器资源。",
    "Face visibility is below the evidence threshold.": "人脸可见比例低于证据阈值。",
    "Illumination quality is below the evidence threshold.": "光照质量低于证据阈值。",
    "Motion may contaminate regional traces.": "运动可能污染区域信号。",
    "Overall acquisition quality requires attention.": "总体采集质量需要处理。",
    "Too few candidates were retained for stable comparison.": "保留的候选过少，无法进行稳定比较。",
    "The analysis pipeline could not complete in the current runtime.": "当前运行环境未能完成分析流程。",
    "Retain the quality evidence and route this case to technical review; do not report HR.": "保留质量证据并转交技术复核；不得发布心率。",
    "Pipeline execution failed after quality qualification; no HR was released.": "质量检查后分析流程执行失败；未发布心率。",
    "file readability": "文件可读性",
    "Choose a supported video file that can be decoded.": "请选择可正常解码的受支持视频文件。",
    "duration": "时长",
    "frame rate": "帧率",
    "resolution": "分辨率",
    "illumination": "光照",
    "motion": "运动",
    "face visibility": "人脸可见性",
    "overall quality": "总体质量",
    "candidate sufficiency": "候选充分性",
    "fraction": "比例",
    "score": "分数",
    "risk score": "风险分数",
    "px short edge": "短边像素",
    "luma": "亮度",
    "mean frame delta": "平均帧差",
    "count": "个",
    "Derived from stored quality evidence; no video duration or frame metadata is implied.": "该面板由已存质量证据派生，不代表真实视频时长或帧元数据。",
    "supports release": "支持放行",
    "supports review": "支持复核",
    "Face visibility": "人脸可见性",
    "Illumination": "光照",
    "Motion": "运动",
    "Candidate agreement": "候选一致性",
    "Harmonic ambiguity": "谐波歧义",
    "Selector support": "选择器支持度",
    "Sufficient visible-face coverage is required before candidate evidence is interpreted.": "解释候选证据前，需要足够的人脸可见覆盖。",
    "Very dark or saturated frames weaken the recoverable color signal.": "过暗或饱和画面会削弱可恢复的颜色信号。",
    "Large frame-to-frame motion can move ROI traces away from physiological variation.": "较大的帧间运动会使 ROI 信号偏离生理变化。",
    "Agreement across routes and regions supports one branch without using reference HR.": "跨路径与区域的一致性可在不使用参考心率的情况下支持某一分支。",
    "A strong half-rate or double-rate alternative keeps the output in review.": "较强的半频或倍频备选会使输出保持在复核状态。",
    "The configured selector score summarizes candidate support under the documented policy.": "配置的选择器分数概括了既定策略下的候选支持度。",
    "pass": "通过",
    "warn": "警告",
    "fail": "失败",
    "release": "放行",
    "review": "复核",
    "retake": "重采",
    "withheld": "未发布",
    "not opened": "未创建",
    "high": "高",
    "urgent": "紧急",
    "routine": "常规",
    "low": "低",
    "open": "待处理",
    "in_review": "复核中",
    "waiting_retake": "等待重采",
    "closed": "已关闭",
    "open work": "未关闭工作",
    "all": "全部",
    "request_retake": "请求重采",
    "retain_for_research_review": "保留用于研究复核",
    "close_without_release": "关闭且不放行",
}


def localize_console_text(value: object, *, language: str = "en") -> str:
    text = "" if value is None else str(value)
    if not language.lower().startswith("zh"):
        return text
    direct = CONSOLE_TEXT_ZH.get(text)
    if direct is not None:
        return direct

    # Preflight can join several complete action sentences into one reason.
    # Replace only sentence-like entries to avoid altering identifiers or units.
    localized = text
    for source, target in sorted(CONSOLE_TEXT_ZH.items(), key=lambda item: len(item[0]), reverse=True):
        if len(source) >= 20 and source in localized:
            localized = localized.replace(source, target)
    return localized.replace("。; ", "；")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def finite_float(value: object, default: float | None = None) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if math.isfinite(output) else default


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _case_id(prefix: str = "case") -> str:
    return f"{prefix}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"


def _candidate(
    candidate_id: str,
    bpm: float,
    *,
    method: str,
    region: str,
    support: int,
    score: float,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "candidate_bpm": bpm,
        "method": method,
        "region": region,
        "support": support,
        "score": score,
    }


def make_demo_cases() -> list[dict[str, Any]]:
    """Return deterministic, explicitly synthetic cases for product-flow testing."""

    now = utc_now()
    base = {
        "schema_version": SCHEMA_VERSION,
        "input_kind": "built_in_demo",
        "source_name": "Synthetic candidate-release demonstration",
        "purpose": "workflow_validation",
        "consent_recorded": True,
        "retention_policy": "session_only",
        "policy_version": POLICY_VERSION,
        "model_version": MODEL_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_scope": "Synthetic workflow demonstration; values are not manuscript metrics.",
        "created_at": now,
        "updated_at": now,
    }
    rows = [
        {
            **base,
            "case_id": "demo_stable_consensus",
            "display_id": "VS-001",
            "decision": "release",
            "priority": "routine",
            "released_hr_bpm": 72.5,
            "selected_candidate_hr_bpm": 72.5,
            "confidence": 0.88,
            "quality_score": 0.91,
            "snr_db": 15.8,
            "face_coverage": 0.96,
            "illumination_score": 0.90,
            "motion_score": 0.09,
            "candidate_count": 5,
            "agreement_fraction": 0.82,
            "harmonic_risk": 0.08,
            "selected_method": "POS + CHROM consensus",
            "selected_region": "forehead + cheeks",
            "review_reason": "",
            "recommended_action": "Retain the evidence packet with the reported estimate.",
            "trend_bpm": [71.8, 72.1, 72.5, 72.7, 72.4],
            "candidates": [
                _candidate("c1", 72.0, method="POS", region="forehead", support=4, score=0.88),
                _candidate("c2", 73.0, method="CHROM", region="left_cheek", support=3, score=0.82),
                _candidate("c3", 109.0, method="GREEN", region="right_cheek", support=1, score=0.21),
            ],
        },
        {
            **base,
            "case_id": "demo_motion_conflict",
            "display_id": "VS-002",
            "decision": "review",
            "priority": "high",
            "released_hr_bpm": None,
            "selected_candidate_hr_bpm": 111.0,
            "confidence": 0.46,
            "quality_score": 0.58,
            "snr_db": 7.9,
            "face_coverage": 0.88,
            "illumination_score": 0.74,
            "motion_score": 0.61,
            "candidate_count": 6,
            "agreement_fraction": 0.31,
            "harmonic_risk": 0.42,
            "selected_method": "candidate pool",
            "selected_region": "mixed ROI",
            "review_reason": "Motion and cross-route disagreement prevent automatic reporting.",
            "recommended_action": "Repeat the recording while the participant remains still.",
            "trend_bpm": [74.0, 82.0, 111.0, 67.0, 96.0],
            "candidates": [
                _candidate("c1", 61.0, method="POS", region="forehead", support=2, score=0.52),
                _candidate("c2", 111.0, method="CHROM", region="left_cheek", support=1, score=0.46),
                _candidate("c3", 74.0, method="GREEN", region="right_cheek", support=2, score=0.49),
            ],
        },
        {
            **base,
            "case_id": "demo_low_light_retake",
            "display_id": "VS-003",
            "decision": "retake",
            "priority": "routine",
            "released_hr_bpm": None,
            "selected_candidate_hr_bpm": None,
            "confidence": 0.20,
            "quality_score": 0.28,
            "snr_db": 2.7,
            "face_coverage": 0.63,
            "illumination_score": 0.18,
            "motion_score": 0.22,
            "candidate_count": 1,
            "agreement_fraction": 0.0,
            "harmonic_risk": 0.64,
            "selected_method": "none",
            "selected_region": "none",
            "review_reason": "Illumination and candidate count are below the acquisition gate.",
            "recommended_action": "Move to even front lighting and record at least 20 seconds again.",
            "trend_bpm": [],
            "candidates": [],
        },
        {
            **base,
            "case_id": "demo_harmonic_review",
            "display_id": "VS-004",
            "decision": "review",
            "priority": "high",
            "released_hr_bpm": None,
            "selected_candidate_hr_bpm": 144.0,
            "confidence": 0.63,
            "quality_score": 0.77,
            "snr_db": 13.1,
            "face_coverage": 0.94,
            "illumination_score": 0.87,
            "motion_score": 0.14,
            "candidate_count": 7,
            "agreement_fraction": 0.48,
            "harmonic_risk": 0.79,
            "selected_method": "CHROM",
            "selected_region": "forehead",
            "review_reason": "A plausible half-rate branch remains unresolved in the candidate pool.",
            "recommended_action": "Inspect the competing 72 and 144 BPM branches or repeat the recording.",
            "trend_bpm": [142.0, 144.0, 73.0, 145.0],
            "candidates": [
                _candidate("c1", 144.0, method="CHROM", region="forehead", support=3, score=0.63),
                _candidate("c2", 72.0, method="POS", region="cheeks", support=3, score=0.61),
                _candidate("c3", 96.0, method="GREEN", region="forehead", support=1, score=0.28),
            ],
        },
        {
            **base,
            "case_id": "demo_second_release",
            "display_id": "VS-005",
            "decision": "release",
            "priority": "routine",
            "released_hr_bpm": 84.1,
            "selected_candidate_hr_bpm": 84.1,
            "confidence": 0.81,
            "quality_score": 0.86,
            "snr_db": 14.2,
            "face_coverage": 0.93,
            "illumination_score": 0.82,
            "motion_score": 0.12,
            "candidate_count": 6,
            "agreement_fraction": 0.76,
            "harmonic_risk": 0.12,
            "selected_method": "multi-ROI consensus",
            "selected_region": "semantic ROI set",
            "review_reason": "",
            "recommended_action": "Retain the evidence packet with the reported estimate.",
            "trend_bpm": [82.8, 83.5, 84.1, 84.5, 84.0],
            "candidates": [
                _candidate("c1", 84.0, method="POS", region="forehead", support=4, score=0.81),
                _candidate("c2", 85.0, method="CHROM", region="cheeks", support=3, score=0.75),
            ],
        },
        {
            **base,
            "case_id": "demo_face_missing",
            "display_id": "VS-006",
            "decision": "retake",
            "priority": "low",
            "released_hr_bpm": None,
            "selected_candidate_hr_bpm": None,
            "confidence": 0.0,
            "quality_score": 0.08,
            "snr_db": None,
            "face_coverage": 0.18,
            "illumination_score": 0.78,
            "motion_score": 0.16,
            "candidate_count": 0,
            "agreement_fraction": 0.0,
            "harmonic_risk": None,
            "selected_method": "none",
            "selected_region": "none",
            "review_reason": "The face was not visible for enough of the recording.",
            "recommended_action": "Center the full face inside the frame and record again.",
            "trend_bpm": [],
            "candidates": [],
        },
    ]
    return [ensure_output_contract(row) for row in rows]


def ensure_output_contract(case: dict[str, Any]) -> dict[str, Any]:
    """Normalize one case and prevent HR publication on non-release states."""

    output = json_safe(dict(case))
    output.setdefault("schema_version", SCHEMA_VERSION)
    output.setdefault("case_id", _case_id())
    output.setdefault("display_id", str(output["case_id"]).upper())
    output.setdefault("decision", "review")
    output.setdefault("priority", "routine")
    output.setdefault("created_at", utc_now())
    output["updated_at"] = utc_now()
    output.setdefault("claim_boundary", CLAIM_BOUNDARY)
    output.setdefault("policy_version", POLICY_VERSION)
    output.setdefault("model_version", MODEL_VERSION)
    output.setdefault("candidates", [])
    output.setdefault("trend_bpm", [])
    output.setdefault("recommended_action", "Inspect the evidence packet before proceeding.")
    decision = str(output["decision"]).lower()
    if decision not in {"release", "review", "retake"}:
        raise ValueError(f"Unsupported decision: {decision}")
    output["decision"] = decision
    if decision != "release":
        output["released_hr_bpm"] = None
    elif finite_float(output.get("released_hr_bpm")) is None:
        raise ValueError("A release decision requires a finite released_hr_bpm value")
    return output


def build_attribution(case: dict[str, Any]) -> dict[str, Any]:
    """Explain evidence and gate behavior without claiming causal attribution."""

    case = ensure_output_contract(case)
    factors: list[dict[str, Any]] = []

    def add(
        factor: str,
        value: float | None,
        unit: str,
        *,
        positive: bool,
        reason: str,
        source_field: str,
    ) -> None:
        status = "supports release" if positive else "supports review"
        factors.append(
            {
                "factor": factor,
                "observed": None if value is None else round(value, 3),
                "unit": unit,
                "status": status,
                "direction": 1 if positive else -1,
                "reason": reason,
                "source_field": source_field,
            }
        )

    face = finite_float(case.get("face_coverage"))
    add(
        "Face visibility",
        face,
        "fraction",
        positive=face is not None and face >= 0.70,
        reason="Sufficient visible-face coverage is required before candidate evidence is interpreted.",
        source_field="face_coverage",
    )
    illumination = finite_float(case.get("illumination_score"))
    add(
        "Illumination",
        illumination,
        "score",
        positive=illumination is not None and illumination >= 0.55,
        reason="Very dark or saturated frames weaken the recoverable color signal.",
        source_field="illumination_score",
    )
    motion = finite_float(case.get("motion_score"))
    add(
        "Motion",
        motion,
        "score",
        positive=motion is not None and motion <= 0.35,
        reason="Large frame-to-frame motion can move ROI traces away from physiological variation.",
        source_field="motion_score",
    )
    agreement = finite_float(case.get("agreement_fraction"))
    add(
        "Candidate agreement",
        agreement,
        "fraction",
        positive=agreement is not None and agreement >= 0.60,
        reason="Agreement across routes and regions supports one branch without using reference HR.",
        source_field="agreement_fraction",
    )
    harmonic = finite_float(case.get("harmonic_risk"))
    add(
        "Harmonic ambiguity",
        harmonic,
        "risk score",
        positive=harmonic is not None and harmonic <= 0.35,
        reason="A strong half-rate or double-rate alternative keeps the output in review.",
        source_field="harmonic_risk",
    )
    confidence = finite_float(case.get("confidence"))
    add(
        "Selector support",
        confidence,
        "score",
        positive=confidence is not None and confidence >= 0.60,
        reason="The configured selector score summarizes candidate support under the documented policy.",
        source_field="confidence",
    )

    factors.sort(key=lambda item: (item["direction"], item["factor"]))
    negative = [item for item in factors if item["direction"] < 0]
    positive = [item for item in factors if item["direction"] > 0]
    return {
        "attribution_type": "evidence_and_policy_attribution",
        "decision": case["decision"],
        "primary_review_drivers": negative[:3],
        "primary_release_support": positive[:3],
        "all_factors": factors,
        "boundary": ATTRIBUTION_BOUNDARY,
    }


def video_preflight(path: str | Path, *, sample_frames: int = 48) -> dict[str, Any]:
    """Inspect video quality before running the slower rPPG pipeline."""

    import cv2

    video_path = Path(path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    sample_indices = np.linspace(0, max(frame_count - 1, 0), num=max(1, min(sample_frames, frame_count)), dtype=int)
    brightness: list[float] = []
    motion: list[float] = []
    faces = 0
    previous = None
    cascade_name = "haarcascade_frontalface_default.xml"
    cascade_candidates = [
        Path(getattr(cv2.data, "haarcascades", "")) / cascade_name,
        Path(__file__).resolve().parent / "assets" / cascade_name,
    ]
    detector = None
    detector_source = None
    for cascade_path in cascade_candidates:
        if not cascade_path.is_file():
            continue
        candidate = cv2.CascadeClassifier(str(cascade_path))
        if not candidate.empty():
            detector = candidate
            detector_source = str(cascade_path)
            break
    try:
        for index in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append(float(gray.mean()))
            small = cv2.resize(gray, (96, 72), interpolation=cv2.INTER_AREA)
            if previous is not None:
                motion.append(float(np.mean(cv2.absdiff(small, previous))))
            previous = small
            if detector is not None:
                detected = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
                faces += int(len(detected) > 0)
    finally:
        capture.release()

    sampled = max(1, len(brightness))
    brightness_mean = float(np.mean(brightness)) if brightness else 0.0
    motion_mean = float(np.mean(motion)) if motion else 0.0
    face_rate = faces / sampled

    checks = [
        _quality_check("duration", duration, "s", pass_if=duration >= 20, warn_if=duration >= 8, fail_message="Record at least 8 seconds; 20-30 seconds is preferred."),
        _quality_check("frame rate", fps, "fps", pass_if=fps >= 20, warn_if=fps >= 15, fail_message="Use a camera or file with at least 15 fps."),
        _quality_check("resolution", min(width, height), "px short edge", pass_if=min(width, height) >= 480, warn_if=min(width, height) >= 240, fail_message="Use at least 320x240 video; 480p or higher is preferred."),
        _quality_check("illumination", brightness_mean, "luma", pass_if=45 <= brightness_mean <= 210, warn_if=25 <= brightness_mean <= 235, fail_message="Use even front lighting and avoid deep shadow or saturation."),
        _quality_check("motion", motion_mean, "mean frame delta", pass_if=motion_mean <= 10, warn_if=motion_mean <= 22, fail_message="Keep the head and camera steady during capture.", reverse=True),
        _quality_check(
            "face visibility",
            face_rate,
            "fraction",
            pass_if=detector is not None and face_rate >= 0.70,
            warn_if=detector is not None and face_rate >= 0.35,
            fail_message=(
                "Center the full face and remove major occlusion."
                if detector is not None
                else "Face-quality inspection is unavailable; reinstall the detector asset before analysis."
            ),
        ),
    ]
    statuses = {item["status"] for item in checks}
    overall = "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "pass")
    return {
        "file_name": video_path.name,
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": round(duration, 3),
        "brightness_mean": round(brightness_mean, 3),
        "motion_mean": round(motion_mean, 3),
        "face_detection_rate": round(face_rate, 3),
        "face_detector_available": detector is not None,
        "face_detector_source": detector_source,
        "overall": overall,
        "checks": checks,
    }


def _quality_check(
    name: str,
    value: float,
    unit: str,
    *,
    pass_if: bool,
    warn_if: bool,
    fail_message: str,
    reverse: bool = False,
) -> dict[str, Any]:
    del reverse
    status = "pass" if pass_if else ("warn" if warn_if else "fail")
    return {
        "check": name,
        "value": round(float(value), 3),
        "unit": unit,
        "status": status,
        "action": "No action required." if status == "pass" else fail_message,
    }


def case_quality_snapshot(case: dict[str, Any]) -> dict[str, Any]:
    """Build a quality panel for a stored or synthetic case without inventing video metadata."""

    normalized = ensure_output_contract(case)
    face = finite_float(normalized.get("face_coverage"), 0.0) or 0.0
    illumination = finite_float(normalized.get("illumination_score"), 0.0) or 0.0
    motion = finite_float(normalized.get("motion_score"), 1.0)
    motion = 1.0 if motion is None else motion
    quality = finite_float(normalized.get("quality_score"), 0.0) or 0.0
    candidates = int(finite_float(normalized.get("candidate_count"), 0.0) or 0)
    checks = [
        _quality_check(
            "face visibility",
            face,
            "fraction",
            pass_if=face >= 0.70,
            warn_if=face >= 0.35,
            fail_message="Face visibility is below the evidence threshold.",
        ),
        _quality_check(
            "illumination",
            illumination,
            "score",
            pass_if=illumination >= 0.55,
            warn_if=illumination >= 0.35,
            fail_message="Illumination quality is below the evidence threshold.",
        ),
        _quality_check(
            "motion",
            motion,
            "score",
            pass_if=motion <= 0.35,
            warn_if=motion <= 0.70,
            fail_message="Motion may contaminate regional traces.",
        ),
        _quality_check(
            "overall quality",
            quality,
            "score",
            pass_if=quality >= 0.75,
            warn_if=quality >= 0.45,
            fail_message="Overall acquisition quality requires attention.",
        ),
        _quality_check(
            "candidate sufficiency",
            float(candidates),
            "count",
            pass_if=candidates >= 3,
            warn_if=candidates >= 1,
            fail_message="Too few candidates were retained for stable comparison.",
        ),
    ]
    statuses = {item["status"] for item in checks}
    overall = "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "pass")
    return {
        "source": "stored_case_evidence",
        "overall": overall,
        "checks": checks,
        "note": "Derived from stored quality evidence; no video duration or frame metadata is implied.",
    }


def case_from_preflight(preflight: dict[str, Any], *, purpose: str, retention_policy: str) -> dict[str, Any]:
    overall = str(preflight.get("overall", "fail"))
    decision = "retake" if overall == "fail" else "review"
    failed = [item for item in preflight.get("checks", []) if item.get("status") != "pass"]
    reason = "; ".join(str(item.get("action")) for item in failed[:3]) or "Quality review required."
    brightness = finite_float(preflight.get("brightness_mean"), 0.0) or 0.0
    motion = finite_float(preflight.get("motion_mean"), 100.0) or 100.0
    face = finite_float(preflight.get("face_detection_rate"), 0.0) or 0.0
    quality_score = float(np.clip((face + np.clip(brightness / 128.0, 0, 1) + (1 - np.clip(motion / 30.0, 0, 1))) / 3, 0, 1))
    case = {
        "schema_version": SCHEMA_VERSION,
        "case_id": _case_id("upload"),
        "display_id": f"VS-{datetime.now(UTC).strftime('%H%M%S')}",
        "input_kind": "uploaded_video",
        "source_name": str(preflight.get("file_name", "uploaded video")),
        "purpose": purpose,
        "consent_recorded": True,
        "retention_policy": retention_policy,
        "decision": decision,
        "priority": "routine" if decision == "retake" else "high",
        "released_hr_bpm": None,
        "selected_candidate_hr_bpm": None,
        "confidence": 0.0,
        "quality_score": quality_score,
        "snr_db": None,
        "face_coverage": face,
        "illumination_score": float(np.clip(1.0 - abs(brightness - 128.0) / 128.0, 0, 1)),
        "motion_score": float(np.clip(motion / 30.0, 0, 1)),
        "candidate_count": 0,
        "agreement_fraction": 0.0,
        "harmonic_risk": None,
        "selected_method": "not run",
        "selected_region": "not run",
        "review_reason": reason,
        "recommended_action": "Correct the listed acquisition issue and run a new recording.",
        "trend_bpm": [],
        "candidates": [],
        "preflight": preflight,
        "policy_version": POLICY_VERSION,
        "model_version": MODEL_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_scope": "Acquisition-quality result only; the HR pipeline was not released.",
        "created_at": utc_now(),
    }
    return ensure_output_contract(case)


def preflight_from_decode_error(file_name: str, error: Exception) -> dict[str, Any]:
    """Create a user-facing quality result when a selected file cannot be decoded."""

    return {
        "file_name": Path(file_name).name,
        "fps": 0.0,
        "frame_count": 0,
        "width": 0,
        "height": 0,
        "duration_sec": 0.0,
        "brightness_mean": 0.0,
        "motion_mean": 0.0,
        "face_detection_rate": 0.0,
        "face_detector_available": False,
        "face_detector_source": None,
        "overall": "fail",
        "checks": [
            {
                "check": "file readability",
                "value": 0.0,
                "unit": "status",
                "status": "fail",
                "action": "Choose a supported video file that can be decoded.",
            }
        ],
        "technical_error": {
            "type": type(error).__name__,
            "message": str(error)[:300],
        },
    }


def case_from_runtime_failure(
    preflight: dict[str, Any],
    error: Exception,
    *,
    purpose: str,
    retention_policy: str,
) -> dict[str, Any]:
    """Route a post-quality pipeline failure to review without leaking an HR value."""

    brightness = finite_float(preflight.get("brightness_mean"), 0.0) or 0.0
    motion = finite_float(preflight.get("motion_mean"), 100.0) or 100.0
    face = finite_float(preflight.get("face_detection_rate"), 0.0) or 0.0
    quality_score = float(
        np.clip(
            (face + np.clip(brightness / 128.0, 0, 1) + (1 - np.clip(motion / 30.0, 0, 1))) / 3,
            0,
            1,
        )
    )
    case = {
        "schema_version": SCHEMA_VERSION,
        "case_id": _case_id("runtime_review"),
        "display_id": f"VS-{datetime.now(UTC).strftime('%H%M%S')}",
        "input_kind": "uploaded_video",
        "source_name": preflight.get("file_name", "uploaded_video"),
        "purpose": purpose,
        "consent_recorded": True,
        "retention_policy": retention_policy,
        "decision": "review",
        "priority": "high",
        "released_hr_bpm": None,
        "selected_candidate_hr_bpm": None,
        "confidence": 0.0,
        "quality_score": quality_score,
        "snr_db": None,
        "face_coverage": face,
        "illumination_score": float(np.clip(1.0 - abs(brightness - 128.0) / 128.0, 0, 1)),
        "motion_score": float(np.clip(motion / 30.0, 0, 1)),
        "candidate_count": 0,
        "agreement_fraction": 0.0,
        "harmonic_risk": None,
        "selected_method": "runtime failure before selection",
        "selected_region": "not established",
        "review_reason": "The analysis pipeline could not complete in the current runtime.",
        "recommended_action": "Retain the quality evidence and route this case to technical review; do not report HR.",
        "trend_bpm": [],
        "candidates": [],
        "preflight": preflight,
        "technical_error": {
            "type": type(error).__name__,
            "message": str(error)[:300],
        },
        "policy_version": POLICY_VERSION,
        "model_version": MODEL_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_scope": "Pipeline execution failed after quality qualification; no HR was released.",
        "created_at": utc_now(),
    }
    return ensure_output_contract(case)


def run_uploaded_video(
    path: str | Path,
    *,
    purpose: str,
    retention_policy: str,
    use_mediapipe: bool = True,
    preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run quality preflight and the existing public product pipeline."""

    preflight = preflight or video_preflight(path)
    if preflight["overall"] == "fail":
        return case_from_preflight(preflight, purpose=purpose, retention_policy=retention_policy)

    from src.product.adult_hr_mvp import AdultHRMVPConfig, run_adult_hr_video

    duration = float(preflight["duration_sec"])
    window = min(20.0, max(8.0, duration))
    config = AdultHRMVPConfig(
        seconds=min(30.0, duration),
        window_sec=window,
        step_sec=max(4.0, window / 2),
        frame_stride=2 if float(preflight["fps"]) >= 24 else 1,
        min_window_sec=8.0,
        use_mediapipe=use_mediapipe,
    )
    result = run_adult_hr_video(path, config=config)
    windows = result.windows.sort_values("window_id") if not result.windows.empty else result.windows
    last = windows.iloc[-1].to_dict() if not windows.empty else {}
    decision = str(last.get("decision", "review"))
    released_hr = finite_float(last.get("product_hr_bpm")) if decision == "release" else None
    candidate_hr = finite_float(last.get("candidate_hr_bpm"))
    clusters = result.clusters.copy()
    candidates: list[dict[str, Any]] = []
    if not clusters.empty:
        score_col = "roi_evidence_v2_score" if "roi_evidence_v2_score" in clusters else "roi_evidence_score"
        ranked = clusters.sort_values(score_col, ascending=False).head(8)
        for index, row in ranked.iterrows():
            candidates.append(
                {
                    "candidate_id": f"cluster_{index}",
                    "candidate_bpm": finite_float(row.get("cluster_bpm")),
                    "method": str(row.get("methods", "")),
                    "region": str(row.get("regions", "")),
                    "support": int(finite_float(row.get("support_rows"), 0) or 0),
                    "score": finite_float(row.get(score_col), 0.0),
                }
            )

    detection = finite_float(result.metadata.get("detector_meta", {}).get("detection_rate"), preflight["face_detection_rate"])
    agreement = finite_float(last.get("roi_evidence_v2_score"), finite_float(last.get("roi_evidence_score"), 0.0))
    case = {
        "schema_version": SCHEMA_VERSION,
        "case_id": _case_id("upload"),
        "display_id": f"VS-{datetime.now(UTC).strftime('%H%M%S')}",
        "input_kind": "uploaded_video",
        "source_name": Path(path).name,
        "purpose": purpose,
        "consent_recorded": True,
        "retention_policy": retention_policy,
        "decision": decision if decision in {"release", "review"} else "review",
        "priority": "routine" if decision == "release" else "high",
        "released_hr_bpm": released_hr,
        "selected_candidate_hr_bpm": candidate_hr,
        "confidence": finite_float(last.get("roi_evidence_v2_score"), finite_float(last.get("roi_evidence_score"), 0.0)),
        "quality_score": float(np.clip((float(preflight["face_detection_rate"]) + (1 - min(float(preflight["motion_mean"]) / 30, 1))) / 2, 0, 1)),
        "snr_db": None,
        "face_coverage": detection,
        "illumination_score": float(np.clip(1.0 - abs(float(preflight["brightness_mean"]) - 128.0) / 128.0, 0, 1)),
        "motion_score": float(np.clip(float(preflight["motion_mean"]) / 30.0, 0, 1)),
        "candidate_count": int(result.metadata.get("n_candidates", len(result.candidates))),
        "agreement_fraction": agreement,
        "harmonic_risk": None,
        "selected_method": str(last.get("max_power_method", "multi-route selector")),
        "selected_region": str(last.get("max_power_region", "multi-ROI")),
        "review_reason": str(last.get("refusal_reason", "")) if decision != "release" else "",
        "recommended_action": "Retain the evidence packet with the reported estimate." if decision == "release" else "Inspect candidate evidence or repeat the recording.",
        "trend_bpm": [finite_float(value) for value in windows.get("product_hr_bpm", pd.Series(dtype=float)).tolist() if finite_float(value) is not None],
        "candidates": candidates,
        "preflight": preflight,
        "window_results": json_safe(windows.to_dict("records")),
        "policy_version": POLICY_VERSION,
        "model_version": MODEL_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_scope": "Label-free public research pipeline run on an uploaded video.",
        "created_at": utc_now(),
    }
    return ensure_output_contract(case)


def build_report_payload(
    case: dict[str, Any],
    *,
    review: dict[str, Any] | None = None,
    audit_events: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized = ensure_output_contract(case)
    return {
        "report_version": REPORT_VERSION,
        "generated_at": utc_now(),
        "case": normalized,
        "attribution": build_attribution(normalized),
        "review": json_safe(review or {}),
        "audit_events": json_safe(list(audit_events or [])),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def build_report_markdown(payload: dict[str, Any], *, language: str = "en") -> str:
    case = payload["case"]
    attribution = payload["attribution"]
    zh = language.lower().startswith("zh")
    text = lambda value: localize_console_text(value, language=language)
    labels = {
        "title": "VitalsSight 证据报告" if zh else "VitalsSight Evidence Report",
        "summary": "结果摘要" if zh else "Result summary",
        "quality": "采集质量" if zh else "Acquisition quality",
        "attribution": "证据与策略归因" if zh else "Evidence and policy attribution",
        "candidates": "候选分支" if zh else "Candidate branches",
        "review": "复核记录" if zh else "Review record",
        "audit": "审计记录" if zh else "Audit trail",
        "boundary": "证据边界" if zh else "Evidence boundary",
    }
    lines = [
        f"# {labels['title']}",
        f"- **{'报告版本' if zh else 'Report version'}:** {payload['report_version']}",
        f"- **{'生成时间' if zh else 'Generated'}:** {payload['generated_at']}",
        "",
        f"## {labels['summary']}",
        f"- {'案例' if zh else 'Case'}: {case['display_id']} ({case['case_id']})",
        f"- {'决策' if zh else 'Decision'}: {text(case['decision'])}",
        f"- {'已发布心率' if zh else 'Released HR'}: {_fmt_hr(case.get('released_hr_bpm'), withheld=text('withheld'))}",
        f"- {'候选心率' if zh else 'Candidate HR'}: {_fmt_hr(case.get('selected_candidate_hr_bpm'))}",
        f"- {'建议操作' if zh else 'Recommended action'}: {text(case.get('recommended_action'))}",
        "",
        f"## {labels['quality']}",
        f"- {'人脸覆盖' if zh else 'Face coverage'}: {_fmt_percent(case.get('face_coverage'))}",
        f"- {'光照分数' if zh else 'Illumination score'}: {_fmt_percent(case.get('illumination_score'))}",
        f"- {'运动分数' if zh else 'Motion score'}: {_fmt_percent(case.get('motion_score'))}",
        f"- {'质量分数' if zh else 'Quality score'}: {_fmt_percent(case.get('quality_score'))}",
        "",
        f"## {labels['attribution']}",
    ]
    for factor in attribution["all_factors"]:
        lines.append(
            f"- {text(factor['factor'])}: {factor['observed']} | "
            f"{text(factor['status'])} - {text(factor['reason'])}"
        )
    lines.extend(["", text(attribution["boundary"]), "", f"## {labels['candidates']}"])
    candidates = case.get("candidates", [])
    if candidates:
        lines.extend(
            f"- {item.get('candidate_id')}: {item.get('candidate_bpm')} BPM | {item.get('method')} | score={item.get('score')}"
            for item in candidates
        )
    else:
        lines.append("- 未保留候选分支。" if zh else "- No candidate branch was retained.")
    lines.extend(
        [
            "",
            f"## {labels['review']}",
            f"- {'状态' if zh else 'Status'}: {text(payload.get('review', {}).get('status', 'not opened'))}",
            f"- {'优先级' if zh else 'Priority'}: {text(payload.get('review', {}).get('priority', ''))}",
            f"- {'负责人' if zh else 'Assignee'}: {payload.get('review', {}).get('assignee', '')}",
            f"- {'处理结果' if zh else 'Resolution'}: {text(payload.get('review', {}).get('resolution', ''))}",
            f"- {'复核备注' if zh else 'Reviewer note'}: {payload.get('review', {}).get('note', '')}",
            "",
            f"## {labels['audit']}",
            "",
        ]
    )
    events = payload.get("audit_events", [])
    if events:
        for event in events:
            lines.append(
                f"- {event.get('created_at', '')} | {event.get('event_type', '')} | "
                f"{event.get('actor', '')} | {json.dumps(event.get('details', {}), ensure_ascii=False)}"
            )
    else:
        lines.append("- 暂无审计事件。" if zh else "- No audit event is available.")
    lines.extend(["", f"## {labels['boundary']}", text(payload["claim_boundary"])])
    return "\n".join(lines)


def build_report_pdf(payload: dict[str, Any], *, language: str = "en") -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    zh = language.lower().startswith("zh")
    text = lambda value: localize_console_text(value, language=language)
    font_name = "Helvetica"
    if zh:
        font_name = "STSong-Light"
        try:
            pdfmetrics.registerFont(UnicodeCIDFont(font_name))
        except Exception:
            font_name = "Helvetica"

    case = payload["case"]
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="VitalsSight Evidence Report",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "VS-Title",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#24343F"),
        alignment=TA_LEFT,
        spaceAfter=10,
    )
    heading = ParagraphStyle(
        "VS-Heading",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#416F7A"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "VS-Body",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#24343F"),
    )
    small = ParagraphStyle(
        "VS-Small",
        parent=body,
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#667680"),
    )
    story: list[Any] = [
        Paragraph("VitalsSight 证据报告" if zh else "VitalsSight Evidence Report", title_style),
        Paragraph(
            f"{case['display_id']} &nbsp;|&nbsp; {payload['generated_at']} &nbsp;|&nbsp; {payload['report_version']}",
            small,
        ),
        Spacer(1, 5 * mm),
    ]
    summary_data = [
        ["决策" if zh else "Decision", text(case.get("decision", ""))],
        ["已发布心率" if zh else "Released HR", _fmt_hr(case.get("released_hr_bpm"), withheld=text("withheld"))],
        ["候选心率" if zh else "Candidate HR", _fmt_hr(case.get("selected_candidate_hr_bpm"))],
        ["建议操作" if zh else "Recommended action", text(case.get("recommended_action", ""))],
    ]
    story.append(Paragraph("结果摘要" if zh else "Result summary", heading))
    story.append(_report_table(summary_data, body, colors))

    quality_data = [
        ["人脸覆盖" if zh else "Face coverage", _fmt_percent(case.get("face_coverage"))],
        ["光照" if zh else "Illumination", _fmt_percent(case.get("illumination_score"))],
        ["运动" if zh else "Motion", _fmt_percent(case.get("motion_score"))],
        ["质量" if zh else "Quality", _fmt_percent(case.get("quality_score"))],
        ["候选数量" if zh else "Candidate count", str(case.get("candidate_count", 0))],
    ]
    story.append(Paragraph("采集质量" if zh else "Acquisition quality", heading))
    story.append(_report_table(quality_data, body, colors))

    story.append(Paragraph("证据与策略归因" if zh else "Evidence and policy attribution", heading))
    attr_data = [["因素" if zh else "Factor", "观测值" if zh else "Observed", "方向" if zh else "Direction", "理由" if zh else "Reason"]]
    for item in payload["attribution"]["all_factors"]:
        attr_data.append(
            [
                text(item["factor"]),
                _fmt_number(item["observed"]),
                text(item["status"]),
                text(item["reason"]),
            ]
        )
    table = Table(
        [[Paragraph(str(cell), body) for cell in row] for row in attr_data],
        colWidths=[32 * mm, 23 * mm, 29 * mm, 90 * mm],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF1F3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#24343F")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD7DB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(text(payload["attribution"]["boundary"]), small))

    candidates = case.get("candidates", [])
    story.append(Paragraph("候选分支" if zh else "Candidate branches", heading))
    if candidates:
        candidate_data = [["ID", "BPM", "方法" if zh else "Method", "区域" if zh else "Region", "支持" if zh else "Support", "分数" if zh else "Score"]]
        for item in candidates[:10]:
            candidate_data.append(
                [
                    str(item.get("candidate_id", "")),
                    _fmt_number(item.get("candidate_bpm")),
                    str(item.get("method", "")),
                    str(item.get("region", "")),
                    str(item.get("support", "")),
                    _fmt_number(item.get("score")),
                ]
            )
        story.append(
            _report_table(candidate_data, body, colors, header=True, widths=[18, 20, 45, 45, 20, 20])
        )
    else:
        story.append(Paragraph("未保留候选分支。" if zh else "No candidate branch was retained.", body))

    review = payload.get("review", {})
    story.append(Paragraph("复核记录" if zh else "Review record", heading))
    review_data = [
        ["状态" if zh else "Status", text(review.get("status", "not opened"))],
        ["优先级" if zh else "Priority", text(review.get("priority", ""))],
        ["负责人" if zh else "Assignee", str(review.get("assignee", ""))],
        ["处理结果" if zh else "Resolution", text(review.get("resolution", ""))],
        ["复核备注" if zh else "Reviewer note", str(review.get("note", ""))],
    ]
    story.append(_report_table(review_data, body, colors))

    story.append(Paragraph("审计记录" if zh else "Audit trail", heading))
    events = payload.get("audit_events", [])
    if events:
        audit_data = [["时间" if zh else "Time", "事件" if zh else "Event", "操作人" if zh else "Actor"]]
        audit_data.extend(
            [str(event.get("created_at", "")), str(event.get("event_type", "")), str(event.get("actor", ""))]
            for event in events
        )
        story.append(_report_table(audit_data, body, colors, header=True, widths=[64, 60, 44]))
    else:
        story.append(Paragraph("暂无审计事件。" if zh else "No audit event is available.", body))

    story.append(Paragraph("证据边界" if zh else "Evidence boundary", heading))
    story.append(Paragraph(text(payload["claim_boundary"]), body))
    document.build(story)
    return buffer.getvalue()


def _report_table(
    data: list[list[Any]],
    body_style: Any,
    colors_module: Any,
    *,
    header: bool = False,
    widths: list[float] | None = None,
) -> Any:
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, Table, TableStyle

    rows = [[Paragraph(str(cell), body_style) for cell in row] for row in data]
    col_widths = [value * mm for value in widths] if widths else [42 * mm, 132 * mm]
    table = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors_module.HexColor("#CBD7DB")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), colors_module.HexColor("#EAF1F3")))
    else:
        commands.append(("BACKGROUND", (0, 0), (0, -1), colors_module.HexColor("#F3F6F7")))
    table.setStyle(TableStyle(commands))
    return table


def _fmt_number(value: object) -> str:
    number = finite_float(value)
    return "NA" if number is None else f"{number:.3f}"


def _fmt_percent(value: object) -> str:
    number = finite_float(value)
    return "NA" if number is None else f"{number:.0%}"


def _fmt_hr(value: object, *, withheld: str = "NA") -> str:
    number = finite_float(value)
    return withheld if number is None else f"{number:.1f} BPM"
