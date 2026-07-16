from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import pandas as pd


SCHEMA_VERSION = "vitalssight.console.case.v1"
REPORT_VERSION = "vitalssight.evidence-report.v2"
POLICY_VERSION = "public_candidate_release_gate.v2"
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
    "The pinned landmark model was unavailable, so candidate evidence was computed from fallback regions.": "固定版本的人脸关键点模型不可用，因此本次候选证据由后备静态区域计算。",
    "Candidate route omissions": "候选路径省略",
    "audited, not relabelled": "已审计且未被重新标记",
    "recorded": "已记录",
    "Failed routes were omitted from the candidate pool and retained in runtime provenance; they were not replaced by another signal under the failed method label.": "失败路径已从候选池中省略并保留在运行溯源中；系统没有用其他信号冒充失败方法的输出。",
    "Install the pinned runtime model with python scripts/setup_runtime_assets.py, then repeat the assessment.": "运行 python scripts/setup_runtime_assets.py 安装固定版本的运行时模型，然后重新评估。",
    "The landmark model was unavailable; fallback regions are retained for review but are not treated as equivalent evidence.": "人脸关键点模型不可用；后备区域结果仅保留供复核，不视为等价证据。",
    "Confirm runtime_metadata.detector_backend is mediapipe_face_landmarker_task on the repeated assessment.": "重新评估后，确认 runtime_metadata.detector_backend 为 mediapipe_face_landmarker_task。",
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
    "Inspect window-level and candidate evidence or repeat the recording under stable conditions.": "检查窗口级与候选证据，或在稳定条件下重新采集。",
    "Window consistency": "窗口一致性",
    "A stable majority of analyzed windows is required before an uploaded-video result is released.": "上传视频结果放行前，分析窗口必须形成稳定多数。",
    "Window-level HR estimates did not form the required stable majority.": "窗口级心率估计未形成所要求的稳定多数。",
    "Review the window-level estimates and repeat the recording if the spread remains unresolved.": "检查窗口级估计；若离散仍无法消解，请重新采集。",
    "Confirm at least two thirds of analyzed windows agree within 10 BPM before release.": "放行前确认至少三分之二的分析窗口在 10 BPM 内一致。",
    "inter_window_hr_disagreement": "窗口间心率估计不一致",
    "no_stable_cross_window_candidate_track": "未形成稳定的跨窗口候选轨迹",
    "competing_cross_window_candidate_tracks": "存在竞争性的跨窗口候选轨迹",
    "Competing candidate tracks": "竞争候选轨迹",
    "More than one cross-window candidate track remained plausible under the documented margin.": "在既定分数间隔内，仍有多条跨窗口候选轨迹具有合理性。",
    "Keep the competing tracks linked to the case and route them to review.": "保留与案例关联的竞争轨迹并转入复核。",
    "Confirm that no competing track remains within the documented score margin before release.": "放行前确认既定分数间隔内不再存在竞争轨迹。",
    "Absence of a similarly supported cross-window track is required for release.": "放行要求不存在证据支持程度相近的跨窗口竞争轨迹。",
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
    "supports pipeline entry": "支持进入候选构建",
    "supports retake": "支持重采",
    "requires attention": "需要关注",
    "not evaluated": "未评估",
    "Candidate construction": "候选构建",
    "not entered": "未进入",
    "preflight pass": "采集前检查通过",
    "Candidate construction was not entered because preflight failed; candidate count is therefore not an acquisition failure.": "由于采集前检查失败，系统未进入候选构建；因此候选数为零不代表候选算法失败。",
    "This acquisition check failed before candidate construction began.": "该采集检查在候选构建开始前失败。",
    "This acquisition check returned a warning and should be corrected before rerunning.": "该采集检查返回警告，建议在重新运行前纠正。",
    "Repeat preflight and confirm this check passes before candidate construction begins.": "重新执行采集前检查，并在候选构建开始前确认该项通过。",
    "The candidate stage was intentionally skipped after the failed acquisition gate.": "采集门控失败后，候选阶段按设计被跳过。",
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
    "Release criteria passed; preserve the result with its evidence packet.": "放行条件已通过；请将结果与证据包一并保留。",
    "Automatic reporting is withheld until the highlighted evidence is reviewed.": "在完成高亮证据复核前，系统不会自动发布结果。",
    "Acquisition criteria were not met; correct the highlighted input conditions and repeat.": "采集条件未达到要求；请修正高亮输入条件后重新采集。",
    "Observed evidence met the configured release conditions. The estimate remains bounded to this evidence packet.": "观测证据满足既定放行条件；该估计仅在本证据包范围内成立。",
    "One or more quality or candidate checks crossed the configured review boundary, so HR remains withheld.": "一个或多个质量或候选检查越过既定复核边界，因此心率保持未发布。",
    "One or more acquisition checks failed before a reportable candidate could be established.": "在形成可报告候选前，一个或多个采集检查未通过。",
    "within target": "目标内",
    "warning": "警告",
    "triggered": "已触发",
    "unavailable": "不可用",
    "This signal met the documented target and did not trigger corrective action.": "该信号已达到既定目标，未触发纠正操作。",
    "This signal was unavailable, so no corrective action was inferred from it.": "该信号不可用，因此未据此推断纠正操作。",
    "Candidate count": "候选数量",
    "Keep the complete evidence packet and report version together with the released estimate.": "将完整证据包和报告版本与已发布估计一并保留。",
    "Confirm that the result is used only within the documented research workflow.": "确认该结果仅用于已说明的研究流程。",
    "Move the full face into the center of the frame and remove major occlusion.": "将完整人脸移至画面中央，并移除明显遮挡。",
    "Face visibility did not reach the policy threshold.": "人脸可见性未达到策略阈值。",
    "Confirm face visibility is at least 70% before rerunning the pipeline.": "重新运行前，确认人脸可见性至少达到 70%。",
    "Use one even, front-facing light source and avoid backlight, deep shadow, or saturation.": "使用均匀的正面单一光源，避免逆光、深阴影或过曝。",
    "Illumination did not reach the evidence threshold.": "光照未达到证据阈值。",
    "Confirm the illumination score is at least 55% before interpretation.": "解释结果前，确认光照分数至少达到 55%。",
    "Stabilize the device and ask the participant to remain still and avoid speaking.": "固定设备，并请受试者保持静止、避免说话。",
    "Motion exceeded the policy limit and may contaminate regional traces.": "运动超过策略上限，可能污染区域信号。",
    "Confirm the motion score is no greater than 35% on the repeat recording.": "确认重采视频的运动分数不高于 35%。",
    "Record a longer, stable window with the full face visible so multiple routes can form candidates.": "在完整人脸可见的条件下录制更长且稳定的视频，使多个路径能够形成候选。",
    "Too few candidate branches were retained for comparison.": "保留的候选分支过少，无法进行比较。",
    "Confirm at least three candidates are retained before automatic reporting is considered.": "考虑自动发布前，确认至少保留三个候选。",
    "Inspect the competing routes and repeat the recording under more stable conditions if disagreement persists.": "检查相互竞争的路径；若分歧持续，请在更稳定的条件下重采。",
    "Cross-route agreement did not reach the release threshold.": "跨路径一致性未达到放行阈值。",
    "Confirm candidate agreement reaches at least 60% before release.": "放行前，确认候选一致性至少达到 60%。",
    "Compare the half-rate and double-rate branches; do not force either branch into release.": "比较半频与倍频分支，不得强制放行其中任何一个。",
    "Harmonic ambiguity exceeded the configured review limit.": "谐波歧义超过既定复核上限。",
    "Confirm harmonic risk is no greater than 35% or retain the case for review.": "确认谐波风险不高于 35%，否则继续保留复核。",
    "Retain the candidate evidence for operator review or repeat the recording to improve support.": "保留候选证据供操作员复核，或重新采集以提高支持度。",
    "Selector support did not reach the release threshold.": "选择器支持度未达到放行阈值。",
    "Confirm selector support reaches at least 60% before release.": "放行前，确认选择器支持度至少达到 60%。",
    "The released estimate stays linked to its quality, candidate, policy, and audit evidence.": "已发布估计将持续关联其质量、候选、策略和审计证据。",
    "A corrected recording should either satisfy the documented thresholds or return a traceable review/retake state without publishing HR.": "修正后的录制应满足既定阈值，或返回可追溯的复核/重采状态且不发布心率。",
    "If later evidence falls outside the thresholds, route the new window to review instead of carrying this release forward.": "若后续证据超出阈值，请将新窗口转入复核，不要沿用本次放行。",
    "If the same trigger persists after one corrected recording, keep HR withheld and assign technical review.": "若完成一次纠正性重采后同一触发项仍存在，请继续隐藏心率并分派技术复核。",
    "If acquisition remains below threshold after correction, do not force a result; inspect the camera, detector, and recording protocol.": "若纠正后采集仍低于阈值，不得强制输出；请检查相机、人脸检测器和录制流程。",
    "Operational acquisition and research-review guidance only; it is not a clinical recommendation.": "仅提供采集与研究复核层面的操作指引，不构成临床建议。",
    "Open guided workflow": "打开引导式流程",
    "Choose a case for evidence-specific guidance or start a consented assessment.": "请选择案例以获取证据级指引，或开始一项已获授权的评估。",
    "Follow the recorded output contract.": "遵循已记录的输出契约。",
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


_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_UNC_ABSOLUTE_PATH = re.compile(r"^\\\\[^\\/]+[\\/][^\\/]+")
_POSIX_ABSOLUTE_PATH = re.compile(r"^/")
_WINDOWS_PATH_FRAGMENT = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:"
    r"[A-Z]:[\\/]+[^\\/\s\"'<>|]+(?:[\\/]+[^\\/\s\"'<>|]+)*"
    r"|\\\\[^\\/\s\"'<>|]+[\\/]+[^\\/\s\"'<>|]+"
    r"(?:[\\/]+[^\\/\s\"'<>|]+)*"
    r")"
)
_POSIX_PATH_FRAGMENT = re.compile(
    r"(?<![A-Za-z0-9:/])/(?!/)[^/\s\"'<>|]+(?:/[^/\s\"'<>|]+)*"
)


def sanitize_report_value(value: Any) -> Any:
    """Remove machine-local absolute paths from exported report payloads."""

    if isinstance(value, dict):
        return {str(key): sanitize_report_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_report_value(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if _WINDOWS_ABSOLUTE_PATH.match(stripped) or _UNC_ABSOLUTE_PATH.match(stripped):
            return PureWindowsPath(stripped).name or "[local path redacted]"
        if _POSIX_ABSOLUTE_PATH.match(stripped):
            return PurePosixPath(stripped).name or "[local path redacted]"
        redacted = _WINDOWS_PATH_FRAGMENT.sub("[local path redacted]", value)
        redacted = _POSIX_PATH_FRAGMENT.sub("[local path redacted]", redacted)
        return redacted
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

    preflight = case.get("preflight") or {}
    preflight_checks = preflight.get("checks") or []
    if case["decision"] == "retake" and preflight_checks:
        for item in preflight_checks:
            check = str(item.get("check") or "acquisition check")
            status = str(item.get("status") or "fail").lower()
            value = finite_float(item.get("value"))
            factors.append(
                {
                    "factor": check,
                    "observed": None if value is None else round(value, 3),
                    "unit": str(item.get("unit") or ""),
                    "status": (
                        "supports pipeline entry"
                        if status == "pass"
                        else ("requires attention" if status == "warn" else "supports retake")
                    ),
                    "direction": 1 if status == "pass" else -1,
                    "reason": (
                        "This signal met the documented target and did not trigger corrective action."
                        if status == "pass"
                        else (
                            "This acquisition check returned a warning and should be corrected before rerunning."
                            if status == "warn"
                            else "This acquisition check failed before candidate construction began."
                        )
                    ),
                    "source_field": f"preflight.checks.{check}",
                }
            )
        factors.append(
            {
                "factor": "Candidate construction",
                "observed": "not entered",
                "unit": "stage",
                "status": "not evaluated",
                "direction": 0,
                "reason": "The candidate stage was intentionally skipped after the failed acquisition gate.",
                "source_field": "pipeline.candidate_construction",
            }
        )
        factors.sort(key=lambda item: (item["direction"], item["factor"]))
        negative = [item for item in factors if item["direction"] < 0]
        positive = [item for item in factors if item["direction"] > 0]
        return {
            "attribution_type": "preflight_evidence_and_policy_attribution",
            "decision": case["decision"],
            "primary_review_drivers": negative[:3],
            "primary_release_support": positive[:3],
            "all_factors": factors,
            "boundary": ATTRIBUTION_BOUNDARY,
        }

    def add(
        factor: str,
        value: float | None,
        unit: str,
        *,
        positive: bool,
        reason: str,
        source_field: str,
    ) -> None:
        if value is None:
            status = "not available"
            direction = 0
        else:
            status = "supports release" if positive else "supports review"
            direction = 1 if positive else -1
        factors.append(
            {
                "factor": factor,
                "observed": None if value is None else round(value, 3),
                "unit": unit,
                "status": status,
                "direction": direction,
                "reason": reason,
                "source_field": source_field,
            }
        )

    raw_preflight_checks = (case.get("preflight") or {}).get("checks") or []
    if raw_preflight_checks:
        factor_labels = {
            "file readability": "File readability",
            "duration": "Duration",
            "frame rate": "Frame rate",
            "resolution": "Resolution",
            "illumination": "Illumination",
            "motion": "Motion",
            "face visibility": "Face visibility",
        }
        for item in raw_preflight_checks:
            check = str(item.get("check") or "acquisition check").lower()
            status = str(item.get("status") or "fail").lower()
            add(
                factor_labels.get(check, check.title()),
                finite_float(item.get("value")),
                str(item.get("unit") or ""),
                positive=status == "pass",
                reason="The acquisition factor is interpreted with the same threshold shown in the quality table.",
                source_field=f"preflight.checks.{check}",
            )
    else:
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

    runtime_metadata = case.get("runtime_metadata") or {}
    detector_backend = str(runtime_metadata.get("detector_backend") or "")
    if detector_backend:
        fallback = detector_backend.startswith("static_roi")
        factors.append(
            {
                "factor": "Landmark backend",
                "observed": detector_backend,
                "unit": "",
                "status": "supports review" if fallback else "supports release",
                "direction": -1 if fallback else 1,
                "reason": (
                    "The landmark model was unavailable, so the run used static fallback regions."
                    if fallback
                    else "The configured face-landmark model supplied the regional evidence."
                ),
                "source_field": "runtime_metadata.detector_backend",
            }
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
    window_consistency = finite_float(case.get("window_consistency_fraction"))
    if window_consistency is not None:
        add(
            "Window consistency",
            window_consistency,
            "fraction",
            positive=window_consistency >= (2 / 3),
            reason="A stable majority of analyzed windows is required before an uploaded-video result is released.",
            source_field="window_consistency_fraction",
        )
    competing_tracks = finite_float(case.get("competing_track_count"))
    if competing_tracks is not None:
        add(
            "Competing candidate tracks",
            competing_tracks,
            "count",
            positive=competing_tracks == 0,
            reason="Absence of a similarly supported cross-window track is required for release.",
            source_field="competing_track_count",
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


def build_action_plan(case: dict[str, Any]) -> dict[str, Any]:
    """Map observed evidence to bounded, verifiable operational next steps."""

    normalized = ensure_output_contract(case)
    evidence: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []

    def add_evidence(
        signal: str,
        source_field: str,
        value: float | None,
        *,
        target: str,
        within_target: bool | None,
        display: str,
        reason: str,
        action: str,
        verification: str,
    ) -> None:
        status = "unavailable" if within_target is None else ("within target" if within_target else "triggered")
        evidence_reason = reason
        if within_target is True:
            evidence_reason = "This signal met the documented target and did not trigger corrective action."
        elif within_target is None:
            evidence_reason = "This signal was unavailable, so no corrective action was inferred from it."
        evidence.append(
            {
                "signal": signal,
                "source_field": source_field,
                "observed": display,
                "target": target,
                "status": status,
                "reason": evidence_reason,
            }
        )
        if within_target is False:
            steps.append(
                {
                    "step": len(steps) + 1,
                    "action": action,
                    "because": reason,
                    "verification": verification,
                    "source_field": source_field,
                }
            )

    preflight = normalized.get("preflight") or {}
    preflight_checks = preflight.get("checks") or []
    if preflight_checks:
        targets = {
            "file readability": "decodable supported video",
            "duration": ">= 20 s preferred; >= 8 s minimum",
            "frame rate": ">= 20 fps preferred; >= 15 fps minimum",
            "resolution": ">= 480 px preferred; >= 240 px warning floor",
            "illumination": "45-210 luma",
            "motion": "<= 10 preferred; <= 22 warning ceiling",
            "face visibility": ">= 70% preferred; >= 35% warning floor",
        }
        for item in preflight_checks:
            check = str(item.get("check") or "acquisition check")
            status = str(item.get("status") or "fail").lower()
            value = finite_float(item.get("value"))
            unit = str(item.get("unit") or "")
            if unit == "fraction" and value is not None:
                display = f"{value:.0%}"
            elif value is None:
                display = "NA"
            else:
                display = f"{value:.3f}".rstrip("0").rstrip(".")
                if unit:
                    display = f"{display} {unit}"
            evidence.append(
                {
                    "signal": check,
                    "source_field": f"preflight.checks.{check}",
                    "observed": display,
                    "target": targets.get(check, "pass preflight"),
                    "status": "within target" if status == "pass" else ("warning" if status == "warn" else "triggered"),
                    "reason": (
                        "This signal met the documented target and did not trigger corrective action."
                        if status == "pass"
                        else (
                            "This acquisition check returned a warning and should be corrected before rerunning."
                            if status == "warn"
                            else "This acquisition check failed before candidate construction began."
                        )
                    ),
                }
            )
            if status != "pass" and normalized["decision"] != "release":
                steps.append(
                    {
                        "step": len(steps) + 1,
                        "action": str(item.get("action") or normalized.get("recommended_action")),
                        "because": (
                            "This acquisition check returned a warning and should be corrected before rerunning."
                            if status == "warn"
                            else "This acquisition check failed before candidate construction began."
                        ),
                        "verification": "Repeat preflight and confirm this check passes before candidate construction begins.",
                        "source_field": f"preflight.checks.{check}",
                    }
                )
        if normalized["decision"] == "retake":
            evidence.append(
                {
                    "signal": "Candidate construction",
                    "source_field": "pipeline.candidate_construction",
                    "observed": "not entered",
                    "target": "preflight pass",
                    "status": "not evaluated",
                    "reason": "Candidate construction was not entered because preflight failed; candidate count is therefore not an acquisition failure.",
                }
            )

    runtime_metadata = normalized.get("runtime_metadata") or {}
    detector_backend = str(runtime_metadata.get("detector_backend") or "")
    if detector_backend.startswith("static_roi"):
        evidence.append(
            {
                "signal": "Face-landmark backend",
                "source_field": "runtime_metadata.detector_backend",
                "observed": detector_backend,
                "target": "mediapipe_face_landmarker_task",
                "status": "triggered",
                "reason": "The pinned landmark model was unavailable, so candidate evidence was computed from fallback regions.",
            }
        )
        steps.append(
            {
                "step": len(steps) + 1,
                "action": "Install the pinned runtime model with python scripts/setup_runtime_assets.py, then repeat the assessment.",
                "because": "The landmark model was unavailable; fallback regions are retained for review but are not treated as equivalent evidence.",
                "verification": "Confirm runtime_metadata.detector_backend is mediapipe_face_landmarker_task on the repeated assessment.",
                "source_field": "runtime_metadata.detector_backend",
            }
        )

    route_failure_count = int(finite_float(runtime_metadata.get("route_failure_count"), 0.0) or 0)
    if route_failure_count:
        evidence.append(
            {
                "signal": "Candidate route omissions",
                "source_field": "runtime_metadata.route_failures",
                "observed": str(route_failure_count),
                "target": "audited, not relabelled",
                "status": "recorded",
                "reason": "Failed routes were omitted from the candidate pool and retained in runtime provenance; they were not replaced by another signal under the failed method label.",
            }
        )

    face = finite_float(normalized.get("face_coverage"))
    if not preflight_checks:
        add_evidence(
            "Face visibility",
            "face_coverage",
            face,
            target=">= 70%",
            within_target=None if face is None else face >= 0.70,
            display="NA" if face is None else f"{face:.0%}",
            reason="Face visibility did not reach the policy threshold.",
            action="Move the full face into the center of the frame and remove major occlusion.",
            verification="Confirm face visibility is at least 70% before rerunning the pipeline.",
        )
    illumination = finite_float(normalized.get("illumination_score"))
    if not preflight_checks:
        add_evidence(
            "Illumination",
            "illumination_score",
            illumination,
            target=">= 55%",
            within_target=None if illumination is None else illumination >= 0.55,
            display="NA" if illumination is None else f"{illumination:.0%}",
            reason="Illumination did not reach the evidence threshold.",
            action="Use one even, front-facing light source and avoid backlight, deep shadow, or saturation.",
            verification="Confirm the illumination score is at least 55% before interpretation.",
        )
    motion = finite_float(normalized.get("motion_score"))
    if not preflight_checks:
        add_evidence(
            "Motion",
            "motion_score",
            motion,
            target="<= 35%",
            within_target=None if motion is None else motion <= 0.35,
            display="NA" if motion is None else f"{motion:.0%}",
            reason="Motion exceeded the policy limit and may contaminate regional traces.",
            action="Stabilize the device and ask the participant to remain still and avoid speaking.",
            verification="Confirm the motion score is no greater than 35% on the repeat recording.",
        )
    candidate_stage_entered = not (normalized["decision"] == "retake" and bool(preflight_checks))
    candidate_count = int(finite_float(normalized.get("candidate_count"), 0.0) or 0)
    if candidate_stage_entered:
        add_evidence(
            "Candidate count",
            "candidate_count",
            float(candidate_count),
            target=">= 3",
            within_target=candidate_count >= 3,
            display=str(candidate_count),
            reason="Too few candidate branches were retained for comparison.",
            action="Record a longer, stable window with the full face visible so multiple routes can form candidates.",
            verification="Confirm at least three candidates are retained before automatic reporting is considered.",
        )
    if candidate_stage_entered and candidate_count > 0:
        agreement = finite_float(normalized.get("agreement_fraction"))
        add_evidence(
            "Candidate agreement",
            "agreement_fraction",
            agreement,
            target=">= 60%",
            within_target=None if agreement is None else agreement >= 0.60,
            display="NA" if agreement is None else f"{agreement:.0%}",
            reason="Cross-route agreement did not reach the release threshold.",
            action="Inspect the competing routes and repeat the recording under more stable conditions if disagreement persists.",
            verification="Confirm candidate agreement reaches at least 60% before release.",
        )
        harmonic = finite_float(normalized.get("harmonic_risk"))
        add_evidence(
            "Harmonic ambiguity",
            "harmonic_risk",
            harmonic,
            target="<= 35%",
            within_target=None if harmonic is None else harmonic <= 0.35,
            display="NA" if harmonic is None else f"{harmonic:.0%}",
            reason="Harmonic ambiguity exceeded the configured review limit.",
            action="Compare the half-rate and double-rate branches; do not force either branch into release.",
            verification="Confirm harmonic risk is no greater than 35% or retain the case for review.",
        )
        window_consistency = finite_float(normalized.get("window_consistency_fraction"))
        if window_consistency is not None:
            add_evidence(
                "Window consistency",
                "window_consistency_fraction",
                window_consistency,
                target=">= 67%",
                within_target=window_consistency >= (2 / 3),
                display=f"{window_consistency:.0%}",
                reason="Window-level HR estimates did not form the required stable majority.",
                action="Review the window-level estimates and repeat the recording if the spread remains unresolved.",
                verification="Confirm at least two thirds of analyzed windows agree within 10 BPM before release.",
            )
        if "competing_track_count" in normalized:
            competing_tracks = int(finite_float(normalized.get("competing_track_count"), 0.0) or 0)
            add_evidence(
                "Competing candidate tracks",
                "competing_track_count",
                float(competing_tracks),
                target="0",
                within_target=competing_tracks == 0,
                display=str(competing_tracks),
                reason="More than one cross-window candidate track remained plausible under the documented margin.",
                action="Keep the competing tracks linked to the case and route them to review.",
                verification="Confirm that no competing track remains within the documented score margin before release.",
            )
    if candidate_stage_entered and normalized.get("selected_candidate_hr_bpm") is not None:
        confidence = finite_float(normalized.get("confidence"))
        add_evidence(
            "Selector support",
            "confidence",
            confidence,
            target=">= 60%",
            within_target=None if confidence is None else confidence >= 0.60,
            display="NA" if confidence is None else f"{confidence:.0%}",
            reason="Selector support did not reach the release threshold.",
            action="Retain the candidate evidence for operator review or repeat the recording to improve support.",
            verification="Confirm selector support reaches at least 60% before release.",
        )

    decision = normalized["decision"]
    if decision == "release":
        headline = "Release criteria passed; preserve the result with its evidence packet."
        rationale = "Observed evidence met the configured release conditions. The estimate remains bounded to this evidence packet."
        steps = [
            {
                "step": 1,
                "action": "Keep the complete evidence packet and report version together with the released estimate.",
                "because": "Observed evidence met the configured release conditions. The estimate remains bounded to this evidence packet.",
                "verification": "The released estimate stays linked to its quality, candidate, policy, and audit evidence.",
                "source_field": "decision",
            },
            {
                "step": 2,
                "action": "Confirm that the result is used only within the documented research workflow.",
                "because": CLAIM_BOUNDARY,
                "verification": "The released estimate stays linked to its quality, candidate, policy, and audit evidence.",
                "source_field": "claim_boundary",
            },
        ]
        expected_outcome = "The released estimate stays linked to its quality, candidate, policy, and audit evidence."
        escalation = "If later evidence falls outside the thresholds, route the new window to review instead of carrying this release forward."
    elif decision == "review":
        headline = "Automatic reporting is withheld until the highlighted evidence is reviewed."
        rationale = "One or more quality or candidate checks crossed the configured review boundary, so HR remains withheld."
        expected_outcome = "A corrected recording should either satisfy the documented thresholds or return a traceable review/retake state without publishing HR."
        escalation = "If the same trigger persists after one corrected recording, keep HR withheld and assign technical review."
    else:
        headline = "Acquisition criteria were not met; correct the highlighted input conditions and repeat."
        rationale = "One or more acquisition checks failed before a reportable candidate could be established."
        expected_outcome = "A corrected recording should either satisfy the documented thresholds or return a traceable review/retake state without publishing HR."
        escalation = "If acquisition remains below threshold after correction, do not force a result; inspect the camera, detector, and recording protocol."

    if decision != "release" and not steps:
        steps.append(
            {
                "step": 1,
                "action": str(normalized.get("recommended_action") or "Inspect the evidence packet before proceeding."),
                "because": rationale,
                "verification": expected_outcome,
                "source_field": "recommended_action",
            }
        )

    return {
        "decision": decision,
        "headline": headline,
        "recommendation": normalized.get("recommended_action"),
        "rationale": rationale,
        "evidence": evidence,
        "steps": steps,
        "expected_outcome": expected_outcome,
        "escalation": escalation,
        "boundary": "Operational acquisition and research-review guidance only; it is not a clinical recommendation.",
    }


def video_preflight(path: str | Path, *, sample_frames: int = 48) -> dict[str, Any]:
    """Inspect video quality before running the slower rPPG pipeline."""

    import cv2

    video_path = Path(path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    metadata_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_count = metadata_frame_count
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    sequential_sampling = frame_count <= 0
    if sequential_sampling:
        # Some valid AVI files omit the index OpenCV uses for CAP_PROP_FRAME_COUNT.
        # Count decodable frames instead of misclassifying them as zero-duration input.
        while capture.grab():
            frame_count += 1
        capture.release()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not reopen video after frame counting: {video_path}")
    duration = frame_count / fps if fps > 0 else 0.0
    sample_indices = np.linspace(0, max(frame_count - 1, 0), num=max(1, min(sample_frames, frame_count)), dtype=int)
    brightness: list[float] = []
    motion: list[float] = []
    faces = 0
    previous = None
    landmark_detector = None
    detector_backend = None
    detector_source = None
    try:
        from src.vision.face_mesh_roi import MediaPipeFaceLandmarkDetector

        candidate_landmark_detector = MediaPipeFaceLandmarkDetector()
        if candidate_landmark_detector.available:
            landmark_detector = candidate_landmark_detector
            detector_backend = candidate_landmark_detector.backend
            detector_source = Path(candidate_landmark_detector.model_path).name or None
        else:
            candidate_landmark_detector.close()
    except Exception:
        landmark_detector = None
    cascade_name = "haarcascade_frontalface_default.xml"
    cascade_candidates = [
        Path(getattr(cv2.data, "haarcascades", "")) / cascade_name,
        Path(__file__).resolve().parent / "assets" / cascade_name,
    ]
    detector = None
    if landmark_detector is None:
        for cascade_path in cascade_candidates:
            if not cascade_path.is_file():
                continue
            candidate = cv2.CascadeClassifier(str(cascade_path))
            if not candidate.empty():
                detector = candidate
                detector_backend = "opencv_haar_cascade"
                detector_source = cascade_path.name
                break
    def inspect_frame(frame: np.ndarray) -> None:
        nonlocal faces, previous
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(float(gray.mean()))
        small = cv2.resize(gray, (96, 72), interpolation=cv2.INTER_AREA)
        if previous is not None:
            motion.append(float(np.mean(cv2.absdiff(small, previous))))
        previous = small
        if landmark_detector is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces += int(landmark_detector.detect(rgb) is not None)
        elif detector is not None:
            detected = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
            faces += int(len(detected) > 0)

    try:
        if sequential_sampling:
            targets = {int(index) for index in sample_indices}
            last_target = max(targets, default=-1)
            frame_index = 0
            while frame_index <= last_target:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index in targets:
                    inspect_frame(frame)
                frame_index += 1
        else:
            for index in sample_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok:
                    continue
                inspect_frame(frame)
    finally:
        capture.release()
        if landmark_detector is not None:
            landmark_detector.close()

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
            pass_if=(landmark_detector is not None or detector is not None) and face_rate >= 0.70,
            warn_if=(landmark_detector is not None or detector is not None) and face_rate >= 0.35,
            fail_message=(
                "Center the full face and remove major occlusion."
                if landmark_detector is not None or detector is not None
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
        "frame_count_source": "decoded_fallback" if sequential_sampling else "container_metadata",
        "width": width,
        "height": height,
        "duration_sec": round(duration, 3),
        "brightness_mean": round(brightness_mean, 3),
        "motion_mean": round(motion_mean, 3),
        "face_detection_rate": round(face_rate, 3),
        "face_detector_available": landmark_detector is not None or detector is not None,
        "face_detector_backend": detector_backend,
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
            "message": sanitize_report_value(str(error)[:300]),
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
            "message": sanitize_report_value(str(error)[:300]),
        },
        "policy_version": POLICY_VERSION,
        "model_version": MODEL_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_scope": "Pipeline execution failed after quality qualification; no HR was released.",
        "created_at": utc_now(),
    }
    return ensure_output_contract(case)


def aggregate_window_output(
    windows: pd.DataFrame,
    *,
    tolerance_bpm: float = 10.0,
) -> dict[str, Any]:
    """Aggregate window-level outputs without letting the final window dominate."""

    if windows.empty:
        return {
            "decision": "review",
            "released_hr_bpm": None,
            "selected_candidate_hr_bpm": None,
            "review_reason": "no_analysis_window",
            "representative": {},
            "consistency_fraction": 0.0,
            "stable_window_count": 0,
            "total_window_count": 0,
            "window_hr_median_bpm": None,
            "window_hr_range_bpm": None,
            "tolerance_bpm": tolerance_bpm,
        }

    ordered = windows.sort_values("window_id").reset_index(drop=True)
    candidate_source = (
        ordered["candidate_hr_bpm"]
        if "candidate_hr_bpm" in ordered
        else pd.Series(np.nan, index=ordered.index, dtype=float)
    )
    candidate_values = pd.to_numeric(candidate_source, errors="coerce")
    finite_candidates = candidate_values[np.isfinite(candidate_values)]
    candidate_median = float(np.median(finite_candidates)) if not finite_candidates.empty else None
    representative = ordered.iloc[-1].to_dict()

    releasable = ordered[
        ordered.get("decision", pd.Series(index=ordered.index, dtype=object)).astype(str).eq("release")
    ].copy()
    release_source = (
        releasable["product_hr_bpm"]
        if "product_hr_bpm" in releasable
        else pd.Series(np.nan, index=releasable.index, dtype=float)
    )
    release_values = pd.to_numeric(release_source, errors="coerce")
    releasable = releasable[np.isfinite(release_values)].copy()
    release_values = pd.to_numeric(releasable.get("product_hr_bpm"), errors="coerce")

    total = int(len(ordered))
    if len(releasable) != total or release_values.empty:
        decision_source = ordered.get("decision", pd.Series("review", index=ordered.index, dtype=object))
        failed = ordered[~decision_source.astype(str).eq("release")]
        if not failed.empty:
            representative = failed.iloc[0].to_dict()
        failed_reasons = [
            str(value)
            for value in failed.get("refusal_reason", pd.Series(dtype=object)).tolist()
            if str(value).strip() and str(value).strip().lower() != "accepted"
        ]
        review_reason = failed_reasons[0] if failed_reasons else "one_or_more_windows_not_releasable"
        return {
            "decision": "review",
            "released_hr_bpm": None,
            "selected_candidate_hr_bpm": candidate_median,
            "review_reason": review_reason,
            "representative": representative,
            "consistency_fraction": float(len(releasable) / max(total, 1)),
            "stable_window_count": int(len(releasable)),
            "total_window_count": total,
            "window_hr_median_bpm": candidate_median,
            "window_hr_range_bpm": float(finite_candidates.max() - finite_candidates.min()) if len(finite_candidates) > 1 else 0.0,
            "tolerance_bpm": tolerance_bpm,
        }

    center = float(np.median(release_values))
    inlier_mask = (release_values - center).abs() <= tolerance_bpm
    inliers = releasable.loc[inlier_mask].copy()
    required = 1 if total == 1 else max(2, math.ceil(total * 2 / 3))
    stable_count = int(len(inliers))
    consistency = float(stable_count / total)
    aggregate_hr = float(np.median(pd.to_numeric(inliers["product_hr_bpm"]))) if stable_count else center

    distances = (pd.to_numeric(releasable["product_hr_bpm"]) - aggregate_hr).abs()
    representative = releasable.loc[distances.idxmin()].to_dict()
    passed = stable_count >= required
    return {
        "decision": "release" if passed else "review",
        "released_hr_bpm": aggregate_hr if passed else None,
        "selected_candidate_hr_bpm": aggregate_hr if passed else center,
        "review_reason": "" if passed else "inter_window_hr_disagreement",
        "representative": representative,
        "consistency_fraction": consistency,
        "stable_window_count": stable_count,
        "total_window_count": total,
        "window_hr_median_bpm": center,
        "window_hr_range_bpm": float(release_values.max() - release_values.min()) if total > 1 else 0.0,
        "tolerance_bpm": tolerance_bpm,
    }


def aggregate_candidate_tracks(
    windows: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    tolerance_bpm: float = 6.0,
    ambiguity_margin: float = 0.20,
) -> dict[str, Any]:
    """Select a coherent cross-window candidate track without reference labels.

    ``tolerance_bpm`` is the maximum full track spread, rather than a radius
    that can silently permit twice that amount of temporal variation.
    """

    if windows.empty:
        fallback = aggregate_window_output(windows)
        fallback["track_diagnostics"] = []
        fallback["selected_track_members"] = []
        fallback["competing_track_count"] = 0
        return fallback
    if clusters.empty or "sample_id" not in clusters or "cluster_bpm" not in clusters:
        fallback = aggregate_window_output(windows)
        fallback.update(
            {
                "decision": "review",
                "released_hr_bpm": None,
                "selected_candidate_hr_bpm": None,
                "review_reason": "no_candidate_generated",
                "track_diagnostics": [],
                "selected_track_members": [],
                "competing_track_count": 0,
            }
        )
        return fallback
    fallback = aggregate_window_output(windows)

    score_col = "roi_evidence_v2_score" if "roi_evidence_v2_score" in clusters else "roi_evidence_score"
    gate_col = "passes_roi_evidence_v2_gate" if "passes_roi_evidence_v2_gate" in clusters else "passes_roi_evidence_gate"
    working = clusters.copy()
    if score_col not in working:
        score_col = "_track_score"
        working[score_col] = 0.0
    if gate_col not in working:
        gate_col = "_track_gate"
        working[gate_col] = 0
    working["cluster_bpm"] = pd.to_numeric(working["cluster_bpm"], errors="coerce")
    working[score_col] = pd.to_numeric(working[score_col], errors="coerce").fillna(0.0)
    gate_values = working[gate_col] if gate_col in working else pd.Series(0, index=working.index)
    working[gate_col] = pd.to_numeric(gate_values, errors="coerce").fillna(0).astype(int)
    working = working[np.isfinite(working["cluster_bpm"])].copy()
    if working.empty:
        fallback.update(
            {
                "decision": "review",
                "released_hr_bpm": None,
                "selected_candidate_hr_bpm": None,
                "review_reason": "no_candidate_generated",
                "track_diagnostics": [],
                "selected_track_members": [],
                "competing_track_count": 0,
            }
        )
        return fallback

    ordered_windows = windows.sort_values("window_id").reset_index(drop=True)
    sample_ids = list(dict.fromkeys(str(value) for value in ordered_windows["sample_id"].tolist()))
    groups = {sample_id: working[working["sample_id"].astype(str).eq(sample_id)] for sample_id in sample_ids}
    selected_bpm_by_sample = {
        str(row["sample_id"]): finite_float(row.get("candidate_hr_bpm"))
        for _, row in ordered_windows.iterrows()
    }
    top_indices: dict[str, int | None] = {}
    for sample_id, group in groups.items():
        if group.empty:
            top_indices[sample_id] = None
            continue
        selected_bpm = selected_bpm_by_sample.get(sample_id)
        if selected_bpm is not None:
            ranked = group.assign(_distance=(group["cluster_bpm"] - selected_bpm).abs()).sort_values(
                ["_distance", gate_col, score_col], ascending=[True, False, False]
            )
        else:
            ranked = group.sort_values([gate_col, score_col], ascending=[False, False])
        top_indices[sample_id] = int(ranked.index[0])
    seen: set[tuple[tuple[str, int], ...]] = set()
    tracks: list[dict[str, Any]] = []
    for _, anchor in working.iterrows():
        anchor_bpm = float(anchor["cluster_bpm"])
        members: list[dict[str, Any]] = []
        signature_parts: list[tuple[str, int]] = []
        for sample_id in sample_ids:
            group = groups[sample_id]
            nearby = group[(group["cluster_bpm"] - anchor_bpm).abs() <= tolerance_bpm]
            if nearby.empty:
                continue
            nearby = nearby.assign(_distance=(nearby["cluster_bpm"] - anchor_bpm).abs()).sort_values(
                [gate_col, "_distance", score_col], ascending=[False, True, False]
            )
            member_index = int(nearby.index[0])
            member = nearby.loc[member_index]
            signature_parts.append((sample_id, member_index))
            members.append(
                {
                    "sample_id": sample_id,
                    "cluster_index": member_index,
                    "candidate_bpm": float(member["cluster_bpm"]),
                    "score": float(member[score_col]),
                    "gate_passed": bool(member[gate_col]),
                    "top_ranked": member_index == top_indices[sample_id],
                    "methods": str(member.get("methods", "")),
                    "regions": str(member.get("regions", "")),
                }
            )
        signature = tuple(signature_parts)
        if not signature or signature in seen:
            continue
        seen.add(signature)
        bpms = [item["candidate_bpm"] for item in members]
        total = len(sample_ids)
        coverage_count = len(members)
        gate_count = sum(item["gate_passed"] for item in members)
        top_count = sum(item["top_ranked"] for item in members)
        coverage_fraction = float(coverage_count / total)
        mean_score = float(np.mean([item["score"] for item in members]))
        spread_bpm = float(max(bpms) - min(bpms)) if len(bpms) > 1 else 0.0
        tracks.append(
            {
                "track_bpm": float(np.median(bpms)),
                "coverage_count": coverage_count,
                "coverage_fraction": coverage_fraction,
                "gate_count": gate_count,
                "gate_fraction": float(gate_count / total),
                "top_count": top_count,
                "top_fraction": float(top_count / total),
                "mean_score": mean_score,
                "support_score": mean_score * coverage_fraction,
                "spread_bpm": spread_bpm,
                "coherent": spread_bpm <= tolerance_bpm + 1e-9,
                "members": members,
            }
        )

    total = len(sample_ids)
    required = 1 if total == 1 else max(2, math.ceil(total * 2 / 3))
    tracks.sort(
        key=lambda item: (
            item["top_fraction"],
            item["gate_fraction"],
            item["coverage_fraction"],
            item["support_score"],
            -item["spread_bpm"],
        ),
        reverse=True,
    )
    eligible = [
        item
        for item in tracks
        if item["coverage_count"] == total
        and item["gate_count"] >= required
        and item["top_count"] >= required
        and item["coherent"]
    ]
    chosen = eligible[0] if eligible else None
    if chosen is None:
        reason = (
            str(fallback.get("review_reason") or "one_or_more_windows_not_releasable")
            if not ordered_windows["decision"].astype(str).eq("release").all()
            else "no_stable_cross_window_candidate_track"
        )
        fallback.update(
            {
                "decision": "review",
                "released_hr_bpm": None,
                "selected_candidate_hr_bpm": None,
                "review_reason": reason,
                "track_diagnostics": [
                    {key: value for key, value in item.items() if key != "members"}
                    for item in tracks[:8]
                ],
                "selected_track_members": [],
                "competing_track_count": 0,
            }
        )
        return fallback

    all_windows_release = ordered_windows["decision"].astype(str).eq("release").all()
    competitors: list[dict[str, Any]] = []
    if total > 1:
        competitors = [
            item
            for item in tracks
            if item is not chosen
            and item["coverage_count"] >= required
            and item["gate_count"] >= required
            and item["coherent"]
            and abs(item["track_bpm"] - chosen["track_bpm"]) > tolerance_bpm
            and item["mean_score"] >= chosen["mean_score"] - ambiguity_margin
        ]

    passed = chosen in eligible and all_windows_release and not competitors
    if not all_windows_release:
        reason = str(fallback.get("review_reason") or "one_or_more_windows_not_releasable")
    elif competitors:
        reason = "competing_cross_window_candidate_tracks"
    else:
        reason = ""

    representative_member = max(chosen["members"], key=lambda item: item["score"])
    representative = {
        "roi_evidence_v2_score": representative_member["score"],
        "max_power_method": representative_member["methods"],
        "max_power_region": representative_member["regions"],
        "refusal_reason": reason,
    }
    diagnostics = [
        {
            key: value
            for key, value in item.items()
            if key != "members"
        }
        for item in tracks[:8]
    ]
    return {
        "decision": "release" if passed else "review",
        "released_hr_bpm": chosen["track_bpm"] if passed else None,
        "selected_candidate_hr_bpm": chosen["track_bpm"],
        "review_reason": reason,
        "representative": representative,
        "consistency_fraction": min(
            chosen["coverage_fraction"],
            chosen["gate_fraction"],
            chosen["top_fraction"],
        ),
        "stable_window_count": sum(member["gate_passed"] for member in chosen["members"]),
        "total_window_count": total,
        "window_hr_median_bpm": chosen["track_bpm"],
        "window_hr_range_bpm": chosen["spread_bpm"],
        "tolerance_bpm": tolerance_bpm,
        "max_track_spread_bpm": tolerance_bpm,
        "track_diagnostics": diagnostics,
        "selected_track_members": chosen["members"],
        "competing_track_count": len(competitors),
    }


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
        fallback_detection_rate=float(preflight["face_detection_rate"]),
    )
    result = run_adult_hr_video(path, config=config)
    detector_meta = result.metadata.get("detector_meta", {})
    detector_model_path = Path(str(detector_meta.get("detector_model_path") or ""))
    detector_model_sha256 = detector_meta.get("detector_model_sha256")
    windows = result.windows.sort_values("window_id") if not result.windows.empty else result.windows
    aggregate = aggregate_candidate_tracks(windows, result.clusters)
    representative = aggregate["representative"]
    decision = str(aggregate["decision"])
    released_hr = finite_float(aggregate.get("released_hr_bpm"))
    candidate_hr = finite_float(aggregate.get("selected_candidate_hr_bpm"))
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

    detection = finite_float(detector_meta.get("detection_rate"), preflight["face_detection_rate"])
    agreement = finite_float(
        representative.get("roi_evidence_v2_score"),
        finite_float(representative.get("roi_evidence_score"), 0.0),
    )
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
        "confidence": agreement,
        "quality_score": float(np.clip((float(preflight["face_detection_rate"]) + (1 - min(float(preflight["motion_mean"]) / 30, 1))) / 2, 0, 1)),
        "snr_db": None,
        "face_coverage": detection,
        "illumination_score": float(np.clip(1.0 - abs(float(preflight["brightness_mean"]) - 128.0) / 128.0, 0, 1)),
        "motion_score": float(np.clip(float(preflight["motion_mean"]) / 30.0, 0, 1)),
        "candidate_count": int(result.metadata.get("n_candidates", len(result.candidates))),
        "agreement_fraction": agreement,
        "window_consistency_fraction": aggregate["consistency_fraction"],
        "window_hr_range_bpm": aggregate["window_hr_range_bpm"],
        "competing_track_count": aggregate.get("competing_track_count", 0),
        "harmonic_risk": None,
        "selected_method": str(representative.get("max_power_method", "multi-route selector")),
        "selected_region": str(representative.get("max_power_region", "multi-ROI")),
        "review_reason": str(aggregate.get("review_reason", "")) if decision != "release" else "",
        "recommended_action": (
            "Retain the evidence packet with the reported estimate."
            if decision == "release"
            else "Inspect window-level and candidate evidence or repeat the recording under stable conditions."
        ),
        "trend_bpm": [
            finite_float(item.get("candidate_bpm"))
            for item in aggregate.get("selected_track_members", [])
            if finite_float(item.get("candidate_bpm")) is not None
        ],
        "candidates": candidates,
        "preflight": preflight,
        "window_results": json_safe(windows.to_dict("records")),
        "runtime_metadata": json_safe(
            {
                "detector_backend": detector_meta.get("detector_backend", "unknown"),
                "detector_model_asset": detector_model_path.name if detector_model_path.name else None,
                "detector_model_sha256": detector_model_sha256,
                "detector_model_integrity": detector_meta.get("detector_model_integrity"),
                "detector_initialization_error": detector_meta.get("detector_initialization_error") or None,
                "fallback_static_roi": detector_meta.get("fallback_static_roi", 0.0),
                "release_eligible_detector": bool(detector_meta.get("release_eligible_detector", False)),
                "route_failure_count": result.metadata.get("route_failure_count", 0),
                "route_failures": result.metadata.get("route_failures", []),
                "analysis_fps": result.metadata.get("analysis_fps"),
                "max_source_frames": result.metadata.get("max_source_frames"),
                "max_analysis_frames": result.metadata.get("max_analysis_frames"),
                "window_aggregation": {
                    "tolerance_bpm": aggregate["tolerance_bpm"],
                    "stable_window_count": aggregate["stable_window_count"],
                    "total_window_count": aggregate["total_window_count"],
                    "consistency_fraction": aggregate["consistency_fraction"],
                    "window_hr_median_bpm": aggregate["window_hr_median_bpm"],
                    "window_hr_range_bpm": aggregate["window_hr_range_bpm"],
                    "competing_track_count": aggregate.get("competing_track_count", 0),
                    "track_diagnostics": aggregate.get("track_diagnostics", []),
                },
            }
        ),
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
    normalized = sanitize_report_value(ensure_output_contract(case))
    return {
        "report_version": REPORT_VERSION,
        "generated_at": utc_now(),
        "case": normalized,
        "attribution": build_attribution(normalized),
        "action_plan": build_action_plan(normalized),
        "review": sanitize_report_value(json_safe(review or {})),
        "audit_events": sanitize_report_value(json_safe(list(audit_events or []))),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _preflight_report_rows(case: dict[str, Any]) -> list[dict[str, str]]:
    preflight = case.get("preflight") or {}
    checks = preflight.get("checks") or []
    if str(case.get("decision")) != "retake" or not checks:
        return []

    rows: list[dict[str, str]] = []
    for item in checks:
        value = finite_float(item.get("value"))
        unit = str(item.get("unit") or "")
        if unit == "fraction" and value is not None:
            observed = f"{value:.0%}"
        elif value is None:
            observed = "NA"
        else:
            observed = f"{value:.3f}".rstrip("0").rstrip(".")
            if unit:
                observed = f"{observed} {unit}"
        rows.append(
            {
                "check": str(item.get("check") or "acquisition check"),
                "observed": observed,
                "status": str(item.get("status") or "fail").lower(),
            }
        )
    rows.append(
        {
            "check": "Candidate construction",
            "observed": "not entered",
            "status": "not evaluated",
        }
    )
    return rows


def build_report_markdown(payload: dict[str, Any], *, language: str = "en") -> str:
    case = payload["case"]
    attribution = payload["attribution"]
    zh = language.lower().startswith("zh")
    text = lambda value: localize_console_text(value, language=language)
    preflight_overall = str((case.get("preflight") or {}).get("overall") or "").lower()
    if str(case.get("decision")) == "retake" or preflight_overall == "fail":
        acquisition_gate = "未通过" if zh else "Not passed"
    elif preflight_overall == "warn":
        acquisition_gate = "通过但有警告" if zh else "Passed with warnings"
    else:
        acquisition_gate = "通过" if zh else "Passed"
    labels = {
        "title": "VitalsSight 证据报告" if zh else "VitalsSight Evidence Report",
        "summary": "结果摘要" if zh else "Result summary",
        "interpretation": "结果解释" if zh else "Operational interpretation",
        "quality": "采集质量" if zh else "Acquisition quality",
        "runtime": "实现与运行溯源" if zh else "Implementation provenance",
        "action_evidence": "建议依据" if zh else "Evidence supporting the recommendation",
        "workflow": "建议操作流程" if zh else "Recommended workflow",
        "attribution": "证据与策略归因" if zh else "Evidence and policy attribution",
        "candidates": "候选分支" if zh else "Candidate branches",
        "review": "复核记录" if zh else "Review record",
        "audit": "审计记录" if zh else "Audit trail",
        "boundary": "证据边界" if zh else "Evidence boundary",
    }
    preflight_rows = _preflight_report_rows(case)
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
        f"## {labels['interpretation']}",
        f"- {'当前结论' if zh else 'Current conclusion'}: {text(payload['action_plan']['headline'])}",
        f"- {'形成原因' if zh else 'Why this state'}: {text(payload['action_plan']['rationale'])}",
        f"- {'预期结果' if zh else 'Expected outcome'}: {text(payload['action_plan']['expected_outcome'])}",
        "",
        f"## {labels['quality']}",
        f"- {'采集门控' if zh else 'Acquisition gate'}: {acquisition_gate}",
    ]
    if preflight_rows:
        lines.extend(
            [
                "",
                f"| {'检查项' if zh else 'Check'} | {'观测值' if zh else 'Observed'} | {'状态' if zh else 'State'} |",
                "|---|---:|---|",
            ]
        )
        for item in preflight_rows:
            lines.append(f"| {text(item['check'])} | {text(item['observed'])} | {text(item['status'])} |")
        lines.extend(
            [
                "",
                text("Candidate construction was not entered because preflight failed; candidate count is therefore not an acquisition failure."),
            ]
        )
    else:
        lines.extend(
            [
                f"- {'人脸覆盖' if zh else 'Face coverage'}: {_fmt_percent(case.get('face_coverage'))}",
                f"- {'光照分数' if zh else 'Illumination score'}: {_fmt_percent(case.get('illumination_score'))}",
                f"- {'运动分数' if zh else 'Motion score'}: {_fmt_percent(case.get('motion_score'))}",
                f"- {'质量分数' if zh else 'Quality score'}: {_fmt_percent(case.get('quality_score'))}",
            ]
        )
    lines.extend(
        [
            "",
            f"## {labels['runtime']}",
            f"- {'模型版本' if zh else 'Model version'}: {case.get('model_version') or 'N/A'}",
            f"- {'策略版本' if zh else 'Policy version'}: {case.get('policy_version') or 'N/A'}",
            f"- {'人脸检测后端' if zh else 'Face detector backend'}: "
            f"{(case.get('runtime_metadata') or {}).get('detector_backend') or (case.get('preflight') or {}).get('face_detector_backend') or 'N/A'}",
            f"- {'检测模型完整性' if zh else 'Detector model integrity'}: "
            f"{(case.get('runtime_metadata') or {}).get('detector_model_integrity') or 'N/A'}",
            f"- {'检测模型 SHA-256' if zh else 'Detector model SHA-256'}: "
            f"{(case.get('runtime_metadata') or {}).get('detector_model_sha256') or 'N/A'}",
            f"- {'已审计的路由省略数' if zh else 'Audited route omissions'}: "
            f"{_fmt_number((case.get('runtime_metadata') or {}).get('route_failure_count'))}",
            f"- {'分析采样率' if zh else 'Analysis sampling rate'}: "
            f"{_fmt_number((case.get('runtime_metadata') or {}).get('analysis_fps')) + ' fps' if (case.get('runtime_metadata') or {}).get('analysis_fps') is not None else 'N/A'}",
            f"- {'分析帧预算' if zh else 'Analysis frame budget'}: "
            f"{_fmt_number((case.get('runtime_metadata') or {}).get('max_analysis_frames'))}",
            "",
            f"## {labels['action_evidence']}",
            f"| {'信号' if zh else 'Signal'} | {'观测值' if zh else 'Observed'} | {'目标' if zh else 'Target'} | {'状态' if zh else 'State'} | {'与建议的关系' if zh else 'Why it matters'} |",
            "|---|---:|---:|---|---|",
        ]
    )
    for item in payload["action_plan"]["evidence"]:
        lines.append(
            f"| {text(item['signal'])} | {text(item['observed'])} | {text(item['target'])} | "
            f"{text(item['status'])} | {text(item['reason'])} |"
        )
    lines.extend(
        [
            "",
            f"## {labels['workflow']}",
        ]
    )
    for item in payload["action_plan"]["steps"]:
        lines.extend(
            [
                f"{item['step']}. {text(item['action'])}",
                f"   - {'依据' if zh else 'Basis'}: {text(item['because'])}",
                f"   - {'复核标准' if zh else 'Verify'}: {text(item['verification'])}",
            ]
        )
    lines.extend(
        [
            f"- **{'仍未解决时' if zh else 'If unresolved'}:** {text(payload['action_plan']['escalation'])}",
            f"- **{'使用边界' if zh else 'Guidance boundary'}:** {text(payload['action_plan']['boundary'])}",
            "",
            f"## {labels['attribution']}",
        ]
    )
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
    action_plan = payload["action_plan"]
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

    interpretation_data = [
        ["当前结论" if zh else "Current conclusion", text(action_plan["headline"])],
        ["形成原因" if zh else "Why this state", text(action_plan["rationale"])],
        ["预期结果" if zh else "Expected outcome", text(action_plan["expected_outcome"])],
    ]
    story.append(Paragraph("结果解释" if zh else "Operational interpretation", heading))
    story.append(_report_table(interpretation_data, body, colors))

    preflight_rows = _preflight_report_rows(case)
    quality_data = [
        [
            "采集门控" if zh else "Acquisition gate",
            (
                ("未通过" if zh else "Not passed")
                if str(case.get("decision")) == "retake" or str((case.get("preflight") or {}).get("overall") or "").lower() == "fail"
                else (
                    ("通过但有警告" if zh else "Passed with warnings")
                    if str((case.get("preflight") or {}).get("overall") or "").lower() == "warn"
                    else ("通过" if zh else "Passed")
                )
            ),
        ],
    ]
    if preflight_rows:
        quality_data.extend(
            [text(item["check"]), f"{text(item['observed'])} ({text(item['status'])})"]
            for item in preflight_rows
        )
    else:
        quality_data.extend(
            [
                ["人脸覆盖" if zh else "Face coverage", _fmt_percent(case.get("face_coverage"))],
                ["光照" if zh else "Illumination", _fmt_percent(case.get("illumination_score"))],
                ["运动" if zh else "Motion", _fmt_percent(case.get("motion_score"))],
                ["质量" if zh else "Quality", _fmt_percent(case.get("quality_score"))],
                ["候选数量" if zh else "Candidate count", str(case.get("candidate_count", 0))],
            ]
        )
    story.append(Paragraph("采集质量" if zh else "Acquisition quality", heading))
    story.append(_report_table(quality_data, body, colors))

    runtime = case.get("runtime_metadata") or {}
    preflight = case.get("preflight") or {}
    runtime_data = [
        ["模型版本" if zh else "Model version", str(case.get("model_version") or "N/A")],
        ["策略版本" if zh else "Policy version", str(case.get("policy_version") or "N/A")],
        [
            "人脸检测后端" if zh else "Face detector backend",
            str(runtime.get("detector_backend") or preflight.get("face_detector_backend") or "N/A"),
        ],
        ["检测模型完整性" if zh else "Detector model integrity", str(runtime.get("detector_model_integrity") or "N/A")],
        ["检测模型 SHA-256" if zh else "Detector model SHA-256", str(runtime.get("detector_model_sha256") or "N/A")],
        ["已审计的路由省略数" if zh else "Audited route omissions", _fmt_number(runtime.get("route_failure_count"))],
        [
            "分析采样率" if zh else "Analysis sampling rate",
            f"{_fmt_number(runtime.get('analysis_fps'))} fps" if runtime.get("analysis_fps") is not None else "N/A",
        ],
        ["分析帧预算" if zh else "Analysis frame budget", _fmt_number(runtime.get("max_analysis_frames"))],
    ]
    story.append(Paragraph("实现与运行溯源" if zh else "Implementation provenance", heading))
    story.append(_report_table(runtime_data, body, colors))

    story.append(Paragraph("建议依据" if zh else "Evidence supporting the recommendation", heading))
    action_evidence = [[
        "信号" if zh else "Signal",
        "观测值" if zh else "Observed",
        "目标" if zh else "Target",
        "状态" if zh else "State",
        "与建议的关系" if zh else "Why it matters",
    ]]
    for item in action_plan["evidence"]:
        action_evidence.append(
            [
                text(item["signal"]),
                item["observed"],
                item["target"],
                text(item["status"]),
                text(item["reason"]),
            ]
        )
    story.append(_report_table(action_evidence, body, colors, header=True, widths=[31, 22, 22, 26, 73]))

    story.append(Paragraph("建议操作流程" if zh else "Recommended workflow", heading))
    for item in action_plan["steps"]:
        story.append(Paragraph(f"<b>{item['step']}. {text(item['action'])}</b>", body))
        story.append(Paragraph(f"{'依据' if zh else 'Basis'}: {text(item['because'])}", small))
        story.append(Paragraph(f"{'复核标准' if zh else 'Verify'}: {text(item['verification'])}", small))
        story.append(Spacer(1, 1.5 * mm))
    story.append(Paragraph(f"<b>{'仍未解决时' if zh else 'If unresolved'}:</b> {text(action_plan['escalation'])}", body))
    story.append(Paragraph(text(action_plan["boundary"]), small))

    story.append(Paragraph("证据与策略归因" if zh else "Evidence and policy attribution", heading))
    attr_data = [["因素" if zh else "Factor", "观测值" if zh else "Observed", "方向" if zh else "Direction", "理由" if zh else "Reason"]]
    for item in payload["attribution"]["all_factors"]:
        observed = text(item["observed"]) if isinstance(item["observed"], str) else _fmt_number(item["observed"])
        attr_data.append(
            [
                text(item["factor"]),
                observed,
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
    review_data = [["状态" if zh else "Status", text(review.get("status", "not opened"))]]
    review_fields = [
        ("优先级" if zh else "Priority", "priority", True),
        ("负责人" if zh else "Assignee", "assignee", False),
        ("处理结果" if zh else "Resolution", "resolution", True),
        ("复核备注" if zh else "Reviewer note", "note", False),
    ]
    for label, key, localize in review_fields:
        value = review.get(key, "")
        if value not in (None, ""):
            review_data.append([label, text(value) if localize else str(value)])
    story.append(_report_table(review_data, body, colors))

    events = payload.get("audit_events", [])
    if events:
        story.append(Paragraph("审计记录" if zh else "Audit trail", heading))
        audit_data = [["时间" if zh else "Time", "事件" if zh else "Event", "操作人" if zh else "Actor"]]
        audit_data.extend(
            [str(event.get("created_at", "")), str(event.get("event_type", "")), str(event.get("actor", ""))]
            for event in events
        )
        story.append(_report_table(audit_data, body, colors, header=True, widths=[64, 60, 44]))

    story.append(Paragraph("证据边界" if zh else "Evidence boundary", heading))
    story.append(Paragraph(text(payload["claim_boundary"]), body))

    def draw_footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D5E0E4"))
        canvas.setLineWidth(0.4)
        canvas.line(18 * mm, 11 * mm, A4[0] - 18 * mm, 11 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#667680"))
        canvas.drawString(18 * mm, 6.5 * mm, "VitalsSight research evidence")
        canvas.drawRightString(A4[0] - 18 * mm, 6.5 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
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
