from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.assistant import AssistantChatRequest, AssistantOrchestrator, MultimodalAssistantService
from src.assistant.multimodal import MediaProcessingError
from src.assistant.schemas import AssistantLanguage, AssistantMediaContext, AssistantRole, ChatTurn
from src.product.console_api import create_app
from src.product.console_service import (
    ATTRIBUTION_BOUNDARY,
    CLAIM_BOUNDARY,
    build_action_plan,
    build_attribution,
    build_report_markdown,
    build_report_payload,
    build_report_pdf,
    case_from_preflight,
    case_from_runtime_failure,
    case_quality_snapshot,
    ensure_output_contract,
    localize_console_text,
    make_demo_cases,
    preflight_from_decode_error,
    run_uploaded_video,
    sanitize_report_value,
    video_preflight,
)
from src.product.console_store import ConsoleStore
from src.product.build_identity import path_fingerprint, source_build_identity


DB_PATH = Path(os.getenv("VITALSSIGHT_DB_PATH", PROJECT / "runtime" / "vitalsight_console.db"))
UPLOAD_DIR = Path(os.getenv("VITALSSIGHT_UPLOAD_DIR", PROJECT / "runtime" / "uploads"))
HEADLINE_METRICS = PROJECT / "reproducibility" / "headline_metrics.csv"
PROTOCOL_SUMMARY = PROJECT / "reproducibility" / "protocol_summary.json"


SECTIONS = [
    "Overview",
    "New assessment",
    "Cases",
    "Review queue",
    "Reports",
    "AI assistant",
    "Evidence",
    "Integrations",
    "Help & settings",
]

ZH = {
    "Overview": "总览",
    "New assessment": "新建评估",
    "Cases": "案例",
    "Review queue": "复核队列",
    "Reports": "报告中心",
    "AI assistant": "AI 助手",
    "Evidence": "证据与性能",
    "Integrations": "系统集成",
    "Help & settings": "帮助与设置",
}


def run() -> None:
    st.set_page_config(
        page_title="VitalsSight Evidence Console",
        page_icon="VS",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    _init_state()
    pending_section = st.session_state.pop("vs_pending_section", "")
    if pending_section in SECTIONS:
        st.session_state["vs_section"] = pending_section
        st.session_state["vs_section_radio"] = pending_section
    store = _store()
    _seed_if_empty(store)

    language = st.sidebar.segmented_control(
        "Language / 语言",
        options=["ZH", "EN"],
        key="vs_language_control",
        on_change=_sync_language,
    )
    st.session_state["vs_language"] = language or "ZH"
    st.sidebar.markdown("<div class='vs-brand'><span>VS</span><b>VitalsSight</b></div>", unsafe_allow_html=True)
    st.sidebar.caption(_ui("Evidence operations console", "证据运营控制台"))
    st.sidebar.markdown(
        f"<div class='vs-side-status'><i></i><div><b>{_escape(_ui('Workspace ready', '工作区就绪'))}</b>"
        f"<span>{_escape(_ui('Evidence store connected', '证据存储已连接'))}</span></div></div>",
        unsafe_allow_html=True,
    )
    section = st.sidebar.radio(
        _ui("Workspace", "工作区"),
        SECTIONS,
        format_func=lambda item: ZH[item] if _is_zh() else item,
        label_visibility="collapsed",
        key="vs_section_radio",
    )
    st.session_state["vs_section"] = section
    if st.session_state.get("vs_rendered_section") != section:
        st.session_state["vs_rendered_section"] = section
        navigation_nonce = int(st.session_state.get("vs_navigation_nonce", 0)) + 1
        st.session_state["vs_navigation_nonce"] = navigation_nonce
        _reset_main_scroll(navigation_nonce)
    if st.sidebar.button(
        _ui("Open guided workflow", "打开操作教学"),
        icon=":material/menu_book:",
        width="stretch",
        key="vs_sidebar_guide",
    ):
        _set_flash(_ui("The role-based guide is open.", "已打开分角色操作教学。"), "info")
        _go("Help & settings")
    st.sidebar.markdown("---")
    st.sidebar.caption(_ui("Research use only", "仅限研究使用"))
    st.sidebar.markdown(
        f"<div class='vs-boundary-small'>{_escape(_ui('No diagnosis, emergency alert, or autonomous clinical release.', '不用于诊断、急救告警或临床自主放行。'))}</div>",
        unsafe_allow_html=True,
    )
    build = source_build_identity()
    st.sidebar.caption(
        f"Build {str(build['commit'])[:12]} · Tree {str(build['tree'])[:12]} · "
        f"{'dirty' if build['dirty'] else 'clean'}"
    )
    st.sidebar.markdown(
        f'<span data-vs-upload-root-fingerprint="{path_fingerprint(UPLOAD_DIR)}" style="display:none"></span>',
        unsafe_allow_html=True,
    )

    _header(section)
    flash = st.session_state.pop("vs_flash", "")
    flash_kind = st.session_state.pop("vs_flash_kind", "success")
    if flash:
        icon = {
            "success": ":material/check_circle:",
            "info": ":material/info:",
            "warning": ":material/warning:",
            "error": ":material/error:",
        }.get(flash_kind, ":material/info:")
        st.toast(flash, icon=icon)
        getattr(st, flash_kind if flash_kind in {"success", "info", "warning", "error"} else "info")(flash)
    if section == "Overview":
        _overview(store)
    elif section == "New assessment":
        _new_assessment(store)
    elif section == "Cases":
        _cases(store)
    elif section == "Review queue":
        _review_queue(store)
    elif section == "Reports":
        _reports(store)
    elif section == "AI assistant":
        _assistant(store)
    elif section == "Evidence":
        _evidence()
    elif section == "Integrations":
        _integrations(store)
    else:
        _help_settings(store)


def _init_state() -> None:
    defaults = {
        "vs_language": "ZH",
        "vs_language_control": "ZH",
        "vs_section": "Overview",
        "vs_section_radio": "Overview",
        "vs_operator": "Research operator",
        "vs_purpose": "workflow_validation",
        "vs_consent": False,
        "vs_retention": "delete_after_analysis",
        "vs_source": "stable",
        "vs_focus_case": "",
        "vs_preflight": None,
        "vs_upload_path": "",
        "vs_upload_widget_version": 0,
        "vs_assessment_result": None,
        "vs_assistant_history": [],
        "vs_assistant_role": "operator",
        "vs_assistant_case": "",
        "vs_assistant_allow_actions": False,
        "vs_assistant_pending_prompt": "",
        "vs_assistant_media_contexts": [],
        "vs_assistant_voice_result": None,
        "vs_assistant_voice_transcript": "",
        "vs_assistant_image_result": None,
        "vs_assistant_audio_widget_version": 0,
        "vs_assistant_image_widget_version": 0,
        "vs_flash": "",
        "vs_flash_kind": "success",
        # Keep the first mobile render interactive; only later workspace changes
        # trigger the one-shot auto-close behavior.
        "vs_rendered_section": "Overview",
        "vs_navigation_nonce": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if not st.session_state.get("vs_upload_cleanup_done"):
        _purge_stale_uploads()
        st.session_state["vs_upload_cleanup_done"] = True


@st.cache_resource(show_spinner=False)
def _store() -> ConsoleStore:
    return ConsoleStore(DB_PATH)


@st.cache_resource(show_spinner=False)
def _assistant_engine() -> AssistantOrchestrator:
    return AssistantOrchestrator(ConsoleStore(DB_PATH), db_path=DB_PATH)


@st.cache_resource(show_spinner=False)
def _multimodal_engine() -> MultimodalAssistantService:
    return MultimodalAssistantService()


def _seed_if_empty(store: ConsoleStore) -> None:
    if store.list_cases():
        return
    for case in make_demo_cases():
        store.upsert_case(case, actor="demo-seed")


def _is_zh() -> bool:
    return st.session_state.get("vs_language", "ZH") == "ZH"


def _sync_language() -> None:
    language = st.session_state.get("vs_language_control", "ZH") or "ZH"
    st.session_state["vs_language"] = language
    suffix = language.lower()
    for field in ("vs_purpose", "vs_consent", "vs_retention", "vs_source"):
        st.session_state[f"{field}_control_{suffix}"] = st.session_state.get(field)


def _sync_assessment_control(field: str, widget_key: str) -> None:
    st.session_state[field] = st.session_state.get(widget_key)


def _ui(en: str, zh: str) -> str:
    return zh if _is_zh() else en


def _data_text(value: object) -> str:
    return localize_console_text(value, language="zh" if _is_zh() else "en")


def _source_text(value: object) -> str:
    if str(value) == "Synthetic candidate-release demonstration":
        return _ui("Built-in synthetic demo", "内置合成演示")
    return _data_text(value)


def _escape(value: object) -> str:
    import html

    return html.escape(str(value))


def _header(section: str) -> None:
    title = ZH[section] if _is_zh() else section
    subtitle = {
        "Overview": _ui("Operational state, quality, and work requiring attention", "查看运行状态、采集质量与待处理工作"),
        "New assessment": _ui("Consent, guided capture, quality qualification, and evidence output", "完成授权、引导采集、质量检查与证据输出"),
        "Cases": _ui("Searchable evidence registry with decision-level provenance", "可搜索的证据登记与决策级溯源"),
        "Review queue": _ui("Assign, document, and close review or retake work", "分派、记录并闭环复核或重采任务"),
        "Reports": _ui("Versioned evidence reports with policy attribution", "生成带策略归因和版本信息的证据报告"),
        "AI assistant": _ui("Evidence-bounded guidance, report explanation, and controlled workflow navigation", "基于证据的引导、报告解释与受控流程导航"),
        "Evidence": _ui("Protocol-bound metrics and non-negotiable claim boundaries", "协议限定的性能指标与不可突破的证据边界"),
        "Integrations": _ui("Validated payloads, OpenAPI schema, and report endpoints", "校验载荷、OpenAPI 规范和报告接口"),
        "Help & settings": _ui("Acquisition guidance, status definitions, privacy, and workspace settings", "采集指引、状态定义、隐私与工作区设置"),
    }[section]
    left, right = st.columns([1, 0.42])
    with left:
        st.markdown(f"<h1 class='vs-page-title'>{_escape(title)}</h1>", unsafe_allow_html=True)
        st.caption(subtitle)
    with right:
        st.markdown(
            f"<div class='vs-env'><div><b>RESEARCH</b><span>candidate-aware HR</span></div>"
            f"<div><b>{_escape(_ui('LOCAL', '本地'))}</b><span>{_escape(_ui('evidence linked', '证据已关联'))}</span></div></div>",
            unsafe_allow_html=True,
        )
        with st.popover(
            _ui("Quick guide", "快速指引"),
            icon=":material/help_outline:",
            width="stretch",
        ):
            st.markdown(
                _ui(
                    "**1. Prepare** the recording and consent.  \n**2. Assess** quality before HR.  \n**3. Act** on release, review, or retake.  \n**4. Export** the evidence report.",
                    "**1. 准备**录制与授权。  \n**2. 评估**采集质量后再运行心率。  \n**3. 处理**放行、复核或重采。  \n**4. 导出**完整证据报告。",
                )
            )
            if st.button(
                _ui("Open full guide", "查看完整教学"),
                icon=":material/arrow_forward:",
                width="stretch",
                key=f"vs_header_guide_{section}",
            ):
                _set_flash(_ui("The full workflow guide is open.", "已打开完整流程教学。"), "info")
                _go("Help & settings")
        if section != "AI assistant" and st.button(
            _ui("Ask AI assistant", "询问 AI 助手"),
            icon=":material/smart_toy:",
            width="stretch",
            key=f"vs_header_assistant_{section}",
        ):
            _go("AI assistant")
    st.markdown("<div class='vs-rule'></div>", unsafe_allow_html=True)


def _overview(store: ConsoleStore) -> None:
    cases = store.list_cases()
    reviews = store.list_reviews(include_closed=False)
    releases = sum(case.get("decision") == "release" for case in cases)
    retakes = sum(case.get("decision") == "retake" for case in cases)
    open_reviews = sum(review.get("status") != "closed" for review in reviews)
    quality = [float(case.get("quality_score") or 0) for case in cases]

    st.markdown(
        f"<div class='vs-workflow-band'><div><span>{_escape(_ui('RECOMMENDED WORKFLOW', '推荐流程'))}</span>"
        f"<b>{_escape(_ui('Capture once, qualify first, then release or route with evidence.', '一次采集，先做质量检查，再凭证据放行或转交处理。'))}</b></div>"
        f"<ol><li>{_escape(_ui('Consent + input', '授权与输入'))}</li><li>{_escape(_ui('Quality gate', '质量门控'))}</li>"
        f"<li>{_escape(_ui('Candidate decision', '候选决策'))}</li><li>{_escape(_ui('Review + report', '复核与报告'))}</li></ol></div>",
        unsafe_allow_html=True,
    )
    quick_a, quick_b, quick_c = st.columns([1, 1, 1])
    if quick_a.button(_ui("Start guided assessment", "开始引导式评估"), type="primary", icon=":material/add_circle:", width="stretch"):
        _set_flash(_ui("Assessment opened. Start with purpose and consent.", "评估已打开，请先确认用途与授权。"), "info")
        _start_assessment()
    if quick_b.button(_ui("Continue review work", "继续复核工作"), icon=":material/fact_check:", width="stretch"):
        _set_flash(_ui("Review queue opened. Select the highest-priority item first.", "已打开复核队列，请优先处理高优先级项目。"), "info")
        _go("Review queue")
    if quick_c.button(_ui("Learn the full workflow", "学习完整流程"), icon=":material/menu_book:", width="stretch"):
        _set_flash(_ui("The role-based guide is open.", "已打开分角色操作教学。"), "info")
        _go("Help & settings")

    columns = st.columns(5)
    _metric(columns[0], _ui("Cases", "案例"), str(len(cases)), _ui("stored evidence packets", "已存证据包"))
    _metric(columns[1], _ui("Released", "已放行"), str(releases), _ui("policy gate passed", "通过策略门控"), tone="teal")
    _metric(columns[2], _ui("Open reviews", "待复核"), str(open_reviews), _ui("operator action required", "需要操作员处理"), tone="amber")
    _metric(columns[3], _ui("Retakes", "需重采"), str(retakes), _ui("capture issue detected", "检测到采集问题"), tone="coral")
    _metric(columns[4], _ui("Median quality", "质量中位数"), f"{pd.Series(quality).median():.0%}", _ui("demonstration workspace", "演示工作区"))

    left, right = st.columns([1.55, 0.9], gap="large")
    with left:
        st.subheader(_ui("Case status", "案例状态"))
        status_rows = []
        for case in cases:
            status_rows.append(
                {
                    _ui("Case", "案例"): case.get("display_id"),
                    _ui("Decision", "决策"): _decision_text(str(case.get("decision"))),
                    _ui("HR", "心率"): _released_hr(case),
                    _ui("Quality", "质量"): _percent(case.get("quality_score")),
                    _ui("Next action", "下一步"): _next_step_text(case),
                }
            )
        st.dataframe(pd.DataFrame(status_rows), hide_index=True, width="stretch", height=325)
        c1, c2 = st.columns(2)
        if c1.button(_ui("Start new assessment", "开始新评估"), type="primary", icon=":material/add_circle:", width="stretch"):
            _set_flash(_ui("Assessment opened. Start with purpose and consent.", "评估已打开，请先确认用途与授权。"), "info")
            _start_assessment()
        if c2.button(_ui("Open review queue", "打开复核队列"), icon=":material/fact_check:", width="stretch"):
            _set_flash(_ui("Review queue opened.", "已打开复核队列。"), "info")
            _go("Review queue")

    with right:
        st.subheader(_ui("Work requiring attention", "需要处理的工作"))
        if not reviews:
            st.success(_ui("No open reviews.", "当前没有待复核项目。"))
        for review in reviews[:5]:
            case = review["case"]
            decision = str(case.get("decision", "review"))
            st.markdown(
                f"<div class='vs-list-row'><div><b>{_escape(case.get('display_id'))}</b>"
                f"<span class='vs-state-pill {decision}'>{_escape(_decision_text(decision))} · {_escape(_data_text(review.get('priority')))}</span></div>"
                f"<small>{_escape(_data_text(case.get('review_reason') or case.get('recommended_action')))}</small></div>",
                unsafe_allow_html=True,
            )

    st.subheader(_ui("Quality and decision distribution", "质量与决策分布"))
    chart_left, chart_right = st.columns([1.35, 1])
    with chart_left:
        _quality_chart(cases)
    with chart_right:
        _decision_chart(cases)
    st.info(
        _ui(
            "This workspace is seeded with explicit synthetic workflow cases. They exercise the interface and do not replace manuscript metrics.",
            "当前工作区使用明确标注的合成流程案例来测试产品交互，不能替代论文实验指标。",
        )
    )


def _new_assessment(store: ConsoleStore) -> None:
    _step_strip()
    st.markdown(
        f"<div class='vs-io-strip'><div><span>{_escape(_ui('INPUT', '输入'))}</span>"
        f"<b>{_escape(_ui('Consented adult RGB face video or a labeled workflow case', '已授权的成人 RGB 人脸视频或明确标注的流程样例'))}</b></div>"
        f"<div><span>{_escape(_ui('OUTPUT', '输出'))}</span>"
        f"<b>{_escape(_ui('Quality result, release/review/retake state, evidence and next action', '质量结果、放行/复核/重采状态、证据与下一步操作'))}</b></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='vs-processing-contract'><div><b>{_escape(_ui('PRIVACY CONTRACT', '隐私契约'))}</b>"
        f"<span>{_escape(_ui('Raw video stays local and is deleted after analysis in the recommended mode.', '推荐模式下，原始视频仅在本地处理并于分析后删除。'))}</span></div>"
        f"<div><b>{_escape(_ui('OUTPUT CONTRACT', '输出契约'))}</b>"
        f"<span>{_escape(_ui('Review and retake states never publish HR.', '复核与重采状态绝不发布心率。'))}</span></div></div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        language_suffix = "zh" if _is_zh() else "en"
        purpose_key = f"vs_purpose_control_{language_suffix}"
        consent_key = f"vs_consent_control_{language_suffix}"
        retention_key = f"vs_retention_control_{language_suffix}"
        source_key = f"vs_source_control_{language_suffix}"
        for field, widget_key in (
            ("vs_purpose", purpose_key),
            ("vs_consent", consent_key),
            ("vs_retention", retention_key),
            ("vs_source", source_key),
        ):
            st.session_state.setdefault(widget_key, st.session_state.get(field))
        st.subheader(_ui("1. Purpose, consent, and retention", "1. 用途、授权与留存"))
        purpose = st.selectbox(
            _ui("Intended research use", "研究用途"),
            ["workflow_validation", "algorithm_evaluation", "research_demo"],
            format_func=lambda value: {
                "workflow_validation": _ui("Workflow validation", "流程验证"),
                "algorithm_evaluation": _ui("Algorithm evaluation", "算法评估"),
                "research_demo": _ui("Research demonstration", "研究演示"),
            }[value],
            key=purpose_key,
            on_change=_sync_assessment_control,
            args=("vs_purpose", purpose_key),
        )
        consent = st.checkbox(
            _ui(
                "I confirm the recording may be processed for the selected research purpose.",
                "我确认该视频可按照所选研究用途进行处理。",
            ),
            key=consent_key,
            on_change=_sync_assessment_control,
            args=("vs_consent", consent_key),
        )
        retention = st.radio(
            _ui("Raw-video handling", "原始视频处理"),
            ["delete_after_analysis", "session_only"],
            format_func=lambda value: {
                "delete_after_analysis": _ui("Delete after analysis; retain derived evidence", "分析后删除，仅保留派生证据"),
                "session_only": _ui("Keep locally until cleared or automatically expired", "本地保留至清除或自动过期"),
            }[value],
            key=retention_key,
            on_change=_sync_assessment_control,
            args=("vs_retention", retention_key),
        )

        st.subheader(_ui("2. Input source", "2. 输入来源"))
        source = st.radio(
            _ui("Choose a source", "选择来源"),
            ["stable", "conflict", "low_light", "upload"],
            horizontal=True,
            format_func=lambda value: {
                "stable": _ui("Stable demo", "稳定样例"),
                "conflict": _ui("Conflict demo", "冲突样例"),
                "low_light": _ui("Low-light demo", "低照样例"),
                "upload": _ui("Upload video", "上传视频"),
            }[value],
            key=source_key,
            on_change=_sync_assessment_control,
            args=("vs_source", source_key),
        )

        uploaded = None
        if source == "upload":
            uploaded = st.file_uploader(
                _ui("Adult RGB face video", "成人 RGB 人脸视频"),
                type=["mp4", "mov", "avi", "mkv", "m4v"],
                key=f"vs_video_upload_{st.session_state['vs_upload_widget_version']}",
                help=_ui(
                    "Use a stable, front-facing 20-30 second recording with even lighting.",
                    "建议使用正面、稳定、光照均匀的 20-30 秒视频。",
                ),
            )
            _capture_guidance()
        else:
            st.caption(
                _ui(
                    "Built-in cases test release, review, and retake behavior without participant media.",
                    "内置案例不包含受试者媒体，用于测试放行、复核和重采流程。",
                )
            )

        action_col, reset_col = st.columns([1, 0.45])
        run_label = _ui("Run assessment", "运行评估")
        if action_col.button(run_label, type="primary", icon=":material/play_arrow:", width="stretch"):
            if not consent:
                message = _ui(
                    "Confirm processing consent before running the assessment.",
                    "运行评估前，请先确认视频处理授权。",
                )
                st.warning(message)
                st.toast(message, icon=":material/privacy_tip:")
            elif source == "upload" and uploaded is None:
                message = _ui("Upload a video before running the assessment.", "运行评估前，请先上传视频。")
                st.warning(message)
                st.toast(message, icon=":material/upload_file:")
            else:
                try:
                    if source == "upload":
                        result = _process_upload(uploaded, purpose=purpose, retention=retention)
                        store.upsert_case(result, actor=st.session_state["vs_operator"])
                        st.session_state["vs_assessment_result"] = result
                        st.session_state["vs_focus_case"] = result["case_id"]
                    else:
                        index = {"stable": 0, "conflict": 1, "low_light": 2}[source]
                        result = make_demo_cases()[index]
                        result["purpose"] = purpose
                        result["retention_policy"] = retention
                        result = ensure_output_contract(result)
                        store.upsert_case(result, actor=st.session_state["vs_operator"])
                        st.session_state["vs_assessment_result"] = result
                        st.session_state["vs_preflight"] = case_quality_snapshot(result)
                        st.session_state["vs_focus_case"] = result["case_id"]
                    message = _ui(
                        f"Assessment completed: {_decision_text(result['decision'])}. Follow the recommended next action.",
                        f"评估完成：{_decision_text(result['decision'])}。请按照推荐的下一步操作处理。",
                    )
                    st.success(message)
                    st.toast(message, icon=":material/check_circle:")
                except Exception as error:
                    st.error(
                        _ui(
                            "The assessment could not be completed. No HR was published.",
                            "本次评估未能完成，系统未发布心率。",
                        )
                    )
                    st.caption(
                        f"{_ui('Technical detail', '技术信息')}: "
                        f"{type(error).__name__}: {sanitize_report_value(str(error)[:180])}"
                    )
        if reset_col.button(_ui("Clear", "清除"), icon=":material/ink_eraser:", width="stretch"):
            _remove_session_upload()
            _reset_upload_widget()
            st.session_state["vs_assessment_result"] = None
            st.session_state["vs_preflight"] = None
            _set_flash(_ui("Assessment input and session result were cleared.", "已清除评估输入和本次会话结果。"), "info")
            st.rerun()
        if not consent:
            st.caption(
                _ui(
                    "Required before run: confirm consent. The button remains active so it can explain what is missing.",
                    "运行前必需：确认授权。按钮保持可点击，以便明确提示缺少的步骤。",
                )
            )

    with right:
        st.subheader(_ui("3. Quality qualification", "3. 质量检查"))
        preflight = st.session_state.get("vs_preflight")
        if preflight:
            _preflight_panel(preflight)
        else:
            st.markdown(
                f"<div class='vs-empty'><b>{_escape(_ui('Waiting for input', '等待输入'))}</b>"
                f"<span>{_escape(_ui('Quality checks run before the HR pipeline.', '质量检查会在心率流程之前运行。'))}</span></div>",
                unsafe_allow_html=True,
            )

        st.subheader(_ui("4. Evidence-linked output", "4. 证据关联输出"))
        result = st.session_state.get("vs_assessment_result")
        if result:
            _result_summary(result)
            _action_plan_panel(build_action_plan(result), compact=True)
            c1, c2 = st.columns(2)
            if c1.button(_ui("Open case", "打开案例"), icon=":material/folder_open:", width="stretch"):
                _set_flash(_ui("Case evidence opened.", "已打开案例证据。"), "info")
                _go("Cases")
            if c2.button(_ui("Build report", "生成报告"), icon=":material/description:", width="stretch"):
                _set_flash(_ui("Evidence report prepared for review and export.", "证据报告已准备，可查看并导出。"), "success")
                _go("Reports")
        else:
            st.caption(_ui("No result has been generated in this session.", "本次会话尚未生成结果。"))


def _process_upload(uploaded: Any, *, purpose: str, retention: str) -> dict[str, Any]:
    _remove_session_upload()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()[:12]
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in uploaded.name)
    path = UPLOAD_DIR / f"{digest}_{safe_name}"
    path.write_bytes(data)
    try:
        with st.spinner(_ui("Checking video quality...", "正在检查视频质量...")):
            try:
                preflight = video_preflight(path)
            except Exception as error:
                preflight = preflight_from_decode_error(uploaded.name, error)
            st.session_state["vs_preflight"] = preflight
        if preflight["overall"] == "fail":
            return case_from_preflight(preflight, purpose=purpose, retention_policy=retention)
        with st.spinner(_ui("Building candidates and applying the release/review gate...", "正在构建候选并执行放行/复核门控...")):
            try:
                return run_uploaded_video(
                    path,
                    purpose=purpose,
                    retention_policy=retention,
                    preflight=preflight,
                )
            except Exception as error:
                return case_from_runtime_failure(
                    preflight,
                    error,
                    purpose=purpose,
                    retention_policy=retention,
                )
    finally:
        if retention == "delete_after_analysis" and path.exists():
            path.unlink()
        elif retention == "session_only" and path.exists():
            st.session_state["vs_upload_path"] = str(path)


def _cases(store: ConsoleStore) -> None:
    cases = store.list_cases()
    if not cases:
        st.warning(_ui("No cases are available.", "当前没有案例。"))
        return
    filter_col, decision_col, source_col = st.columns([1.3, 0.8, 1])
    query = filter_col.text_input(_ui("Search", "搜索"), placeholder=_ui("Case ID or source", "案例编号或来源"))
    decision = decision_col.selectbox(
        _ui("Decision", "决策"),
        ["all", "release", "review", "retake"],
        format_func=lambda value: _data_text(value) if value == "all" else _decision_text(value),
    )
    sources = sorted({str(case.get("source_name", "")) for case in cases})
    source = source_col.selectbox(
        _ui("Source", "来源"),
        ["all", *sources],
        format_func=lambda value: _data_text(value) if value == "all" else _source_text(value),
    )

    filtered = []
    for case in cases:
        text = f"{case.get('display_id')} {case.get('case_id')} {case.get('source_name')}".lower()
        if query and query.lower() not in text:
            continue
        if decision != "all" and case.get("decision") != decision:
            continue
        if source != "all" and case.get("source_name") != source:
            continue
        filtered.append(case)
    if not filtered:
        st.info(_ui("No case matches the filters.", "没有符合筛选条件的案例。"))
        return

    rows = pd.DataFrame(
        [
            {
                _ui("Case", "案例"): case.get("display_id"),
                _ui("Decision", "决策"): _decision_text(case.get("decision")),
                _ui("Published HR", "已发布心率"): _released_hr(case),
                _ui("Quality", "质量"): _percent(case.get("quality_score")),
                _ui("Source", "来源"): _source_text(case.get("source_name")),
                _ui("Updated", "更新时间"): case.get("updated_at"),
            }
            for case in filtered
        ]
    )
    st.dataframe(rows, hide_index=True, width="stretch")
    labels = [f"{case.get('display_id')} | {_decision_text(case.get('decision'))} | {_source_text(case.get('source_name'))}" for case in filtered]
    focus_id = st.session_state.get("vs_focus_case")
    default_index = next((i for i, case in enumerate(filtered) if case.get("case_id") == focus_id), 0)
    selected_label = st.selectbox(_ui("Open case", "打开案例"), labels, index=default_index)
    selected = filtered[labels.index(selected_label)]
    st.session_state["vs_focus_case"] = selected["case_id"]
    _case_detail(selected, store)


def _case_detail(case: dict[str, Any], store: ConsoleStore) -> None:
    st.markdown("<div class='vs-section-rule'></div>", unsafe_allow_html=True)
    top, action = st.columns([1, 0.38])
    with top:
        st.subheader(f"{case.get('display_id')} · {_decision_text(case.get('decision'))}")
        st.caption(f"{case.get('case_id')} | {_source_text(case.get('source_name'))} | {case.get('policy_version')}")
    with action:
        if st.button(
            _ui("Open report", "打开报告"),
            icon=":material/description:",
            width="stretch",
            key=f"report_{case['case_id']}",
        ):
            st.session_state["vs_focus_case"] = case["case_id"]
            _set_flash(_ui("Evidence report opened for this case.", "已打开该案例的证据报告。"), "info")
            _go("Reports")

    _result_summary(case)
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.subheader(_ui("Trend and candidate branches", "趋势与候选分支"))
        _trend_chart(case)
        candidates = case.get("candidates", [])
        if candidates:
            st.dataframe(pd.DataFrame(candidates), hide_index=True, width="stretch")
        else:
            st.caption(_ui("No candidate branch was retained.", "没有保留候选分支。"))
    with right:
        st.subheader(_ui("Evidence attribution", "证据归因"))
        attribution = build_attribution(case)
        for factor in attribution["all_factors"]:
            tone = "good" if factor["direction"] > 0 else "warn"
            st.markdown(
                f"<div class='vs-factor {tone}'><b>{_escape(_data_text(factor['factor']))}</b>"
                f"<span>{_escape(_data_text(factor['status']))} · {_escape(factor['observed'])}</span>"
                f"<small>{_escape(_data_text(factor['reason']))}</small></div>",
                unsafe_allow_html=True,
            )
        st.caption(_data_text(ATTRIBUTION_BOUNDARY))

    st.subheader(_ui("Recommended next action", "推荐的下一步操作"))
    _action_plan_panel(build_action_plan(case), compact=True)

    with st.expander(_ui("Audit trail", "审计记录"), expanded=False):
        events = store.audit_events(case["case_id"])
        st.dataframe(pd.DataFrame(events), hide_index=True, width="stretch")


def _review_queue(store: ConsoleStore) -> None:
    reviews = store.list_reviews(include_closed=True)
    if not reviews:
        st.success(_ui("No review records.", "当前没有复核记录。"))
        return
    f1, f2 = st.columns(2)
    status_filter = f1.selectbox(
        _ui("Status", "状态"),
        ["open work", "all", "open", "in_review", "waiting_retake", "closed"],
        format_func=_data_text,
    )
    priority_filter = f2.selectbox(
        _ui("Priority", "优先级"),
        ["all", "urgent", "high", "routine", "low"],
        format_func=_data_text,
    )
    filtered = []
    for item in reviews:
        if status_filter == "open work" and item["status"] == "closed":
            continue
        if status_filter not in {"all", "open work"} and item["status"] != status_filter:
            continue
        if priority_filter != "all" and item["priority"] != priority_filter:
            continue
        filtered.append(item)
    if not filtered:
        st.info(_ui("No review matches the filters.", "没有符合筛选条件的复核项目。"))
        return
    table = pd.DataFrame(
        [
            {
                _ui("Case", "案例"): item["display_id"],
                _ui("Status", "状态"): _data_text(item["status"]),
                _ui("Priority", "优先级"): _data_text(item["priority"]),
                _ui("Assignee", "负责人"): item["assignee"] or _ui("Unassigned", "未分派"),
                _ui("Reason", "原因"): _data_text(item["case"].get("review_reason")),
                _ui("Updated", "更新时间"): item["updated_at"],
            }
            for item in filtered
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")
    labels = [f"{item['display_id']} | {_data_text(item['priority'])} | {_data_text(item['status'])}" for item in filtered]
    selected = filtered[labels.index(st.selectbox(_ui("Review item", "复核项目"), labels))]
    case = selected["case"]
    left, right = st.columns([1, 0.9], gap="large")
    with left:
        _result_summary(case)
        st.markdown(f"**{_ui('Reason','原因')}**: {_data_text(case.get('review_reason'))}")
        st.markdown(f"**{_ui('Recommended action','建议操作')}**: {_data_text(case.get('recommended_action'))}")
        with st.expander(_ui("Candidate evidence", "候选证据"), expanded=True):
            candidates = case.get("candidates", [])
            if candidates:
                st.dataframe(pd.DataFrame(candidates), hide_index=True, width="stretch")
            else:
                st.caption(_ui("No candidates were retained.", "未保留候选。"))
    with right:
        st.subheader(_ui("Review action", "复核操作"))
        with st.form(f"review_form_{case['case_id']}"):
            status = st.selectbox(
                _ui("Status", "状态"),
                ["open", "in_review", "waiting_retake", "closed"],
                index=["open", "in_review", "waiting_retake", "closed"].index(selected["status"]),
                format_func=_data_text,
            )
            priority = st.selectbox(
                _ui("Priority", "优先级"),
                ["urgent", "high", "routine", "low"],
                index=["urgent", "high", "routine", "low"].index(selected["priority"]),
                format_func=_data_text,
            )
            assignee = st.text_input(_ui("Assignee", "负责人"), value=selected["assignee"])
            note = st.text_area(_ui("Reviewer note", "复核备注"), value=selected["note"], height=100)
            resolution = st.selectbox(
                _ui("Resolution", "处理结果"),
                ["", "request_retake", "retain_for_research_review", "close_without_release"],
                index=["", "request_retake", "retain_for_research_review", "close_without_release"].index(selected["resolution"]) if selected["resolution"] in ["", "request_retake", "retain_for_research_review", "close_without_release"] else 0,
                format_func=lambda value: _ui("Not set", "未设置") if value == "" else _data_text(value),
            )
            submitted = st.form_submit_button(
                _ui("Save review", "保存复核"),
                type="primary",
                icon=":material/save:",
                width="stretch",
            )
        if submitted:
            store.update_review(
                case["case_id"],
                status=status,
                priority=priority,
                assignee=assignee,
                note=note,
                resolution=resolution,
                actor=st.session_state["vs_operator"],
            )
            _set_flash(
                _ui(
                "Review record saved with an audit event.",
                "复核记录已保存并写入审计事件。",
                ),
                "success",
            )
            st.rerun()
        st.subheader(_ui("Audit trail", "审计记录"))
        st.dataframe(pd.DataFrame(store.audit_events(case["case_id"])), hide_index=True, width="stretch")


def _reports(store: ConsoleStore) -> None:
    cases = store.list_cases()
    if not cases:
        st.warning(_ui("No cases are available for reporting.", "当前没有可生成报告的案例。"))
        return
    labels = [f"{case.get('display_id')} | {_decision_text(case.get('decision'))}" for case in cases]
    focus_id = st.session_state.get("vs_focus_case")
    default_index = next((i for i, case in enumerate(cases) if case.get("case_id") == focus_id), 0)
    selected_label = st.selectbox(_ui("Case report", "案例报告"), labels, index=default_index)
    case = cases[labels.index(selected_label)]
    st.session_state["vs_focus_case"] = case["case_id"]
    review = next((item for item in store.list_reviews() if item["case_id"] == case["case_id"]), None)
    payload = build_report_payload(case, review=review, audit_events=store.audit_events(case["case_id"]))
    language = "zh" if _is_zh() else "en"
    markdown = build_report_markdown(payload, language=language)
    pdf = build_report_pdf(payload, language=language)

    _report_preview(payload)
    action_left, action_right = st.columns([1, 1])
    if case.get("decision") == "release":
        if action_left.button(
            _ui("Open case evidence", "打开案例证据"),
            icon=":material/folder_open:",
            width="stretch",
            key="report_open_case",
        ):
            _set_flash(_ui("Case evidence opened.", "已打开案例证据。"), "info")
            _go("Cases")
    else:
        if action_left.button(
            _ui("Open review workflow", "打开复核流程"),
            icon=":material/fact_check:",
            width="stretch",
            key="report_open_review",
        ):
            _set_flash(_ui("Review workflow opened for the selected case.", "已打开所选案例的复核流程。"), "info")
            _go("Review queue")
    if action_right.button(
        _ui("Start a corrected recording", "开始纠正性重采"),
        icon=":material/videocam:",
        width="stretch",
        key="report_start_retake",
    ):
        _set_flash(_ui("Assessment opened. Apply the report recommendations before recording.", "评估已打开，请在录制前落实报告中的建议。"), "info")
        _start_assessment()

    st.subheader(_ui("Export evidence package", "导出证据包"))
    st.caption(
        _ui(
            "PDF is the human-readable report; JSON preserves the full contract; Markdown supports review; CSV contains the case-level row.",
            "PDF 用于人工阅读；JSON 保留完整数据契约；Markdown 便于复核；CSV 提供案例级数据行。",
        )
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button(_ui("Report PDF", "报告 PDF"), pdf, file_name=f"{case['display_id']}_evidence_report.pdf", mime="application/pdf", icon=":material/picture_as_pdf:", width="stretch")
    c2.download_button(_ui("Evidence JSON", "证据 JSON"), json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), file_name=f"{case['display_id']}_evidence_report.json", mime="application/json", icon=":material/data_object:", width="stretch")
    c3.download_button(_ui("Review Markdown", "复核 Markdown"), markdown.encode("utf-8"), file_name=f"{case['display_id']}_evidence_report.md", mime="text/markdown", icon=":material/article:", width="stretch")
    c4.download_button(
        _ui("Case CSV", "案例 CSV"),
        pd.DataFrame([payload["case"]]).drop(columns=[col for col in ["candidates", "trend_bpm", "preflight", "window_results"] if col in payload["case"]]).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{case['display_id']}_case.csv",
        mime="text/csv",
        icon=":material/table_view:",
        width="stretch",
    )
    tabs = st.tabs(
        [
            _ui("Report detail", "报告详情"),
            _ui("Evidence to action", "证据到行动"),
            _ui("Attribution", "归因"),
            _ui("Review & audit", "复核与审计"),
            _ui("Structured data", "结构化数据"),
        ]
    )
    with tabs[0]:
        st.subheader(_ui("Trend and retained candidates", "趋势与保留候选"))
        _trend_chart(case)
        candidates = case.get("candidates", [])
        if candidates:
            st.dataframe(pd.DataFrame(candidates), hide_index=True, width="stretch")
        else:
            st.info(_ui("No candidate branch was retained for this case.", "该案例未保留候选分支。"))
        st.subheader(_ui("Implementation provenance", "实现与运行溯源"))
        runtime = case.get("runtime_metadata") or {}
        preflight = case.get("preflight") or {}
        provenance_rows = [
            [_ui("Model version", "模型版本"), case.get("model_version") or "N/A"],
            [_ui("Policy version", "策略版本"), case.get("policy_version") or "N/A"],
            [
                _ui("Face detector backend", "人脸检测后端"),
                runtime.get("detector_backend") or preflight.get("face_detector_backend") or "N/A",
            ],
            [
                _ui("Detector model integrity", "检测模型完整性"),
                runtime.get("detector_model_integrity") or "N/A",
            ],
            [
                _ui("Audited route omissions", "已审计的路由省略数"),
                runtime.get("route_failure_count", "N/A"),
            ],
            [
                _ui("Analysis sampling rate", "分析采样率"),
                f"{runtime['analysis_fps']} fps" if runtime.get("analysis_fps") is not None else "N/A",
            ],
            [
                _ui("Analysis frame budget", "分析帧预算"),
                runtime.get("max_analysis_frames", "N/A"),
            ],
        ]
        st.dataframe(
            pd.DataFrame(
                provenance_rows,
                columns=[_ui("Field", "字段"), _ui("Value", "内容")],
            ),
            hide_index=True,
            width="stretch",
        )
    with tabs[1]:
        _action_plan_panel(payload["action_plan"], compact=False)
    with tabs[2]:
        attribution = payload["attribution"]
        attribution_rows = [
            {
                _ui("Factor", "因素"): _data_text(item.get("factor")),
                _ui("Observed", "观测值"): _data_text(item.get("observed")),
                _ui("Direction", "方向"): _data_text(item.get("status")),
                _ui("Reason", "理由"): _data_text(item.get("reason")),
                _ui("Source field", "来源字段"): item.get("source_field"),
            }
            for item in attribution["all_factors"]
        ]
        st.dataframe(pd.DataFrame(attribution_rows), hide_index=True, width="stretch")
        st.info(_data_text(attribution["boundary"]))
    with tabs[3]:
        review = payload.get("review", {})
        review_rows = [
            [_ui("Status", "状态"), _data_text(review.get("status", "not opened"))],
            [_ui("Priority", "优先级"), _data_text(review.get("priority", ""))],
            [_ui("Assignee", "负责人"), review.get("assignee") or _ui("Unassigned", "未分派")],
            [_ui("Resolution", "处理结果"), _data_text(review.get("resolution", ""))],
            [_ui("Reviewer note", "复核备注"), review.get("note") or _ui("No note", "暂无备注")],
        ]
        st.dataframe(pd.DataFrame(review_rows, columns=[_ui("Field", "字段"), _ui("Value", "内容")]), hide_index=True, width="stretch")
        events = payload.get("audit_events", [])
        if events:
            st.dataframe(pd.DataFrame(events), hide_index=True, width="stretch")
        else:
            st.info(_ui("No audit event is available.", "暂无审计事件。"))
    with tabs[4]:
        st.json(payload, expanded=False)


def _report_preview(payload: dict[str, Any]) -> None:
    case = payload["case"]
    plan = payload["action_plan"]
    decision = str(case.get("decision"))
    released = _released_hr(case)
    acquisition_gate = _acquisition_gate_text(case)
    st.markdown(
        f"<div class='vs-report-sheet'><header><div><span>VITALSSIGHT / { _escape(payload['report_version']) }</span>"
        f"<h2>{_escape(_ui('Evidence report', '证据报告'))}</h2>"
        f"<p>{_escape(case.get('display_id'))} · {_escape(payload['generated_at'])}</p></div>"
        f"<strong class='vs-status {decision}'>{_escape(_decision_text(decision))}</strong></header>"
        f"<section class='vs-report-hero'><div><small>{_escape(_ui('Released HR', '已发布心率'))}</small><b>{_escape(released)}</b></div>"
        f"<div><small>{_escape(_ui('Acquisition gate', '采集门控'))}</small><b>{_escape(acquisition_gate)}</b></div>"
        f"<div><small>{_escape(_ui('Policy', '策略'))}</small><b>{_escape(case.get('policy_version'))}</b></div></section>"
        f"<section class='vs-report-narrative'><span>{_escape(_ui('CURRENT INTERPRETATION', '当前解释'))}</span>"
        f"<h3>{_escape(_data_text(plan['headline']))}</h3><p>{_escape(_data_text(plan['rationale']))}</p></section>"
        f"<section class='vs-report-recommendation'><span>{_escape(_ui('RECOMMENDED NEXT ACTION', '推荐的下一步操作'))}</span>"
        f"<b>{_escape(_data_text(plan['recommendation']))}</b><p>{_escape(_data_text(plan['expected_outcome']))}</p></section>"
        f"<footer>{_escape(_data_text(plan['boundary']))}</footer></div>",
        unsafe_allow_html=True,
    )


def _action_plan_panel(plan: dict[str, Any], *, compact: bool) -> None:
    decision = str(plan.get("decision", "review"))
    st.markdown(
        f"<div class='vs-action-head {decision}'><span>{_escape(_ui('WHY THIS ACTION', '为什么这样处理'))}</span>"
        f"<b>{_escape(_data_text(plan.get('headline')))}</b><p>{_escape(_data_text(plan.get('rationale')))}</p></div>",
        unsafe_allow_html=True,
    )
    evidence = plan.get("evidence", [])
    if compact:
        evidence = [item for item in evidence if item.get("status") == "triggered"][:3]
    if evidence:
        rows = [
            {
                _ui("Signal", "信号"): _data_text(item.get("signal")),
                _ui("Observed", "观测值"): _data_text(item.get("observed")),
                _ui("Target", "目标"): _data_text(item.get("target")),
                _ui("State", "状态"): _data_text(item.get("status")),
                _ui("Reason", "原因"): _data_text(item.get("reason")),
            }
            for item in evidence
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    steps = plan.get("steps", [])[:3] if compact else plan.get("steps", [])
    for item in steps:
        st.markdown(
            f"<div class='vs-action-step'><b>{item.get('step')}</b><div><strong>{_escape(_data_text(item.get('action')))}</strong>"
            f"<span>{_escape(_ui('Basis', '依据'))}: {_escape(_data_text(item.get('because')))}</span>"
            f"<small>{_escape(_ui('Verify', '复核标准'))}: {_escape(_data_text(item.get('verification')))}</small></div></div>",
            unsafe_allow_html=True,
        )
    if not compact:
        st.markdown(
            f"<div class='vs-escalation'><b>{_escape(_ui('If the issue persists', '问题仍存在时'))}</b>"
            f"<span>{_escape(_data_text(plan.get('escalation')))}</span></div>",
            unsafe_allow_html=True,
        )
        st.caption(_data_text(plan.get("boundary")))


def _set_media_context(raw_context: dict[str, Any]) -> None:
    context = AssistantMediaContext.model_validate(raw_context)
    existing = [
        item
        for item in st.session_state.get("vs_assistant_media_contexts", [])
        if str(item.get("kind")) != context.kind.value
    ]
    st.session_state["vs_assistant_media_contexts"] = [*existing, context.model_dump(mode="json")][-2:]


def _multimodal_intake(multimodal: MultimodalAssistantService) -> None:
    health = multimodal.health()
    image_state = _ui("Image ready", "图片就绪") if health.image.available else _ui("Technical fallback", "图片技术降级")
    speech_state = _ui("Voice ready", "语音就绪") if health.speech.available else _ui("Voice unavailable", "语音不可用")
    st.markdown(
        "<div class='vs-modalities'>"
        f"<div class='{'ready' if health.speech.available else 'degraded'}'><span>{_escape(_ui('VOICE', '语音'))}</span><b>{_escape(speech_state)}</b><small>{_escape(health.speech.model)}</small></div>"
        f"<div class='{'ready' if health.image.available else 'degraded'}'><span>{_escape(_ui('IMAGE', '图片'))}</span><b>{_escape(image_state)}</b><small>{_escape(health.image.model)}</small></div>"
        f"<div class='bounded'><span>{_escape(_ui('MEDIA POLICY', '媒体策略'))}</span><b>{_escape(_ui('Transient, non-authoritative context', '临时、非权威上下文'))}</b><small>{_escape(_ui('Raw media is not retained', '不保留原始媒体'))}</small></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander(_ui("Voice and image input", "语音与图片输入"), expanded=False, icon=":material/add_photo_alternate:"):
        st.caption(
            _ui(
                "Review a transcript or image summary before sending it. Media can support workflow guidance only; it cannot measure vitals, identify a person, diagnose, or override the recorded gate.",
                "发送前请核对转写或图片摘要。媒体只用于工作流引导，不能测量生命体征、识别人、诊断或覆盖系统已记录的门控结果。",
            )
        )
        voice_tab, image_tab = st.tabs([_ui("Voice", "语音"), _ui("Image", "图片")])

        with voice_tab:
            audio = st.audio_input(
                _ui("Record a question or instruction", "录制问题或操作要求"),
                key=f"vs_assistant_audio_{st.session_state['vs_assistant_audio_widget_version']}",
                disabled=not health.speech.available,
                help=_ui(
                    "The recording is transcribed locally and deleted immediately after processing.",
                    "录音在本机转写，处理后立即删除。",
                ),
            )
            if not health.speech.available:
                st.info(_ui(health.speech.details, "本地语音转写组件尚未就绪，仍可使用文字输入。"))
            if audio is not None:
                if st.button(
                    _ui("Transcribe locally", "本地转写"),
                    icon=":material/graphic_eq:",
                    type="primary",
                    key="vs_assistant_transcribe",
                ):
                    try:
                        with st.spinner(_ui("Transcribing and checking speech quality...", "正在转写并检查语音质量……")):
                            result = multimodal.transcribe_audio(
                                audio.getvalue(),
                                filename=getattr(audio, "name", "voice.wav") or "voice.wav",
                                content_type=getattr(audio, "type", "audio/wav") or "audio/wav",
                                language=AssistantLanguage.zh if _is_zh() else AssistantLanguage.en,
                            )
                        st.session_state["vs_assistant_voice_result"] = result.model_dump(mode="json")
                        st.session_state["vs_assistant_voice_transcript"] = result.transcript
                        st.rerun()
                    except MediaProcessingError as error:
                        st.error(str(error))

            voice_result = st.session_state.get("vs_assistant_voice_result")
            if voice_result:
                transcript = st.text_area(
                    _ui("Review and edit transcript", "核对并编辑转写"),
                    key="vs_assistant_voice_transcript",
                    height=110,
                    max_chars=4000,
                )
                quality = str(voice_result.get("quality") or "uncertain")
                if quality == "uncertain":
                    st.warning(_ui("Speech quality was uncertain. Correct the transcript before sending.", "语音质量不确定，请在发送前校正转写内容。"))
                else:
                    st.success(
                        _ui(
                            f"Transcript ready · {voice_result.get('duration_seconds', 0):.1f}s · {voice_result.get('detected_language', 'unknown')}",
                            f"转写就绪 · {voice_result.get('duration_seconds', 0):.1f} 秒 · {voice_result.get('detected_language', 'unknown')}",
                        )
                    )
                use_voice, clear_voice = st.columns([1, 0.42])
                if use_voice.button(
                    _ui("Use transcript as question", "将转写作为问题"),
                    type="primary",
                    icon=":material/send:",
                    width="stretch",
                    disabled=not bool(transcript.strip()),
                    key="vs_assistant_use_voice",
                ):
                    context = dict(voice_result["context"])
                    context["summary"] = transcript.strip()
                    _set_media_context(context)
                    st.session_state["vs_assistant_pending_prompt"] = transcript.strip()
                    st.rerun()
                if clear_voice.button(
                    _ui("Discard transcript", "丢弃转写"),
                    icon=":material/close:",
                    width="stretch",
                    key="vs_assistant_clear_voice",
                ):
                    st.session_state["vs_assistant_voice_result"] = None
                    st.rerun()

        with image_tab:
            image_file = st.file_uploader(
                _ui("Upload a screenshot, report image, or capture frame", "上传界面截图、报告图片或采集帧"),
                type=["jpg", "jpeg", "png", "webp"],
                key=f"vs_assistant_image_{st.session_state['vs_assistant_image_widget_version']}",
                help=_ui(
                    "The image is normalized in memory, EXIF is removed, and the original is not retained.",
                    "图片仅在内存中标准化并移除 EXIF，原图不会被保留。",
                ),
            )
            image_question = st.text_input(
                _ui("What should the assistant focus on?", "希望助手重点看什么？"),
                placeholder=_ui("For example: explain the visible workflow issue", "例如：解释图中可见的工作流问题"),
                key="vs_assistant_image_question",
                max_chars=600,
            )
            if image_file is not None:
                preview_col, action_col = st.columns([1.2, 0.8])
                preview_col.image(image_file, caption=image_file.name, width="stretch")
                if action_col.button(
                    _ui("Analyze safely", "安全分析"),
                    icon=":material/image_search:",
                    type="primary",
                    width="stretch",
                    key="vs_assistant_analyze_image",
                ):
                    try:
                        with st.spinner(_ui("Removing metadata and analyzing visible content...", "正在移除元数据并分析可见内容……")):
                            result = multimodal.analyze_image(
                                image_file.getvalue(),
                                filename=image_file.name,
                                content_type=image_file.type or "application/octet-stream",
                                question=image_question,
                                language=AssistantLanguage.zh if _is_zh() else AssistantLanguage.en,
                            )
                        st.session_state["vs_assistant_image_result"] = result.model_dump(mode="json")
                        st.rerun()
                    except MediaProcessingError as error:
                        st.error(str(error))

            image_result = st.session_state.get("vs_assistant_image_result")
            if image_result:
                if image_result.get("degraded"):
                    st.warning(_ui("Vision model fallback was used; only technical intake checks are available.", "视觉模型已降级；当前仅提供技术接入检查。"))
                st.markdown(f"**{_ui('Visual summary', '图片摘要')}**  \n{image_result.get('summary', '')}")
                if image_result.get("workflow_relevance"):
                    st.caption(f"{_ui('Workflow relevance', '工作流关联')}: {image_result.get('workflow_relevance')}")
                if image_result.get("visible_text"):
                    with st.expander(_ui("Visible text (untrusted)", "可见文字（非可信）")):
                        st.write(image_result.get("visible_text"))
                checks = image_result.get("technical_checks") or {}
                if checks:
                    st.dataframe(
                        pd.DataFrame(
                            [{_ui("Check", "检查"): key, _ui("Result", "结果"): value} for key, value in checks.items()]
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                attach_col, ask_col, clear_col = st.columns([0.8, 0.9, 0.42])
                if attach_col.button(
                    _ui("Attach to next question", "附加到下个问题"),
                    icon=":material/attach_file:",
                    width="stretch",
                    key="vs_assistant_attach_image",
                ):
                    _set_media_context(image_result["context"])
                    _set_flash(_ui("Image context attached. Type a question below.", "图片上下文已附加，请在下方输入问题。"), "info")
                    st.rerun()
                if ask_col.button(
                    _ui("Ask with this image", "结合图片提问"),
                    icon=":material/send:",
                    type="primary",
                    width="stretch",
                    key="vs_assistant_ask_image",
                ):
                    _set_media_context(image_result["context"])
                    st.session_state["vs_assistant_pending_prompt"] = image_question.strip() or _ui(
                        "Explain how this image relates to the VitalsSight workflow and what the user should do next.",
                        "解释这张图片与 VitalsSight 工作流的关系，以及用户下一步应该做什么。",
                    )
                    st.rerun()
                if clear_col.button(
                    _ui("Clear", "清除"),
                    icon=":material/close:",
                    width="stretch",
                    key="vs_assistant_clear_image",
                ):
                    st.session_state["vs_assistant_image_result"] = None
                    st.session_state["vs_assistant_image_widget_version"] += 1
                    st.rerun()

    attached = st.session_state.get("vs_assistant_media_contexts", [])
    if attached:
        labels = ", ".join(
            _ui("voice transcript", "语音转写") if item.get("kind") == "audio_transcript" else _ui("image context", "图片上下文")
            for item in attached
        )
        status_col, clear_col = st.columns([1, 0.22])
        status_col.info(_ui(f"Attached to the next question: {labels}", f"已附加到下个问题：{labels}"))
        if clear_col.button(
            _ui("Remove", "移除"),
            icon=":material/link_off:",
            width="stretch",
            key="vs_assistant_remove_media",
        ):
            st.session_state["vs_assistant_media_contexts"] = []
            st.rerun()


def _assistant(store: ConsoleStore) -> None:
    engine = _assistant_engine()
    multimodal = _multimodal_engine()
    health = engine.health()
    ready = health.model_available
    status_class = "ready" if ready else "degraded"
    status_title = _ui("Qwen assistant ready", "Qwen 助手已就绪") if ready else _ui("Evidence fallback active", "证据降级模式已启用")
    status_detail = (
        _ui(
            f"{health.provider} · {health.model} · local knowledge {health.knowledge_chunks} chunks",
            f"{health.provider} · {health.model} · 本地知识 {health.knowledge_chunks} 个片段",
        )
        if ready
        else _ui(
            "The language model is offline or unavailable. Case tools and deterministic guidance remain operational.",
            "语言模型离线或不可用；案例工具和确定性证据引导仍可正常运行。",
        )
    )
    st.markdown(
        f"<div class='vs-assistant-status {status_class}'><i></i><div><b>{_escape(status_title)}</b>"
        f"<span>{_escape(status_detail)}</span></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='vs-assistant-contract'><div><span>{_escape(_ui('SYSTEM FACTS', '系统事实'))}</span>"
        f"<b>{_escape(_ui('Measurements, output state, policy and audit remain authoritative.', '测量值、输出状态、策略和审计记录始终是权威来源。'))}</b></div>"
        f"<div><span>{_escape(_ui('AI EXPLANATION', 'AI 解释'))}</span>"
        f"<b>{_escape(_ui('The assistant explains and navigates; it cannot override the gate.', '助手负责解释与导航，不能覆盖门控决策。'))}</b></div></div>",
        unsafe_allow_html=True,
    )
    _multimodal_intake(multimodal)

    cases = store.list_cases()
    case_lookup = {str(case["case_id"]): case for case in cases}
    case_options = [""] + list(case_lookup)
    focus = str(st.session_state.get("vs_focus_case") or "")
    if not st.session_state.get("vs_assistant_case") and focus in case_lookup:
        st.session_state["vs_assistant_case"] = focus

    role_col, case_col, mode_col = st.columns([0.75, 1.35, 0.9])
    role_labels = {
        "operator": _ui("Capture operator", "采集操作员"),
        "reviewer": _ui("Evidence reviewer", "证据复核人员"),
        "clinician": _ui("Clinical reader", "医护阅读者"),
        "admin": _ui("Administrator", "管理员"),
    }
    role = role_col.selectbox(
        _ui("Assistant role", "助手角色"),
        list(role_labels),
        format_func=lambda value: role_labels[value],
        key="vs_assistant_role",
    )
    selected_case = case_col.selectbox(
        _ui("Evidence context", "证据上下文"),
        case_options,
        format_func=lambda value: (
            _ui("General guidance (no case)", "通用指引（不指定案例）")
            if not value
            else f"{case_lookup[value].get('display_id')} · {_decision_text(case_lookup[value].get('decision'))}"
        ),
        key="vs_assistant_case",
    )
    if selected_case:
        st.session_state["vs_focus_case"] = selected_case
    can_propose = role in {"reviewer", "admin"} and health.actions_enabled
    allow_actions = mode_col.toggle(
        _ui("Prepare review updates", "准备复核更新"),
        value=bool(st.session_state.get("vs_assistant_allow_actions")) if can_propose else False,
        disabled=not can_propose,
        help=_ui(
            "The assistant can only prepare a change. A second explicit confirmation is always required.",
            "助手只能准备变更，真正保存始终需要第二次明确确认。",
        ),
        key="vs_assistant_allow_actions",
    )
    mode_col.caption(
        _ui("Actions enabled by policy", "策略已启用动作")
        if health.actions_enabled
        else _ui("Read-only policy", "只读策略")
    )

    st.subheader(_ui("Ask from the workflow", "从工作流直接提问"))
    quick_prompts = [
        (_ui("Explain this state", "解释当前状态"), _ui("Why is this case in its current state?", "为什么这个案例处于当前状态？"), ":material/help_center:"),
        (_ui("Retake guidance", "重拍指引"), _ui("What exactly should be corrected before the next recording?", "下一次录制前具体需要纠正什么？"), ":material/replay:"),
        (_ui("Report summary", "报告摘要"), _ui("Summarize this evidence report and its limitations.", "总结这份证据报告及其局限性。"), ":material/summarize:"),
        (_ui("Failed metrics", "失败指标"), _ui("Which recorded checks failed or warned, and why?", "哪些已记录检查失败或警告，原因是什么？"), ":material/monitor_heart:"),
    ]
    quick_cols = st.columns(4)
    for index, (label, prompt, icon) in enumerate(quick_prompts):
        if quick_cols[index].button(label, icon=icon, width="stretch", key=f"vs_assistant_quick_{index}"):
            st.session_state["vs_assistant_pending_prompt"] = prompt
            st.rerun()

    history: list[dict[str, Any]] = st.session_state.get("vs_assistant_history", [])
    toolbar_left, toolbar_right = st.columns([1, 0.25])
    toolbar_left.caption(
        _ui(
            "Answers cite case, report, policy, or local knowledge evidence. Conversation text stays in this browser session; audit stores hashes, not raw chat.",
            "回答引用案例、报告、策略或本地知识证据。对话文本仅保留在当前浏览器会话；审计仅保存哈希，不保存原始对话。",
        )
    )
    if toolbar_right.button(
        _ui("Clear chat", "清空对话"),
        icon=":material/delete_sweep:",
        width="stretch",
        disabled=not bool(history),
        key="vs_assistant_clear",
    ):
        st.session_state["vs_assistant_history"] = []
        st.session_state["vs_assistant_media_contexts"] = []
        st.rerun()

    for index, entry in enumerate(history):
        role_name = str(entry.get("role") or "assistant")
        with st.chat_message(role_name):
            if role_name == "user":
                st.markdown(str(entry.get("content") or ""))
                media_items = entry.get("media_contexts") or []
                if media_items:
                    labels = ", ".join(
                        _ui("voice transcript", "语音转写")
                        if item.get("kind") == "audio_transcript"
                        else _ui("image context", "图片上下文")
                        for item in media_items
                    )
                    st.caption(_ui(f"Transient context: {labels}", f"临时上下文：{labels}"))
            else:
                payload = entry.get("response") or {}
                st.markdown(str(payload.get("answer") or entry.get("content") or ""))
                provider_label = f"{payload.get('provider', 'unknown')} · {payload.get('model', 'unknown')}"
                if payload.get("degraded"):
                    st.caption(_ui(f"Fallback response · {provider_label}", f"降级回答 · {provider_label}"))
                else:
                    st.caption(_ui(f"Local model · {provider_label}", f"本地模型 · {provider_label}"))
                refs = payload.get("evidence_refs") or []
                if refs:
                    with st.expander(_ui(f"Evidence cited ({len(refs)})", f"引用证据（{len(refs)}）")):
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        _ui("ID", "编号"): item.get("evidence_id"),
                                        _ui("Evidence", "证据"): item.get("label"),
                                        _ui("Source", "来源"): item.get("source"),
                                        _ui("Recorded value", "记录值"): item.get("value"),
                                    }
                                    for item in refs
                                ]
                            ),
                            hide_index=True,
                            width="stretch",
                        )
                actions = payload.get("recommended_actions") or []
                if actions:
                    with st.expander(_ui("Operational next steps", "操作下一步")):
                        for action_index, item in enumerate(actions, start=1):
                            st.markdown(
                                f"**{action_index}. {_escape(item.get('label'))}**  \n"
                                f"{_escape(item.get('rationale'))}"
                            )
                            if item.get("verification"):
                                st.caption(f"{_ui('Verify', '复核标准')}: {item.get('verification')}")
                navigation = payload.get("navigation_target")
                if navigation in SECTIONS and st.button(
                    _ui(f"Open {navigation}", f"打开{ZH.get(navigation, navigation)}"),
                    icon=":material/arrow_forward:",
                    key=f"vs_assistant_nav_{index}_{navigation}",
                ):
                    _go(str(navigation))
                pending = payload.get("pending_action")
                if pending:
                    st.warning(
                        _ui(
                            f"Pending only: {pending.get('summary')}. No record has changed.",
                            f"仅为待确认动作：{pending.get('summary')}。当前尚未修改任何记录。",
                        )
                    )
                    confirm_col, reject_col = st.columns(2)
                    if confirm_col.button(
                        _ui("Confirm and save", "确认并保存"),
                        icon=":material/check_circle:",
                        type="primary",
                        width="stretch",
                        key=f"vs_assistant_confirm_{index}",
                    ):
                        try:
                            result = engine.confirm(str(pending["token"]), actor=st.session_state["vs_operator"])
                            _set_flash(_data_text(result.message), "success")
                        except (PermissionError, KeyError, ValueError) as error:
                            _set_flash(str(error), "error")
                        st.rerun()
                    if reject_col.button(
                        _ui("Reject change", "拒绝变更"),
                        icon=":material/cancel:",
                        width="stretch",
                        key=f"vs_assistant_reject_{index}",
                    ):
                        try:
                            result = engine.reject(str(pending["token"]), actor=st.session_state["vs_operator"])
                            _set_flash(_data_text(result.message), "info")
                        except (KeyError, ValueError) as error:
                            _set_flash(str(error), "error")
                        st.rerun()
                st.caption(f"Trace {payload.get('tool_trace_id', 'N/A')} · {_data_text(payload.get('warning_or_boundary', ''))}")

    prompt = st.chat_input(_ui("Ask about a case, report, retake, review, or workflow", "询问案例、报告、重拍、复核或操作流程"))
    queued = str(st.session_state.pop("vs_assistant_pending_prompt", "") or "")
    prompt = queued or prompt
    if prompt:
        media_contexts = [
            AssistantMediaContext.model_validate(item)
            for item in st.session_state.get("vs_assistant_media_contexts", [])
        ]
        prior_turns = [
            ChatTurn(role=item["role"], content=str(item.get("content") or (item.get("response") or {}).get("answer") or ""))
            for item in history[-8:]
            if item.get("role") in {"user", "assistant"}
        ]
        history.append(
            {
                "role": "user",
                "content": prompt,
                "media_contexts": [item.model_dump(mode="json") for item in media_contexts],
            }
        )
        request = AssistantChatRequest(
            message=prompt,
            case_id=selected_case or None,
            role=AssistantRole(role),
            language=AssistantLanguage.zh if _is_zh() else AssistantLanguage.en,
            history=prior_turns,
            actor=st.session_state["vs_operator"],
            conversation_id=st.session_state.get("vs_assistant_conversation_id"),
            allow_action_proposals=bool(allow_actions),
            media_contexts=media_contexts,
        )
        with st.spinner(_ui("Checking evidence and composing a bounded answer...", "正在核对证据并生成受约束回答……")):
            response = engine.chat(request)
        st.session_state["vs_assistant_conversation_id"] = response.conversation_id
        history.append({"role": "assistant", "content": response.answer, "response": response.model_dump(mode="json")})
        st.session_state["vs_assistant_history"] = history
        st.session_state["vs_assistant_media_contexts"] = []
        st.rerun()


def _evidence() -> None:
    metrics = pd.read_csv(HEADLINE_METRICS, encoding="utf-8-sig") if HEADLINE_METRICS.exists() else pd.DataFrame()
    protocol = json.loads(PROTOCOL_SUMMARY.read_text(encoding="utf-8")) if PROTOCOL_SUMMARY.exists() else {}
    st.warning(_data_text(CLAIM_BOUNDARY))
    if metrics.empty:
        st.info(_ui("No public protocol metrics were found.", "没有找到公开协议指标。"))
        return
    st.subheader(_ui("Protocol-bound headline metrics", "协议限定的核心指标"))
    st.dataframe(metrics, hide_index=True, width="stretch")
    chart_data = metrics.copy()
    chart_data["mae_numeric"] = pd.to_numeric(chart_data["mae_bpm"].astype(str).str.extract(r"([0-9.]+)")[0], errors="coerce")
    chart_data = chart_data.dropna(subset=["mae_numeric"])
    fig = go.Figure(
        go.Bar(
            x=chart_data["protocol_key"],
            y=chart_data["mae_numeric"],
            marker_color="#5D8196",
            text=chart_data["mae_bpm"],
            textposition="outside",
            hovertext=chart_data["interpretation"],
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=30, r=20, t=24, b=100),
        yaxis_title="MAE (BPM)",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#526773", family="Segoe UI, sans-serif", size=12),
    )
    fig.update_xaxes(tickangle=-28)
    fig.update_yaxes(gridcolor="#E7EDF0", zeroline=False)
    st.plotly_chart(fig, width="stretch")

    st.subheader(_ui("Protocol invariants", "协议不变量"))
    invariants = protocol.get("invariants", {}) if isinstance(protocol, dict) else {}
    invariant_rows = [{_ui("Invariant", "不变量"): key, _ui("Locked", "锁定"): bool(value)} for key, value in invariants.items()]
    st.dataframe(pd.DataFrame(invariant_rows), hide_index=True, width="stretch")
    st.caption(_data_text(protocol.get("claim_boundary", CLAIM_BOUNDARY)))


def _integrations(store: ConsoleStore) -> None:
    cases = store.list_cases()
    labels = [f"{case.get('display_id')} | {case.get('case_id')}" for case in cases]
    selected = cases[labels.index(st.selectbox(_ui("Payload case", "载荷案例"), labels))]
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader(_ui("Validated case payload", "已校验案例载荷"))
        try:
            payload = sanitize_report_value(ensure_output_contract(selected))
            st.success(_ui("The release/review contract is valid.", "放行/复核契约校验通过。"))
            st.json(payload, expanded=False)
        except ValueError as error:
            st.error(sanitize_report_value(str(error)))
        if st.button(_ui("Write integration audit event", "写入集成审计事件"), icon=":material/history:", width="stretch"):
            store.log_event(selected["case_id"], "integration.payload_validated", actor=st.session_state["vs_operator"], details={"schema_version": selected.get("schema_version")})
            message = _ui("Audit event recorded for this payload.", "已为该载荷写入审计事件。")
            st.success(message)
            st.toast(message, icon=":material/check_circle:")

    with right:
        st.subheader(_ui("REST API", "REST 接口"))
        endpoints = pd.DataFrame(
            [
                ["GET", "/health", _ui("Service health and boundary", "服务健康与边界")],
                ["POST", "/api/v1/assessments/video", _ui("Run video quality and evidence workflow", "运行视频质量与证据流程")],
                ["GET", "/api/v1/cases", _ui("Case registry", "案例登记")],
                ["GET", "/api/v1/cases/{case_id}", _ui("Evidence packet", "证据包")],
                ["GET", "/api/v1/reviews", _ui("Review queue", "复核队列")],
                ["PUT", "/api/v1/reviews/{case_id}", _ui("Review update", "复核更新")],
                ["GET", "/api/v1/cases/{case_id}/report?format=pdf", _ui("PDF report", "PDF 报告")],
                ["GET", "/api/v1/assistant/health", _ui("Local model and fallback health", "本地模型与降级状态")],
                ["GET", "/api/v1/assistant/multimodal/health", _ui("Image and speech capability health", "图片与语音能力状态")],
                ["POST", "/api/v1/assistant/chat", _ui("Evidence-bounded assistant response", "基于证据的助手回答")],
                ["POST", "/api/v1/assistant/transcribe", _ui("Transient local speech transcription", "临时本地语音转写")],
                ["POST", "/api/v1/assistant/analyze-image", _ui("Transient bounded image analysis", "临时受约束图片分析")],
                ["POST", "/api/v1/assistant/confirm", _ui("Explicitly confirm a pending review update", "明确确认待处理复核更新")],
                ["POST", "/api/v1/assistant/reject", _ui("Reject a pending review update", "拒绝待处理复核更新")],
            ],
            columns=["Method", "Path", _ui("Purpose", "用途")],
        )
        st.dataframe(endpoints, hide_index=True, width="stretch")
        api = create_app(DB_PATH, seed_demo=False)
        openapi = api.openapi()
        st.download_button(
            _ui("Download OpenAPI schema", "下载 OpenAPI 规范"),
            json.dumps(openapi, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="vitalssight_openapi.json",
            mime="application/json",
            icon=":material/download:",
            width="stretch",
        )
        st.code("uvicorn app.api_server:app --host 127.0.0.1 --port 8010", language="bash")
        st.caption(_ui("The API uses the same SQLite evidence and audit store as this page.", "API 与本页面使用同一个 SQLite 证据和审计存储。"))


def _help_settings(store: ConsoleStore) -> None:
    st.markdown(
        f"<div class='vs-guide-intro'><span>{_escape(_ui('GUIDED WORKFLOW', '引导式工作流'))}</span>"
        f"<h2>{_escape(_ui('Choose your role and follow one complete path', '选择你的角色，按照一条完整路径操作'))}</h2>"
        f"<p>{_escape(_ui('Each step states the required input, the action to take, the output to expect, and where to go next.', '每一步都说明所需输入、执行操作、预期输出以及下一步去向。'))}</p></div>",
        unsafe_allow_html=True,
    )
    guide_tabs = st.tabs(
        [
            _ui("Capture operator", "采集操作员"),
            _ui("Evidence reviewer", "证据复核员"),
            _ui("Report & integration", "报告与集成"),
            _ui("AI assistant", "AI 助手"),
        ]
    )
    with guide_tabs[0]:
        _guide_rows(
            [
                (_ui("Set purpose and consent", "设置用途与授权"), _ui("Research purpose, consent confirmation, retention choice", "研究用途、授权确认、留存方式"), _ui("Choose the purpose, confirm consent, then select raw-video handling.", "选择用途、确认授权，再选择原始视频处理方式。"), _ui("A documented processing contract", "已记录的数据处理契约"), _ui("Choose an input source", "选择输入来源")),
                (_ui("Prepare the recording", "准备录制"), _ui("Adult RGB face video or a labeled demo case", "成人 RGB 人脸视频或明确标注的演示案例"), _ui("Use even front light, center the full face, remain still, and record 20-30 seconds.", "使用均匀正面光、完整人脸居中、保持静止并录制 20-30 秒。"), _ui("A usable video window", "可用的视频窗口"), _ui("Run assessment", "运行评估")),
                (_ui("Run quality qualification", "运行质量检查"), _ui("Consented input", "已授权输入"), _ui("Click Run assessment; the system checks duration, fps, resolution, light, motion, and face visibility.", "点击运行评估；系统检查时长、帧率、分辨率、光照、运动和人脸可见性。"), _ui("Pass, warning, or retake guidance", "通过、警告或重采指引"), _ui("Read the output state", "查看输出状态")),
                (_ui("Interpret the state", "解释状态"), _ui("Quality and candidate evidence", "质量与候选证据"), _ui("Release shows HR; review and retake always withhold HR and show the trigger.", "放行会显示心率；复核和重采始终隐藏心率并显示触发原因。"), _ui("A traceable release, review, or retake result", "可追溯的放行、复核或重采结果"), _ui("Follow the recommended action", "执行推荐操作")),
                (_ui("Correct or route", "纠正或转交"), _ui("Triggered signal, observed value, and target threshold", "触发信号、观测值与目标阈值"), _ui("Correct capture problems or route unresolved candidate evidence to review.", "纠正采集问题，或将未消解的候选证据转入复核。"), _ui("A corrected recording or assigned review", "纠正后的录制或已分派复核"), _ui("Build the report", "生成报告")),
                (_ui("Export the evidence package", "导出证据包"), _ui("Completed case and optional review record", "已完成案例及可选复核记录"), _ui("Open Reports and export PDF, JSON, Markdown, or CSV for the intended audience.", "打开报告中心，并按受众导出 PDF、JSON、Markdown 或 CSV。"), _ui("Versioned report with action rationale and audit trail", "带行动依据和审计轨迹的版本化报告"), _ui("Retain or integrate", "保留或集成")),
            ]
        )
        if st.button(_ui("Start this workflow", "开始该流程"), type="primary", icon=":material/play_arrow:", width="stretch", key="guide_start_assessment"):
            _set_flash(_ui("Guided assessment opened at purpose and consent.", "引导式评估已从用途与授权步骤打开。"), "info")
            _start_assessment()
    with guide_tabs[1]:
        _guide_rows(
            [
                (_ui("Prioritize work", "确定优先级"), _ui("Open review queue", "未关闭的复核队列"), _ui("Filter by status and priority, then select the highest-priority unresolved item.", "按状态与优先级筛选，再选择最高优先级的未解决项目。"), _ui("One selected review case", "一个已选择的复核案例"), _ui("Inspect evidence", "检查证据")),
                (_ui("Inspect why HR was withheld", "检查心率为何未发布"), _ui("Reason, candidates, thresholds, and attribution", "原因、候选、阈值与归因"), _ui("Compare the observed values with policy targets; never infer from candidate HR alone.", "将观测值与策略目标比较，不得仅凭候选心率下结论。"), _ui("A documented evidence assessment", "已记录的证据评估"), _ui("Choose an action", "选择处理操作")),
                (_ui("Document the decision", "记录处理决定"), _ui("Status, priority, assignee, note, resolution", "状态、优先级、负责人、备注、处理结果"), _ui("Complete every field that applies and save the review.", "填写所有适用字段并保存复核。"), _ui("Timestamped review and audit event", "带时间戳的复核与审计事件"), _ui("Verify the report", "核对报告")),
                (_ui("Close the loop", "闭环处理"), _ui("Saved review and corrected capture if requested", "已保存复核及按需完成的纠正性重采"), _ui("Confirm the report includes the reason, recommendation, evidence basis, and escalation path.", "确认报告包含原因、建议、证据依据和升级路径。"), _ui("An auditable closed or waiting-retake state", "可审计的已关闭或等待重采状态"), _ui("Export or continue monitoring", "导出或继续观察")),
            ]
        )
        if st.button(_ui("Open review queue", "打开复核队列"), type="primary", icon=":material/fact_check:", width="stretch", key="guide_open_review"):
            _set_flash(_ui("Review queue opened. Start with priority and status.", "已打开复核队列，请先查看优先级和状态。"), "info")
            _go("Review queue")
    with guide_tabs[2]:
        _guide_rows(
            [
                (_ui("Select the audience", "确定报告受众"), _ui("Completed case", "已完成案例"), _ui("Use PDF for people, JSON for systems, Markdown for review, and CSV for case-level analysis.", "PDF 面向人工阅读，JSON 面向系统，Markdown 面向复核，CSV 面向案例级分析。"), _ui("Correct export format", "正确的导出格式"), _ui("Review content", "检查内容")),
                (_ui("Verify the evidence chain", "核对证据链"), _ui("Interpretation, thresholds, actions, attribution, and audit", "解释、阈值、操作、归因与审计"), _ui("Confirm every recommendation names its source signal, observed value, target, and verification condition.", "确认每条建议都标明来源信号、观测值、目标和复核条件。"), _ui("A self-explanatory report", "可自解释的报告"), _ui("Export or integrate", "导出或集成")),
                (_ui("Validate the API contract", "验证 API 契约"), _ui("Case payload or OpenAPI schema", "案例载荷或 OpenAPI 规范"), _ui("Validate that non-release states contain no released HR, then write an audit event.", "验证非放行状态不包含已发布心率，再写入审计事件。"), _ui("Validated payload and audit record", "已校验载荷与审计记录"), _ui("Connect downstream", "连接下游系统")),
                (_ui("Preserve boundaries", "保留使用边界"), _ui("Report version, model/policy versions, claim boundary", "报告版本、模型/策略版本和证据边界"), _ui("Keep the boundary and versions with every exported or integrated result.", "每次导出或集成时都保留边界和版本信息。"), _ui("Traceable research output", "可追溯的研究输出"), _ui("Archive or continue review", "归档或继续复核")),
            ]
        )
        c1, c2 = st.columns(2)
        if c1.button(_ui("Open reports", "打开报告中心"), type="primary", icon=":material/description:", width="stretch", key="guide_open_reports"):
            _set_flash(_ui("Report center opened. Select a case and review the interpretation first.", "已打开报告中心，请先选择案例并查看结果解释。"), "info")
            _go("Reports")
        if c2.button(_ui("Open integrations", "打开系统集成"), icon=":material/api:", width="stretch", key="guide_open_integrations"):
            _set_flash(_ui("Integration workspace opened.", "已打开系统集成工作区。"), "info")
            _go("Integrations")
    with guide_tabs[3]:
        _guide_rows(
            [
                (_ui("Select a role", "选择角色"), _ui("Operator, reviewer, clinical reader, or administrator", "操作员、复核人员、医护阅读者或管理员"), _ui("Choose the role whose workflow and permissions match the current task.", "选择与当前任务和权限相符的角色。"), _ui("Role-bounded tool access", "受角色约束的工具访问"), _ui("Choose evidence context", "选择证据上下文")),
                (_ui("Select a case", "选择案例"), _ui("One case or general guidance", "一个案例或通用指引"), _ui("Use a case for metric-level explanations; use general guidance for tutorials and policy questions.", "需要指标级解释时选择案例；教程和策略问题使用通用指引。"), _ui("Case, report, policy, and knowledge citations", "案例、报告、策略和知识引用"), _ui("Ask or use a quick prompt", "提问或使用快捷问题")),
                (_ui("Read facts separately from explanation", "区分事实与解释"), _ui("Answer, evidence IDs, action steps, provider status", "回答、证据编号、操作步骤和模型状态"), _ui("Verify system facts and evidence citations before acting on the natural-language explanation.", "根据自然语言解释采取行动前，先核对系统事实和证据引用。"), _ui("Traceable answer with a tool trace ID", "带工具追踪编号的可追溯回答"), _ui("Navigate or prepare an action", "导航或准备动作")),
                (_ui("Confirm any change", "确认任何变更"), _ui("Pending review update token", "待处理复核更新令牌"), _ui("The assistant only prepares the update. Inspect it, then explicitly confirm or reject it.", "助手只准备更新；检查后必须明确确认或拒绝。"), _ui("Timestamped review and assistant audit event", "带时间戳的复核与助手审计事件"), _ui("Verify the saved review", "核对已保存复核")),
            ]
        )
        if st.button(_ui("Open AI assistant", "打开 AI 助手"), type="primary", icon=":material/smart_toy:", width="stretch", key="guide_open_assistant"):
            _set_flash(_ui("AI assistant opened in evidence-bounded mode.", "已用证据约束模式打开 AI 助手。"), "info")
            _go("AI assistant")

    st.markdown("<div class='vs-section-rule'></div>", unsafe_allow_html=True)
    left, right = st.columns([1, 0.8], gap="large")
    with left:
        st.subheader(_ui("Capture checklist", "采集检查清单"))
        checklist = [
            _ui("Use even front lighting; avoid strong backlight.", "使用均匀正面光，避免强逆光。"),
            _ui("Keep the full face visible and remove major occlusion.", "保持完整人脸可见，避免明显遮挡。"),
            _ui("Remain still and avoid speaking during the recording.", "采集时保持静止并避免说话。"),
            _ui("Record 20-30 seconds at 15 fps or higher.", "建议录制 20-30 秒，帧率不低于 15 fps。"),
            _ui("Repeat the recording when the system returns retake.", "系统返回重采时，请修正问题后重新录制。"),
        ]
        for index, item in enumerate(checklist, 1):
            st.markdown(f"<div class='vs-check'><b>{index}</b><span>{_escape(item)}</span></div>", unsafe_allow_html=True)
        st.subheader(_ui("Output states", "输出状态"))
        states = pd.DataFrame(
            [
                [_decision_text("release"), _ui("HR may be displayed with its evidence packet.", "可以连同证据包显示心率。")],
                [_decision_text("review"), _ui("HR is withheld; an operator reviews the evidence.", "心率不发布，由操作员复核证据。")],
                [_decision_text("retake"), _ui("The recording does not meet the acquisition gate.", "录制未达到采集门槛，需要重新采集。")],
            ],
            columns=[_ui("State", "状态"), _ui("Meaning", "含义")],
        )
        st.dataframe(states, hide_index=True, width="stretch")
        st.info(_data_text(CLAIM_BOUNDARY))

    with right:
        st.subheader(_ui("Workspace settings", "工作区设置"))
        operator = st.text_input(_ui("Operator name", "操作员名称"), value=st.session_state["vs_operator"])
        if st.button(_ui("Save operator", "保存操作员"), type="primary", icon=":material/person_check:", width="stretch"):
            st.session_state["vs_operator"] = operator.strip() or "Research operator"
            message = _ui("Operator saved for future audit events.", "操作员已保存，将用于后续审计事件。")
            st.success(message)
            st.toast(message, icon=":material/check_circle:")
        st.markdown("---")
        st.subheader(_ui("Data handling", "数据处理"))
        st.markdown(
            _ui(
                "Uploaded raw video is processed locally. The recommended mode deletes it after analysis and stores only derived evidence, the decision, and the audit trail.",
                "上传的原始视频在本地处理。推荐模式会在分析后删除视频，仅保留派生证据、决策和审计记录。",
            )
        )
        if st.button(_ui("Restore built-in demo cases", "恢复内置演示案例"), icon=":material/refresh:", width="stretch"):
            for case in make_demo_cases():
                store.upsert_case(case, actor=st.session_state["vs_operator"])
            message = _ui("Built-in cases restored without deleting user cases.", "已恢复内置案例，未删除用户案例。")
            st.success(message)
            st.toast(message, icon=":material/check_circle:")

        with st.expander(_ui("What should I do if a click appears to do nothing?", "如果点击后看起来没有反应怎么办？")):
            st.markdown(
                _ui(
                    "Every command now either navigates, downloads a file, or shows a success/warning message. If no message appears, refresh once and check the browser console before repeating the action.",
                    "现在每个命令都会跳转、下载文件或显示成功/警告信息。如果没有出现任何信息，请先刷新一次并检查浏览器控制台，再重复操作。",
                )
            )


def _guide_rows(rows: list[tuple[str, str, str, str, str]]) -> None:
    for index, (title, input_text, action, output, next_step) in enumerate(rows, 1):
        st.markdown(
            f"<div class='vs-guide-row'><b>{index}</b><div><h3>{_escape(title)}</h3>"
            f"<dl><dt>{_escape(_ui('Input', '输入'))}</dt><dd>{_escape(input_text)}</dd>"
            f"<dt>{_escape(_ui('Action', '操作'))}</dt><dd>{_escape(action)}</dd>"
            f"<dt>{_escape(_ui('Output', '输出'))}</dt><dd>{_escape(output)}</dd>"
            f"<dt>{_escape(_ui('Next', '下一步'))}</dt><dd>{_escape(next_step)}</dd></dl></div></div>",
            unsafe_allow_html=True,
        )


def _result_summary(case: dict[str, Any]) -> None:
    decision = str(case.get("decision", "review"))
    tone = {"release": "teal", "review": "amber", "retake": "coral"}.get(decision, "neutral")
    hr = _released_hr(case)
    acquisition_gate = _acquisition_gate_text(case)
    st.markdown(
        f"<div class='vs-result {tone}'><div><small>{_escape(_ui('Decision','决策'))}</small>"
        f"<b>{_escape(_decision_text(decision))}</b></div>"
        f"<div><small>{_escape(_ui('Published HR','已发布心率'))}</small><b>{_escape(hr)}</b></div>"
        f"<div><small>{_escape(_ui('Acquisition gate','采集门控'))}</small><b>{_escape(acquisition_gate)}</b></div>"
        f"<div><small>{_escape(_ui('Next action','下一步'))}</small><span>{_escape(_data_text(case.get('recommended_action')))}</span></div></div>",
        unsafe_allow_html=True,
    )


def _acquisition_gate_text(case: dict[str, Any]) -> str:
    overall = str((case.get("preflight") or {}).get("overall") or "").lower()
    if str(case.get("decision")) == "retake" or overall == "fail":
        return _ui("Not passed", "未通过")
    if overall == "warn":
        return _ui("Passed with warnings", "通过但有警告")
    return _ui("Passed", "通过")


def _preflight_panel(preflight: dict[str, Any]) -> None:
    overall = str(preflight.get("overall", "fail"))
    status_text = {"pass": _ui("Passed", "通过"), "warn": _ui("Passed with warnings", "通过但有警告"), "fail": _ui("Retake required", "需要重采")}.get(overall, overall)
    st.markdown(f"**{_ui('Overall','总体')}**: {status_text}")
    rows = []
    for item in preflight.get("checks", []):
        rows.append(
            {
                _ui("Check", "检查项"): _data_text(item["check"]),
                _ui("Value", "数值"): f"{item['value']} {_data_text(item['unit'])}",
                _ui("Status", "状态"): _data_text(item["status"]),
                _ui("Action", "操作"): _data_text(item["action"]),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if preflight.get("note"):
        st.caption(_data_text(preflight["note"]))


def _capture_guidance() -> None:
    st.markdown(
        "<div class='vs-guidance-grid'>"
        f"<div><b>1</b><span>{_escape(_ui('Face centered', '人脸居中'))}</span></div>"
        f"<div><b>2</b><span>{_escape(_ui('Even lighting', '均匀光照'))}</span></div>"
        f"<div><b>3</b><span>{_escape(_ui('Remain still', '保持静止'))}</span></div>"
        f"<div><b>4</b><span>{_escape(_ui('20-30 seconds', '20-30 秒'))}</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _step_strip() -> None:
    labels = [
        _ui("Consent", "授权"),
        _ui("Capture", "采集"),
        _ui("Quality", "质量"),
        _ui("Result or review", "结果或复核"),
    ]
    if st.session_state.get("vs_assessment_result"):
        current = 4
    elif st.session_state.get("vs_preflight"):
        current = 3
    elif st.session_state.get("vs_consent"):
        current = 2
    else:
        current = 1
    state_labels = {
        "done": _ui("Complete", "已完成"),
        "current": _ui("Current", "当前"),
        "pending": _ui("Next", "待进行"),
    }
    blocks = []
    for index, label in enumerate(labels, 1):
        state = "done" if index < current else ("current" if index == current else "pending")
        blocks.append(
            f"<div class='{state}'><b>{index}</b><span>{_escape(label)}<small>{_escape(state_labels[state])}</small></span></div>"
        )
    html = "".join(blocks)
    st.markdown(f"<div class='vs-step-strip'>{html}</div>", unsafe_allow_html=True)


def _metric(container: Any, label: str, value: str, note: str, *, tone: str = "neutral") -> None:
    container.markdown(
        f"<div class='vs-metric {tone}'><span>{_escape(label)}</span><b>{_escape(value)}</b><small>{_escape(note)}</small></div>",
        unsafe_allow_html=True,
    )


def _quality_chart(cases: list[dict[str, Any]]) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[case.get("display_id") for case in cases],
            y=[case.get("quality_score") or 0 for case in cases],
            marker_color=[_decision_color(str(case.get("decision"))) for case in cases],
            text=[_percent(case.get("quality_score")) for case in cases],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=330,
        margin=dict(l=20, r=15, t=18, b=45),
        yaxis=dict(range=[0, 1.08], tickformat=".0%"),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#526773", family="Segoe UI, sans-serif", size=12),
    )
    fig.update_yaxes(gridcolor="#E7EDF0", zeroline=False)
    st.plotly_chart(fig, width="stretch")


def _decision_chart(cases: list[dict[str, Any]]) -> None:
    counts = pd.Series([case.get("decision") for case in cases]).value_counts()
    labels = [item for item in ["release", "review", "retake"] if item in counts]
    fig = go.Figure(
        go.Pie(
            labels=[_decision_text(label) for label in labels],
            values=[int(counts[label]) for label in labels],
            hole=0.62,
            marker_colors=[_decision_color(label) for label in labels],
            textinfo="label+value",
        )
    )
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#526773", family="Segoe UI, sans-serif", size=12),
    )
    st.plotly_chart(fig, width="stretch")


def _trend_chart(case: dict[str, Any]) -> None:
    values = [value for value in case.get("trend_bpm", []) if value is not None]
    if not values:
        st.caption(_ui("No releasable trend is available.", "没有可发布的趋势。"))
        return
    fig = go.Figure(
        go.Scatter(
            x=list(range(1, len(values) + 1)),
            y=values,
            mode="lines+markers",
            line=dict(color="#4F7E95", width=2.4),
            marker=dict(size=7, color="#4F7E95", line=dict(color="#FFFFFF", width=1.2)),
        )
    )
    fig.update_layout(
        height=270,
        margin=dict(l=25, r=15, t=14, b=35),
        xaxis_title=_ui("Window", "窗口"),
        yaxis_title="HR (BPM)",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#526773", family="Segoe UI, sans-serif", size=12),
    )
    fig.update_yaxes(gridcolor="#E7EDF0", zeroline=False)
    st.plotly_chart(fig, width="stretch")


def _decision_text(decision: str) -> str:
    labels = {
        "release": _ui("Released", "已放行"),
        "review": _ui("Review", "复核"),
        "retake": _ui("Retake", "重采"),
    }
    return labels.get(str(decision), str(decision))


def _decision_color(decision: str) -> str:
    return {"release": "#5E8F88", "review": "#B18B58", "retake": "#B8736D"}.get(decision, "#738894")


def _next_step_text(case: dict[str, Any]) -> str:
    return {
        "release": _ui("Retain evidence", "保留证据"),
        "review": _ui("Open review", "进入复核"),
        "retake": _ui("Repeat capture", "重新采集"),
    }.get(str(case.get("decision")), _ui("Inspect case", "检查案例"))


def _released_hr(case: dict[str, Any]) -> str:
    value = case.get("released_hr_bpm") if case.get("decision") == "release" else None
    return "Withheld" if value is None and not _is_zh() else ("未发布" if value is None else f"{float(value):.1f} BPM")


def _percent(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.0%}"


def _go(section: str) -> None:
    if section not in SECTIONS:
        raise ValueError(f"Unknown console section: {section}")
    st.session_state["vs_pending_section"] = section
    st.rerun()


def _reset_main_scroll(navigation_nonce: int) -> None:
    """Start each workspace view at its first instruction after navigation."""
    script = """
        <script>
        const navigationNonce = __NAVIGATION_NONCE__;
        let sidebarCloseRequested = false;
        const resetMainScroll = () => {
            const main = window.parent.document.querySelector('[data-testid="stMain"]');
            if (main) main.scrollTo({ top: 0, left: 0, behavior: 'instant' });
        };
        const closeMobileSidebar = () => {
            if (window.parent.innerWidth > 900 || sidebarCloseRequested) return;
            const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) return;
            const expanded = sidebar.getAttribute('aria-expanded') ?? sidebar.getAttribute('aria');
            const rect = sidebar.getBoundingClientRect();
            const style = window.parent.getComputedStyle(sidebar);
            const visiblyOpen = rect.width > 120 && rect.right > 0 && style.visibility !== 'hidden';
            if (expanded === 'false' || !visiblyOpen) return;
            const collapse = sidebar.querySelector('[data-testid="stSidebarCollapseButton"] button');
            if (!collapse || !collapse.innerText.includes('keyboard_double_arrow_left')) return;
            sidebarCloseRequested = true;
            collapse.click();
        };
        resetMainScroll();
        window.requestAnimationFrame(() => {
            resetMainScroll();
            closeMobileSidebar();
        });
        window.setTimeout(() => {
            resetMainScroll();
            closeMobileSidebar();
        }, 150);
        </script>
        """.replace("__NAVIGATION_NONCE__", str(int(navigation_nonce)))
    st.iframe(
        script,
        height=1,
        width=1,
    )


def _set_flash(message: str, kind: str = "success") -> None:
    st.session_state["vs_flash"] = message
    st.session_state["vs_flash_kind"] = kind if kind in {"success", "info", "warning", "error"} else "info"


def _start_assessment() -> None:
    """Open a clean acquisition flow without discarding stored cases."""
    _remove_session_upload()
    _reset_upload_widget()
    for key in ("vs_assessment_result", "vs_preflight", "vs_upload_path"):
        st.session_state.pop(key, None)
    st.session_state["vs_consent"] = False
    st.session_state["vs_source"] = "stable"
    st.session_state["vs_retention"] = "delete_after_analysis"
    for suffix in ("zh", "en"):
        st.session_state[f"vs_consent_control_{suffix}"] = False
        st.session_state[f"vs_source_control_{suffix}"] = "stable"
        st.session_state[f"vs_retention_control_{suffix}"] = "delete_after_analysis"
    _go("New assessment")


def _remove_session_upload() -> None:
    raw_path = st.session_state.get("vs_upload_path", "")
    if raw_path:
        path = Path(raw_path)
        try:
            if path.resolve().is_relative_to(UPLOAD_DIR.resolve()) and path.is_file():
                path.unlink()
        except (OSError, ValueError):
            pass
    st.session_state["vs_upload_path"] = ""


def _reset_upload_widget() -> None:
    """Force Streamlit to rebuild the uploader after an assessment reset."""
    current = int(st.session_state.get("vs_upload_widget_version", 0))
    st.session_state["vs_upload_widget_version"] = current + 1


def _purge_stale_uploads(*, max_age_seconds: int = 2 * 60 * 60) -> None:
    if not UPLOAD_DIR.exists():
        return
    threshold = time.time() - max_age_seconds
    for path in UPLOAD_DIR.rglob("*"):
        try:
            if path.is_file() and path.stat().st_mtime < threshold:
                path.unlink()
        except OSError:
            continue


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #1e3440;
            --ink-soft: #415762;
            --muted: #6b7d86;
            --line: #d8e2e6;
            --line-strong: #b9cad1;
            --paper: #ffffff;
            --canvas: #f4f7f8;
            --sidebar: #f9fbfc;
            --primary: #4b7893;
            --primary-dark: #315f77;
            --primary-soft: #eaf1f5;
            --steel: #728c9c;
            --steel-soft: #eef3f5;
            --teal: #5e8f88;
            --teal-soft: #edf5f3;
            --review: #b18b58;
            --review-soft: #f7f2e9;
            --rose: #b8736d;
            --rose-soft: #f8efed;
            --shadow: 0 10px 30px rgba(32, 58, 70, 0.07);
            --shadow-soft: 0 4px 14px rgba(32, 58, 70, 0.045);
        }
        html, body, [class*="css"] { font-family: "Inter", "Segoe UI", "Microsoft YaHei", sans-serif; }
        .stApp { background: var(--canvas); color: var(--ink); font-size: 0.96rem; }
        header[data-testid="stHeader"] { background: rgba(244, 247, 248, 0.96); border-bottom: 1px solid rgba(216, 226, 230, 0.86); backdrop-filter: blur(10px); }
        [data-testid="stToolbar"] { display: flex !important; }
        [data-testid="stAppDeployButton"], [data-testid="stMainMenu"] { display: none !important; }
        [data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stExpandSidebarButton"] {
            visibility: visible !important; opacity: 1 !important;
        }
        [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] {
            display: flex !important; visibility: visible !important; opacity: 1 !important;
            position: fixed !important; top: 0.72rem !important; left: 0.72rem !important; z-index: 100001 !important;
        }
        [data-testid="stSidebarCollapsedControl"] button,
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stExpandSidebarButton"] {
            background: #ffffff !important; border: 1px solid var(--line-strong) !important;
            border-radius: 7px !important; box-shadow: 0 3px 12px rgba(36, 66, 79, 0.10) !important;
            color: var(--primary-dark) !important; min-width: 36px !important; min-height: 36px !important;
        }
        .block-container { padding-top: 3.65rem; padding-bottom: 4rem; max-width: 1480px; }
        section[data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--line); }
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 0.2rem; }
        section[data-testid="stSidebar"] .stRadio > div { gap: 0.18rem; }
        section[data-testid="stSidebar"] .stRadio label {
            padding: 0.58rem 0.72rem; border-radius: 6px; color: var(--ink-soft); border-left: 3px solid transparent;
            transition: background 140ms ease, color 140ms ease, border-color 140ms ease;
        }
        section[data-testid="stSidebar"] .stRadio label:hover { background: #edf3f6; color: var(--ink); }
        section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
            background: var(--primary-soft); color: var(--primary-dark); border-left-color: var(--primary); font-weight: 700;
        }
        .vs-brand { display: flex; align-items: center; gap: 0.72rem; margin: 0.9rem 0 0.08rem; font-size: 1.08rem; color: var(--ink); }
        .vs-brand span { width: 36px; height: 36px; display: inline-grid; place-items: center; background: var(--primary); color: white; border-radius: 6px; font-weight: 800; box-shadow: 0 6px 16px rgba(71,120,141,0.20); }
        .vs-side-status { display:flex; align-items:center; gap:0.62rem; margin:0.72rem 0 0.7rem; padding:0.58rem 0.68rem; border:1px solid var(--line); background:#fff; border-radius:6px; box-shadow:var(--shadow-soft); }
        .vs-side-status i { width:8px; height:8px; border-radius:50%; background:var(--teal); box-shadow:0 0 0 4px var(--teal-soft); flex:0 0 auto; }
        .vs-side-status b, .vs-side-status span { display:block; }
        .vs-side-status b { color:var(--ink); font-size:0.76rem; }
        .vs-side-status span { color:var(--muted); font-size:0.68rem; margin-top:0.06rem; }
        .vs-boundary-small { font-size: 0.76rem; line-height: 1.5; color: var(--muted); padding: 0.55rem 0; }
        .vs-page-title { font-size: 1.78rem !important; line-height: 1.15; margin: 0 !important; color: var(--ink); letter-spacing: 0; }
        .vs-env { margin-top: 0.2rem; border: 1px solid var(--line); background: var(--paper); padding: 0.52rem 0.7rem; border-radius: 6px; display: grid; grid-template-columns:1fr 1fr; gap: 0; color: var(--muted); font-size: 0.72rem; box-shadow: var(--shadow-soft); }
        .vs-env > div { display:flex; flex-direction:column; align-items:flex-start; gap:0.04rem; min-width:0; padding:0 0.6rem; border-right:1px solid var(--line); }
        .vs-env span { white-space:nowrap; }
        .vs-env > div:first-child { padding-left:0; }
        .vs-env > div:last-child { border-right:0; padding-right:0; }
        .vs-env b { color: var(--primary); letter-spacing: 0; }
        .vs-rule { border-top: 1px solid var(--line); margin: 0.8rem 0 1.15rem; }
        .vs-section-rule { border-top: 1px solid var(--line); margin: 1.5rem 0; }
        h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
        h2 { font-size: 1.18rem !important; }
        h3 { font-size: 1rem !important; }
        p, li { color: var(--ink-soft); }
        .vs-workflow-band { display:grid; grid-template-columns:minmax(260px,0.85fr) minmax(420px,1.15fr); gap:1.5rem; align-items:center; background:#ffffff; border:1px solid var(--line); border-left:5px solid var(--primary); border-radius:6px; padding:1rem 1.15rem; margin:0.1rem 0 0.75rem; box-shadow:var(--shadow); }
        .vs-workflow-band span, .vs-io-strip span, .vs-guide-intro > span, .vs-report-narrative > span, .vs-report-recommendation > span, .vs-action-head > span { display:block; color:var(--primary); font-size:0.67rem; font-weight:800; letter-spacing:0; }
        .vs-workflow-band b { display:block; margin-top:0.28rem; font-size:0.94rem; line-height:1.45; color:var(--ink); }
        .vs-workflow-band ol { margin:0; padding:0; display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); list-style:none; counter-reset:flow; }
        .vs-workflow-band li { counter-increment:flow; padding:0.42rem 0.7rem; border-left:1px solid var(--line); font-size:0.78rem; color:var(--ink-soft); }
        .vs-workflow-band li::before { content:counter(flow); display:block; color:var(--primary); font-weight:800; margin-bottom:0.12rem; }
        .vs-io-strip { display:grid; grid-template-columns:1fr 1fr; gap:1px; border:1px solid var(--line); background:var(--line); border-radius:7px; overflow:hidden; margin-bottom:1rem; }
        .vs-io-strip > div { background:var(--paper); padding:0.72rem 0.86rem; }
        .vs-io-strip b { display:block; margin-top:0.18rem; font-size:0.82rem; color:var(--ink-soft); }
        .vs-processing-contract { display:grid; grid-template-columns:1fr 1fr; gap:1px; margin:-0.35rem 0 1.05rem; border:1px solid var(--line); border-radius:6px; overflow:hidden; background:var(--line); }
        .vs-processing-contract > div { background:var(--steel-soft); padding:0.58rem 0.82rem; }
        .vs-processing-contract b, .vs-processing-contract span { display:block; }
        .vs-processing-contract b { color:var(--primary); font-size:0.65rem; }
        .vs-processing-contract span { color:var(--ink-soft); font-size:0.75rem; margin-top:0.14rem; line-height:1.4; }
        .vs-metric { background: var(--paper); border: 1px solid var(--line); border-top: 3px solid var(--steel); border-radius: 6px; padding: 0.82rem 0.92rem; min-height: 108px; box-shadow: var(--shadow-soft); }
        .vs-metric.teal { border-top-color: var(--teal); }
        .vs-metric.amber { border-top-color: var(--review); }
        .vs-metric.coral { border-top-color: var(--rose); }
        .vs-metric span { display:block; color: var(--muted); font-size: 0.76rem; }
        .vs-metric b { display:block; color: var(--ink); font-size: 1.68rem; line-height: 1.35; margin-top: 0.12rem; }
        .vs-metric small { color: var(--muted); font-size: 0.71rem; }
        .vs-list-row { background: var(--paper); border-bottom: 1px solid var(--line); padding: 0.78rem 0.82rem; }
        .vs-list-row:first-of-type { border-top: 1px solid var(--line); border-radius:7px 7px 0 0; }
        .vs-list-row:last-of-type { border-radius:0 0 7px 7px; }
        .vs-list-row div { display:flex; justify-content:space-between; gap: 0.5rem; }
        .vs-list-row span, .vs-list-row small { color: var(--muted); font-size: 0.75rem; }
        .vs-state-pill { display:inline-flex !important; width:max-content; align-items:center; padding:0.16rem 0.42rem; border-radius:999px; border:1px solid var(--line); background:var(--steel-soft); }
        .vs-state-pill.release { color:#3f756e !important; border-color:#b9d2ce; background:var(--teal-soft); }
        .vs-state-pill.review { color:#84652f !important; border-color:#dbc9aa; background:var(--review-soft); }
        .vs-state-pill.retake { color:#955a55 !important; border-color:#dfbbb7; background:var(--rose-soft); }
        .vs-step-strip { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 0; border: 1px solid var(--line); background: var(--paper); border-radius:7px; overflow:hidden; margin-bottom: 0.75rem; }
        .vs-step-strip div { display:flex; align-items:center; gap:0.6rem; padding:0.68rem 0.82rem; border-right:1px solid var(--line); border-top:3px solid transparent; }
        .vs-step-strip div:last-child { border-right: 0; }
        .vs-step-strip div.current { border-top-color:var(--primary); background:var(--primary-soft); }
        .vs-step-strip div.done { border-top-color:var(--teal); }
        .vs-step-strip b { width:27px; height:27px; border:1px solid var(--line-strong); color:var(--muted); display:grid; place-items:center; border-radius:50%; font-size:0.74rem; flex:0 0 auto; }
        .vs-step-strip .current b { border-color:var(--primary); color:var(--primary); background:#fff; }
        .vs-step-strip .done b { border-color:var(--teal); color:#fff; background:var(--teal); }
        .vs-step-strip span { font-size:0.8rem; font-weight:700; }
        .vs-step-strip small { display:block; margin-top:0.05rem; color:var(--muted); font-size:0.62rem; font-weight:500; }
        .vs-guidance-grid { display:grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap:0.5rem; margin:0.65rem 0 0.85rem; }
        .vs-guidance-grid div { border:1px solid var(--line); background:var(--paper); padding:0.62rem; border-radius:7px; display:flex; gap:0.45rem; align-items:center; }
        .vs-guidance-grid b { color:var(--primary); }
        .vs-guidance-grid span { font-size:0.74rem; }
        .vs-result { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:1px; border:1px solid var(--line); border-left:5px solid var(--steel); border-radius:7px; overflow:hidden; background:var(--line); margin:0.45rem 0 0.85rem; box-shadow:0 4px 16px rgba(34,63,76,0.04); }
        .vs-result.teal { border-left-color:var(--teal); }
        .vs-result.amber { border-left-color:var(--review); }
        .vs-result.coral { border-left-color:var(--rose); }
        .vs-result > div { padding:0.7rem 0.82rem; min-width:0; background:var(--paper); }
        .vs-result small, .vs-result span { display:block; color:var(--muted); font-size:0.73rem; line-height:1.4; overflow-wrap:anywhere; }
        .vs-result b { display:block; font-size:1rem; color:var(--ink); margin-top:0.15rem; }
        .vs-empty { min-height:155px; display:grid; place-content:center; text-align:center; border:1px dashed var(--line-strong); border-radius:7px; background:#f9fbfc; color:var(--muted); padding:1rem; }
        .vs-empty b, .vs-empty span { display:block; }
        .vs-factor { border-left:4px solid var(--review); background:var(--paper); border-top:1px solid var(--line); border-right:1px solid var(--line); border-bottom:1px solid var(--line); border-radius:0 7px 7px 0; padding:0.62rem 0.74rem; margin-bottom:0.48rem; }
        .vs-factor.good { border-left-color:var(--teal); }
        .vs-factor b, .vs-factor span, .vs-factor small { display:block; }
        .vs-factor span { color:var(--muted); font-size:0.75rem; }
        .vs-factor small { color:var(--ink-soft); margin-top:0.25rem; line-height:1.4; }
        .vs-report-sheet { background:var(--paper); border:1px solid var(--line-strong); border-radius:8px; overflow:hidden; box-shadow:var(--shadow); margin:0.35rem 0 0.9rem; }
        .vs-report-sheet header { display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; padding:1.15rem 1.25rem; border-bottom:1px solid var(--line); background:#fbfcfd; }
        .vs-report-sheet header span { color:var(--primary); font-size:0.67rem; font-weight:800; }
        .vs-report-sheet header h2 { margin:0.22rem 0 0.1rem; font-size:1.3rem !important; }
        .vs-report-sheet header p { margin:0; font-size:0.75rem; color:var(--muted); }
        .vs-status { display:inline-flex; align-items:center; padding:0.36rem 0.64rem; border:1px solid; border-radius:999px; font-size:0.76rem; white-space:nowrap; }
        .vs-status.release { color:#477771; border-color:#8eb7b2; background:var(--teal-soft); }
        .vs-status.review { color:#84652f; border-color:#dbc9aa; background:var(--review-soft); }
        .vs-status.retake { color:#965f65; border-color:#d3a7aa; background:var(--rose-soft); }
        .vs-report-hero { display:grid; grid-template-columns:0.65fr 0.65fr 1.7fr; gap:1px; background:var(--line); border-bottom:1px solid var(--line); }
        .vs-report-hero > div { background:#ffffff; padding:0.82rem 1.05rem; min-width:0; }
        .vs-report-hero small { display:block; color:var(--muted); font-size:0.7rem; }
        .vs-report-hero b { display:block; color:var(--ink); margin-top:0.18rem; overflow-wrap:anywhere; }
        .vs-report-narrative, .vs-report-recommendation { padding:1rem 1.25rem; border-bottom:1px solid var(--line); }
        .vs-report-narrative h3 { margin:0.28rem 0 0.25rem; font-size:1.05rem !important; }
        .vs-report-narrative p, .vs-report-recommendation p { margin:0; line-height:1.5; font-size:0.82rem; }
        .vs-report-recommendation { background:var(--primary-soft); }
        .vs-report-recommendation b { display:block; margin:0.28rem 0 0.18rem; color:var(--primary-dark); }
        .vs-report-sheet footer { padding:0.62rem 1.25rem; color:var(--muted); font-size:0.69rem; background:#fbfcfd; }
        .vs-action-head { border-left:5px solid var(--review); background:var(--review-soft); padding:0.85rem 1rem; border-radius:0 7px 7px 0; margin:0.2rem 0 0.75rem; }
        .vs-action-head.release { border-left-color:var(--teal); background:var(--teal-soft); }
        .vs-action-head.retake { border-left-color:var(--rose); background:var(--rose-soft); }
        .vs-action-head b, .vs-action-head p { display:block; margin:0.22rem 0 0; }
        .vs-action-head p { font-size:0.8rem; line-height:1.45; }
        .vs-action-step { display:grid; grid-template-columns:32px 1fr; gap:0.7rem; padding:0.72rem 0; border-bottom:1px solid var(--line); }
        .vs-action-step > b { width:28px; height:28px; display:grid; place-items:center; border-radius:50%; background:var(--primary-soft); color:var(--primary); font-size:0.75rem; }
        .vs-action-step strong, .vs-action-step span, .vs-action-step small { display:block; }
        .vs-action-step strong { color:var(--ink); }
        .vs-action-step span { color:var(--ink-soft); margin-top:0.2rem; font-size:0.77rem; line-height:1.4; }
        .vs-action-step small { color:var(--muted); margin-top:0.16rem; line-height:1.4; }
        .vs-escalation { display:flex; gap:0.65rem; align-items:flex-start; margin:0.8rem 0 0.35rem; padding:0.72rem 0.82rem; background:#f7f9fa; border:1px solid var(--line); border-radius:7px; }
        .vs-escalation b { white-space:nowrap; color:var(--rose); }
        .vs-escalation span { color:var(--ink-soft); line-height:1.45; font-size:0.79rem; }
        .vs-guide-intro { padding:0.35rem 0 0.9rem; max-width:900px; }
        .vs-guide-intro h2 { margin:0.28rem 0 0.25rem; font-size:1.34rem !important; }
        .vs-guide-intro p { margin:0; line-height:1.55; }
        .vs-guide-row { display:grid; grid-template-columns:38px 1fr; gap:0.8rem; padding:0.85rem 0; border-bottom:1px solid var(--line); }
        .vs-guide-row > b { width:32px; height:32px; display:grid; place-items:center; border:1px solid var(--primary); border-radius:50%; color:var(--primary); font-size:0.78rem; }
        .vs-guide-row h3 { margin:0 0 0.45rem; font-size:0.98rem !important; }
        .vs-guide-row dl { display:grid; grid-template-columns:74px minmax(0,1fr); gap:0.28rem 0.65rem; margin:0; }
        .vs-guide-row dt { color:var(--primary); font-size:0.71rem; font-weight:800; }
        .vs-guide-row dd { margin:0; color:var(--ink-soft); font-size:0.78rem; line-height:1.45; }
        .vs-check { display:flex; gap:0.7rem; align-items:flex-start; border-bottom:1px solid var(--line); padding:0.68rem 0; }
        .vs-check b { width:27px; height:27px; border:1px solid var(--primary); color:var(--primary); display:grid; place-items:center; border-radius:50%; font-size:0.73rem; flex:0 0 auto; }
        .vs-check span { line-height:1.5; }
        .vs-assistant-status { display:flex; align-items:center; gap:0.78rem; border:1px solid var(--line); border-left:5px solid var(--primary); background:#fff; padding:0.78rem 0.9rem; border-radius:7px; box-shadow:var(--shadow-soft); margin-bottom:0.75rem; }
        .vs-assistant-status i { width:10px; height:10px; border-radius:50%; background:var(--teal); box-shadow:0 0 0 5px var(--teal-soft); flex:0 0 auto; }
        .vs-assistant-status.degraded { border-left-color:var(--review); }
        .vs-assistant-status.degraded i { background:var(--review); box-shadow:0 0 0 5px var(--review-soft); }
        .vs-assistant-status b, .vs-assistant-status span { display:block; }
        .vs-assistant-status b { color:var(--ink); font-size:0.86rem; }
        .vs-assistant-status span { color:var(--muted); font-size:0.73rem; margin-top:0.12rem; line-height:1.45; }
        .vs-assistant-contract { display:grid; grid-template-columns:1fr 1fr; gap:1px; border:1px solid var(--line); background:var(--line); border-radius:7px; overflow:hidden; margin-bottom:1.1rem; }
        .vs-assistant-contract > div { background:var(--paper); padding:0.7rem 0.86rem; }
        .vs-assistant-contract span { display:block; color:var(--primary); font-size:0.64rem; font-weight:800; }
        .vs-assistant-contract b { display:block; color:var(--ink-soft); font-size:0.76rem; line-height:1.45; margin-top:0.18rem; }
        .vs-modalities { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:0.6rem; margin:0.2rem 0 0.85rem; }
        .vs-modalities > div { position:relative; min-height:74px; border:1px solid var(--line); border-top:3px solid var(--primary); background:#fff; border-radius:7px; padding:0.62rem 0.72rem; box-shadow:0 3px 12px rgba(32,58,70,0.03); }
        .vs-modalities > div.ready { border-top-color:var(--teal); }
        .vs-modalities > div.degraded { border-top-color:var(--review); }
        .vs-modalities > div.bounded { border-top-color:var(--primary); background:var(--primary-soft); }
        .vs-modalities span, .vs-modalities b, .vs-modalities small { display:block; }
        .vs-modalities span { color:var(--muted); font-size:0.61rem; font-weight:800; letter-spacing:0; }
        .vs-modalities b { color:var(--ink); font-size:0.78rem; margin-top:0.18rem; line-height:1.3; }
        .vs-modalities small { color:var(--muted); font-size:0.66rem; margin-top:0.14rem; overflow-wrap:anywhere; }
        [data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:7px; background:var(--paper); padding:0.25rem 0.55rem; box-shadow:0 3px 12px rgba(32,58,70,0.035); }
        [data-testid="stChatMessage"] + [data-testid="stChatMessage"] { margin-top:0.55rem; }
        [data-testid="stChatInput"] { border-color:var(--line-strong); }
        div[data-testid="stMetric"] { border:1px solid var(--line); background:var(--paper); border-radius:7px; padding:0.58rem 0.68rem; }
        .stButton > button, .stDownloadButton > button { border-radius:7px; min-height:2.52rem; font-weight:680; border-color:var(--line-strong); transition:background 140ms ease, border-color 140ms ease, box-shadow 140ms ease; }
        .stButton > button p, .stDownloadButton > button p { color:inherit !important; }
        .stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--primary); color:var(--primary-dark); box-shadow:0 4px 12px rgba(71,120,141,0.10); }
        .stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline:3px solid rgba(71,120,141,0.22); outline-offset:2px; }
        .stButton > button[kind="primary"] { background:var(--primary); border-color:var(--primary); color:#fff; }
        .stButton > button[kind="primary"]:hover { background:var(--primary-dark); border-color:var(--primary-dark); color:#fff; }
        div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:7px; overflow:hidden; background:#fff; }
        div[data-testid="stAlert"] { border-radius:7px; border-width:1px; }
        div[data-baseweb="tab-list"] { gap:0.2rem; border-bottom:1px solid var(--line); }
        button[data-baseweb="tab"] { padding-left:0.85rem; padding-right:0.85rem; }
        @media (max-width: 1050px) {
            .vs-workflow-band { grid-template-columns:1fr; }
            .vs-workflow-band ol { border-top:1px solid var(--line); padding-top:0.45rem; }
        }
        @media (max-width: 900px) {
            .block-container { padding-top: 3.6rem; padding-left: 1rem; padding-right: 1rem; }
            .vs-page-title { font-size: 1.48rem !important; }
            .vs-env { margin-top: 0; }
            .vs-step-strip { grid-template-columns: repeat(2,minmax(0,1fr)); }
            .vs-step-strip div:nth-child(2) { border-right:0; }
            .vs-step-strip div:nth-child(-n+2) { border-bottom:1px solid var(--line); }
            .vs-guidance-grid { grid-template-columns: repeat(2,minmax(0,1fr)); }
            .vs-result { grid-template-columns: repeat(2,minmax(0,1fr)); }
            .vs-metric { min-height: 104px; }
            .vs-report-hero { grid-template-columns:1fr 1fr; }
            .vs-report-hero > div:last-child { grid-column:1 / -1; }
        }
        @media (max-width: 620px) {
            .vs-workflow-band ol { grid-template-columns:repeat(2,minmax(0,1fr)); }
            .vs-workflow-band li:nth-child(3) { border-top:1px solid var(--line); }
            .vs-workflow-band li:nth-child(4) { border-top:1px solid var(--line); }
            .vs-step-strip, .vs-guidance-grid, .vs-io-strip, .vs-processing-contract { grid-template-columns: 1fr; }
            .vs-assistant-contract { grid-template-columns:1fr; }
            .vs-modalities { grid-template-columns:1fr; }
            .vs-step-strip div { border-right: 0; border-bottom: 1px solid var(--line); }
            .vs-step-strip div:last-child { border-bottom: 0; }
            .vs-env { grid-template-columns:1fr; }
            .vs-env > div { padding:0.18rem 0; border-right:0; border-bottom:1px solid var(--line); }
            .vs-env > div:last-child { border-bottom:0; }
            .vs-result { grid-template-columns: 1fr; }
            .vs-report-sheet header { flex-direction:column; }
            .vs-report-hero { grid-template-columns:1fr; }
            .vs-report-hero > div:last-child { grid-column:auto; }
            .vs-guide-row dl { grid-template-columns:1fr; }
            .vs-escalation { display:block; }
            .vs-escalation span { display:block; margin-top:0.25rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    run()
