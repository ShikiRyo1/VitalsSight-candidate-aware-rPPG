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

from src.product.console_api import create_app
from src.product.console_service import (
    ATTRIBUTION_BOUNDARY,
    CLAIM_BOUNDARY,
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
    video_preflight,
)
from src.product.console_store import ConsoleStore


DB_PATH = Path(os.getenv("VITALSSIGHT_DB_PATH", PROJECT / "runtime" / "vitalsight_console.db"))
UPLOAD_DIR = PROJECT / "runtime" / "uploads"
HEADLINE_METRICS = PROJECT / "reproducibility" / "headline_metrics.csv"
PROTOCOL_SUMMARY = PROJECT / "reproducibility" / "protocol_summary.json"


SECTIONS = [
    "Overview",
    "New assessment",
    "Cases",
    "Review queue",
    "Reports",
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
        default=st.session_state["vs_language"],
        key="vs_language_control",
    )
    st.session_state["vs_language"] = language or "ZH"
    st.sidebar.markdown("<div class='vs-brand'><span>VS</span><b>VitalsSight</b></div>", unsafe_allow_html=True)
    st.sidebar.caption(_ui("Evidence operations console", "证据运营控制台"))
    section = st.sidebar.radio(
        _ui("Workspace", "工作区"),
        SECTIONS,
        format_func=lambda item: ZH[item] if _is_zh() else item,
        label_visibility="collapsed",
        key="vs_section_radio",
    )
    st.session_state["vs_section"] = section
    st.sidebar.markdown("---")
    st.sidebar.caption(_ui("Research use only", "仅限研究使用"))
    st.sidebar.markdown(
        f"<div class='vs-boundary-small'>{_escape(_ui('No diagnosis, emergency alert, or autonomous clinical release.', '不用于诊断、急救告警或临床自主放行。'))}</div>",
        unsafe_allow_html=True,
    )

    _header(section)
    flash = st.session_state.pop("vs_flash", "")
    if flash:
        st.success(flash)
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
    elif section == "Evidence":
        _evidence()
    elif section == "Integrations":
        _integrations(store)
    else:
        _help_settings(store)


def _init_state() -> None:
    defaults = {
        "vs_language": "ZH",
        "vs_section": "Overview",
        "vs_section_radio": "Overview",
        "vs_operator": "Research operator",
        "vs_focus_case": "",
        "vs_preflight": None,
        "vs_upload_path": "",
        "vs_assessment_result": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if not st.session_state.get("vs_upload_cleanup_done"):
        _purge_stale_uploads()
        st.session_state["vs_upload_cleanup_done"] = True


@st.cache_resource(show_spinner=False)
def _store() -> ConsoleStore:
    return ConsoleStore(DB_PATH)


def _seed_if_empty(store: ConsoleStore) -> None:
    if store.list_cases():
        return
    for case in make_demo_cases():
        store.upsert_case(case, actor="demo-seed")


def _is_zh() -> bool:
    return st.session_state.get("vs_language", "ZH") == "ZH"


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
        "Evidence": _ui("Protocol-bound metrics and non-negotiable claim boundaries", "协议限定的性能指标与不可突破的证据边界"),
        "Integrations": _ui("Validated payloads, OpenAPI schema, and report endpoints", "校验载荷、OpenAPI 规范和报告接口"),
        "Help & settings": _ui("Acquisition guidance, status definitions, privacy, and workspace settings", "采集指引、状态定义、隐私与工作区设置"),
    }[section]
    left, right = st.columns([1, 0.32])
    with left:
        st.markdown(f"<h1 class='vs-page-title'>{_escape(title)}</h1>", unsafe_allow_html=True)
        st.caption(subtitle)
    with right:
        st.markdown(
            "<div class='vs-env'><b>RESEARCH</b><span>candidate-aware HR</span></div>",
            unsafe_allow_html=True,
        )
    st.markdown("<div class='vs-rule'></div>", unsafe_allow_html=True)


def _overview(store: ConsoleStore) -> None:
    cases = store.list_cases()
    reviews = store.list_reviews(include_closed=False)
    releases = sum(case.get("decision") == "release" for case in cases)
    retakes = sum(case.get("decision") == "retake" for case in cases)
    open_reviews = sum(review.get("status") != "closed" for review in reviews)
    quality = [float(case.get("quality_score") or 0) for case in cases]

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
        if c1.button(_ui("Start new assessment", "开始新评估"), type="primary", width="stretch"):
            _start_assessment()
        if c2.button(_ui("Open review queue", "打开复核队列"), width="stretch"):
            _go("Review queue")

    with right:
        st.subheader(_ui("Work requiring attention", "需要处理的工作"))
        if not reviews:
            st.success(_ui("No open reviews.", "当前没有待复核项目。"))
        for review in reviews[:5]:
            case = review["case"]
            st.markdown(
                f"<div class='vs-list-row'><div><b>{_escape(case.get('display_id'))}</b>"
                f"<span>{_escape(_decision_text(case.get('decision')))} · {_escape(_data_text(review.get('priority')))}</span></div>"
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
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.subheader(_ui("1. Purpose, consent, and retention", "1. 用途、授权与留存"))
        purpose = st.selectbox(
            _ui("Intended research use", "研究用途"),
            ["workflow_validation", "algorithm_evaluation", "research_demo"],
            format_func=lambda value: {
                "workflow_validation": _ui("Workflow validation", "流程验证"),
                "algorithm_evaluation": _ui("Algorithm evaluation", "算法评估"),
                "research_demo": _ui("Research demonstration", "研究演示"),
            }[value],
        )
        consent = st.checkbox(
            _ui(
                "I confirm the recording may be processed for the selected research purpose.",
                "我确认该视频可按照所选研究用途进行处理。",
            )
        )
        retention = st.radio(
            _ui("Raw-video handling", "原始视频处理"),
            ["delete_after_analysis", "session_only"],
            format_func=lambda value: {
                "delete_after_analysis": _ui("Delete after analysis; retain derived evidence", "分析后删除，仅保留派生证据"),
                "session_only": _ui("Keep locally until cleared or automatically expired", "本地保留至清除或自动过期"),
            }[value],
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
        )

        uploaded = None
        if source == "upload":
            uploaded = st.file_uploader(
                _ui("Adult RGB face video", "成人 RGB 人脸视频"),
                type=["mp4", "mov", "avi", "mkv", "m4v"],
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
        if action_col.button(run_label, type="primary", disabled=not consent, width="stretch"):
            try:
                if source == "upload":
                    if uploaded is None:
                        st.warning(_ui("Upload a video first.", "请先上传视频。"))
                    else:
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
                st.success(_ui("Assessment completed and saved.", "评估已完成并保存。"))
            except Exception as error:
                st.error(
                    _ui(
                        "The assessment could not be completed. No HR was published.",
                        "本次评估未能完成，系统未发布心率。",
                    )
                )
                st.caption(
                    f"{_ui('Technical detail', '技术信息')}: "
                    f"{type(error).__name__}: {str(error)[:180]}"
                )
        if reset_col.button(_ui("Clear", "清除"), width="stretch"):
            _remove_session_upload()
            st.session_state["vs_assessment_result"] = None
            st.session_state["vs_preflight"] = None
            st.rerun()

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
            c1, c2 = st.columns(2)
            if c1.button(_ui("Open case", "打开案例"), width="stretch"):
                _go("Cases")
            if c2.button(_ui("Build report", "生成报告"), width="stretch"):
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
        if st.button(_ui("Open report", "打开报告"), width="stretch", key=f"report_{case['case_id']}"):
            st.session_state["vs_focus_case"] = case["case_id"]
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
            submitted = st.form_submit_button(_ui("Save review", "保存复核"), type="primary", width="stretch")
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
            st.session_state["vs_flash"] = _ui(
                "Review record saved with an audit event.",
                "复核记录已保存并写入审计事件。",
            )
            st.rerun()
        st.subheader(_ui("Audit trail", "审计记录"))
        st.dataframe(pd.DataFrame(store.audit_events(case["case_id"])), hide_index=True, width="stretch")


def _reports(store: ConsoleStore) -> None:
    cases = store.list_cases()
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

    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("PDF", pdf, file_name=f"{case['display_id']}_evidence_report.pdf", mime="application/pdf", width="stretch")
    c2.download_button("JSON", json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), file_name=f"{case['display_id']}_evidence_report.json", mime="application/json", width="stretch")
    c3.download_button("Markdown", markdown.encode("utf-8"), file_name=f"{case['display_id']}_evidence_report.md", mime="text/markdown", width="stretch")
    c4.download_button(
        "CSV",
        pd.DataFrame([case]).drop(columns=[col for col in ["candidates", "trend_bpm", "preflight", "window_results"] if col in case]).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{case['display_id']}_case.csv",
        mime="text/csv",
        width="stretch",
    )
    tabs = st.tabs([_ui("Preview", "预览"), _ui("Attribution", "归因"), _ui("Structured data", "结构化数据")])
    with tabs[0]:
        st.markdown(markdown)
    with tabs[1]:
        attribution = payload["attribution"]
        attribution_rows = [
            {
                _ui("Factor", "因素"): _data_text(item.get("factor")),
                _ui("Observed", "观测值"): item.get("observed"),
                _ui("Direction", "方向"): _data_text(item.get("status")),
                _ui("Reason", "理由"): _data_text(item.get("reason")),
            }
            for item in attribution["all_factors"]
        ]
        st.dataframe(pd.DataFrame(attribution_rows), hide_index=True, width="stretch")
        st.info(_data_text(attribution["boundary"]))
    with tabs[2]:
        st.json(payload, expanded=False)


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
            marker_color="#5E8792",
            text=chart_data["mae_bpm"],
            textposition="outside",
            hovertext=chart_data["interpretation"],
        )
    )
    fig.update_layout(height=360, margin=dict(l=30, r=20, t=20, b=100), yaxis_title="MAE (BPM)", showlegend=False)
    fig.update_xaxes(tickangle=-28)
    fig.update_yaxes(gridcolor="#E3E9EB", zeroline=False)
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
            payload = ensure_output_contract(selected)
            st.success(_ui("The release/review contract is valid.", "放行/复核契约校验通过。"))
            st.json(payload, expanded=False)
        except ValueError as error:
            st.error(str(error))
        if st.button(_ui("Write integration audit event", "写入集成审计事件"), width="stretch"):
            store.log_event(selected["case_id"], "integration.payload_validated", actor=st.session_state["vs_operator"], details={"schema_version": selected.get("schema_version")})
            st.success(_ui("Audit event recorded.", "审计事件已记录。"))

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
            width="stretch",
        )
        st.code("uvicorn app.api_server:app --host 127.0.0.1 --port 8010", language="bash")
        st.caption(_ui("The API uses the same SQLite evidence and audit store as this page.", "API 与本页面使用同一个 SQLite 证据和审计存储。"))


def _help_settings(store: ConsoleStore) -> None:
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
        if st.button(_ui("Save operator", "保存操作员"), type="primary", width="stretch"):
            st.session_state["vs_operator"] = operator.strip() or "Research operator"
            st.success(_ui("Operator saved for audit events.", "操作员已用于后续审计事件。"))
        st.markdown("---")
        st.subheader(_ui("Data handling", "数据处理"))
        st.markdown(
            _ui(
                "Uploaded raw video is processed locally. The recommended mode deletes it after analysis and stores only derived evidence, the decision, and the audit trail.",
                "上传的原始视频在本地处理。推荐模式会在分析后删除视频，仅保留派生证据、决策和审计记录。",
            )
        )
        if st.button(_ui("Restore built-in demo cases", "恢复内置演示案例"), width="stretch"):
            for case in make_demo_cases():
                store.upsert_case(case, actor=st.session_state["vs_operator"])
            st.success(_ui("Built-in cases restored without deleting user cases.", "已恢复内置案例，未删除用户案例。"))


def _result_summary(case: dict[str, Any]) -> None:
    decision = str(case.get("decision", "review"))
    tone = {"release": "teal", "review": "amber", "retake": "coral"}.get(decision, "neutral")
    hr = _released_hr(case)
    st.markdown(
        f"<div class='vs-result {tone}'><div><small>{_escape(_ui('Decision','决策'))}</small>"
        f"<b>{_escape(_decision_text(decision))}</b></div>"
        f"<div><small>{_escape(_ui('Published HR','已发布心率'))}</small><b>{_escape(hr)}</b></div>"
        f"<div><small>{_escape(_ui('Quality','质量'))}</small><b>{_escape(_percent(case.get('quality_score')))}</b></div>"
        f"<div><small>{_escape(_ui('Next action','下一步'))}</small><span>{_escape(_data_text(case.get('recommended_action')))}</span></div></div>",
        unsafe_allow_html=True,
    )


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
    html = "".join(f"<div><b>{index}</b><span>{_escape(label)}</span></div>" for index, label in enumerate(labels, 1))
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
    fig.update_layout(height=330, margin=dict(l=20, r=15, t=10, b=45), yaxis=dict(range=[0, 1.08], tickformat=".0%"), showlegend=False)
    fig.update_yaxes(gridcolor="#E4EAEC", zeroline=False)
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
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
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
            line=dict(color="#5E8792", width=2.3),
            marker=dict(size=7, color="#5E8792"),
        )
    )
    fig.update_layout(height=270, margin=dict(l=25, r=15, t=10, b=35), xaxis_title=_ui("Window", "窗口"), yaxis_title="HR (BPM)", showlegend=False)
    fig.update_yaxes(gridcolor="#E4EAEC", zeroline=False)
    st.plotly_chart(fig, width="stretch")


def _decision_text(decision: str) -> str:
    labels = {
        "release": _ui("Released", "已放行"),
        "review": _ui("Review", "复核"),
        "retake": _ui("Retake", "重采"),
    }
    return labels.get(str(decision), str(decision))


def _decision_color(decision: str) -> str:
    return {"release": "#6B9B91", "review": "#B69A67", "retake": "#C8837A"}.get(decision, "#82919A")


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


def _start_assessment() -> None:
    """Open a clean acquisition flow without discarding stored cases."""
    _remove_session_upload()
    for key in ("vs_assessment_result", "vs_preflight", "vs_upload_path"):
        st.session_state.pop(key, None)
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
            --ink: #23333d;
            --muted: #687780;
            --line: #d9e2e5;
            --paper: #ffffff;
            --canvas: #f4f7f8;
            --blue: #5e8792;
            --blue-soft: #eaf1f3;
            --teal: #6b9b91;
            --teal-soft: #edf5f2;
            --amber: #b69a67;
            --amber-soft: #f6f1e8;
            --coral: #c8837a;
            --coral-soft: #f8eeeb;
        }
        .stApp { background: var(--canvas); color: var(--ink); }
        [data-testid="stToolbar"] { display: none; }
        .block-container { padding-top: 3rem; padding-bottom: 3rem; max-width: 1480px; }
        section[data-testid="stSidebar"] { background: #eef3f4; border-right: 1px solid var(--line); }
        section[data-testid="stSidebar"] .stRadio > div { gap: 0.18rem; }
        section[data-testid="stSidebar"] .stRadio label {
            padding: 0.55rem 0.65rem; border-radius: 5px; color: var(--ink);
        }
        section[data-testid="stSidebar"] .stRadio label:has(input:checked) { background: #dfe9ec; font-weight: 700; }
        .vs-brand { display: flex; align-items: center; gap: 0.65rem; margin: 1rem 0 0.1rem; font-size: 1.05rem; }
        .vs-brand span { width: 30px; height: 30px; display: inline-grid; place-items: center; background: #456f7a; color: white; border-radius: 5px; font-weight: 800; }
        .vs-boundary-small { font-size: 0.76rem; line-height: 1.4; color: #6d7c84; padding: 0.55rem 0; }
        .vs-page-title { font-size: 1.75rem !important; line-height: 1.15; margin: 0 !important; color: var(--ink); letter-spacing: 0; }
        .vs-env { margin-top: 0.45rem; border: 1px solid var(--line); background: var(--paper); padding: 0.55rem 0.75rem; border-radius: 5px; display: flex; justify-content: space-between; gap: 1rem; color: var(--muted); font-size: 0.78rem; }
        .vs-env b { color: var(--blue); letter-spacing: 0.04em; }
        .vs-rule { border-top: 1px solid var(--line); margin: 0.65rem 0 1rem; }
        .vs-section-rule { border-top: 1px solid var(--line); margin: 1.1rem 0; }
        h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
        h2 { font-size: 1.12rem !important; }
        h3 { font-size: 1rem !important; }
        .vs-metric { background: var(--paper); border: 1px solid var(--line); border-top: 3px solid #8a9aa2; border-radius: 6px; padding: 0.8rem 0.9rem; min-height: 112px; }
        .vs-metric.teal { border-top-color: var(--teal); }
        .vs-metric.amber { border-top-color: var(--amber); }
        .vs-metric.coral { border-top-color: var(--coral); }
        .vs-metric span { display:block; color: var(--muted); font-size: 0.78rem; }
        .vs-metric b { display:block; color: var(--ink); font-size: 1.65rem; line-height: 1.35; margin-top: 0.15rem; }
        .vs-metric small { color: var(--muted); font-size: 0.72rem; }
        .vs-list-row { background: var(--paper); border-bottom: 1px solid var(--line); padding: 0.7rem 0.75rem; }
        .vs-list-row:first-of-type { border-top: 1px solid var(--line); }
        .vs-list-row div { display:flex; justify-content:space-between; gap: 0.5rem; }
        .vs-list-row span, .vs-list-row small { color: var(--muted); font-size: 0.76rem; }
        .vs-step-strip { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 0; border: 1px solid var(--line); background: var(--paper); margin-bottom: 1rem; }
        .vs-step-strip div { display:flex; align-items:center; gap:0.55rem; padding:0.65rem 0.8rem; border-right:1px solid var(--line); }
        .vs-step-strip div:last-child { border-right: 0; }
        .vs-step-strip b { width:25px; height:25px; border:1px solid var(--blue); color:var(--blue); display:grid; place-items:center; border-radius:50%; font-size:0.76rem; }
        .vs-step-strip span { font-size:0.82rem; font-weight:650; }
        .vs-guidance-grid { display:grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap:0.45rem; margin:0.6rem 0 0.8rem; }
        .vs-guidance-grid div { border:1px solid var(--line); background:var(--paper); padding:0.55rem; border-radius:5px; display:flex; gap:0.4rem; align-items:center; }
        .vs-guidance-grid b { color:var(--blue); }
        .vs-guidance-grid span { font-size:0.75rem; }
        .vs-result { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:1px; border:1px solid var(--line); border-left:5px solid #82919a; background:var(--line); margin:0.4rem 0 0.8rem; }
        .vs-result.teal { border-left-color:var(--teal); }
        .vs-result.amber { border-left-color:var(--amber); }
        .vs-result.coral { border-left-color:var(--coral); }
        .vs-result > div { padding:0.65rem 0.8rem; min-width:0; background:var(--paper); }
        .vs-result small, .vs-result span { display:block; color:var(--muted); font-size:0.74rem; line-height:1.35; overflow-wrap:anywhere; }
        .vs-result b { display:block; font-size:1rem; color:var(--ink); margin-top:0.15rem; }
        .vs-empty { min-height:155px; display:grid; place-content:center; text-align:center; border:1px dashed #b8c5ca; background:#f9fbfb; color:var(--muted); }
        .vs-empty b, .vs-empty span { display:block; }
        .vs-factor { border-left:4px solid var(--amber); background:var(--paper); border-top:1px solid var(--line); border-right:1px solid var(--line); border-bottom:1px solid var(--line); padding:0.55rem 0.7rem; margin-bottom:0.45rem; }
        .vs-factor.good { border-left-color:var(--teal); }
        .vs-factor b, .vs-factor span, .vs-factor small { display:block; }
        .vs-factor span { color:var(--muted); font-size:0.76rem; }
        .vs-factor small { color:#50616a; margin-top:0.25rem; line-height:1.35; }
        .vs-check { display:flex; gap:0.7rem; align-items:flex-start; border-bottom:1px solid var(--line); padding:0.65rem 0; }
        .vs-check b { width:25px; height:25px; border:1px solid var(--blue); color:var(--blue); display:grid; place-items:center; border-radius:50%; font-size:0.75rem; flex:0 0 auto; }
        .vs-check span { line-height:1.5; }
        div[data-testid="stMetric"] { border:1px solid var(--line); background:var(--paper); border-radius:6px; padding:0.55rem 0.65rem; }
        .stButton > button, .stDownloadButton > button { border-radius:5px; min-height:2.45rem; font-weight:650; }
        .stButton > button[kind="primary"] { background:#557f88; border-color:#557f88; color:#fff; }
        .stButton > button[kind="primary"]:hover { background:#476f78; border-color:#476f78; color:#fff; }
        div[data-testid="stDataFrame"] { border:1px solid var(--line); }
        div[data-testid="stAlert"] { border-radius:5px; }
        @media (max-width: 900px) {
            .block-container { padding-top: 2.2rem; padding-left: 1rem; padding-right: 1rem; }
            .vs-page-title { font-size: 1.45rem !important; }
            .vs-env { margin-top: 0; flex-wrap: wrap; gap: 0.25rem 0.7rem; }
            .vs-step-strip { grid-template-columns: repeat(2,minmax(0,1fr)); }
            .vs-step-strip div:nth-child(2) { border-right:0; }
            .vs-guidance-grid { grid-template-columns: repeat(2,minmax(0,1fr)); }
            .vs-result { grid-template-columns: repeat(2,minmax(0,1fr)); }
            .vs-metric { min-height: 100px; }
        }
        @media (max-width: 520px) {
            .vs-step-strip, .vs-guidance-grid { grid-template-columns: 1fr; }
            .vs-step-strip div { border-right: 0; border-bottom: 1px solid var(--line); }
            .vs-step-strip div:last-child { border-bottom: 0; }
            .vs-result { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    run()
