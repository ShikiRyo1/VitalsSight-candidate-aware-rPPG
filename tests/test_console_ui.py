"""Executable UI regressions using isolated local stores and synthetic data only."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.product.console_service import make_demo_cases
from src.product.console_store import ConsoleStore


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def console_ui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VITALSSIGHT_AUTH_MODE", "disabled")
    monkeypatch.setenv("VITALSSIGHT_DB_PATH", str(tmp_path / "console.db"))
    monkeypatch.setenv("VITALSSIGHT_UPLOAD_DIR", str(tmp_path / "uploads"))
    st.cache_resource.clear()

    def create(language: str = "EN") -> AppTest:
        app = AppTest.from_file(str(PROJECT / "app" / "product_console.py"), default_timeout=30)
        app.session_state["vs_language"] = language
        app.session_state["vs_language_control"] = language
        return app.run()

    yield create
    st.cache_resource.clear()


def click_label(app: AppTest, label: str) -> AppTest:
    return next(button for button in app.button if button.label == label).click().run()


@pytest.mark.parametrize("language,title", [("EN", "Runtime setup checks"), ("ZH", "运行环境检查")])
def test_overview_onboarding_and_readiness_render_in_both_languages(console_ui, language: str, title: str) -> None:
    app = console_ui(language)
    assert not app.exception
    assert app.radio(key="vs_overview_scope").value == "demo"
    assert any(title in item.label for item in app.expander)
    assert any("vs-welcome" in item.value for item in app.markdown)
    assert len(app.dataframe[0].value) == len(make_demo_cases())


def test_upload_entry_selects_upload_without_granting_consent(console_ui) -> None:
    app = console_ui()
    app.button(key="vs_start_upload").click().run()
    assert not app.exception
    assert app.session_state["vs_section"] == "New assessment"
    assert app.radio(key="vs_source_control_en").value == "upload"
    assert not app.checkbox(key="vs_consent_control_en").value
    assert len(app.get("file_uploader")) == 1
    assert any("not a replay" in item.value for item in app.info)
    assert any("a video file" in item.value for item in app.caption)


def test_assessment_still_requires_consent(console_ui) -> None:
    app = console_ui()
    click_label(app, "Start guided assessment")
    click_label(app, "Run assessment")
    assert not app.exception
    assert app.session_state["vs_assessment_result"] is None
    assert any("Confirm processing consent" in item.value for item in app.warning)


@pytest.mark.parametrize("source,decision", [("stable", "release"), ("conflict", "review"), ("low_light", "retake")])
def test_demo_flow_preserves_decision_and_non_release_contract(console_ui, source: str, decision: str) -> None:
    app = console_ui()
    click_label(app, "Start guided assessment")
    app.radio(key="vs_source_control_en").set_value(source).run()
    app.checkbox(key="vs_consent_control_en").check().run()
    click_label(app, "Run assessment")
    assert not app.exception
    result = app.session_state["vs_assessment_result"]
    assert result["decision"] == decision
    assert result["input_kind"] == "built_in_demo"
    if decision != "release":
        assert result["released_hr_bpm"] is None
    assert any("Synthetic demo" in item.value for item in app.caption)


def test_changing_input_clears_preview_not_saved_cases(console_ui, tmp_path: Path) -> None:
    app = console_ui()
    click_label(app, "Start guided assessment")
    app.checkbox(key="vs_consent_control_en").check().run()
    click_label(app, "Run assessment")
    assert app.session_state["vs_assessment_result"]
    app.radio(key="vs_source_control_en").set_value("conflict").run()
    assert not app.exception
    assert app.session_state["vs_assessment_result"] is None
    assert app.session_state["vs_preflight"] is None
    assert len(ConsoleStore(tmp_path / "console.db").list_cases()) == len(make_demo_cases())
    assert any("No assessment for this input yet" in item.value for item in app.markdown)


def test_other_scope_has_helpful_empty_state_and_no_nan_metric(console_ui) -> None:
    app = console_ui()
    app.radio(key="vs_overview_scope").set_value("other").run()
    assert not app.exception
    assert not app.dataframe
    assert any("No evidence in this scope yet" in item.value for item in app.info)
    assert not any("nan%" in item.value.lower() for item in app.markdown)


def test_scope_filters_counts_table_and_pending_reviews(console_ui, tmp_path: Path) -> None:
    store = ConsoleStore(tmp_path / "console.db")
    for case in make_demo_cases():
        store.upsert_case(case, actor="test")
    other = make_demo_cases()[1]
    other.update(case_id="uploaded-fixture", display_id="UP-001", input_kind="uploaded_video", source_name="test-fixture.mp4")
    store.upsert_case(other, actor="test")
    app = console_ui()
    assert not app.exception
    assert app.radio(key="vs_overview_scope").value == "other"
    assert list(app.dataframe[0].value["Case"]) == ["UP-001"]
    assert any("UP-001" in item.value and "vs-list-row" in item.value for item in app.markdown)
    assert not any("VS-002" in item.value and "vs-list-row" in item.value for item in app.markdown)
    app.radio(key="vs_overview_scope").set_value("all").run()
    assert not app.exception
    assert len(app.dataframe[0].value) == len(make_demo_cases()) + 1


@pytest.mark.parametrize("filename,data,limit", [("empty.mp4", b"", 10), ("wrong.txt", b"x", 10), ("large.mp4", b"x" * 11, 10)])
def test_processing_revalidates_actual_bytes_before_creating_upload(filename: str, data: bytes, limit: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app import product_console as console

    target = tmp_path / "not-created"
    monkeypatch.setattr(console, "UPLOAD_DIR", target)
    monkeypatch.setattr(console, "_remove_session_upload", lambda: None)
    monkeypatch.setattr(console, "_runtime_snapshot", lambda: {"upload_policy": {"max_upload_bytes": limit}})
    monkeypatch.setattr(console, "_ui", lambda english, chinese: english)
    uploaded = SimpleNamespace(name=filename, getvalue=lambda: data)
    with pytest.raises(ValueError):
        console._process_upload(uploaded, purpose="workflow_validation", retention="delete_after_analysis")
    assert not target.exists()


def test_new_onboarding_styles_include_small_screen_layout() -> None:
    source = (PROJECT / "app" / "product_console.py").read_text(encoding="utf-8")
    assert ".vs-start-grid { grid-template-columns:1fr; }" in source
    assert ".vs-welcome { grid-template-columns:1fr; }" in source
    assert "on_change=_invalidate_session_assessment" in source


@pytest.mark.parametrize("change", ["organization", "participant", "purpose", "consent_id", "version", "status", "withdrawn", "affirmation"])
def test_assessment_context_invalidates_old_preview_on_identity_or_consent_change(change: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from app import product_console as console

    context = {
        "organization_id": "org-a",
        "participant_id": "participant-a",
        "purpose": "workflow_validation",
        "consent": True,
        "active_consent": {"consent_id": "consent-a", "document_version": "v1", "status": "active"},
    }
    first = console._assessment_context_fingerprint(**context)
    state = {"vs_assessment_context": first, "vs_assessment_result": {"case_id": "old"}, "vs_preflight": {"overall": "pass"}}
    monkeypatch.setattr(console.st, "session_state", state)
    monkeypatch.setattr(console, "_remove_session_upload", lambda: None)
    if change == "organization":
        context["organization_id"] = "org-b"
    elif change == "participant":
        context["participant_id"] = "participant-b"
    elif change == "purpose":
        context["purpose"] = "algorithm_evaluation"
    elif change == "consent_id":
        context["active_consent"]["consent_id"] = "consent-b"
    elif change == "version":
        context["active_consent"]["document_version"] = "v2"
    elif change == "status":
        context["active_consent"]["status"] = "withdrawn"
    elif change == "withdrawn":
        context["active_consent"] = None
        context["consent"] = False
    else:
        context["consent"] = False
    assert console._sync_assessment_context(console._assessment_context_fingerprint(**context))
    assert state["vs_assessment_result"] is None
    assert state["vs_preflight"] is None


def test_unchanged_assessment_context_preserves_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import product_console as console

    state = {"vs_assessment_context": "same", "vs_assessment_result": {"case_id": "current"}, "vs_preflight": {"overall": "pass"}}
    monkeypatch.setattr(console.st, "session_state", state)
    assert not console._sync_assessment_context("same")
    assert state["vs_assessment_result"]["case_id"] == "current"


def test_preexisting_unbound_preview_is_invalidated_on_first_context_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import product_console as console

    state = {"vs_assessment_result": {"case_id": "unknown-context"}, "vs_preflight": {"overall": "pass"}}
    monkeypatch.setattr(console.st, "session_state", state)
    monkeypatch.setattr(console, "_remove_session_upload", lambda: None)
    assert console._sync_assessment_context("known-context")
    assert state["vs_assessment_result"] is None
    assert state["vs_preflight"] is None


def test_participant_switch_clears_session_preview_in_ui(console_ui, tmp_path: Path) -> None:
    store = ConsoleStore(tmp_path / "console.db")
    for name in ("P-001", "P-002"):
        store.upsert_participant(pseudonym=name, study_id="test-study", organization_id="local-research", created_by="test")
    app = console_ui()
    click_label(app, "Start guided assessment")
    app.checkbox(key="vs_consent_control_en").check().run()
    click_label(app, "Run assessment")
    assert not app.exception
    assert app.session_state["vs_assessment_result"]
    select = next(item for item in app.selectbox if item.label == "Participant context")
    other = next(option for option in select.options if option != select.value)
    select.set_value(other).run()
    assert not app.exception
    assert app.session_state["vs_assessment_result"] is None
    assert app.session_state["vs_preflight"] is None
