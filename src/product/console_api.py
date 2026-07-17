from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.assistant import (
    AssistantChatRequest,
    AssistantChatResponse,
    AssistantConfirmRequest,
    AssistantConfirmResponse,
    AssistantHealthResponse,
    AssistantMultimodalHealthResponse,
    AudioTranscriptionResponse,
    ImageAnalysisResponse,
    MultimodalAssistantService,
    AssistantOrchestrator,
)
from src.assistant.multimodal import MediaProcessingError
from src.assistant.provider import ChatProvider
from src.product.console_service import (
    CLAIM_BOUNDARY,
    REPORT_VERSION,
    build_report_payload,
    build_report_pdf,
    case_from_preflight,
    case_from_runtime_failure,
    make_demo_cases,
    preflight_from_decode_error,
    run_uploaded_video,
    sanitize_report_value,
    video_preflight,
)
from src.product.build_identity import path_fingerprint, source_build_identity
from src.product.console_store import ConsoleStore


class ReviewUpdate(BaseModel):
    status: str = Field(pattern="^(open|in_review|waiting_retake|closed)$")
    priority: str = Field(pattern="^(urgent|high|routine|low)$")
    assignee: str = ""
    note: str = ""
    resolution: str = ""
    actor: str = "api-user"


def default_db_path() -> Path:
    configured = os.getenv("VITALSSIGHT_DB_PATH")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "runtime" / "vitalsight_console.db"


def create_app(
    db_path: str | Path | None = None,
    *,
    seed_demo: bool = True,
    assistant_provider: ChatProvider | None = None,
    assistant_actions_enabled: bool | None = None,
    multimodal_service: MultimodalAssistantService | None = None,
) -> FastAPI:
    resolved_db_path = Path(db_path or default_db_path())
    store = ConsoleStore(resolved_db_path)
    assistant = AssistantOrchestrator(
        store,
        db_path=resolved_db_path,
        provider=assistant_provider,
        actions_enabled=assistant_actions_enabled,
    )
    multimodal = multimodal_service or MultimodalAssistantService()
    upload_dir = Path(os.getenv("VITALSSIGHT_UPLOAD_DIR", resolved_db_path.parent / "uploads" / "api"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    max_upload_bytes = int(os.getenv("VITALSSIGHT_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
    if seed_demo and not store.list_cases():
        for case in make_demo_cases():
            store.upsert_case(case, actor="demo-seed")

    app = FastAPI(
        title="VitalsSight Evidence API",
        version="1.1.0",
        description=(
            "Research product API for candidate-aware adult HR evidence, review workflow, report export, "
            "and transient voice/image assistant intake. It does not provide clinical decisions."
        ),
    )
    app.state.store = store
    app.state.assistant = assistant
    app.state.multimodal = multimodal

    @app.get("/health")
    def health() -> dict[str, Any]:
        return sanitize_report_value({
            "status": "ok",
            "service": "vitalssight-console",
            "report_version": REPORT_VERSION,
            "build": source_build_identity(),
            "storage": {
                "upload_dir_fingerprint": path_fingerprint(upload_dir),
                "raw_video_policy": "delete_after_analysis",
            },
            "claim_boundary": CLAIM_BOUNDARY,
        })

    @app.get("/api/v1/cases")
    def list_cases() -> dict[str, Any]:
        cases = store.list_cases()
        return sanitize_report_value({"count": len(cases), "items": cases, "claim_boundary": CLAIM_BOUNDARY})

    @app.get("/api/v1/assistant/health", response_model=AssistantHealthResponse)
    def assistant_health() -> AssistantHealthResponse:
        return assistant.health()

    @app.get("/api/v1/assistant/multimodal/health", response_model=AssistantMultimodalHealthResponse)
    def assistant_multimodal_health() -> AssistantMultimodalHealthResponse:
        return multimodal.health()

    @app.post("/api/v1/assistant/transcribe", response_model=AudioTranscriptionResponse)
    def assistant_transcribe(
        file: UploadFile = File(...),
        language: str = Form("auto"),
    ) -> AudioTranscriptionResponse:
        try:
            if language not in {"auto", "zh", "en"}:
                raise HTTPException(status_code=422, detail="language must be auto, zh, or en")
            data = file.file.read(multimodal.max_audio_bytes + 1)
            if len(data) > multimodal.max_audio_bytes:
                raise HTTPException(status_code=413, detail="Audio exceeds the configured upload limit")
            return multimodal.transcribe_audio(
                data,
                filename=file.filename or "voice.wav",
                content_type=file.content_type or "audio/wav",
                language=None if language == "auto" else language,
            )
        except MediaProcessingError as error:
            raise HTTPException(status_code=error.status_code, detail=str(error)) from error
        finally:
            file.file.close()

    @app.post("/api/v1/assistant/analyze-image", response_model=ImageAnalysisResponse)
    def assistant_analyze_image(
        file: UploadFile = File(...),
        question: str = Form(""),
        language: str = Form("zh"),
    ) -> ImageAnalysisResponse:
        try:
            if language not in {"zh", "en"}:
                raise HTTPException(status_code=422, detail="language must be zh or en")
            data = file.file.read(multimodal.max_image_bytes + 1)
            if len(data) > multimodal.max_image_bytes:
                raise HTTPException(status_code=413, detail="Image exceeds the configured upload limit")
            return multimodal.analyze_image(
                data,
                filename=file.filename or "image.png",
                content_type=file.content_type or "application/octet-stream",
                question=question,
                language=language,
            )
        except MediaProcessingError as error:
            raise HTTPException(status_code=error.status_code, detail=str(error)) from error
        finally:
            file.file.close()

    @app.post("/api/v1/assistant/chat", response_model=AssistantChatResponse)
    def assistant_chat(request: AssistantChatRequest) -> AssistantChatResponse:
        return assistant.chat(request)

    @app.post("/api/v1/assistant/confirm", response_model=AssistantConfirmResponse)
    def assistant_confirm(request: AssistantConfirmRequest) -> AssistantConfirmResponse:
        try:
            return assistant.confirm(request.token, actor=request.actor)
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/assistant/reject", response_model=AssistantConfirmResponse)
    def assistant_reject(request: AssistantConfirmRequest) -> AssistantConfirmResponse:
        try:
            return assistant.reject(request.token, actor=request.actor)
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/assessments/video", status_code=201)
    def assess_video(
        file: UploadFile = File(...),
        consent_recorded: bool = Form(...),
        purpose: str = Form("workflow_validation"),
        retention_policy: str = Form("delete_after_analysis"),
        actor: str = Form("api-user"),
    ) -> dict[str, Any]:
        if not consent_recorded:
            file.file.close()
            raise HTTPException(status_code=422, detail="Explicit research-processing consent is required")
        if purpose not in {"workflow_validation", "algorithm_evaluation", "research_demo"}:
            file.file.close()
            raise HTTPException(status_code=422, detail="Unsupported research purpose")
        if retention_policy != "delete_after_analysis":
            file.file.close()
            raise HTTPException(
                status_code=422,
                detail="The upload API accepts delete_after_analysis only; raw video is never retained",
            )

        original_name = Path(file.filename or "upload.bin").name
        safe_name = "".join(character if character.isalnum() or character in "._-" else "_" for character in original_name)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}:
            file.file.close()
            raise HTTPException(status_code=415, detail="Supported video types: mp4, mov, avi, mkv, m4v")

        upload_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = upload_dir / f"{uuid4().hex}_{safe_name}"
        size = 0
        try:
            with temporary_path.open("wb") as target:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_upload_bytes:
                        raise HTTPException(status_code=413, detail="Video exceeds the configured upload limit")
                    target.write(chunk)
            if size == 0:
                raise HTTPException(status_code=422, detail="Uploaded video is empty")

            try:
                preflight = video_preflight(temporary_path)
            except Exception as error:
                preflight = preflight_from_decode_error(original_name, error)

            if preflight["overall"] == "fail":
                case = case_from_preflight(
                    preflight,
                    purpose=purpose,
                    retention_policy=retention_policy,
                )
            else:
                try:
                    case = run_uploaded_video(
                        temporary_path,
                        purpose=purpose,
                        retention_policy=retention_policy,
                        preflight=preflight,
                    )
                except Exception as error:
                    case = case_from_runtime_failure(
                        preflight,
                        error,
                        purpose=purpose,
                        retention_policy=retention_policy,
                    )
            case["source_name"] = safe_name
            case = sanitize_report_value(case)
            store.upsert_case(case, actor=actor.strip() or "api-user")
            return sanitize_report_value(
                {"item": case, "raw_video_retained": False, "claim_boundary": CLAIM_BOUNDARY}
            )
        finally:
            file.file.close()
            if temporary_path.exists():
                temporary_path.unlink()

    @app.get("/api/v1/cases/{case_id}")
    def get_case(case_id: str) -> dict[str, Any]:
        case = store.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        return sanitize_report_value(case)

    @app.get("/api/v1/reviews")
    def list_reviews(include_closed: bool = True) -> dict[str, Any]:
        reviews = store.list_reviews(include_closed=include_closed)
        return sanitize_report_value({"count": len(reviews), "items": reviews, "claim_boundary": CLAIM_BOUNDARY})

    @app.put("/api/v1/reviews/{case_id}")
    def update_review(case_id: str, update: ReviewUpdate) -> dict[str, Any]:
        try:
            store.update_review(case_id, **update.model_dump())
        except KeyError as error:
            raise HTTPException(status_code=404, detail=sanitize_report_value(str(error))) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=sanitize_report_value(str(error))) from error
        review = next((item for item in store.list_reviews() if item["case_id"] == case_id), None)
        return sanitize_report_value({"item": review, "claim_boundary": CLAIM_BOUNDARY})

    @app.get("/api/v1/cases/{case_id}/report")
    def get_report(case_id: str, format: str = "json", language: str = "en") -> Any:
        case = store.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        review = next((item for item in store.list_reviews() if item["case_id"] == case_id), None)
        payload = build_report_payload(
            case,
            review=review,
            audit_events=store.audit_events(case_id),
        )
        if format.lower() == "pdf":
            pdf = build_report_pdf(payload, language=language)
            return Response(
                content=pdf,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{case_id}.pdf"'},
            )
        if format.lower() != "json":
            raise HTTPException(status_code=422, detail="format must be json or pdf")
        return sanitize_report_value(payload)

    return app


app = create_app()
