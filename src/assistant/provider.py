from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import os
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ProviderToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderReply:
    content: str = ""
    tool_calls: tuple[ProviderToolCall, ...] = ()


@dataclass(frozen=True)
class ProviderStatus:
    available: bool
    provider: str
    model: str
    details: str = ""


class ChatProvider(Protocol):
    provider_name: str
    model: str

    def status(self) -> ProviderStatus: ...

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> ProviderReply: ...


class VisionProvider(Protocol):
    provider_name: str
    model: str

    def status(self) -> ProviderStatus: ...

    def analyze(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> ProviderReply: ...


class OllamaProvider:
    """Small Ollama client with no additional SDK dependency."""

    provider_name = "ollama"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("VITALSSIGHT_OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")
        self.model = model or os.getenv("VITALSSIGHT_ASSISTANT_MODEL") or "qwen3.6:35b"
        self.timeout_seconds = float(timeout_seconds or os.getenv("VITALSSIGHT_ASSISTANT_TIMEOUT", "300"))
        self.thinking_enabled = os.getenv("VITALSSIGHT_ASSISTANT_THINKING", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.context_tokens = int(os.getenv("VITALSSIGHT_ASSISTANT_NUM_CTX", "8192"))
        self.answer_tokens = int(os.getenv("VITALSSIGHT_ASSISTANT_NUM_PREDICT", "768"))
        self.reasoning_temperature = float(os.getenv("VITALSSIGHT_ASSISTANT_TEMPERATURE", "0.6"))
        self.keep_alive = os.getenv("VITALSSIGHT_ASSISTANT_KEEP_ALIVE", "0")

    def _request(self, path: str, payload: dict[str, Any] | None = None, *, timeout: float | None = None) -> Any:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urlopen(request, timeout=timeout or self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {error.code}: {detail[:500]}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"Ollama is unavailable at {self.base_url}: {error}") from error

    def status(self) -> ProviderStatus:
        try:
            payload = self._request("/api/tags", timeout=2.5)
        except RuntimeError as error:
            return ProviderStatus(False, self.provider_name, self.model, str(error))
        names = {str(item.get("name")) for item in payload.get("models", [])}
        aliases = {self.model, f"{self.model}:latest"}
        available = bool(names & aliases)
        reasoning = "deep reasoning enabled" if self.thinking_enabled else "deep reasoning disabled"
        detail = (
            f"model ready; {reasoning}; context={self.context_tokens}; max_output={self.answer_tokens}; "
            f"keep_alive={self.keep_alive}"
            if available
            else f"Ollama is running, but {self.model} is not installed"
        )
        return ProviderStatus(available, self.provider_name, self.model, detail)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> ProviderReply:
        # Tool selection stays deterministic and short; deeper reasoning is reserved
        # for the schema-constrained final explanation.
        final_composition = response_schema is not None and not tools
        think = self.thinking_enabled and final_composition
        nonthinking_budget = min(768, self.answer_tokens) if final_composition else min(384, self.answer_tokens)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": think,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.reasoning_temperature if think else 0.0,
                "top_p": 0.95 if think else 0.8,
                "top_k": 20,
                "num_ctx": self.context_tokens,
                "num_predict": self.answer_tokens if think else nonthinking_budget,
            },
        }
        if tools:
            payload["tools"] = tools
        if response_schema:
            payload["format"] = response_schema
        result = self._request("/api/chat", payload)
        message = result.get("message") or {}
        if think and response_schema:
            content = str(message.get("content") or "")
            try:
                structured_answer_ready = isinstance(json.loads(content), dict)
            except json.JSONDecodeError:
                structured_answer_ready = False
            if not structured_answer_ready:
                retry_payload = dict(payload)
                retry_payload["think"] = False
                retry_payload["options"] = {
                    **payload["options"],
                    "temperature": 0.0,
                    "top_p": 0.8,
                    "num_predict": self.answer_tokens,
                }
                result = self._request("/api/chat", retry_payload)
                message = result.get("message") or {}
        parsed_calls: list[ProviderToolCall] = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if function.get("name"):
                parsed_calls.append(ProviderToolCall(str(function["name"]), dict(arguments)))
        return ProviderReply(content=str(message.get("content") or ""), tool_calls=tuple(parsed_calls))


class OllamaVisionProvider(OllamaProvider):
    """Bounded vision sidecar using Ollama's documented base64 image API."""

    provider_name = "ollama-vision"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model or os.getenv("VITALSSIGHT_ASSISTANT_VISION_MODEL") or "qwen3-vl:4b-instruct",
            timeout_seconds=timeout_seconds or float(os.getenv("VITALSSIGHT_VISION_TIMEOUT", "180")),
        )
        self.keep_alive = os.getenv("VITALSSIGHT_VISION_KEEP_ALIVE", "0")

    def analyze(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> ProviderReply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                }
            ],
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.0, "top_p": 0.8, "num_ctx": 8192, "num_predict": 768},
        }
        if response_schema:
            payload["format"] = response_schema
        result = self._request("/api/chat", payload)
        message = result.get("message") or {}
        return ProviderReply(content=str(message.get("content") or ""))


class UnavailableProvider:
    provider_name = "none"
    model = "deterministic-guidance"

    def status(self) -> ProviderStatus:
        return ProviderStatus(False, self.provider_name, self.model, "No model provider was configured")

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> ProviderReply:
        raise RuntimeError("No language-model provider is available")
