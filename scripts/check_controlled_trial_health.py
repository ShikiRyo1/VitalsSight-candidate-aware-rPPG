from __future__ import annotations

import argparse
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(url: str, *, token: str = "", timeout: float = 10.0) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check VitalsSight controlled-trial services.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8010")
    parser.add_argument("--token", default="", help="OIDC access token for protected checks.")
    parser.add_argument("--require-auth", action="store_true")
    parser.add_argument("--require-model", action="store_true")
    parser.add_argument("--require-multimodal", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.api_url.rstrip("/")
    checks: dict[str, Any] = {}
    try:
        service = request_json(f"{base}/health")
        checks["service"] = service
        auth_mode = str((service.get("auth") or {}).get("mode") or "")
        if args.require_auth and auth_mode != "required":
            raise RuntimeError(f"Expected required auth mode, found {auth_mode or 'unknown'}")
        assistant = request_json(f"{base}/api/v1/assistant/health", token=args.token)
        checks["assistant"] = assistant
        if args.require_model and not assistant.get("model_available"):
            raise RuntimeError("The configured language model is unavailable")
        multimodal = request_json(
            f"{base}/api/v1/assistant/multimodal/health",
            token=args.token,
        )
        checks["multimodal"] = multimodal
        multimodal_ready = bool((multimodal.get("image") or {}).get("available")) and bool(
            (multimodal.get("speech") or {}).get("available")
        )
        if args.require_multimodal and not multimodal_ready:
            raise RuntimeError("One or more multimodal sidecars are unavailable")
    except (HTTPError, URLError, TimeoutError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "failed", "error": str(error), "checks": checks}, indent=2))
        return 1
    print(json.dumps({"status": "passed", "checks": checks}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
