from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser


DEFAULT_UI_PORT = 8561
DEFAULT_API_PORT = 8061


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def user_data_root() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    target = base / "VitalsSightResearchDemo"
    target.mkdir(parents=True, exist_ok=True)
    return target


def port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def choose_ports(ui_port: int, api_port: int) -> tuple[int, int]:
    for offset in range(20):
        ui = ui_port + offset
        api = api_port + offset
        if not port_is_open(ui) and not port_is_open(api):
            return ui, api
    raise RuntimeError("No free local VitalsSight port pair was found")


def http_json(url: str, timeout: float = 2.0) -> dict[str, object] | list[object] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                return None
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def http_ready(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def configure_environment(ui_port: int, api_port: int) -> dict[str, str]:
    root = repository_root()
    data = user_data_root()
    logs = data / "logs"
    uploads = data / "uploads"
    logs.mkdir(parents=True, exist_ok=True)
    uploads.mkdir(parents=True, exist_ok=True)
    values = {
        "VITALSSIGHT_DB_PATH": str(data / "vitalsight_console.db"),
        "VITALSSIGHT_UPLOAD_DIR": str(uploads),
        "VITALSSIGHT_AUTH_MODE": "disabled",
        "VITALSSIGHT_ASSISTANT_MODEL": os.getenv("VITALSSIGHT_ASSISTANT_MODEL", "qwen3:8b"),
        "VITALSSIGHT_ASSISTANT_VISION_MODEL": os.getenv(
            "VITALSSIGHT_ASSISTANT_VISION_MODEL", "qwen3-vl:4b-instruct"
        ),
        "VITALSSIGHT_ASSISTANT_ACTIONS_ENABLED": "false",
        "VITALSSIGHT_OLLAMA_URL": os.getenv("VITALSSIGHT_OLLAMA_URL", "http://127.0.0.1:11434"),
        "VITALSSIGHT_BUILD_COMMIT": git_value("rev-parse", "HEAD"),
        "VITALSSIGHT_BUILD_TREE": git_value("rev-parse", "HEAD^{tree}"),
        "VITALSSIGHT_BUILD_DIRTY": "false" if not git_value("status", "--porcelain") else "true",
        "VITALSSIGHT_DESKTOP_MODE": "true",
        "VITALSSIGHT_UI_PORT": str(ui_port),
        "VITALSSIGHT_API_PORT": str(api_port),
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_GLOBAL_DEVELOPMENT_MODE": "false",
        "PYTHONPATH": str(root),
    }
    os.environ.update(values)
    return values


def git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root()), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or "clean"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"


class LocalServices:
    def __init__(self, ui_port: int, api_port: int) -> None:
        self.ui_port = ui_port
        self.api_port = api_port
        self.ui_url = f"http://127.0.0.1:{ui_port}"
        self.api_url = f"http://127.0.0.1:{api_port}"
        self.processes: list[subprocess.Popen[bytes]] = []
        self.logs: list[object] = []

    def start(self) -> None:
        root = repository_root()
        logs = user_data_root() / "logs"
        commands = (
            (
                "api",
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "app.api_server:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(self.api_port),
                    "--log-level",
                    "warning",
                    "--no-access-log",
                ],
            ),
            (
                "ui",
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    "app/streamlit_app.py",
                    "--server.address",
                    "127.0.0.1",
                    "--server.port",
                    str(self.ui_port),
                    "--server.headless",
                    "true",
                    "--global.developmentMode",
                    "false",
                    "--browser.gatherUsageStats",
                    "false",
                ],
            ),
        )
        for name, command in commands:
            stdout = (logs / f"local_{name}.stdout.log").open("wb")
            stderr = (logs / f"local_{name}.stderr.log").open("wb")
            self.logs.extend((stdout, stderr))
            self.processes.append(
                subprocess.Popen(
                    command,
                    cwd=str(root),
                    env=os.environ.copy(),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                )
            )

    def wait_ready(self, timeout: float) -> dict[str, object]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            exited = [process.returncode for process in self.processes if process.poll() is not None]
            if exited:
                raise RuntimeError(f"A VitalsSight service exited during startup: {exited}")
            health = http_json(f"{self.api_url}/api/v1/assistant/health", timeout=1.5)
            if isinstance(health, dict) and http_ready(self.ui_url, timeout=1.5):
                return health
            time.sleep(0.5)
        raise TimeoutError(f"VitalsSight did not become ready within {timeout:.0f} seconds")

    def stop(self) -> None:
        for process in reversed(self.processes):
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 8
        for process in reversed(self.processes):
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
        for handle in self.logs:
            try:
                handle.close()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the local VitalsSight research workflow")
    parser.add_argument("--ui-port", type=int, default=DEFAULT_UI_PORT)
    parser.add_argument("--api-port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    ui_port, api_port = choose_ports(args.ui_port, args.api_port)
    configure_environment(ui_port, api_port)
    services = LocalServices(ui_port, api_port)
    stopping = False

    def stop_services(*_unused: object) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        services.stop()

    signal.signal(signal.SIGINT, stop_services)
    signal.signal(signal.SIGTERM, stop_services)

    try:
        services.start()
        assistant = services.wait_ready(args.startup_timeout)
        cases = http_json(f"{services.api_url}/api/v1/cases", timeout=5)
        case_items = cases.get("items", []) if isinstance(cases, dict) else cases
        result = {
            "status": "PASS",
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "ui": services.ui_url,
            "api": services.api_url,
            "assistant": assistant,
            "seeded_case_count": len(case_items) if isinstance(case_items, list) else None,
            "data_root": str(user_data_root()),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        if args.smoke_test:
            return 0
        if not args.no_browser:
            webbrowser.open(services.ui_url)
        print("VitalsSight is ready. Press Control-C in this window to stop it.", flush=True)
        while not stopping:
            exited = [process.returncode for process in services.processes if process.poll() is not None]
            if exited:
                raise RuntimeError(f"A VitalsSight service stopped unexpectedly: {exited}")
            time.sleep(0.5)
        return 0
    finally:
        stop_services()


if __name__ == "__main__":
    raise SystemExit(main())
