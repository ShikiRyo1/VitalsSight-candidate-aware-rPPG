from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser


APP_NAME = "VitalsSight Research Demo"
DEFAULT_UI_PORT = 8561
DEFAULT_API_PORT = 8061


def bundle_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")).resolve()
    return Path(__file__).resolve().parents[1]


def user_data_root() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    root = base / "VitalsSightResearchDemo"
    root.mkdir(parents=True, exist_ok=True)
    return root


def log_root() -> Path:
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Logs" / "VitalsSight"
    else:
        root = user_data_root() / "logs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def stage_log(message: str) -> None:
    try:
        with (log_root() / "desktop_launcher.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} pid={os.getpid()} {message}\n")
    except OSError:
        pass


def metadata() -> dict[str, str]:
    path = bundle_root() / "packaging" / "build_metadata.json"
    if not path.is_file():
        return {"version": "development", "commit": "source", "tree": "source"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": "unknown", "commit": "unknown", "tree": "unknown"}


def port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def choose_ports(ui_port: int, api_port: int) -> tuple[int, int]:
    for offset in range(30):
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


def installed_ollama_models() -> set[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
            timeout=12,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return set()
    models: set[str] = set()
    for line in result.stdout.splitlines()[1:]:
        fields = line.split()
        if fields:
            models.add(fields[0])
    return models


def configure_environment(ui_port: int, api_port: int) -> dict[str, str]:
    root = bundle_root()
    data = user_data_root()
    uploads = data / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    build = metadata()
    models = installed_ollama_models()
    language = next(
        (name for name in ("qwen3.6:35b", "qwen3:8b", "qwen3:4b") if name in models),
        "qwen3:8b",
    )
    values = {
        "VITALSSIGHT_DB_PATH": str(data / "vitalsight_console.db"),
        "VITALSSIGHT_UPLOAD_DIR": str(uploads),
        "VITALSSIGHT_AUTH_MODE": "disabled",
        "VITALSSIGHT_ASSISTANT_MODEL": os.getenv("VITALSSIGHT_ASSISTANT_MODEL", language),
        "VITALSSIGHT_ASSISTANT_VISION_MODEL": os.getenv(
            "VITALSSIGHT_ASSISTANT_VISION_MODEL", "qwen3-vl:4b-instruct"
        ),
        "VITALSSIGHT_ASSISTANT_ACTIONS_ENABLED": "false",
        "VITALSSIGHT_OLLAMA_URL": os.getenv("VITALSSIGHT_OLLAMA_URL", "http://127.0.0.1:11434"),
        "VITALSSIGHT_BUILD_COMMIT": str(build.get("commit", "unknown")),
        "VITALSSIGHT_BUILD_TREE": str(build.get("tree", "unknown")),
        "VITALSSIGHT_BUILD_DIRTY": "false",
        "VITALSSIGHT_DESKTOP_MODE": "true",
        "VITALSSIGHT_UI_PORT": str(ui_port),
        "VITALSSIGHT_API_PORT": str(api_port),
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_GLOBAL_DEVELOPMENT_MODE": "false",
        "PYTHONPATH": str(root),
    }
    os.environ.update(values)
    return values


def run_api_worker(port: int) -> int:
    root = bundle_root()
    os.chdir(root)
    sys.path.insert(0, str(root))
    stage_log(f"API worker starting on 127.0.0.1:{port}")
    import uvicorn
    from app.api_server import app

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    return 0


def run_ui_worker(port: int) -> int:
    root = bundle_root()
    os.chdir(root)
    sys.path.insert(0, str(root))
    from streamlit.web import cli as streamlit_cli

    script = user_data_root() / "runtime" / "streamlit_bootstrap.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    bootstrap = "from app.product_console import run\n\nrun()\n"
    if not script.is_file() or script.read_text(encoding="utf-8") != bootstrap:
        script.write_text(bootstrap, encoding="utf-8")
    sys.argv = [
        "streamlit",
        "run",
        str(script),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--global.developmentMode",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    return int(streamlit_cli.main() or 0)


def worker_command(worker: str, port: int) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--worker", worker, "--port", str(port)]
    return [sys.executable, str(Path(__file__).resolve()), "--worker", worker, "--port", str(port)]


class ServicePair:
    def __init__(self, ui_port: int, api_port: int) -> None:
        self.ui_port = ui_port
        self.api_port = api_port
        self.ui_url = f"http://127.0.0.1:{ui_port}"
        self.api_url = f"http://127.0.0.1:{api_port}"
        self.processes: list[subprocess.Popen[bytes]] = []
        self.logs: list[object] = []

    def start(self) -> None:
        for worker, port in (("api", self.api_port), ("ui", self.ui_port)):
            stdout = (log_root() / f"desktop_{worker}.stdout.log").open("wb")
            stderr = (log_root() / f"desktop_{worker}.stderr.log").open("wb")
            self.logs.extend((stdout, stderr))
            self.processes.append(
                subprocess.Popen(
                    worker_command(worker, port),
                    cwd=str(bundle_root()),
                    env=os.environ.copy(),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                )
            )
        state = {
            "ui_url": self.ui_url,
            "api_url": self.api_url,
            "pids": [process.pid for process in self.processes],
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (user_data_root() / "desktop_services.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )

    def wait_ready(self, timeout: float = 150.0) -> dict[str, object]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(process.poll() is not None for process in self.processes):
                raise RuntimeError("A VitalsSight service exited during startup; inspect the application logs")
            health = http_json(f"{self.api_url}/api/v1/assistant/health", timeout=1.5)
            if isinstance(health, dict) and http_ready(self.ui_url, timeout=1.5):
                return health
            time.sleep(0.5)
        raise TimeoutError("VitalsSight did not become ready within 150 seconds; inspect the application logs")

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
        try:
            (user_data_root() / "desktop_services.json").unlink(missing_ok=True)
        except OSError:
            pass


def smoke_test(ui_port: int, api_port: int) -> int:
    ui_port, api_port = choose_ports(ui_port, api_port)
    configure_environment(ui_port, api_port)
    services = ServicePair(ui_port, api_port)
    try:
        services.start()
        assistant = services.wait_ready()
        cases = http_json(f"{services.api_url}/api/v1/cases", timeout=5)
        case_items = cases.get("items", []) if isinstance(cases, dict) else cases
        result = {
            "status": "PASS",
            "platform": sys.platform,
            "architecture": platform.machine(),
            "ui": services.ui_url,
            "api": services.api_url,
            "assistant": assistant,
            "seeded_case_count": len(case_items) if isinstance(case_items, list) else None,
        }
        target = user_data_root() / "native_app_smoke_test.json"
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        stage_log(f"Native app smoke test passed: {target}")
        return 0
    finally:
        services.stop()


def desktop(ui_port: int, api_port: int, no_browser: bool) -> int:
    import tkinter as tk
    from tkinter import messagebox

    window = tk.Tk()
    window.title(APP_NAME)
    window.geometry("700x390")
    window.minsize(700, 390)
    window.configure(bg="#F2F7F8")

    tk.Label(
        window,
        text="VitalsSight",
        bg="#F2F7F8",
        fg="#20323D",
        font=("Helvetica Neue", 30, "bold"),
    ).pack(pady=(36, 3))
    tk.Label(
        window,
        text="Evidence-linked camera heart-rate research workflow",
        bg="#F2F7F8",
        fg="#59707C",
        font=("Helvetica Neue", 13),
    ).pack()
    status = tk.StringVar(value="Starting local evidence services...")
    tk.Label(
        window,
        textvariable=status,
        bg="#F2F7F8",
        fg="#39778A",
        font=("Helvetica Neue", 13, "bold"),
    ).pack(pady=(42, 10))
    detail = tk.StringVar(value="No Python installation or remote server is required.")
    tk.Label(
        window,
        textvariable=detail,
        wraplength=610,
        justify="center",
        bg="#F2F7F8",
        fg="#59707C",
        font=("Helvetica Neue", 10),
    ).pack()
    buttons = tk.Frame(window, bg="#F2F7F8")
    buttons.pack(pady=34)
    open_button = tk.Button(
        buttons,
        text="Open VitalsSight",
        state="disabled",
        bg="#39778A",
        fg="white",
        activebackground="#2F6676",
        activeforeground="white",
        relief="flat",
        padx=26,
        pady=10,
        font=("Helvetica Neue", 11, "bold"),
    )
    open_button.grid(row=0, column=0, padx=8)
    api_button = tk.Button(
        buttons,
        text="API documentation",
        state="disabled",
        bg="#FFFFFF",
        fg="#20323D",
        relief="solid",
        borderwidth=1,
        padx=20,
        pady=10,
        font=("Helvetica Neue", 11),
    )
    api_button.grid(row=0, column=1, padx=8)
    stop_button = tk.Button(
        buttons,
        text="Stop",
        bg="#FFFFFF",
        fg="#A05D57",
        relief="solid",
        borderwidth=1,
        padx=22,
        pady=10,
        font=("Helvetica Neue", 11),
    )
    stop_button.grid(row=0, column=2, padx=8)

    state: dict[str, object] = {"services": None}

    def stop_and_close() -> None:
        services = state.get("services")
        if isinstance(services, ServicePair):
            status.set("Stopping VitalsSight...")
            services.stop()
        window.destroy()

    stop_button.configure(command=stop_and_close)
    window.protocol("WM_DELETE_WINDOW", stop_and_close)

    def start_services() -> None:
        try:
            selected_ui, selected_api = choose_ports(ui_port, api_port)
            values = configure_environment(selected_ui, selected_api)
            services = ServicePair(selected_ui, selected_api)
            state["services"] = services
            services.start()
            health = services.wait_ready()
            provider = str(health.get("provider", "deterministic"))
            model = str(health.get("model", values["VITALSSIGHT_ASSISTANT_MODEL"]))
            model_ready = bool(health.get("model_available", False))

            def ready() -> None:
                status.set("VitalsSight is ready")
                detail.set(
                    f"Assistant: {provider} / {model}"
                    if model_ready
                    else "Deterministic evidence assistant active; optional local AI is not required."
                )
                open_button.configure(state="normal", command=lambda: webbrowser.open(services.ui_url))
                api_button.configure(
                    state="normal", command=lambda: webbrowser.open(f"{services.api_url}/docs")
                )
                if not no_browser:
                    webbrowser.open(services.ui_url)

            window.after(0, ready)
        except Exception as error:
            stage_log(f"Startup failed: {error!r}")
            message = str(error)

            def failed() -> None:
                status.set("Startup failed")
                detail.set(message)
                messagebox.showerror(APP_NAME, f"{message}\n\nLogs: {log_root()}")

            window.after(0, failed)

    threading.Thread(target=start_services, daemon=True).start()
    window.mainloop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("api", "ui"))
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ui-port", type=int, default=DEFAULT_UI_PORT)
    parser.add_argument("--api-port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if args.port <= 0:
            raise SystemExit("--port is required for worker mode")
        configure_environment(
            int(os.getenv("VITALSSIGHT_UI_PORT", DEFAULT_UI_PORT)),
            int(os.getenv("VITALSSIGHT_API_PORT", DEFAULT_API_PORT)),
        )
        return run_api_worker(args.port) if args.worker == "api" else run_ui_worker(args.port)
    if args.smoke_test:
        return smoke_test(args.ui_port, args.api_port)
    return desktop(args.ui_port, args.api_port, args.no_browser)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
