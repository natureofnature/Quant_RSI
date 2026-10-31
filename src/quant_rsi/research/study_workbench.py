"""Single-user loopback launcher. Studies remain CLI processes and ordinary directories."""

import json
import re
import secrets
import subprocess
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
from pathlib import Path

from quant_rsi.core.artifacts import atomic_write, canonical
from quant_rsi.research.study_resume import run_lock
from quant_rsi.research.study_ui import _read_json, page, read_state


class Workbench:
    def __init__(self, workspace):
        self.root = Path(workspace).resolve()
        if self.root.is_relative_to(Path(__file__).resolve().parents[3]):
            raise ValueError("Keep workbench tasks outside the source checkout")
        self.tasks = self.root / "tasks"
        self.tasks.mkdir(parents=True, exist_ok=True)
        self.children = {}

    def task(self, task_id):
        if not re.fullmatch(r"[0-9a-f]{24}", task_id):
            raise ValueError("Invalid task ID")
        task = self.tasks / task_id
        if not (task / "task.json").is_file():
            raise ValueError("Task not found in this workspace")
        return task

    def item(self, task):
        item = _read_json(task / "task.json", {})
        study = task / "study"
        runtime = _read_json(study / "runtime.json", {})
        process = self.children.get(task.name)
        running = process is not None and process.poll() is None
        if study.exists():
            try:
                with run_lock(study):
                    pass
            except ValueError:
                running = True
        status = "running" if running else runtime.get("status", "interrupted")
        if status in {"running", "starting"} and not running:
            status = "interrupted"
        failure = _read_json(study / "failure.json", {})
        error = failure.get("error")
        if not running and status != "completed" and (task / "worker.log").exists():
            with (task / "worker.log").open("rb") as stream:
                stream.seek(max(0, (task / "worker.log").stat().st_size - 2048))
                lines = stream.read().decode("utf-8", errors="replace").splitlines()
            # CLI's final diagnostic explains initialization/resume refusal too.
            if lines:
                error = lines[-1][:1000]
        return item | {
            "id": task.name,
            "status": status,
            "stage": runtime.get("current_stage"),
            "error": error,
            "resumable": not running
            and status in {"failed", "interrupted"}
            and (study / "checkpoints/identity.json").exists(),
            "reused_operations": runtime.get("reused_operations", 0),
        }

    def listing(self):
        return sorted(
            (self.item(p.parent) for p in self.tasks.glob("*/task.json")),
            key=lambda item: item["created_at"],
            reverse=True,
        )

    def _launch(self, task, arguments):
        with (task / "worker.log").open("ab") as log:
            self.children[task.name] = subprocess.Popen(
                [sys.executable, "-m", "quant_rsi", *arguments],
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )

    def start(self, options):
        if any(item["status"] == "running" for item in self.listing()):
            raise ValueError("A study is already running; wait for it before starting another")
        if not isinstance(options, dict) or set(options) - {
            "symbol",
            "llm",
            "model",
            "data",
            "llm_model",
            "news_research",
        }:
            raise ValueError("Unexpected study options")
        symbol = str(options.get("symbol", "")).strip().upper()
        llm, model, data = (
            options.get(k, d)
            for k, d in (("llm", "codex"), ("model", "lightgbm"), ("data", "daily"))
        )
        if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", symbol):
            raise ValueError("Enter a stock symbol, for example KO or AAPL")
        if (
            not all(isinstance(value, str) for value in (llm, model, data))
            or llm not in {"codex", "api", "simulated"}
            or model not in {"lightgbm", "ridge"}
            or data not in {"daily", "synthetic"}
        ):
            raise ValueError("Unsupported provider, model or data mode")
        llm_model = options.get("llm_model", "")
        if (
            not isinstance(llm_model, str)
            or len(llm_model) > 120
            or (llm == "api" and not llm_model.strip())
        ):
            raise ValueError(
                "API mode requires a model name (credentials come from the server environment)"
            )
        if type(options.get("news_research", True)) is not bool:
            raise ValueError("news_research must be boolean")
        task = self.tasks / secrets.token_hex(12)
        task.mkdir()
        atomic_write(
            task / "task.json",
            canonical(
                {
                    "symbol": symbol,
                    "llm": llm,
                    "model": model,
                    "data": data,
                    "created_at": time.time(),
                }
            ),
        )
        arguments = [
            "research-study",
            symbol,
            "--output",
            str(task / "study"),
            "--llm",
            llm,
            "--model",
            model,
            "--prices",
            data,
            "--no-ui",
            "--factor-root",
            str(Path(__file__).resolve().parents[3] / "instruments"),
            "--history-dir",
            str(self.root / "history" / data),
        ]
        if data == "synthetic":
            arguments += ["--news", "simulated", "--related", "simulated"]
        if llm_model:
            arguments += ["--llm-model=" + llm_model]
        if options.get("news_research", True):
            arguments += ["--news-research"]
        self._launch(task, arguments)
        return self.item(task)

    def retry(self, task_id):
        task = self.task(task_id)
        if not self.item(task)["resumable"]:
            raise ValueError("Only an interrupted or failed resumable study can continue")
        if any(item["status"] == "running" for item in self.listing()):
            raise ValueError("A study is already running")
        self._launch(task, ["research-resume", str(task / "study")])
        return self.item(task)


def make_server(workbench, token, port=0):
    prefix = f"/{token}/"

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body, content_type="application/json"):
            raw = (json.dumps(body) if content_type == "application/json" else body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def route(self):
            if self.headers.get(
                "Host"
            ) != f"127.0.0.1:{self.server.server_port}" or not self.path.startswith(prefix):
                raise ValueError("Use the local workbench URL printed by the command")
            return self.path[len(prefix) :]

        def do_GET(self):
            try:
                route = self.route()
                if route == "":
                    self.respond(
                        200,
                        files("quant_rsi.research").joinpath("study_workbench.html").read_text(),
                        "text/html",
                    )
                elif route == "tasks":
                    self.respond(200, {"tasks": workbench.listing()})
                elif re.fullmatch(r"tasks/[0-9a-f]{24}/(state)?", route):
                    task = workbench.task(route.split("/")[1])
                    state = read_state(task / "study")
                    item = workbench.item(task)
                    if item["status"] == "interrupted":
                        state["runtime"]["status"] = "failed"
                        state["failure"] = {"error": "Worker stopped. Continue from the workbench."}
                    if route.endswith("/state"):
                        self.respond(200, state)
                    else:
                        self.respond(200, page(state, live=True), "text/html")
                else:
                    self.respond(404, {"error": "Not found"})
            except (ValueError, OSError) as exc:
                self.respond(400, {"error": str(exc)})

        def do_POST(self):
            try:
                route = self.route()
                origin = f"http://127.0.0.1:{self.server.server_port}"
                if (
                    self.headers.get("Origin") != origin
                    or self.headers.get("Content-Type") != "application/json"
                ):
                    raise ValueError("Start studies from this local workbench")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size")
                options = json.loads(self.rfile.read(length))
                if route == "tasks":
                    result = workbench.start(options)
                elif re.fullmatch(r"tasks/[0-9a-f]{24}/retry", route) and options == {}:
                    result = workbench.retry(route.split("/")[1])
                else:
                    raise ValueError("Unknown action")
                self.respond(202, result)
            except (ValueError, OSError) as exc:
                self.respond(400, {"error": str(exc)})

        def log_message(self, *args):
            pass

    return HTTPServer(("127.0.0.1", port), Handler)


def serve(workspace, *, port=0, open_browser=True):
    if not 0 <= port <= 65535:
        raise ValueError("Port must be between 0 and 65535")
    token = secrets.token_urlsafe(24)
    workbench = Workbench(workspace)
    with run_lock(workbench.root), make_server(workbench, token, port) as server:
        url = f"http://127.0.0.1:{server.server_port}/{token}/"
        print(f"Research workbench: {url}", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Workbench stopped. Research workers continue; reopen the same workspace.")
