"""Read-only local study viewer. Existing runtime/trace artifacts own all progress."""

import argparse
import json
import re
import secrets
import select
import subprocess
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path

from quant_rsi.core.artifacts import atomic_write
from quant_rsi.research.study_ui_views import visual_output


def _read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _discovery_output(stage, output, previous, symbol):
    """Project retained discovery stages through existing table/card renderers."""
    if stage.endswith("_factor_library"):
        return visual_output("factor_library", output, previous, symbol)
    if re.fullmatch(r"discovery_fit_\d+", stage):
        prediction = output.get("prediction", {})
        metrics = prediction.get("development", {}).get("metrics", {})
        return {
            "kind": "facts",
            "facts": {
                "fit": stage.removeprefix("discovery_fit_"),
                "price_mae": metrics.get("mae_price"),
                "development_observations": metrics.get("sample_count"),
                "input_factors": (output.get("actual_features") or {}).get("input_factor_ids", []),
                "input_bundle_refs": (output.get("actual_features") or {}).get(
                    "input_bundle_refs", []
                ),
            },
            "note": "Compare only with the linked baseline on the same retained sample. "
            "See the full report for comparison groups and strategy diagnostics.",
        }
    if stage.startswith("discovery_decision_"):
        return {
            "kind": "facts",
            "facts": {
                "action": output.get("action"),
                "reason": output.get("reason"),
                "candidates": [c.get("name") for c in output.get("candidates", [])],
                "requested_inputs": [d.get("input_id") for d in output.get("data_requests", [])],
                "query": output.get("query"),
            },
            "note": "Decision output; a proposal is not yet an executed or validated factor.",
        }
    return {
        "kind": "facts",
        "facts": {
            key: output.get(key)
            for key in (
                "status",
                "request",
                "distinct_observations",
                "distinct_values",
                "covered_rows",
                "gaps",
                "counts",
                "identity",
            )
            if key in output
        },
        "note": "Source evidence, availability and input lineage are retained in the full report.",
    }


def _discovery_state(root, config, runtime, result):
    if config.get("research_mode") != "discovery":
        return None
    stages = runtime.get("stages", [])
    detail = result.get("discovery", {})
    source_counts = _read_json(root / "discovery" / "source_usage.json", {}).get("usage", {}).get(
        "counts", {}
    ) or detail.get("source_counts", {})
    decisions = [s for s in stages if s["stage"].startswith("discovery_decision_")]
    fits = [s for s in stages if re.fullmatch(r"discovery_fit_\d+", s["stage"])]
    simulated = config.get("llm") == "simulated"
    counts = runtime.setdefault("counts", {})
    counts["model_stages_completed"] = sum(s["status"] == "completed" for s in fits)
    counts["search_requests_started"] = source_counts.get("search", 0)
    counts["llm_requests_started"] = (
        0
        if simulated
        else len(decisions) + source_counts.get("extract", 0) + source_counts.get("analysis", 0)
    )
    usage = runtime.get("token_usage", {})
    if usage.get("reported_responses") and counts["llm_requests_started"]:
        usage["status"] = (
            "complete"
            if usage["reported_responses"] == counts["llm_requests_started"]
            and usage.get("cached_input_tokens") is not None
            else "partial"
        )
    selected = detail.get("selected")
    baseline_metric = selected_metric = None
    selected_round = selected.get("result", {}).get("round") if selected else None
    for group in detail.get("groups", {}).values():
        incumbent = group.get("incumbent", {}).get("result", {})
        if selected_round is not None and incumbent.get("round") == selected_round:
            baseline_metric = group.get("baseline", {}).get("result", {}).get("metric")
            selected_metric = incumbent.get("metric")
    return {
        "decisions": len(decisions),
        "completed_decisions": sum(s["status"] == "completed" for s in decisions),
        "source_counts": source_counts,
        "group_count": len(detail.get("groups", {})),
        "selection_note": detail.get("selection_note"),
        "selected_round": selected_round,
        "baseline_metric": baseline_metric,
        "selected_metric": selected_metric,
        "metric": config.get("selection_metric", "mae"),
        "counts": detail.get("counts", {}),
    }


def read_state(root):
    config = _read_json(root / "config.json", {})
    runtime = _read_json(root / "runtime.json", {"status": "starting", "stages": []})
    events = []
    previous = {}

    def present(stage, output):
        text = json.dumps(output, indent=2, ensure_ascii=False)
        return {
            "stage": stage,
            "text": text[:18000],
            "truncated": len(text) > 18000,
            "view": (
                _discovery_output(stage, output, previous, config.get("symbol", "…"))
                if stage.startswith("discovery_")
                else visual_output(stage, output, previous, config.get("symbol", "…"))
            ),
        }

    try:
        lines = (root / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # a writer may still be appending the last line
        events.append(present(event["stage"], event["output"]))
        previous[event["stage"]] = event["output"]
    for stage in runtime.get("stages", []):
        if "output" in stage:
            events.insert(0, present(stage["stage"], stage["output"]))
    result = _read_json(root / "result.json", {}) if runtime["status"] == "completed" else {}
    discovery = _discovery_state(root, config, runtime, result)
    if discovery and result:
        memory = result.get("discovery", {}).get("memory", [])
        for name, facts, note in (
            (
                "discovery_memory",
                {"retained_records": len(memory)},
                "Candidate proposals, source receipts and experiment feedback are retained.",
            ),
            (
                "discovery_summary",
                discovery,
                "Only matched experiments are compared. Predictive gain is not established.",
            ),
        ):
            event = present(name, facts)
            event["view"] = {"kind": "facts", "facts": facts, "note": note}
            events.append(event)
    report = ""
    if result:
        try:
            report = (root / "report.html").read_text(encoding="utf-8")
        except FileNotFoundError:
            pass
    try:
        updated = (root / "runtime.json").stat().st_mtime
    except FileNotFoundError:
        updated = time.time()
    history = None
    if config.get("history_dir"):
        from quant_rsi.research.study_history import history_view

        try:
            history = history_view(Path(config["history_dir"]), config.get("symbol"))
        except (OSError, ValueError) as exc:
            history = {"entries": [], "notice": f"Study history unavailable: {exc}"}
    return {
        "symbol": config.get("symbol", "…"),
        "config": {
            key: config.get(key)
            for key in (
                "llm",
                "model",
                "compare_models",
                "prices",
                "news",
                "news_research",
                "research_rounds",
                "fit_budget",
                "comparison_fit_budget",
                "publication_fit_budget",
                "walk_forward_fit_budget",
                "question",
                "research_mode",
            )
        },
        "iterations": {
            key: result.get("iterations", {}).get(key)
            for key in ("completed_fits", "selected_round", "stop_reason")
        },
        "runtime": runtime,
        "discovery": discovery,
        "updated_at": updated,
        "events": events,
        "failure": _read_json(root / "failure.json", None),
        "conclusion": result.get("study", {}).get("conclusion"),
        "selection": result.get("study", {}).get("selection"),
        "report": report,
        "history": history,
    }


def page(state, *, live=False):
    # No provider text is executable markup, including a literal closing script tag.
    bootstrap = json.dumps({"state": state, "live": live}, ensure_ascii=False).replace(
        "<", "\\u003c"
    )
    template = files("quant_rsi.research").joinpath("study_ui.html").read_text(encoding="utf-8")
    cards = files("quant_rsi.research").joinpath("study_ui_cards.js").read_text(encoding="utf-8")
    template = template.replace("__STUDY_CARDS__", cards)
    return template.replace("__STUDY_BOOTSTRAP__", bootstrap)


def save_view(root):
    atomic_write(root / "live.html", page(read_state(root)).encode("utf-8"))


def make_server(root, token):
    prefix = f"/{token}/"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path, separator, query = self.path.partition("?")
            if path == prefix[:-1]:
                # Relative state polling requires the page URL's trailing slash.
                self.server.last_request = time.monotonic()
                self.send_response(302)
                self.send_header("Location", prefix + separator + query)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if path not in {prefix, prefix + "state"}:
                self.send_error(404)
                return
            self.server.last_request = time.monotonic()
            state = read_state(root)
            is_state = path == prefix + "state"
            body = json.dumps(state, ensure_ascii=False) if is_state else page(state, live=True)
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json; charset=utf-8" if is_state else "text/html; charset=utf-8",
            )
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body.encode("utf-8"))))
            self.end_headers()
            try:
                self.wfile.write(body.encode("utf-8"))
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    # Browsers may preconnect without sending a request; keep polling responsive.
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.timeout = 1
    server.last_request = time.monotonic()
    return server


def start_view(root):
    """Start an isolated read-only viewer; never block or fail the research on UI errors."""
    process = None
    try:
        save_view(root)
        with (root / "ui-server.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", __name__, "--serve", str(root)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                start_new_session=True,
            )
        if not select.select([process.stdout], [], [], 5)[0]:
            raise OSError("viewer startup timed out")
        url = process.stdout.readline().strip()
        process.stdout.close()
        if not url.startswith("http://127.0.0.1:"):
            raise OSError("viewer failed to start; see ui-server.log")
        print(f"[quant-rsi] Live UI: {url}", file=sys.stderr, flush=True)
        try:
            if not webbrowser.open(url, new=2):
                print(
                    "[quant-rsi] Open the Live UI URL in a browser on this machine.",
                    file=sys.stderr,
                )
        except webbrowser.Error:
            print(
                "[quant-rsi] Browser unavailable; open the Live UI URL manually.", file=sys.stderr
            )
        return process
    except OSError as exc:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)
        print(
            f"[quant-rsi] UI unavailable: {exc}; research continues in the terminal.",
            file=sys.stderr,
        )
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", type=Path, required=True)
    args = parser.parse_args()
    token = secrets.token_urlsafe(24)
    with make_server(args.serve.resolve(), token) as server:
        print(f"http://127.0.0.1:{server.server_port}/{token}/", flush=True)
        # The child outlives the research command so its final page can still be read.
        # Closing the tab (or a completed viewer stopping its polling) releases it.
        while time.monotonic() - server.last_request < 120:
            server.handle_request()


if __name__ == "__main__":
    main()
