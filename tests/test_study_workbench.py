"""Local HTTP controls launch the same CLI and reuse its recoverable artifacts."""

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_research_study import command

from quant_rsi.cli import main
from quant_rsi.research import study
from quant_rsi.research.study_workbench import Workbench, make_server


def test_launcher_real_offline_process_history_and_duplicate_guard(tmp_path):
    workbench = Workbench(tmp_path)
    item = workbench.start({"symbol": "AAPL", "llm": "simulated", "data": "synthetic"})
    with pytest.raises(ValueError, match="already running"):
        workbench.start({"symbol": "KO"})
    process = workbench.children[item["id"]]
    assert process.wait(timeout=30) == 0
    restarted = Workbench(tmp_path)
    saved = restarted.listing()[0]
    assert saved["status"] == "completed" and saved["symbol"] == "AAPL"
    with pytest.raises(ValueError, match="Only an interrupted"):
        restarted.retry(item["id"])
    assert (restarted.task(item["id"]) / "study/report.html").is_file()


def test_retry_in_new_process_reuses_completed_fit(tmp_path, monkeypatch):
    workbench = Workbench(tmp_path)
    task = workbench.tasks / ("a" * 24)
    task.mkdir()
    (task / "task.json").write_text(json.dumps({"symbol": "KO", "created_at": 0}))
    original = study.simulated_hypothesis

    def fail_revision(*args, **kwargs):
        if kwargs.get("previous"):
            raise ValueError("Provider temporarily unavailable")
        return original(*args, **kwargs)

    monkeypatch.setattr(study, "simulated_hypothesis", fail_revision)
    assert main(command(task / "study", "--llm", "simulated")) == 1
    checkpoint = task / "study/checkpoints/fit_1.json"
    before = checkpoint.stat().st_mtime_ns
    item = Workbench(tmp_path).listing()[0]
    assert item["status"] == "failed" and item["resumable"]
    workbench.retry(item["id"])
    with pytest.raises(ValueError):
        workbench.retry(item["id"])
    assert workbench.children[item["id"]].wait(timeout=30) == 0
    assert checkpoint.stat().st_mtime_ns == before
    assert Workbench(tmp_path).listing()[0]["status"] == "completed"


def test_http_origin_token_inputs_and_history(tmp_path):
    workbench = Workbench(tmp_path)
    with make_server(workbench, "test-token") as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        origin = f"http://127.0.0.1:{server.server_port}"
        base = origin + "/test-token/"
        try:
            with urlopen(base) as response:
                assert b"Start a study" in response.read()
            with urlopen(base + "tasks") as response:
                assert json.load(response) == {"tasks": []}
            for path, headers, body in (
                ("tasks", {"Origin": "https://other.example"}, {"symbol": "KO"}),
                ("tasks", {}, {"symbol": "KO"}),
                ("tasks", {"Origin": origin}, {"symbol": "../../escape"}),
                ("tasks", {"Origin": origin}, {"symbol": "KO", "llm": []}),
                ("tasks", {"Origin": origin}, {"symbol": "KO", "output": "/tmp/arbitrary"}),
                ("tasks/../../retry", {"Origin": origin}, {}),
            ):
                request = Request(
                    base + path,
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json", **headers},
                )
                with pytest.raises(HTTPError) as error:
                    urlopen(request)
                assert error.value.code == 400
            with pytest.raises(HTTPError):
                urlopen(origin + "/wrong-token/tasks")
            assert workbench.listing() == []
        finally:
            server.shutdown()
            thread.join(timeout=5)
