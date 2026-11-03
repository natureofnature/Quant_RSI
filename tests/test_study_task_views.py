"""Compare retained evidence, never rank different tasks by superficially similar metrics."""

import hashlib
import json
import shutil
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_study_onboarding import run_workbench_script
from test_study_queue import finish_queue

from quant_rsi.research.study_onboarding import starter_studies
from quant_rsi.research.study_workbench import Workbench, make_server


@pytest.fixture(scope="module")
def original_study(tmp_path_factory):
    root = tmp_path_factory.mktemp("retained-comparison")
    w = Workbench(root)
    starter = starter_studies()[0]
    batch = w.batch(starter["symbols"], starter["options"])
    finish_queue(w)
    task = w.task(batch["tasks"][-1]["id"])
    assert w.item(task)["status"] == "completed"
    return task


@pytest.fixture
def pair(tmp_path, original_study):
    w = Workbench(tmp_path)
    tasks = []
    for char in ("a", "b"):
        task = w.tasks / (char * 24)
        shutil.copytree(original_study, task)
        tasks.append(task)
    return w, tasks


def edit(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def selected(result):
    return result["iterations"]["attempts"][result["iterations"]["selected_round"] - 1][
        "prediction"
    ]


@pytest.mark.parametrize(
    "missing_risk", [None, float("nan"), float("inf"), "NaN", "Infinity", "unavailable", True]
)
def test_identical_retained_evidence_and_undefined_risk(pair, missing_risk):
    w, (left, right) = pair
    edit(
        right / "study/result.json",
        lambda r: selected(r)["strategy"]["strategy"]["metrics"].update(sharpe_ratio=missing_risk),
    )
    result = w.compare(left.name, right.name)
    assert result["comparison"]["development"]["comparable"]
    assert result["comparison"]["development"]["differences"]["mae_price"] == 0
    assert "sharpe_ratio" not in result["comparison"]["development"]["differences"]
    assert result["studies"][1]["segments"]["development"]["metrics"]["sharpe_ratio"] is None
    json.dumps(result, allow_nan=False)
    assert result["comparison"]["terminal"]["comparable"]
    assert result["studies"][0]["input_coverage"]["retained_price_sessions"] > 0
    original = json.loads((left / "study/result.json").read_bytes())
    for segment in ("development", "terminal"):
        strategy = (
            selected(original)["strategy"]
            if segment == "development"
            else original["terminal_evaluation"]["strategy"]
        )
        metrics = result["studies"][0]["segments"][segment]["metrics"]
        for name in ("net_return", "max_drawdown"):
            assert isinstance(strategy["strategy"]["metrics"][name], str)
            assert metrics[name] == float(strategy["strategy"]["metrics"][name])
            assert result["comparison"][segment]["differences"][name] == 0
        assert metrics["buy_and_hold_return"] == float(
            strategy["buy_and_hold"]["metrics"]["net_return"]
        )


@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda r: r.update(target="AAPL"), "symbol differs"),
        (lambda r: selected(r).update(horizon_sessions=20), "horizon sessions differs"),
        (lambda r: r["configuration"].update(prices="daily"), "data differs"),
        (
            lambda r: r["configuration"].update(information_mode="retained"),
            "information mode differs",
        ),
        (
            lambda r: selected(r)["development"]["rows"][0].update(observed_price=999),
            "Evaluation observations differs",
        ),
        (
            lambda r: selected(r)["strategy"]["account_rules"]["buy_costs"].update(
                commission_bps="10"
            ),
            "Execution policy and costs differs",
        ),
        (
            lambda r: selected(r)["strategy"].update(threshold=0.01),
            "Execution policy and costs differs",
        ),
        (lambda r: selected(r)["development"].update(currency="EUR"), "Metric units differs"),
    ],
)
def test_incompatible_results_keep_values_but_never_compute_differences(pair, change, reason):
    w, (left, right) = pair
    edit(right / "study/result.json", change)
    comparison = w.compare(left.name, right.name)["comparison"]["development"]
    assert not comparison["comparable"]
    assert reason in comparison["reasons"]
    assert comparison["differences"] == {}


def test_open_price_changes_detected_even_when_forecast_labels_are_unchanged(pair):
    w, (left, right) = pair
    result = json.loads((right / "study/result.json").read_text())
    first = selected(result)["strategy"]["period"]["start"]

    def alter(data):
        data["opens"][data["dates"].index(first) + 1] *= 1.01

    edit(right / "study/inputs.json", alter)
    comparison = w.compare(left.name, right.name)["comparison"]["development"]
    assert comparison["reasons"] == ["Actual account price path differs"]
    assert comparison["differences"] == {}


def test_missing_prices_or_terminal_never_look_like_zero_difference(pair):
    w, (left, right) = pair
    (right / "study/inputs.json").unlink()
    edit(
        right / "study/result.json", lambda r: r.update(terminal_evaluation={"status": "disabled"})
    )
    result = w.compare(left.name, right.name)
    assert not result["comparison"]["development"]["comparable"]
    assert not result["comparison"]["terminal"]["comparable"]
    assert result["studies"][1]["segments"]["terminal"]["status"] == "unavailable"


def test_reuse_settings_reads_frozen_config_and_never_copies_paths_or_mutates_task(pair):
    w, (left, _) = pair
    edit(
        left / "study/config.json",
        lambda c: c.update(initial_cash="25000", llm_model=None, secret="never-return"),
    )
    before = {
        p.relative_to(left): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in left.rglob("*")
        if p.is_file()
    }
    settings = w.study_settings(left.name)
    assert settings["options"]["initial_cash"] == "25000"
    assert settings["options"]["llm_model"] == ""
    assert settings["source_task_id"] == left.name
    assert "never-return" not in json.dumps(settings)
    assert not {"output", "history_dir", "arguments", "secret"} & settings["options"].keys()
    assert before == {
        p.relative_to(left): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in left.rglob("*")
        if p.is_file()
    }
    batch = w.batch(settings["symbols"], settings["options"])
    fresh = w.task(batch["tasks"][-1]["id"])
    assert fresh != left and w.item(fresh)["status"] == "queued"
    finish_queue(w)
    assert w.item(fresh)["status"] == "completed"
    assert json.loads((fresh / "study/config.json").read_text())["initial_cash"] == "25000"
    assert before == {
        p.relative_to(left): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in left.rglob("*")
        if p.is_file()
    }


def test_settings_and_comparison_http_preserve_queue_and_validate_requests(pair):
    w, (left, right) = pair
    server = make_server(w, "retained-test")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    base = origin + "/retained-test/"
    try:
        with urlopen(base + f"tasks/{left.name}/settings") as response:
            assert json.load(response)["source_task_id"] == left.name

        def compare(body, caller=origin):
            request = Request(
                base + "comparisons",
                data=json.dumps(body).encode(),
                headers={"Origin": caller, "Content-Type": "application/json"},
            )
            return urlopen(request)

        with compare({"left": left.name, "right": right.name}) as response:
            assert response.status == 200
            assert json.load(response)["comparison"]["development"]["comparable"]
        for body in (
            {"left": [], "right": right.name},
            {"left": left.name, "right": left.name},
            {"left": "../outside", "right": right.name},
            {"left": left.name},
        ):
            with pytest.raises(HTTPError) as error:
                compare(body)
            assert error.value.code == 400
        with pytest.raises(HTTPError):
            compare({"left": left.name, "right": right.name}, "https://other.example")
        assert len(w.listing()) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_full_script_uses_task_settings_and_shows_retained_comparison(pair):
    w, (left, right) = pair
    comparison = w.compare(left.name, right.name)
    edit(right / "study/result.json", lambda r: selected(r)["strategy"].update(threshold=0.5))
    incompatible = w.compare(left.name, right.name)
    run_workbench_script(
        r"""
(async()=>{
  await new Promise(setImmediate);
  const texts=element=>[element.textContent||'',...element.children.flatMap(texts)].join(' ');
  await useTaskSettings({id:fixture.taskSettings.source_task_id,symbol:'KO'});
  assert.equal(nodes['source-note'].hidden,false);
  assert.match(nodes['source-note'].textContent,/does not replay or resume/);
  assert.equal(fields.enabled.checked,false);
  assert.equal(requests.filter(r=>r.method==='POST').length,0);
  nodes['compare-left'].value=fixture.comparison.studies[0].id;
  nodes['compare-right'].value=fixture.comparison.studies[1].id;
  await nodes['compare-studies'].onclick();
  assert.match(texts(nodes['pair-development']),/Price MAE/);
  assert.match(texts(nodes['pair-development']),/0.0000/);
  assert.match(nodes['pair-development-status'].textContent,/Same evaluation observations/);
  assert.equal(requests.filter(r=>r.method==='POST'&&r.path==='batches').length,0);
  renderRetainedComparison(fixture.incompatible);
  assert.match(nodes['pair-development-status'].textContent,/Execution policy and costs differs/);
  const rows=nodes['pair-development'].children[0].children[1].children;
  assert(rows.every(row=>row.children[3].textContent==='—'));
  const failed={id:fixture.taskSettings.source_task_id,symbol:'KO',status:'failed',resumable:true,
    error:'Study code, dependencies or configuration changed; start a new study'};
  renderTasks([failed]);
  const buttons=nodes.tasks.children[0].children[1].children.filter(n=>n.tag==='button');
  assert(buttons.some(n=>n.textContent==='Use settings'));
  assert(!buttons.some(n=>n.textContent==='Continue study'));
})().catch(error=>{console.error(error);process.exitCode=1;});
""",
        {
            "taskSettings": w.study_settings(left.name),
            "comparison": comparison,
            "incompatible": incompatible,
        },
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["configuration"].pop("prices"),
        lambda r: selected(r)["development"].pop("metric_units"),
        lambda r: selected(r)["strategy"].pop("policy"),
        lambda r: selected(r)["strategy"].pop("buy_and_hold"),
    ],
)
def test_incomplete_old_result_evidence_is_not_treated_as_a_match(pair, change):
    w, (left, right) = pair
    edit(right / "study/result.json", change)
    result = w.compare(left.name, right.name)
    assert not result["comparison"]["development"]["comparable"]
    assert not result["comparison"]["development"]["differences"]


def test_comparison_rejects_queued_or_non_research_tasks(pair):
    w, (left, right) = pair
    settings = w.study_settings(left.name)
    task = w._enqueue("KO", settings["options"])
    with pytest.raises(ValueError, match="completed research"):
        w.compare(left.name, task.name)
    edit(right / "task.json", lambda m: m.update(action="check_inputs"))
    with pytest.raises(ValueError, match="completed research"):
        w.compare(left.name, right.name)
    with pytest.raises(ValueError, match="Only research"):
        w.study_settings(right.name)


def test_two_real_estimators_compare_on_the_same_retained_task(pair):
    w, (left, _) = pair
    settings = w.study_settings(left.name)
    batch = w.batch(settings["symbols"], settings["options"] | {"model": "lightgbm"})
    finish_queue(w)
    right = w.task(batch["tasks"][-1]["id"])
    assert w.item(right)["status"] == "completed"
    result = w.compare(left.name, right.name)
    assert [s["model"] for s in result["studies"]] == ["ridge", "lightgbm"]
    for segment in ("development", "terminal"):
        comparison = result["comparison"][segment]
        assert comparison["comparable"], comparison["reasons"]
        a, b = (s["segments"][segment]["metrics"]["mae_price"] for s in result["studies"])
        assert comparison["differences"]["mae_price"] == pytest.approx(b - a)
    assert result["comparison"]["development"]["differences"]["mae_price"] != 0
