"""Independent review reproductions through the connected discovery entry point."""

import json

from test_research_study import command
from test_study_discovery import arguments

from quant_rsi.cli import main
from quant_rsi.research import study_discovery
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.discovery_factors import DiscoveryFactor


def test_invented_evidence_cannot_enter_a_discovery_fit(tmp_path, monkeypatch):
    original = study_discovery.simulated_discovery_decision

    def invented(context):
        action = original(context)
        for candidate in action.get("candidates", []):
            candidate["evidence_ids"] = ["source-never-retrieved"]
        return action

    monkeypatch.setattr(study_discovery, "simulated_discovery_decision", invented)
    root = tmp_path / "invented"
    assert main(arguments(root)) == 0
    result = json.loads((root / "result.json").read_text())
    fitted = [
        row
        for row in result["discovery"]["memory"]
        if row.get("kind") == "experiment" and row.get("spec", {}).get("kind") == "candidate"
    ]
    assert not fitted, "Candidates citing nonexistent source evidence were fitted"


def test_completed_development_feedback_is_reused_without_forecast_maturity(tmp_path, monkeypatch):
    history = tmp_path / "history"
    first, second = tmp_path / "first", tmp_path / "second"
    options = (
        "--research-mode",
        "discovery",
        "--llm",
        "simulated",
        "--no-ui",
        "--history-dir",
        str(history),
    )
    assert main(command(first, *options)) == 0
    old = json.loads((first / "result.json").read_text())
    previous = {
        row["ref"]: row["result"]["metric"]
        for row in old["discovery"]["memory"]
        if row["kind"] == "experiment"
    }
    captured = []
    original = study_discovery.simulated_discovery_decision

    def observe(context):
        if not captured:
            captured.append(context)
        return original(context)

    monkeypatch.setattr(study_discovery, "simulated_discovery_decision", observe)
    assert main(command(second, *options)) == 0
    reused = [row for row in captured[0]["memory"] if row.get("projection_of") in previous]
    assert reused, "Completed development experiments disappeared until forecast settlement"
    assert all(row["result"]["metric"] == previous[row["projection_of"]] for row in reused)
    assert all("terminal_evaluation" not in json.dumps(row) for row in reused)
    assert all(row["prior_study"]["symbol"] == "KO" for row in reused)
    assert all(row["experiment_context"]["metric"] == "mae" for row in reused)
    assert all(row["experiment_context"]["comparison_key"] for row in reused)
    expected_parents = {
        DiscoveryFactor.model_validate_json(json.dumps(row["spec"]["candidate"])).recipe_ref
        for row in old["discovery"]["memory"]
        if row["kind"] == "experiment" and row["spec"].get("candidate")
    }
    assert {row["recipe_ref"] for row in captured[0]["visible_candidates"]} == expected_parents


def test_latest_price_publication_does_not_enter_development_feedback(tmp_path, monkeypatch):
    original_predict = LightGBMDemo.predict
    original_decide = study_discovery.simulated_discovery_decision
    marker = "latest_price_after_reserved_terminal"
    contexts = []

    def predict(self, *args, **kwargs):
        result = original_predict(self, *args, **kwargs)
        # Real retained prices produce this branch; synthetic fixtures normally do not.
        result["latest_forecast"] = {
            "origin_session": "2099-01-01",
            "reference_price": 123.0,
            "outcome": "not_observed",
            "feature_values": {marker: 789.0},
        }
        return result

    def decide(context):
        contexts.append(context)
        assert marker not in json.dumps(context)
        return original_decide(context)

    monkeypatch.setattr(LightGBMDemo, "predict", predict)
    monkeypatch.setattr(study_discovery, "simulated_discovery_decision", decide)
    root = tmp_path / "publication"
    assert main(arguments(root)) == 0
    result = json.loads((root / "result.json").read_text())
    assert len(contexts) >= 3
    assert marker not in json.dumps(result["discovery"]["memory"])
    assert marker in json.dumps(result["iterations"]["attempts"])
