"""Completed incomparable research remains history without inventing a publication."""

import json
from copy import deepcopy

import lightgbm as lgb
from test_research_study import command

from quant_rsi.cli import main
from quant_rsi.research import study_discovery
from quant_rsi.research.study_history import (
    history_view,
    ledger_studies,
    mature_feedback,
    observe_prices,
    register_study,
)


def test_incomparable_completed_study_is_registered_and_reused(tmp_path, monkeypatch):
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
        "--rounds",
        "5",
        "--max-fits",
        "4",
    )
    original_decide, original_train = study_discovery.simulated_discovery_decision, lgb.train
    calls, fits = [], []

    def two_groups(context):
        calls.append(context)
        number = len(calls)
        if number == 5:
            return {"action": "stop", "reason": "Two separate domains completed."}
        acquire = number in (1, 3)
        action = original_decide(context | {"input_bundles": []} if acquire else context)
        if number >= 3:
            action["candidates"][0]["inputs"][0]["factor_id"] = "acquired.second_demand"
            for request in action["data_requests"]:
                request["input_id"] = "acquired.second_demand"
        return action

    def train(*args, **kwargs):
        fits.append(1)
        return original_train(*args, **kwargs)

    monkeypatch.setattr(study_discovery, "simulated_discovery_decision", two_groups)
    monkeypatch.setattr(lgb, "train", train)
    assert main(command(first, *options)) == 0
    result = json.loads((first / "result.json").read_text())
    assert len(fits) == result["iterations"]["completed_fits"] == 4
    assert result["discovery"]["counts"]["acquisitions"] == 2
    assert len(result["discovery"]["groups"]) == 2
    assert result["iterations"]["selected_round"] is None
    view = history_view(history, "KO")
    assert len(view["entries"]) == 1, "Completed research was lost without a global winner"
    entry = view["entries"][0]
    assert entry["selected_round"] is None
    assert entry["factor_ids"] == []
    assert entry["forecast"] is None
    assert entry["status"] == "unavailable"
    assert entry["reason"] == "no_globally_selected_forecast"

    # The immutable registration is idempotent; absence of publication prevents settlement.
    retained = deepcopy(result)
    retained["study"].pop("history")
    registration = register_study(history, first, retained)
    repeated = register_study(history, first, retained)
    assert registration["result_ref"] == repeated["result_ref"]
    assert registration["publication"] is registration["source_ref"] is None
    observed = observe_prices(history, "KO", {"info": {"received_at": "2020-01-01T00:00:00Z"}})
    assert observed["entries"][0]["assessment_count"] == 0
    assert mature_feedback(history, "KO")["entries"] == []
    attempts = ledger_studies(history, "KO")[0]["attempts"]
    assert len(attempts) == 4
    assert not any(row["selected"] for row in attempts)

    prior = {
        row["ref"]: row["spec"]["comparison_key"]
        for row in result["discovery"]["memory"]
        if row["kind"] == "experiment"
    }
    captured = []

    def capture(context):
        captured.append(context)
        return original_decide(context)

    monkeypatch.setattr(study_discovery, "simulated_discovery_decision", capture)
    assert main(command(second, *options)) == 0
    reused = [row for row in captured[0]["memory"] if row.get("projection_of") in prior]
    assert {row["experiment_context"]["comparison_key"] for row in reused} == set(prior.values())
    assert all("terminal_evaluation" not in json.dumps(row) for row in reused)
