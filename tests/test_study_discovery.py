"""Same CLI, real fits and accounts, explicit offline news/acquisition fixtures."""

import json
from copy import deepcopy

import lightgbm as lgb
import pytest
from test_research_study import command

from quant_rsi.cli import main
from quant_rsi.research import study_discovery
from quant_rsi.research.discovery_engine import DiscoveryExperimentRunner


def arguments(root, *extra):
    return command(
        root,
        "--research-mode",
        "discovery",
        "--llm",
        "simulated",
        "--no-ui",
        "--no-history",
        *extra,
    )


def test_discovery_acquired_lineage_real_fits_and_frozen_rows(tmp_path, monkeypatch):
    engines = []
    original = DiscoveryExperimentRunner.__init__

    def capture(self, *args, **kwargs):
        original(self, *args, **kwargs)
        engines.append(self)

    monkeypatch.setattr(DiscoveryExperimentRunner, "__init__", capture)
    root = tmp_path / "study"
    assert main(arguments(root)) == 0
    result = json.loads((root / "result.json").read_text())
    discovery = result["discovery"]
    assert result["target"] == "KO"
    assert discovery["counts"]["fits"] == 3
    assert discovery["counts"]["acquisitions"] == 1
    assert len(json.loads((root / "discovery_catalog.json").read_text())) == 32
    attempts = result["iterations"]["attempts"]
    assert [a["prediction"]["training_feature_columns"] for a in attempts] == [3, 4, 4]
    windows = result["study"]["research_windows"]
    assert all(
        a["prediction"]["training_rows"] == len(windows["partitions"]["train"]["origin_sessions"])
        for a in attempts
    )
    records = discovery["memory"]
    acquisition = next(m["result"] for m in records if m["kind"] == "acquire")
    features = [
        m["result"]["actual_features"]
        for m in records
        if m["kind"] == "experiment" and m["result"].get("actual_features")
    ]
    assert len(features) == 2
    assert all(acquisition["identity"] in f["input_bundle_refs"] for f in features)
    assert all("acquired.reported_demand" in f["definition"]["input_factor_ids"] for f in features)
    assert all(f["distinct_training_values"] > 1 for f in features)
    assert features[0]["definition"]["recipe_ref"] != features[1]["definition"]["recipe_ref"]
    second = [
        m for m in records if m["kind"] == "interpretation" and m["decision"]["action"] == "revise"
    ][0]
    first_feedback = next(
        m for m in records if m["kind"] == "experiment" and m["spec"]["kind"] == "candidate"
    )
    assert first_feedback["ref"] in second["consumed_memory_refs"]
    parent_ref = features[0]["definition"]["recipe_ref"]
    assert second["decision"]["candidates"][0]["parent_recipe_refs"] == [parent_ref]
    assert second["candidate_lineage"][0]["status"] == "accepted"
    assert features[1]["definition"]["parent_recipe_refs"] == [parent_ref]
    assert result["terminal_evaluation"]["status"] == "completed"
    assert "Executed feature values" in (root / "report.html").read_text()
    assert "Candidate revisions" in (root / "report.html").read_text()
    assert parent_ref in (root / "report.html").read_text()
    # The runner independently verifies source-bound alignment, not just provider claims.
    corrupt = deepcopy(acquisition)
    corrupt["columns"]["acquired.reported_demand"][40] = 999.0
    candidate = next(
        m["decision"]["candidates"][0]
        for m in records
        if m["kind"] == "interpretation" and m["decision"]["action"] == "test"
    )
    assert engines[0].validate(candidate, [corrupt])["status"] == "rejected"


def test_discovery_resume_reuses_completed_training_and_original_feedback(tmp_path, monkeypatch):
    root = tmp_path / "study"
    original, train = study_discovery.simulated_discovery_decision, lgb.train
    fits, calls = [], []
    fail = True

    def decide(context):
        calls.append(context["iteration"])
        if context["iteration"] == 3 and fail:
            raise ValueError("Injected decision outage after first candidate")
        return original(context)

    def fit(*args, **kwargs):
        fits.append(1)
        return train(*args, **kwargs)

    monkeypatch.setattr(study_discovery, "simulated_discovery_decision", decide)
    monkeypatch.setattr(lgb, "train", fit)
    assert main(arguments(root)) == 1
    assert len(fits) == 2
    first = (root / "checkpoints/fit_1.json").read_bytes()
    memory = (root / "discovery/memory/candidate_2_0.json").read_bytes()
    fail = False
    assert main(["research-resume", str(root)]) == 0
    assert len(fits) == 3
    assert (root / "checkpoints/fit_1.json").read_bytes() == first
    assert (root / "discovery/memory/candidate_2_0.json").read_bytes() == memory
    assert calls == [1, 2, 3, 3, 4]


def test_discovery_missing_candidates_finishes_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        study_discovery,
        "simulated_discovery_decision",
        lambda _: {
            "action": "stop",
            "reason": "No supported measurement",
            "candidates": [],
            "data_requests": [],
            "query": None,
        },
    )
    root = tmp_path / "baseline"
    assert main(arguments(root)) == 0
    result = json.loads((root / "result.json").read_text())
    assert len(result["iterations"]["attempts"]) == 1
    assert result["discovery"]["counts"]["proposed"] == 0


@pytest.mark.parametrize(
    "option", [["--latest-fit", "refit"], ["--compare-models"], ["--patience", "1"]]
)
def test_discovery_rejects_unimplemented_legacy_controls(tmp_path, option):
    assert main(arguments(tmp_path / "study", *option)) == 1


def test_reporting_failure_after_cached_fit_does_not_charge_fit_twice(tmp_path, monkeypatch):
    root = tmp_path / "handoff"
    original_init = DiscoveryExperimentRunner.__init__
    original_train = lgb.train
    fits = []
    fail = True

    def init(self, model, args, cutoffs, emit):
        def report(stage, output):
            if fail and stage == "discovery_fit_2" and output is not None:
                raise ValueError("reporting failed after completed candidate fit")
            return emit(stage, output)

        return original_init(self, model, args, cutoffs, report)

    def fit(*args, **kwargs):
        fits.append(1)
        return original_train(*args, **kwargs)

    monkeypatch.setattr(DiscoveryExperimentRunner, "__init__", init)
    monkeypatch.setattr(lgb, "train", fit)
    assert main(arguments(root)) == 1
    assert len(fits) == 2
    receipts = list((root / "discovery/fit_failures").glob("*.json"))
    failure = json.loads(receipts[0].read_text())
    assert failure["fit_count"] == 0
    assert failure["charge_note"] == "completed_fit_receipt"
    fail = False
    assert main(["research-resume", str(root)]) == 0
    assert len(fits) == 3
    result = json.loads((root / "result.json").read_text())
    assert result["discovery"]["counts"]["fits"] == 3
    assert result["discovery"]["counts"]["failed_fits"] == 0
