"""Zero-fit discovery runs source and feature code without reading outcome feedback."""

import json
from copy import deepcopy

import numpy as np
import pytest
from test_research_study import command

from quant_rsi.cli import main
from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research import study, study_discovery
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.discovery_engine import DiscoveryExperimentRunner
from quant_rsi.research.study_iterations import validate_limits


def arguments(root, *extra):
    return command(
        root,
        "--llm",
        "simulated",
        "--research-mode",
        "discovery",
        "--max-fits",
        "0",
        "--horizon-sessions",
        "3",
        "--no-ui",
        "--history-dir",
        str(root.parent / "history"),
        *extra,
    )


def forbid_outcomes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Preparation must not fit, evaluate, publish or load outcome feedback")

    monkeypatch.setattr(DiscoveryExperimentRunner, "__call__", forbidden)
    monkeypatch.setattr(LightGBMDemo, "predict", forbidden)
    monkeypatch.setattr(LightGBMDemo, "evaluate_terminal", forbidden)
    for name in ("observe_prices", "mature_feedback", "history_view"):
        monkeypatch.setattr(study, name, forbidden)
    monkeypatch.setattr(study_discovery, "_prior_memory", forbidden)


@pytest.mark.parametrize("case", ["ready", "partial", "stop", "semantic_gap"])
def test_native_preparation_has_actual_columns_or_explicit_gaps_and_never_fits(
    tmp_path, monkeypatch, case
):
    forbid_outcomes(monkeypatch)
    root = tmp_path / case
    original_sources = study_discovery.simulated_discovery_sources
    original_decide = study_discovery.simulated_discovery_decision
    engines, contexts = [], []
    init = DiscoveryExperimentRunner.__init__

    def capture(self, *args, **kwargs):
        init(self, *args, **kwargs)
        engines.append(self)

    def sources(*args, **kwargs):
        provider = original_sources(*args, **kwargs)
        if case == "partial":
            middle = len(provider.articles) // 2
            provider.articles = provider.articles[middle : middle + 1]
            provider.limits["search"] = 0
        return provider

    def decide(context):
        contexts.append(deepcopy(context))
        if case == "stop":
            return {"action": "stop", "reason": "No justified candidate", "candidates": []}
        result = original_decide(context)
        if case == "semantic_gap" and result["action"] == "test":
            result["candidates"][0]["evidence_ids"] = []
        return result

    monkeypatch.setattr(DiscoveryExperimentRunner, "__init__", capture)
    monkeypatch.setattr(study_discovery, "simulated_discovery_sources", sources)
    monkeypatch.setattr(study_discovery, "simulated_discovery_decision", decide)
    assert main(arguments(root)) == 0
    result = json.loads((root / "result.json").read_text())
    packet = json.loads((root / "preparation.json").read_text())
    assert result["discovery"]["mode"] == "preparation_only"
    assert result["discovery"]["counts"]["fits"] == 0
    assert result["discovery"]["groups"] == {}
    assert result["iterations"]["attempts"] == []
    assert result["iterations"]["selected_round"] is None
    assert "terminal_evaluation" not in result
    assert "latest_forecast" not in result["summary"]
    assert not list((root / "checkpoints").glob("fit_*.json"))
    assert not (root / "terminal_evaluation.json").exists()
    assert all(c["prior_memory"] == [] and c["preparation_only"] for c in contexts)
    assert all(c["forecast_target"]["horizon_sessions"] == 3 for c in contexts)
    assert packet["forecast_target"]["target"] == "3_session_close_log_return"
    assert "preparation only" in result["summary"]["text"]
    assert "actual LightGBM fits" not in result["notice"]
    assert "Experiments are incomparable" not in result["summary"]["text"]
    assert "complete required feature coverage" in (root / "report.html").read_text()
    records = packet["records"]
    if case == "stop":
        assert records == []
        return
    assert len(records) == 1
    checked = records[0]["result"]
    feedback = contexts[-1]["memory"][-1]
    assert feedback["kind"] == "preparation"
    assert feedback["result"] == checked
    if case == "semantic_gap":
        assert checked["status"] == "rejected"
        assert checked["reason"] == "candidate_requires_admitted_evidence"
        assert "preparation" not in checked
        return
    prepared = checked["preparation"]
    assert checked["status"] == ("ready" if case == "ready" else "needs_data")
    coverage = prepared["coverage"]
    assert coverage["computed_rows"] > 0
    if case == "partial":
        assert coverage["computed_rows"] < coverage["required_rows"]
        assert checked["reason"] == "incomplete_frozen_population"
    else:
        assert coverage["computed_rows"] == coverage["required_rows"]
    model = engines[0].model
    indices = np.concatenate((model.train, model.origins))
    acquisition = next(m["result"] for m in result["discovery"]["memory"] if m["kind"] == "acquire")
    _, values, _ = engines[0].prepare(records[0]["candidate"], [acquisition])
    expected = [float(values[i]) if np.isfinite(values[i]) else None for i in indices]
    assert [r["value"] for r in prepared["rows"]] == expected
    assert [r["session"] for r in prepared["rows"]] == [model.data.dates[i] for i in indices]
    assert not set(r["session"] for r in prepared["rows"]).intersection(
        model.data.dates[i] for i in model.terminal_origins
    )
    assert prepared["input_bundle_refs"] == [acquisition["identity"]]
    assert prepared["source_ids"] == sorted({o["source_id"] for o in acquisition["observations"]})
    assert prepared["column_ref"] == digest(canonical(prepared["rows"]))
    assert all(
        set(r) == {"session", "prediction_time", "partition", "value"} for r in prepared["rows"]
    )


def test_preparation_resume_twice_keeps_columns_and_does_not_repeat_source_calls(
    tmp_path, monkeypatch
):
    forbid_outcomes(monkeypatch)
    root = tmp_path / "resume"
    source_calls = []
    factory = study_discovery.simulated_discovery_sources
    decide = study_discovery.simulated_discovery_decision
    failures_remaining = 2

    def sources(*args, **kwargs):
        source = factory(*args, **kwargs)
        structured = source.llm.structured

        def counted(*args, **kwargs):
            source_calls.append(1)
            return structured(*args, **kwargs)

        source.llm.structured = counted
        return source

    def decision(context):
        nonlocal failures_remaining
        if context["iteration"] == 3 and failures_remaining:
            failures_remaining -= 1
            raise ValueError("Injected outage after completed feature preparation")
        return decide(context)

    monkeypatch.setattr(study_discovery, "simulated_discovery_sources", sources)
    monkeypatch.setattr(study_discovery, "simulated_discovery_decision", decision)
    assert main(arguments(root)) == 1
    assert source_calls
    calls = len(source_calls)
    snapshots = {p: p.read_bytes() for p in (root / "discovery" / "memory").glob("*.json")}
    assert any("preparation" in p.name for p in snapshots)
    assert main(["research-resume", str(root)]) == 1
    assert len(source_calls) == calls
    assert all(p.read_bytes() == data for p, data in snapshots.items())
    assert main(["research-resume", str(root)]) == 0
    first = (root / "preparation.json").read_bytes()
    # Completed runs remain immutable; another resume must not perform work.
    assert main(["research-resume", str(root)]) == 1
    assert (root / "preparation.json").read_bytes() == first
    assert len(source_calls) == calls
    assert all(p.read_bytes() == data for p, data in snapshots.items())
    result = json.loads((root / "result.json").read_text())
    assert result["iterations"]["completed_fits"] == 0


@pytest.mark.parametrize("mode", ["legacy", "incremental"])
def test_zero_fit_is_not_silently_enabled_for_other_modes(tmp_path, mode):
    root = tmp_path / mode
    assert main(arguments(root, "--research-mode", mode)) == 1
    assert not root.exists()
    with pytest.raises(ValueError, match="fit_budget"):
        validate_limits(3, 0, None)
