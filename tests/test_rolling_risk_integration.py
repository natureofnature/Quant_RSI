"""Offline measured-risk inputs exercise real fits, cutoff isolation and retained resume."""

import argparse
import json
import shutil
from copy import deepcopy

import numpy as np
import pytest
import test_monthly_rolling as monthly
import test_rolling_news_integration as news_tests
from test_risk_data import write_bundle

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.market.risk_data import align_risk_features, load_risk_history
from quant_rsi.research import rolling
from quant_rsi.research.rolling_windows import monthly_plan

retained = monthly.retained


def risk_snapshot(directory, dates):
    write_bundle(
        directory,
        [(day, 15 + (i % 13) / 5) for i, day in enumerate(dates)],
        [(day, 3 + (i % 17) / 20) for i, day in enumerate(dates)],
    )
    return load_risk_history(directory)


def risk_decision(*, news=True):
    if not news:
        candidate = monthly.candidate()
        candidate["inputs"][0]["factor_id"] = "risk.vix_close"
        return {
            "action": "replace",
            "basis": "price_only",
            "reason": "Measured volatility",
            "candidates": [candidate],
            "news_inputs": [],
        }
    decision = news_tests.proposal()
    candidate = decision["candidates"][0]
    candidate["inputs"].append({"name": "vix", "factor_id": "risk.vix_close"})
    candidate["expression"]["nodes"] = [
        {"name": "news", "op": "leaf", "leaf": "pressure"},
        {"name": "risk", "op": "leaf", "leaf": "vix"},
        {"name": "result", "op": "multiply", "inputs": ["news", "risk"]},
    ]
    return decision


def test_news_risk_interaction_reaches_fit_without_changing_baseline(retained, tmp_path):
    _, data, info, calendar = retained
    bundle = risk_snapshot(tmp_path / "risk", data.dates)
    plan = monthly_plan(data, calendar, "2026-07", "2026-07")

    def run(name, history):
        root = tmp_path / name
        root.mkdir()
        cache, llm = monthly.MemoryCache(), monthly.DecisionLLM([risk_decision()])
        folds = rolling.run_months(
            data,
            info,
            {},
            plan,
            cache,
            root,
            model="ridge",
            llm=llm,
            news_history=news_tests.history(),
            news_required=True,
            risk_history=history,
        )
        return folds[0], cache, llm

    fold, cache, llm = run("first", bundle)
    assert fold["candidate_status"] == "fitted"
    fixed, adaptive = cache.calls["2026-07_fixed_fit"], cache.calls["2026-07_adaptive_fit"]
    np.testing.assert_array_equal(np.asarray(adaptive["x"])[:, :2], fixed["x"])
    assert adaptive["factor_ids"] == fixed["factor_ids"] + ["discovery.export_pressure"]
    dates = plan[0]["history_sessions"]
    columns, _, _ = align_risk_features(bundle, dates)
    expected = columns["risk.vix_close"][plan[0]["training_indices"]] * np.asarray(
        fold["news_training"]["news.export_pressure"]["training_values"]
    )
    np.testing.assert_allclose(np.asarray(adaptive["x"])[:, -1], expected)
    assert len(fold["risk_inputs"]["coverage"]) == 4
    assert "risk.treasury_10y_change_20" in canonical(llm.contexts[0]).decode()

    changed = deepcopy(bundle)
    for values in changed["series"].values():
        for row in values:
            if row["date"] > "2026-06-30" and row["value"] is not None:
                row["value"] += 1000
    second, later_cache, _ = run("future", changed)
    assert second["forecasts"] == fold["forecasts"]
    assert later_cache.calls == cache.calls
    missing = deepcopy(bundle)
    missing["series"]["vix"] = [
        row for row in missing["series"]["vix"] if not "2026-05-01" <= row["date"] <= "2026-05-20"
    ]
    unavailable, missing_cache, _ = run("missing", missing)
    assert unavailable["candidate_status"] == "rejected"
    assert "unavailable" in unavailable["candidate_error"].lower()
    assert "2026-07_adaptive_fit" not in missing_cache.calls
    assert missing_cache.calls["2026-07_fixed_fit"] == fixed
    coverage = unavailable["risk_inputs"]["coverage"][0]
    assert coverage["training_available"] < coverage["training_samples"]


def test_cli_retains_risk_snapshot_and_binds_resume_to_verified_content(
    retained, tmp_path, monkeypatch
):
    directory, data, _, _ = retained
    source = tmp_path / "risk"
    bundle = risk_snapshot(source, data.dates)
    parser = argparse.ArgumentParser()
    rolling.add_arguments(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        [
            "research-rolling",
            "SOXX",
            "--prices-dir",
            str(directory),
            "--first-month",
            "2026-07",
            "--last-month",
            "2026-08",
            "--model",
            "ridge",
            "--llm",
            "codex",
            "--risk-history",
            str(source),
            "--output",
            str(tmp_path / "study"),
        ]
    )
    llm = monthly.DecisionLLM([risk_decision(news=False), RuntimeError("retry provider")])
    monkeypatch.setattr(rolling, "LLMHypothesizer", lambda *a, **kw: llm)
    with pytest.raises(RuntimeError, match="retry provider"):
        rolling.run_rolling(args)
    config = json.loads((args.output / "config.json").read_bytes())
    assert config["input_identity"]["risk_sha256"] == digest(canonical(bundle))
    assert "gpr_history" not in config
    assert "gpr_sha256" not in config["input_identity"]
    identity = json.loads((args.output / "checkpoints/identity.json").read_bytes())
    assert "xlrd" not in identity["dependencies"]
    snapshot = args.output / "risk_history"
    assert load_risk_history(snapshot) == bundle
    assert "Measured market-risk inputs" in (args.output / "report.html").read_text()
    original = (args.output / "2026-07/prediction.json").read_bytes()
    shutil.rmtree(source)
    args.resume = True

    receipts = snapshot / "receipts.json"
    original_receipts = receipts.read_bytes()
    changed = json.loads(original_receipts)
    changed[0]["received_at"] = "2026-11-05T20:00:00+00:00"
    receipts.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="configuration changed"):
        rolling.run_rolling(args)
    receipts.write_bytes(original_receipts)
    result = rolling.run_rolling(args)
    assert result["status"] == "completed"
    assert (args.output / "2026-07/prediction.json").read_bytes() == original
    assert [context["forecast_target"] for context in llm.contexts] == [
        "2026-07-31",
        "2026-08-31",
        "2026-08-31",
    ]
