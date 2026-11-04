"""Synthetic news/GPR reaches real Ridge fits; no claim of forecasting quality."""

import argparse
import json
import shutil
from copy import deepcopy

import numpy as np
import pytest
import test_monthly_rolling as monthly
import test_rolling_news_integration as news_tests
from test_geopolitical_data import write_gpr_bundle

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.market.geopolitical_data import align_gpr_features, load_gpr_history
from quant_rsi.research import rolling, study_resume
from quant_rsi.research.rolling_windows import monthly_plan

retained = monthly.retained


def proposal(*, news=True):
    if not news:
        candidate = monthly.candidate()
        candidate["inputs"][0]["factor_id"] = "risk.gpr_level"
        return {
            "action": "replace",
            "basis": "price_only",
            "reason": "Test released geopolitical news index",
            "candidates": [candidate],
            "news_inputs": [],
        }
    decision = news_tests.proposal()
    candidate = decision["candidates"][0]
    candidate["inputs"].append({"name": "gpr", "factor_id": "risk.gpr_level"})
    candidate["expression"]["nodes"] = [
        {"name": "news", "op": "leaf", "leaf": "pressure"},
        {"name": "risk", "op": "leaf", "leaf": "gpr"},
        {"name": "result", "op": "multiply", "inputs": ["news", "risk"]},
    ]
    return decision


def test_news_gpr_uses_available_vintages_in_actual_fit(retained, tmp_path):
    _, data, info, calendar = retained
    write_gpr_bundle(tmp_path / "gpr")
    bundle = load_gpr_history(tmp_path / "gpr")
    plan = monthly_plan(data, calendar, "2026-07", "2026-07")

    def run(name, history):
        root = tmp_path / name
        root.mkdir()
        cache, llm = monthly.MemoryCache(), monthly.DecisionLLM([proposal()])
        fold = rolling.run_months(
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
            gpr_history=history,
        )[0]
        return fold, cache, llm

    fold, cache, llm = run("original", bundle)
    assert fold["candidate_status"] == "fitted"
    fixed = cache.calls["2026-07_fixed_fit"]
    adaptive = cache.calls["2026-07_adaptive_fit"]
    np.testing.assert_array_equal(np.asarray(adaptive["x"])[:, :2], fixed["x"])
    columns, _, _ = align_gpr_features(bundle, plan[0]["history_sessions"])
    expected = columns["risk.gpr_level"][plan[0]["training_indices"]] * np.asarray(
        fold["news_training"]["news.export_pressure"]["training_values"]
    )
    np.testing.assert_allclose(np.asarray(adaptive["x"])[:, -1], expected)
    assert "risk.gpr_level" in canonical(llm.contexts[0]).decode()
    assert fold["risk_inputs"]["coverage"][0]["prediction_available"]

    future = deepcopy(bundle)
    for vintage in future["vintages"]:
        if vintage["available_after_date"] >= "2026-06-30":
            for observation in vintage["observations"]:
                if observation["value"] is not None:
                    observation["value"] += 10000
    later, later_cache, _ = run("future", future)
    assert later["forecasts"] == fold["forecasts"]
    assert later_cache.calls == cache.calls

    missing = deepcopy(bundle)
    missing["vintages"] = [v for v in missing["vintages"] if v["release_date"] > "2026-06-30"]
    unavailable, missing_cache, _ = run("missing", missing)
    assert unavailable["candidate_status"] == "rejected"
    assert "2026-07_adaptive_fit" not in missing_cache.calls
    assert missing_cache.calls["2026-07_fixed_fit"] == fixed


def test_cli_retains_gpr_and_resumes_without_external_source(retained, tmp_path, monkeypatch):
    directory, _, _, _ = retained
    source = tmp_path / "gpr"
    write_gpr_bundle(source)
    bundle = load_gpr_history(source)
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
            "--gpr-history",
            str(source),
            "--output",
            str(tmp_path / "study"),
        ]
    )
    llm = monthly.DecisionLLM([proposal(news=False), RuntimeError("provider unavailable")])
    monkeypatch.setattr(rolling, "LLMHypothesizer", lambda *a, **kw: llm)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        rolling.run_rolling(args)
    config = json.loads((args.output / "config.json").read_bytes())
    assert config["input_identity"]["gpr_sha256"] == digest(canonical(bundle))
    identity = json.loads((args.output / "checkpoints/identity.json").read_bytes())
    assert identity["dependencies"]["xlrd"] == study_resume.version("xlrd")
    snapshot = args.output / "gpr_history"
    assert load_gpr_history(snapshot) == bundle
    report = (args.output / "report.html").read_text()
    assert "risk.gpr_level" in report
    original = (args.output / "2026-07/prediction.json").read_bytes()
    shutil.rmtree(source)
    args.resume = True
    original_version = study_resume.version
    with monkeypatch.context() as patch:
        patch.setattr(
            study_resume,
            "version",
            lambda name: "changed-reader" if name == "xlrd" else original_version(name),
        )
        with pytest.raises(ValueError, match="dependencies or configuration changed"):
            rolling.run_rolling(args)
    assert len(llm.contexts) == 2
    receipts = snapshot / "receipts.json"
    before = receipts.read_bytes()
    changed = json.loads(before)
    changed[0]["received_at"] = "2026-11-05T20:00:00+00:00"
    receipts.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="configuration changed"):
        rolling.run_rolling(args)
    receipts.write_bytes(before)
    assert rolling.run_rolling(args)["status"] == "completed"
    assert (args.output / "2026-07/prediction.json").read_bytes() == original
