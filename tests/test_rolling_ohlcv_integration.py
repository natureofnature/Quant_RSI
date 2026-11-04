"""Retained OHLCV enters real tiny Ridge fits; fixtures make no predictive-quality claim."""

import argparse
import json
from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest
import test_monthly_rolling as monthly
import test_rolling_news_integration as news_tests
from test_daily_data import chart, install

from quant_rsi.market import daily_data
from quant_rsi.research import rolling
from quant_rsi.research.demo_prices import read_daily_history
from quant_rsi.research.rolling_ohlcv import load_ohlcv_history
from quant_rsi.research.rolling_windows import monthly_plan, read_calendar


@pytest.fixture
def retained_ohlcv(tmp_path, monkeypatch):
    payload = chart("2026-11-03")
    item = payload["chart"]["result"][0]
    item["meta"]["symbol"], item["events"] = "SOXX", {}
    quote = item["indicators"]["quote"][0]
    quote["high"] = [c + 0.5 + (i % 7) / 10 for i, c in enumerate(quote["close"])]
    quote["low"] = [o - 0.2 - (i % 11) / 20 for i, o in enumerate(quote["open"])]
    quote["volume"] = [1000 + (i % 13) * 70 for i in range(len(quote["open"]))]
    install(monkeypatch, payload, now="2026-11-04T12:00:00+00:00")
    directory = tmp_path / "prices"
    daily_data.collect_daily(
        "SOXX",
        directory,
        historical_sessions=209,
        future_sessions=20,
        as_of=datetime.fromisoformat("2026-10-31T23:59:59-04:00"),
    )
    data, info = read_daily_history("SOXX", directory)
    bundle = load_ohlcv_history("SOXX", directory, data, info)
    return directory, data, info, read_calendar(directory), bundle


def proposal():
    decision = news_tests.proposal()
    common = decision["candidates"][0]
    range_candidate = deepcopy(common)
    range_candidate.update(
        name="news_range", description="News attention times normalized daily range"
    )
    range_candidate["inputs"].append({"name": "range", "factor_id": "survey.qlib.klen"})
    range_candidate["expression"]["nodes"] = [
        {"name": "news", "op": "leaf", "leaf": "pressure"},
        {"name": "range", "op": "leaf", "leaf": "range"},
        {"name": "result", "op": "multiply", "inputs": ["news", "range"]},
    ]
    volume_candidate = deepcopy(common)
    volume_candidate.update(
        name="news_relative_volume", description="News attention times relative volume"
    )
    volume_candidate["inputs"].append({"name": "volume", "factor_id": "price.volume"})
    volume_candidate["expression"]["nodes"] = [
        {"name": "news", "op": "leaf", "leaf": "pressure"},
        {"name": "volume", "op": "leaf", "leaf": "volume"},
        {"name": "mean", "op": "rolling_mean", "inputs": ["volume"], "window": 5},
        {"name": "relative", "op": "divide", "inputs": ["volume", "mean"]},
        {"name": "result", "op": "multiply", "inputs": ["news", "relative"]},
    ]
    decision["candidates"] = [range_candidate, volume_candidate]
    return decision


def test_news_ohlcv_candidates_fit_same_rows_and_future_inputs_cannot_change_them(
    retained_ohlcv, tmp_path
):
    _, data, info, calendar, bundle = retained_ohlcv
    plan = monthly_plan(data, calendar, "2026-07", "2026-07")

    def run(name, inputs, llm=None):
        root = tmp_path / name
        root.mkdir()
        cache = monthly.MemoryCache()
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
            news_required=llm is not None,
            ohlcv_history=inputs,
        )[0]
        return fold, cache

    llm = monthly.DecisionLLM([proposal()])
    fold, cache = run("enabled", bundle, llm)
    assert fold["candidate_status"] == "fitted"
    _, old_cache = run("disabled", None)
    fixed, adaptive = cache.calls["2026-07_fixed_fit"], cache.calls["2026-07_adaptive_fit"]
    assert fixed == old_cache.calls["2026-07_fixed_fit"]
    assert adaptive["y"] == fixed["y"] and adaptive["window"] == fixed["window"]
    np.testing.assert_array_equal(np.asarray(adaptive["x"])[:, :2], fixed["x"])
    assert adaptive["factor_ids"] == fixed["factor_ids"] + [
        "discovery.news_range",
        "discovery.news_relative_volume",
    ]
    train = np.asarray(plan[0]["training_indices"])
    news = np.asarray(fold["news_training"]["news.export_pressure"]["training_values"])
    columns = bundle["columns"]
    ranges = (columns["price.high"] - columns["price.low"]) / data.opens[:, 0]
    volume = columns["price.volume"]
    relative_volume = [volume[i] / volume[i - 4 : i + 1].mean() for i in train]
    np.testing.assert_allclose(np.asarray(adaptive["x"])[:, -2], news * ranges[train])
    np.testing.assert_allclose(np.asarray(adaptive["x"])[:, -1], news * relative_volume)
    assert len(fold["ohlcv_inputs"]["features"]) == 3
    assert fold["fits"]["adaptive"]["training_samples"] == len(train)

    future = deepcopy(bundle)
    mask = np.asarray(data.dates) > "2026-06-30"
    for values in future["columns"].values():
        values[mask] *= 1000
    later_llm = monthly.DecisionLLM([proposal()])
    later, later_cache = run("future", future, later_llm)
    assert later_llm.contexts == llm.contexts
    assert later_cache.calls == cache.calls
    assert later["forecasts"] == fold["forecasts"]
    assert (tmp_path / "enabled/2026-07/prediction.json").read_bytes() == (
        tmp_path / "future/2026-07/prediction.json"
    ).read_bytes()

    missing = deepcopy(bundle)
    missing["columns"]["price.volume"][train[5]] = np.nan
    partial, partial_cache = run("missing", missing, monthly.DecisionLLM([proposal()]))
    assert partial["candidate_status"] == "fitted_with_rejections"
    kept = partial_cache.calls["2026-07_adaptive_fit"]
    assert kept["factor_ids"] == fixed["factor_ids"] + ["discovery.news_range"]
    np.testing.assert_array_equal(kept["x"], np.asarray(adaptive["x"])[:, :3])
    assert kept["y"] == adaptive["y"]
    assert partial_cache.calls["2026-07_fixed_fit"] == fixed
    assert any("news_relative_volume" in str(row) for row in partial["candidate_rejections"])


def test_cli_ohlcv_provenance_resume_binding_and_flag_off_compatibility(
    retained_ohlcv, tmp_path, monkeypatch
):
    directory, _, _, _, bundle = retained_ohlcv
    parser = argparse.ArgumentParser()
    rolling.add_arguments(parser.add_subparsers(dest="command"))
    arguments = [
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
        "--output",
        str(tmp_path / "study"),
        "--ohlcv-features",
    ]
    args = parser.parse_args(arguments)
    selected = monthly.candidate()
    selected["inputs"][0]["factor_id"] = "survey.qlib.klen"
    selected["expression"]["nodes"] = [{"name": "result", "op": "leaf", "leaf": "source"}]
    decision = {
        "action": "replace",
        "basis": "price_only",
        "reason": "Retained bar range",
        "candidates": [selected],
        "news_inputs": [],
    }
    llm = monthly.DecisionLLM([decision, RuntimeError("provider stopped")])
    monkeypatch.setattr(rolling, "LLMHypothesizer", lambda *a, **kw: llm)
    with pytest.raises(RuntimeError, match="provider stopped"):
        rolling.run_rolling(args)
    config = json.loads((args.output / "config.json").read_bytes())
    assert config["ohlcv_features"] is True
    assert config["input_identity"]["ohlcv"] == bundle["provenance"]
    assert (
        json.loads((args.output / "ohlcv_inputs.json").read_bytes())["provenance"]
        == bundle["provenance"]
    )
    assert "Retained OHLCV inputs" in (args.output / "report.html").read_text()
    original = (args.output / "2026-07/prediction.json").read_bytes()
    args.resume = True
    # Even a validated loader returning changed provenance cannot reuse old checkpoints.
    changed = deepcopy(bundle)
    changed["provenance"]["raw_sha256"] = "f" * 64
    with monkeypatch.context() as patch:
        patch.setattr(rolling, "load_ohlcv_history", lambda *a, **kw: changed)
        with pytest.raises(ValueError, match="configuration changed"):
            rolling.run_rolling(args)
    result = rolling.run_rolling(args)
    assert result["status"] == "completed"
    assert (args.output / "2026-07/prediction.json").read_bytes() == original

    disabled = parser.parse_args(
        [
            "research-rolling",
            "SOXX",
            "--prices-dir",
            str(directory),
            "--first-month",
            "2026-07",
            "--last-month",
            "2026-07",
            "--model",
            "ridge",
            "--llm",
            "none",
            "--output",
            str(tmp_path / "disabled"),
        ]
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            rolling, "load_ohlcv_history", lambda *a, **kw: pytest.fail("flag-off loaded OHLCV")
        )
        plain = rolling.run_rolling(disabled)
    assert "ohlcv_features" not in plain["configuration"]
    assert "ohlcv" not in plain["configuration"]["input_identity"]
    assert "ohlcv_inputs" not in plain["folds"][0]
    assert not (disabled.output / "ohlcv_inputs.json").exists()
    assert plain["folds"][0]["forecasts"]["fixed"] == result["folds"][0]["forecasts"]["fixed"]
