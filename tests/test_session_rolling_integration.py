"""Offline three-session news/model/feedback/resume checks, not evidence of alpha."""

import argparse
import json
from copy import deepcopy

import numpy as np
import pytest
from test_monthly_rolling import DecisionLLM, MemoryCache
from test_monthly_rolling import retained as retained_fixture
from test_rolling_news_integration import history, proposal, retain_news

from quant_rsi.core.data import MarketData
from quant_rsi.research import rolling
from quant_rsi.research.rolling_windows import session_plan

retained = retained_fixture


def run(retained, root, *, data=None, news=None, periods=2):
    _, original, info, calendar = retained
    data = original if data is None else data
    plan = session_plan(data, calendar, "2026-07", "2026-07")[:periods]
    cache = MemoryCache()
    llm = DecisionLLM([proposal()] + [retain_news()] * (len(plan) - 1))
    root.mkdir()
    folds = rolling.run_months(
        data,
        info,
        {},
        plan,
        cache,
        root,
        model="ridge",
        llm=llm,
        news_history=history() if news is None else news,
        news_required=True,
        news_mode="reconstructed",
        max_candidates=16,
    )
    return folds, cache, llm


def test_news_enters_real_ridge_and_same_month_periods_keep_distinct_feedback(retained, tmp_path):
    folds, cache, llm = run(retained, tmp_path / "news")
    assert [fold["period_id"] for fold in folds] == ["2026-07-06", "2026-07-09"]
    assert all(fold["candidate_status"] == "fitted" for fold in folds)
    assert [len(context["mature_feedback"]) for context in llm.contexts] == [0, 1]
    assert folds[0]["feedback_periods"] == []
    assert folds[1]["feedback_periods"] == ["2026-07-06"]
    feedback = llm.contexts[1]["mature_feedback"][0]
    assert feedback["period_id"] == "2026-07-06" and "month" not in feedback
    assert feedback["prediction_sha256"] == folds[0]["prediction_sha256"]
    assert feedback["target_session"] <= folds[1]["origin_session"]
    assert len(cache.calls) == 6  # Two actual fits and one fake-provider decision per period.
    for fold, context in zip(folds, llm.contexts, strict=True):
        key = fold["period_id"]
        fixed, adaptive = cache.calls[key + "_fixed_fit"], cache.calls[key + "_adaptive_fit"]
        x = np.asarray(adaptive["x"])
        np.testing.assert_array_equal(x[:, :-1], fixed["x"])
        assert adaptive["y"] == fixed["y"] and adaptive["window"] == fixed["window"]
        assert adaptive["factor_ids"][-1] == "discovery.export_pressure"
        numerical = fold["news_training"]["news.export_pressure"]
        assert numerical["training_values"] == x[:, -1].tolist()
        assert numerical["distinct_training_values"] >= 2
        assert numerical["prediction_value"] == adaptive["prediction_x"][-1]
        assert context["horizon_sessions"] == 3 and context["max_candidates"] == 16
        assert context["period_id"] == key and context["period_kind"] == "trading_sessions"
        frozen = json.loads((tmp_path / "news" / key / "prediction.json").read_bytes())
        assert "outcome" not in frozen
        assert all("observed_price" not in row for row in frozen["forecasts"].values())
    first_ids = {row["id"] for row in llm.contexts[0]["news"]["items"]}
    next_ids = {row["id"] for row in llm.contexts[1]["news"]["items"]}
    assert "after_close" not in first_ids and "after_close" in next_ids
    assert "future" not in first_ids | next_ids
    assert folds[0]["news_training"]["news.export_pressure"]["prediction_value"] == 0
    assert folds[1]["news_training"]["news.export_pressure"]["prediction_value"] == 1


def test_future_prices_and_news_do_not_change_first_context_or_frozen_forecast(retained, tmp_path):
    original, old_cache, old_llm = run(retained, tmp_path / "original", periods=1)
    _, data, _, _ = retained
    opens, closes = data.opens.copy(), data.closes.copy()
    future = [i for i, day in enumerate(data.dates) if day > "2026-06-30"]
    opens[future] *= 2
    closes[future] *= 3
    changed_data = MarketData(data.dates, data.instruments, opens, closes)
    changed_news = deepcopy(history())
    for row in changed_news["items"]:
        if row["id"] in {"after_close", "future"}:
            row["title"] = row["text"] = "Changed future material cannot reach the first decision"
    changed, new_cache, new_llm = run(
        retained, tmp_path / "changed", data=changed_data, news=changed_news, periods=1
    )
    assert old_llm.contexts == new_llm.contexts
    assert old_cache.calls == new_cache.calls
    assert original[0]["prediction_sha256"] == changed[0]["prediction_sha256"]
    assert (tmp_path / "original/2026-07-06/prediction.json").read_bytes() == (
        tmp_path / "changed/2026-07-06/prediction.json"
    ).read_bytes()
    assert (
        original[0]["forecasts"]["fixed"]["observed_price"]
        != (changed[0]["forecasts"]["fixed"]["observed_price"])
    )


def test_pending_three_day_target_has_no_accuracy_or_account_effect(retained, tmp_path):
    _, data, _, _ = retained
    indices = [i for i, day in enumerate(data.dates) if day <= "2026-07-07"]
    partial = MarketData(
        tuple(data.dates[i] for i in indices),
        data.instruments,
        data.opens[indices],
        data.closes[indices],
    )
    folds, _, llm = run(retained, tmp_path / "pending", data=partial)
    assert [f["outcome"] for f in folds] == ["observed", "not_observed"]
    assert len(llm.contexts[1]["mature_feedback"]) == 1
    metrics, accounts = rolling.aggregate(partial, folds, {})
    assert set(metrics) == {"fixed", "adaptive", "unchanged_price", "training_label_mean"}
    assert all(value["sample_count"] == 1 for value in metrics.values())
    for account in accounts.values():
        assert account["period"]["end"] == "2026-07-06"
        assert account["prediction_horizon_sessions"] == 3
        assert account["strategy"]["valuations"][-1]["session_id"] == "2026-07-06"
    assert all("observed_price" not in row for row in folds[1]["forecasts"].values())


def args_for(directory, output):
    parser = argparse.ArgumentParser()
    rolling.add_arguments(parser.add_subparsers(dest="command"))
    return parser.parse_args(
        [
            "research-rolling",
            "SOXX",
            "--prices-dir",
            str(directory),
            "--first-month",
            "2026-07",
            "--last-month",
            "2026-07",
            "--horizon-sessions",
            "3",
            "--model",
            "ridge",
            "--llm",
            "codex",
            "--news-research",
            "--max-candidates",
            "16",
            "--output",
            str(output),
        ]
    )


def test_cli_failure_resume_reuses_successful_news_and_fits_and_rejects_horizon_change(
    retained, tmp_path, monkeypatch
):
    directory, _, _, _ = retained
    args = args_for(directory, tmp_path / "resume")
    calls, fits = [], []
    llm = DecisionLLM([proposal(), RuntimeError("offline provider failure")] + [retain_news()] * 6)
    real_fit = rolling.fit_forecast

    def collect(*a, **kw):
        calls.append(1)
        return history()

    def fit(*a, **kw):
        fits.append(1)
        return real_fit(*a, **kw)

    monkeypatch.setattr(rolling, "collect_history", collect)
    monkeypatch.setattr(rolling, "fit_forecast", fit)
    monkeypatch.setattr(rolling, "LLMHypothesizer", lambda *a, **kw: llm)
    with pytest.raises(RuntimeError, match="offline provider failure"):
        rolling.run_rolling(args)
    assert len(fits) == 3 and len(calls) == 1 and len(llm.contexts) == 2
    assert (args.output / "session_plan.json").exists()
    assert not (args.output / "monthly_plan.json").exists()
    frozen = (args.output / "2026-07-06/prediction.json").read_bytes()
    saved_config = (args.output / "config.json").read_bytes()
    saved_runtime = (args.output / "runtime.json").read_bytes()
    args.resume = True
    args.horizon_sessions = 5
    with pytest.raises(ValueError, match="configuration changed"):
        rolling.run_rolling(args)
    assert len(fits) == 3 and len(calls) == 1 and len(llm.contexts) == 2
    assert (args.output / "config.json").read_bytes() == saved_config
    assert (args.output / "runtime.json").read_bytes() == saved_runtime
    args.horizon_sessions = 3
    result = rolling.run_rolling(args)
    assert result["kind"] == "SessionRollingStudy" and result["status"] == "completed"
    assert len(result["folds"]) == 7
    assert len(fits) == 14 and len(calls) == 1 and len(llm.contexts) == 8
    assert [context["forecast_target"] for context in llm.contexts[:3]] == [
        "2026-07-06",
        "2026-07-09",
        "2026-07-09",
    ]
    assert (args.output / "2026-07-06/prediction.json").read_bytes() == frozen
    assert all(fold["candidate_status"] == "fitted" for fold in result["folds"])
    assert result["configuration"]["horizon_sessions"] == 3
    assert result["metrics"]["adaptive"]["sample_count"] == 7
    assert result["accounts"]["adaptive"]["prediction_horizon_sessions"] == 3
    assert (args.output / "report.html").stat().st_size > 1000
