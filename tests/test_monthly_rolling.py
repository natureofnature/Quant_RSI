"""Offline monthly chronology and resume checks; synthetic prices do not establish alpha."""

import argparse
import json
from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest
from test_daily_data import chart, install

from quant_rsi.core.artifacts import canonical
from quant_rsi.core.data import MarketData
from quant_rsi.market import daily_data
from quant_rsi.research import rolling
from quant_rsi.research.demo_prices import read_daily_history
from quant_rsi.research.rolling_windows import monthly_plan, read_calendar


@pytest.fixture
def retained(monkeypatch, tmp_path):
    payload = chart("2026-11-03")
    quote = payload["chart"]["result"][0]
    quote["meta"]["symbol"] = "SOXX"
    quote["events"] = {}
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
    return directory, data, info, read_calendar(directory)


class MemoryCache:
    def __init__(self):
        self.calls = {}

    def call(self, name, inputs, operation):
        self.calls[name] = deepcopy(inputs)
        return operation()


class DecisionLLM:
    def __init__(self, decisions=()):
        self.decisions = list(decisions)
        self.contexts = []

    def structured(self, prompt, schema):
        self.contexts.append(json.loads(prompt.split("INPUT_JSON:\n", 1)[1]))
        value = (
            self.decisions.pop(0)
            if self.decisions
            else {
                "action": "retain",
                "basis": "price_only",
                "reason": "Keep the fixed baseline",
                "candidates": [],
            }
        )
        if isinstance(value, Exception):
            raise value
        return json.dumps(value)


def execute(retained, root, *, last="2026-11", llm=None, snapshots=(), data=None, peers=None):
    _, original, info, calendar = retained
    data = data or original
    plan = monthly_plan(data, calendar, "2026-07", last)
    cache = MemoryCache()
    root.mkdir()
    folds = rolling.run_months(
        data, info, peers or {}, plan, cache, root, model="ridge", llm=llm, snapshots=snapshots
    )
    return folds, cache


def candidate():
    return {
        "name": "trailing_close",
        "description": "Price-only lagged level hypothesis",
        "inputs": [{"name": "source", "factor_id": "price.close"}],
        "expression": {
            "schema_version": 2,
            "nodes": [
                {"name": "a", "op": "leaf", "leaf": "source"},
                {"name": "result", "op": "lag", "inputs": ["a"], "window": 3},
            ],
        },
        "evidence_ids": [],
    }


def test_official_calendar_windows_are_fixed_width_and_labels_mature(retained):
    _, data, info, calendar = retained
    assert len(data.dates) == 209
    assert data.dates[0] == "2026-01-02"
    assert data.dates[-1] == "2026-10-30"
    assert "horizon_sessions" not in info
    assert "latest_target_session" not in info
    plan = monthly_plan(data, calendar, "2026-07", "2026-11")
    assert [fold["horizon_sessions"] for fold in plan] == [22, 21, 21, 22, 20]
    assert [fold["training_window"]["start"] for fold in plan] == [
        "2026-01-01",
        "2026-02-01",
        "2026-03-01",
        "2026-04-01",
        "2026-05-01",
    ]
    assert plan[0]["training_window"]["end"] == "2026-06-30"
    for fold in plan:
        assert fold["training_indices"][0] == 30
        assert fold["training_origins"][0] == fold["history_sessions"][30]
        assert fold["training_targets"][-1] == fold["origin_session"]
        assert all(target <= fold["origin_session"] for target in fold["training_targets"])
        assert set(fold["unmatured_origins"]).isdisjoint(fold["training_origins"])
        for origin, target in zip(fold["training_origins"], fold["training_targets"], strict=True):
            assert (
                fold["history_sessions"].index(target) - fold["history_sessions"].index(origin)
                == fold["horizon_sessions"]
            )


def test_future_target_and_peer_prices_cannot_change_earlier_forecast(retained, tmp_path):
    _, data, _, _ = retained
    peer = data.closes[:, 0] * 1.1 + np.sin(np.arange(len(data.dates))) * 0.1
    original, _ = execute(retained, tmp_path / "original", last="2026-07", peers={"SPY": peer})
    future = np.array([day > "2026-06-30" for day in data.dates])
    opens, closes = data.opens.copy(), data.closes.copy()
    opens[future] *= 1.7
    closes[future] *= 1.7
    peer[future] *= 0.6
    changed = MarketData(data.dates, data.instruments, opens, closes)
    revised, _ = execute(
        retained, tmp_path / "changed", last="2026-07", data=changed, peers={"SPY": peer}
    )
    assert (tmp_path / "original/2026-07/prediction.json").read_bytes() == (
        tmp_path / "changed/2026-07/prediction.json"
    ).read_bytes()
    assert (
        original[0]["forecasts"]["fixed"]["observed_price"]
        != revised[0]["forecasts"]["fixed"]["observed_price"]
    )


def test_default_five_fits_pending_month_excluded_from_metrics_and_account(retained, tmp_path):
    _, data, _, _ = retained
    folds, cache = execute(retained, tmp_path / "study")
    assert len(cache.calls) == 5
    assert all(name.endswith("_fixed_fit") for name in cache.calls)
    assert [fold["outcome"] for fold in folds] == ["observed"] * 4 + ["not_observed"]
    assert "observed_price" not in folds[-1]["forecasts"]["fixed"]
    metrics, accounts = rolling.aggregate(data, folds, {})
    settled_metrics, settled_accounts = rolling.aggregate(data, folds[:-1], {})
    assert canonical(metrics) == canonical(settled_metrics)
    assert canonical(accounts) == canonical(settled_accounts)
    assert set(accounts) == {"fixed"}
    assert accounts["fixed"]["prediction_horizon_sessions"] == [22, 21, 21, 22]
    assert "adaptive" not in metrics


def test_llm_receives_only_mature_feedback_and_cutoff_eligible_news(retained, tmp_path):
    old = {
        "id": "old",
        "received_at": "2026-06-10T12:00:00Z",
        "published_at": "2026-06-09T12:00:00Z",
    }
    later = {
        "id": "later",
        "received_at": "2026-07-10T12:00:00Z",
        "published_at": "2026-06-09T12:00:00Z",
    }
    future = {
        "id": "future",
        "received_at": "2026-06-10T12:00:00Z",
        "published_at": "2026-07-10T12:00:00Z",
    }
    snapshots = [
        {"received_at": "2026-06-20T12:00:00Z", "items": [old, later, future]},
        {"received_at": "2026-08-01T12:00:00Z", "items": [{**old, "id": "late_snapshot"}]},
    ]
    llm = DecisionLLM()
    folds, _ = execute(retained, tmp_path / "study", last="2026-08", llm=llm, snapshots=snapshots)
    july, august = llm.contexts
    assert july["mature_feedback"] == []
    assert [item["id"] for item in july["news"]["items"]] == ["old"]
    assert {item["id"] for item in august["news"]["items"]} == {"old", "later", "future"}
    assert [item["month"] for item in august["mature_feedback"]] == ["2026-07"]
    assert august["mature_feedback"][0]["prediction_sha256"] == folds[0]["prediction_sha256"]
    assert (
        august["mature_feedback"][0]["forecasts"]["fixed"]["observed_price"]
        == folds[0]["forecasts"]["fixed"]["observed_price"]
    )


def test_new_expression_is_fitted_and_forecast_frozen_before_settlement(
    retained, tmp_path, monkeypatch
):
    llm = DecisionLLM(
        [
            {
                "action": "replace",
                "basis": "price_only",
                "reason": "Test a new lag",
                "candidates": [candidate()],
            }
        ]
    )
    native_settle = rolling.settle
    seen = []

    def inspect_freeze(fold, data):
        frozen = json.loads((tmp_path / "study" / fold["month"] / "prediction.json").read_bytes())
        assert all("observed_price" not in row for row in frozen["forecasts"].values())
        assert "outcome" not in frozen
        seen.append(fold["month"])
        return native_settle(fold, data)

    monkeypatch.setattr(rolling, "settle", inspect_freeze)
    folds, cache = execute(retained, tmp_path / "study", last="2026-08", llm=llm)
    assert seen == ["2026-07", "2026-08"]
    for fold in folds:
        assert fold["candidate_status"] == "fitted"
        assert "discovery.trailing_close" in fold["fits"]["adaptive"]["factor_ids"]
        fixed = cache.calls[f"{fold['month']}_fixed_fit"]
        adaptive = cache.calls[f"{fold['month']}_adaptive_fit"]
        assert np.asarray(adaptive["x"]).shape == (len(fixed["x"]), len(fixed["x"][0]) + 1)
        assert np.all(np.isfinite(adaptive["x"]))
        data = retained[1]
        expected = [
            data.closes[data.dates.index(origin) - 3, 0] for origin in fold["training_origins"]
        ]
        np.testing.assert_array_equal(np.asarray(adaptive["x"])[:, -1], expected)
        assert (
            fold["forecasts"]["adaptive"]["predicted_price"]
            != fold["forecasts"]["fixed"]["predicted_price"]
        )


def test_cutoff_ineligible_candidate_rejected_without_extra_fit(retained, tmp_path):
    value = candidate() | {"evidence_ids": ["future-news"]}
    llm = DecisionLLM(
        [
            {
                "action": "replace",
                "basis": "news",
                "reason": "Unavailable news",
                "candidates": [value],
            }
        ]
    )
    folds, cache = execute(retained, tmp_path / "study", last="2026-07", llm=llm)
    fold = folds[0]
    assert fold["candidate_status"] == "rejected"
    assert "cutoff-ineligible" in fold["candidate_error"]
    assert fold["decision"]["action"] == "rejected"
    assert fold["decision"]["response_checkpoint"] == "checkpoints/2026-07_decision.json"
    assert fold["forecasts"]["adaptive"] == fold["forecasts"]["fixed"]
    assert fold["fits"]["adaptive"]["factor_ids"] == fold["fits"]["fixed"]["factor_ids"]
    assert fold["fits"]["adaptive"]["reused_arm"] == "fixed"
    assert "Research incumbent" in fold["candidate_recipe_role"]
    assert set(cache.calls) == {"2026-07_fixed_fit", "2026-07_decision"}


def test_calendar_hash_and_missing_training_session_rejected(retained):
    directory, data, _, calendar = retained
    missing = MarketData(data.dates[1:], data.instruments, data.opens[1:], data.closes[1:])
    with pytest.raises(ValueError, match="Incomplete six-month history"):
        monthly_plan(missing, calendar, "2026-07", "2026-07")
    path = directory / "calendar_evidence.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="differs from collection receipt"):
        read_calendar(directory)


def test_resume_reuses_completed_llm_and_fits_after_provider_failure(
    retained, tmp_path, monkeypatch
):
    directory, _, _, _ = retained
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
            "2026-09",
            "--model",
            "ridge",
            "--llm",
            "codex",
            "--output",
            str(tmp_path / "resume"),
        ]
    )
    llm = DecisionLLM(
        [
            {
                "action": "retain",
                "basis": "price_only",
                "reason": "Keep baseline",
                "candidates": [],
            },
            RuntimeError("transient provider failure"),
        ]
    )
    monkeypatch.setattr(rolling, "LLMHypothesizer", lambda *a, **kw: llm)
    real_fit = rolling.fit_forecast
    fitted = []

    def count_fit(*a, **kw):
        fitted.append(1)
        return real_fit(*a, **kw)

    monkeypatch.setattr(rolling, "fit_forecast", count_fit)
    with pytest.raises(RuntimeError, match="transient provider failure"):
        rolling.run_rolling(args)
    assert "transient provider failure" in (args.output / "report.html").read_text()
    original = (args.output / "2026-07/prediction.json").read_bytes()
    assert len(fitted) == 2
    args.resume = True
    result = rolling.run_rolling(args)
    assert result["status"] == "completed"
    assert len(fitted) == 3
    assert [context["forecast_target"] for context in llm.contexts] == [
        "2026-07-31",
        "2026-08-31",
        "2026-08-31",
        "2026-09-30",
    ]
    assert (args.output / "2026-07/prediction.json").read_bytes() == original
