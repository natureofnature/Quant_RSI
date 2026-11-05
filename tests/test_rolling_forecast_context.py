"""Offline integration of saved-model accounting; fixture forecasts do not establish alpha."""

import argparse
import json
from copy import deepcopy

import numpy as np
import pytest
import test_monthly_rolling as monthly
from test_monthly_rolling import DecisionLLM, candidate, execute

from quant_rsi.core.data import MarketData
from quant_rsi.research import rolling
from quant_rsi.research.rolling_diagnostics import summarize_forecast

retained = monthly.retained


def replacement():
    return {
        "action": "replace",
        "basis": "price_only",
        "reason": "Exercise an actual additional numerical column",
        "candidates": [candidate()],
    }


def record_fits(monkeypatch):
    actual, calls = rolling.fit_forecast, []

    def counted(*args, **kwargs):
        calls.append(list(args[4]))
        return actual(*args, **kwargs)

    monkeypatch.setattr(rolling, "fit_forecast", counted)
    return calls


def test_current_baseline_and_mature_paired_accounting_need_no_extra_fits(
    retained, tmp_path, monkeypatch
):
    calls = record_fits(monkeypatch)
    llm = DecisionLLM([replacement()])
    folds, cache = execute(retained, tmp_path / "study", last="2026-08", llm=llm)
    assert len(calls) == 4  # Exactly two baseline and two candidate fits.
    assert len([key for key in cache.calls if key.endswith("_fit")]) == 4
    assert len(llm.contexts) == 2
    july, august = folds
    assert llm.contexts[0]["mature_feedback"] == []
    assert llm.contexts[0]["baseline_diagnostics"] == summarize_forecast(
        july["forecast_diagnostics"]["fixed"]
    )
    for fold in folds:
        fixed, adaptive = (fold["forecast_diagnostics"][arm] for arm in ("fixed", "adaptive"))
        assert fixed["comparison_identity"] == adaptive["comparison_identity"]
        assert adaptive["candidate_columns"] == 1
        for arm in ("fixed", "adaptive"):
            diagnostic = fold["forecast_diagnostics"][arm]
            assert [c["factor_id"] for c in diagnostic["columns"]] == fold["fits"][arm][
                "factor_ids"
            ]
            assert diagnostic["training_samples"] == len(fold["training_origins"])
            decomposition = diagnostic["decomposition"]
            assert decomposition["units"] == "log_return" and decomposition["available"]
            assert sum(
                decomposition[k] for k in ("intercept", "baseline_group", "candidate_group")
            ) == (pytest.approx(fold["forecasts"][arm]["predicted_log_return"]))
        paired = fold["paired_forecast_decomposition"]
        assert paired["available"]
        assert sum(
            paired[k] for k in ("intercept_change", "baseline_refit_change", "candidate_group")
        ) == (pytest.approx(paired["predicted_log_return_change"]))
        frozen = json.loads((tmp_path / "study" / fold["month"] / "prediction.json").read_bytes())
        assert frozen["forecast_diagnostics"] == fold["forecast_diagnostics"]
        assert "observed_price" not in json.dumps(frozen["forecast_diagnostics"])
    feedback = llm.contexts[1]["mature_feedback"]
    assert [row["month"] for row in feedback] == [july["month"]]
    assert feedback[0]["forecast_diagnostics"] == {
        arm: summarize_forecast(value) for arm, value in july["forecast_diagnostics"].items()
    }
    assert feedback[0]["paired_forecast_decomposition"] == july["paired_forecast_decomposition"]
    assert llm.contexts[1]["baseline_diagnostics"] == summarize_forecast(
        august["forecast_diagnostics"]["fixed"]
    )


def test_future_prices_and_news_cannot_change_current_or_mature_diagnostics(retained, tmp_path):
    _, data, _, _ = retained
    news = [
        {
            "received_at": "2026-06-01T12:00:00Z",
            "items": [
                {
                    "id": "future",
                    "published_at": "2026-08-10T12:00:00Z",
                    "received_at": "2026-08-10T12:00:00Z",
                    "title": "future event",
                }
            ],
        }
    ]
    first_llm = DecisionLLM([replacement()])
    first, _ = execute(
        retained, tmp_path / "original", last="2026-08", llm=first_llm, snapshots=news
    )
    future = np.asarray([day > "2026-07-31" for day in data.dates])
    opens, closes = data.opens.copy(), data.closes.copy()
    opens[future] *= 1.8
    closes[future] *= 1.8
    altered = MarketData(data.dates, data.instruments, opens, closes)
    changed_news = deepcopy(news)
    changed_news[0]["items"][0]["title"] = "different future business outcome"
    second_llm = DecisionLLM([replacement()])
    second, _ = execute(
        retained,
        tmp_path / "changed",
        last="2026-08",
        llm=second_llm,
        snapshots=changed_news,
        data=altered,
    )
    assert first_llm.contexts == second_llm.contexts
    for left, right in zip(first, second, strict=True):
        assert left["forecast_diagnostics"] == right["forecast_diagnostics"]
        assert left["paired_forecast_decomposition"] == right["paired_forecast_decomposition"]
        assert (tmp_path / "original" / left["month"] / "prediction.json").read_bytes() == (
            tmp_path / "changed" / right["month"] / "prediction.json"
        ).read_bytes()
    assert (
        first[-1]["forecasts"]["fixed"]["observed_price"]
        != second[-1]["forecasts"]["fixed"]["observed_price"]
    )


@pytest.mark.parametrize("action", ["retain", "withdraw", "malformed"])
def test_baseline_fallback_clones_diagnostics_without_an_extra_fit(
    retained, tmp_path, monkeypatch, action
):
    calls = record_fits(monkeypatch)
    decision = {"action": action, "basis": "price_only", "reason": "No candidate", "candidates": []}
    folds, _ = execute(retained, tmp_path / action, last="2026-07", llm=DecisionLLM([decision]))
    assert len(calls) == 1
    fold = folds[0]
    assert fold["fits"]["adaptive"]["reused_arm"] == "fixed"
    fixed, adaptive = (fold["forecast_diagnostics"][arm] for arm in ("fixed", "adaptive"))
    assert fixed == adaptive and fixed is not adaptive
    paired = fold["paired_forecast_decomposition"]
    assert paired["available"]
    assert all(
        paired[key] == 0
        for key in (
            "intercept_change",
            "baseline_refit_change",
            "candidate_group",
            "predicted_log_return_change",
        )
    )
    adaptive["columns"][0]["prediction_value"] = -123
    assert adaptive != fixed


def test_cached_resume_preserves_diagnostics_and_does_not_refit(retained, tmp_path, monkeypatch):
    parser = argparse.ArgumentParser()
    rolling.add_arguments(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        [
            "research-rolling",
            "SOXX",
            "--prices-dir",
            str(retained[0]),
            "--first-month",
            "2026-07",
            "--last-month",
            "2026-08",
            "--model",
            "ridge",
            "--llm",
            "codex",
            "--output",
            str(tmp_path / "resume"),
        ]
    )
    llm = DecisionLLM([replacement(), RuntimeError("offline interrupted provider")])
    monkeypatch.setattr(rolling, "LLMHypothesizer", lambda *a, **kw: llm)
    calls = record_fits(monkeypatch)
    with pytest.raises(RuntimeError, match="offline interrupted provider"):
        rolling.run_rolling(args)
    assert len(calls) == 3  # July B/candidate; August B completed before provider failure.
    july_bytes = (args.output / "2026-07/prediction.json").read_bytes()
    failed_context = deepcopy(llm.contexts[-1])
    args.resume = True
    result = rolling.run_rolling(args)
    assert result["status"] == "completed"
    assert len(calls) == 4 and len(llm.contexts) == 3
    assert llm.contexts[-1] == failed_context
    assert (args.output / "2026-07/prediction.json").read_bytes() == july_bytes
    assert failed_context["mature_feedback"][0]["forecast_diagnostics"]
    complete = {
        month: (args.output / month / "prediction.json").read_bytes()
        for month in ("2026-07", "2026-08")
    }
    with pytest.raises(ValueError, match="Completed studies are immutable"):
        rolling.run_rolling(args)
    assert len(calls) == 4 and len(llm.contexts) == 3
    assert complete == {
        month: (args.output / month / "prediction.json").read_bytes() for month in complete
    }
