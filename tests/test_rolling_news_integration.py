"""Offline end-to-end news-to-matrix regressions; synthetic fixtures do not prove alpha."""

import argparse
from calendar import monthrange
from copy import deepcopy

import numpy as np
import pytest
import test_monthly_rolling as monthly_fixtures
from test_monthly_rolling import DecisionLLM, MemoryCache, candidate

from quant_rsi.research import rolling
from quant_rsi.research.rolling_windows import monthly_plan

retained = monthly_fixtures.retained


def history():
    def article(identifier, title, published):
        return {
            "id": identifier,
            "symbol": "SOXX",
            "title": title,
            "text": title,
            "url": "https://example.com/" + identifier,
            "published_at": published,
            "received_at": "2026-11-04T18:00:00+00:00",
            "version_available_at": "2026-11-04T18:00:00+00:00",
        }

    return {
        "provider": "historical_google_news_search",
        "symbol": "SOXX",
        "coverage_status": "partial_search_results",
        "searches": [
            {
                "start": f"2026-{month:02d}-01",
                "end": f"2026-{month:02d}-{monthrange(2026, month)[1]}",
                "status": "partial_results",
            }
            for month in range(1, 11)
        ],
        "items": [
            article("first", "Chip export restrictions", "2026-03-02T12:00:00+00:00"),
            article("second", "More export restrictions", "2026-05-20T12:00:00+00:00"),
            article("third", "Chip export restrictions extended", "2026-06-05T12:00:00+00:00"),
            article("after_close", "Export restrictions after close", "2026-06-30T21:00:00+00:00"),
            article("future", "Future export restrictions", "2026-11-01T12:00:00+00:00"),
        ],
    }


def proposal():
    return {
        "action": "replace",
        "basis": "news",
        "reason": "Test the observed export-restriction headline intensity",
        "news_inputs": [
            {
                "input_id": "news.export_pressure",
                "hypothesis": "Export-restriction attention may affect semiconductor returns",
                "operation": "count",
                "terms": ["export restrictions"],
                "negative_terms": [],
                "window_days": 20,
                "evidence_ids": ["first", "second", "third"],
            }
        ],
        "candidates": [
            {
                "name": "export_pressure",
                "description": "Trailing retrieved export-restriction headline count",
                "inputs": [{"name": "pressure", "factor_id": "news.export_pressure"}],
                "expression": {
                    "schema_version": 2,
                    "nodes": [{"name": "result", "op": "leaf", "leaf": "pressure"}],
                },
                "evidence_ids": ["first", "second", "third"],
            }
        ],
    }


def retain_news():
    return {
        "action": "retain",
        "basis": "news",
        "reason": "Keep the news hypothesis",
        "candidates": [],
        "news_inputs": [],
    }


def execute(retained, path, *, news=None, decisions=None, last="2026-08"):
    _, data, info, calendar = retained
    plan = monthly_plan(data, calendar, "2026-07", last)
    cache = MemoryCache()
    llm = DecisionLLM(decisions or [proposal(), retain_news()])
    path.mkdir()
    folds = rolling.run_months(
        data,
        info,
        {},
        plan,
        cache,
        path,
        model="ridge",
        llm=llm,
        news_history=history() if news is None else news,
        news_mode="reconstructed",
        news_required=True,
    )
    return folds, cache, llm


def test_news_candidate_really_enters_matched_training_and_next_month(retained, tmp_path):
    folds, cache, llm = execute(retained, tmp_path / "study")
    assert [fold["candidate_status"] for fold in folds] == ["fitted", "fitted"]
    assert [fold["decision"]["action"] for fold in folds] == ["replace", "retain"]
    assert [len(context["mature_feedback"]) for context in llm.contexts] == [0, 1]
    for fold in folds:
        fixed = cache.calls[fold["month"] + "_fixed_fit"]
        adaptive = cache.calls[fold["month"] + "_adaptive_fit"]
        x = np.asarray(adaptive["x"])
        np.testing.assert_array_equal(x[:, :-1], fixed["x"])
        assert adaptive["y"] == fixed["y"]
        assert adaptive["window"] == fixed["window"]
        actual = fold["news_training"]["news.export_pressure"]
        assert actual["training_values"] == x[:, -1].tolist()
        assert actual["distinct_training_values"] >= 2
        assert adaptive["prediction_x"][-1] == actual["prediction_value"]
        assert adaptive["factor_ids"][-1] == "discovery.export_pressure"
        diagnostic = fold["news_features"][0]
        assert diagnostic["measurement"] == "retrieved_headline_match_count"
        assert diagnostic["information_mode"] == "reconstructed"
        for instant, ids in zip(
            diagnostic["prediction_times"], diagnostic["row_evidence_ids"], strict=True
        ):
            if instant < "2026-03-02T12:00:00+00:00":
                assert not ids
    assert folds[1]["candidate_news_inputs"] == folds[0]["candidate_news_inputs"]
    context_ids = {item["id"] for item in llm.contexts[0]["news"]["items"]}
    assert context_ids == {"first", "second", "third"}
    assert "after_close" not in context_ids


def test_future_article_changes_cannot_change_earlier_prompt_or_prediction(retained, tmp_path):
    first, _, original_llm = execute(retained, tmp_path / "before", last="2026-07")
    changed = history()
    changed["items"][-1]["title"] = "Unknown future fact"
    changed["items"][-2]["title"] = "Changed same-day post-close headline"
    second, _, changed_llm = execute(retained, tmp_path / "after", news=changed, last="2026-07")
    assert original_llm.contexts == changed_llm.contexts
    assert first == second
    assert (tmp_path / "before/2026-07/prediction.json").read_bytes() == (
        tmp_path / "after/2026-07/prediction.json"
    ).read_bytes()


def test_required_news_rejects_price_only_and_news_without_numerical_input(retained, tmp_path):
    price = {
        "action": "replace",
        "basis": "price_only",
        "reason": "Try lag",
        "candidates": [candidate()],
    }
    folds, cache, _ = execute(retained, tmp_path / "price", decisions=[price], last="2026-07")
    assert folds[0]["candidate_status"] == "rejected"
    assert "price-only" in folds[0]["candidate_error"]
    assert "2026-07_adaptive_fit" not in cache.calls
    unsupported = deepcopy(price)
    unsupported["basis"] = "news"
    unsupported["candidates"][0]["evidence_ids"] = ["first"]
    folds, cache, _ = execute(retained, tmp_path / "fake", decisions=[unsupported], last="2026-07")
    assert folds[0]["candidate_status"] == "rejected"
    assert "actual news input" in folds[0]["candidate_error"]
    assert folds[0]["forecasts"]["adaptive"] == folds[0]["forecasts"]["fixed"]
    assert "2026-07_adaptive_fit" not in cache.calls


def test_acquisition_and_llm_failures_resume_without_repeating_successful_work(
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
            "2026-08",
            "--model",
            "ridge",
            "--llm",
            "codex",
            "--news-research",
            "--output",
            str(tmp_path / "resume"),
        ]
    )
    acquisitions, fits = [], []

    def acquire(*a, **kw):
        acquisitions.append(1)
        if len(acquisitions) == 1:
            raise RuntimeError("news transport unavailable")
        return history()

    llm = DecisionLLM([proposal(), RuntimeError("LLM unavailable"), retain_news()])
    real_fit = rolling.fit_forecast

    def fit(*a, **kw):
        fits.append(1)
        return real_fit(*a, **kw)

    monkeypatch.setattr(rolling, "collect_history", acquire)
    monkeypatch.setattr(rolling, "LLMHypothesizer", lambda *a, **kw: llm)
    monkeypatch.setattr(rolling, "fit_forecast", fit)
    with pytest.raises(RuntimeError, match="news transport unavailable"):
        rolling.run_rolling(args)
    assert not fits and not llm.contexts
    args.resume = True
    with pytest.raises(RuntimeError, match="LLM unavailable"):
        rolling.run_rolling(args)
    frozen = (args.output / "2026-07/prediction.json").read_bytes()
    result = rolling.run_rolling(args)
    assert result["status"] == "completed"
    assert len(acquisitions) == 2 and len(fits) == 4 and len(llm.contexts) == 3
    assert (args.output / "2026-07/prediction.json").read_bytes() == frozen
    assert all(fold["candidate_status"] == "fitted" for fold in result["folds"])


def test_removing_an_eligible_headline_changes_only_adaptive_feature_inputs(retained, tmp_path):
    decision = proposal()
    # Keep the same valid recipe and citations when removing an uncited matching article.
    decision["news_inputs"][0]["evidence_ids"] = ["first", "third"]
    decision["candidates"][0]["evidence_ids"] = ["first", "third"]
    original, original_cache, _ = execute(
        retained, tmp_path / "original", decisions=[decision], last="2026-07"
    )
    reduced = history()
    reduced["items"] = [item for item in reduced["items"] if item["id"] != "second"]
    changed, changed_cache, _ = execute(
        retained, tmp_path / "removed", news=reduced, decisions=[decision], last="2026-07"
    )
    assert original[0]["candidate_status"] == changed[0]["candidate_status"] == "fitted"
    assert original_cache.calls["2026-07_fixed_fit"] == changed_cache.calls["2026-07_fixed_fit"]
    before = np.asarray(original_cache.calls["2026-07_adaptive_fit"]["x"])
    after = np.asarray(changed_cache.calls["2026-07_adaptive_fit"]["x"])
    np.testing.assert_array_equal(before[:, :-1], after[:, :-1])
    assert np.any(before[:, -1] != after[:, -1])
    np.testing.assert_array_equal(
        before[:, -1] - after[:, -1], (before[:, -1] > after[:, -1]).astype(float)
    )
    assert original[0]["forecasts"]["fixed"] == changed[0]["forecasts"]["fixed"]


def test_incomplete_news_query_coverage_rejects_candidate_instead_of_inventing_zero(
    retained, tmp_path
):
    missing = history()
    missing["searches"] = [
        interval for interval in missing["searches"] if interval["start"] != "2026-05-01"
    ]
    folds, cache, _ = execute(retained, tmp_path / "missing", news=missing, last="2026-07")
    fold = folds[0]
    assert fold["candidate_status"] == "rejected"
    assert "unavailable" in fold["candidate_error"]
    assert "2026-07_adaptive_fit" not in cache.calls
    diagnostic = fold["news_features"][0]
    missing_rows = [
        i for i, status in enumerate(diagnostic["row_status"]) if status == "missing_query_coverage"
    ]
    assert missing_rows
    assert all(diagnostic["values"][i] is None for i in missing_rows)
    assert fold["forecasts"]["adaptive"] == fold["forecasts"]["fixed"]


def test_cancelling_news_binding_cannot_smuggle_price_only_feature_into_news_arm(
    retained, tmp_path
):
    decision = proposal()
    feature = decision["candidates"][0]
    feature["inputs"].append({"name": "price", "factor_id": "price.close"})
    feature["expression"]["nodes"] = [
        {"name": "news", "op": "leaf", "leaf": "pressure"},
        {"name": "price", "op": "leaf", "leaf": "price"},
        {"name": "zero", "op": "subtract", "inputs": ["news", "news"]},
        {"name": "product", "op": "multiply", "inputs": ["zero", "price"]},
        {"name": "result", "op": "add", "inputs": ["product", "price"]},
    ]
    folds, cache, _ = execute(
        retained, tmp_path / "cancelled-news", decisions=[decision], last="2026-07"
    )
    assert folds[0]["candidate_status"] == "rejected"
    assert "does not numerically depend" in folds[0]["candidate_error"]
    assert "2026-07_adaptive_fit" not in cache.calls
    assert folds[0]["forecasts"]["adaptive"] == folds[0]["forecasts"]["fixed"]


def test_news_execution_report_shows_actual_inputs_and_evidence(retained, tmp_path):
    from quant_rsi.research.rolling_report import render_rolling

    folds, _, _ = execute(retained, tmp_path / "report", last="2026-07")
    html = render_rolling({"symbol": "SOXX", "folds": folds, "status": "completed"})
    assert "1 of 1 monthly models used executable news inputs" in html
    assert "news.export_pressure" in html
    assert "Distinct training values" in html
    assert "Chip export restrictions" in html
    assert "2026-11-04T18:00:00+00:00" in html
    assert "not observed revenue" in html
