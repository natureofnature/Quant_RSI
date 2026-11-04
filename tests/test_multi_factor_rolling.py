"""Joint factor pools exercise the real monthly runner; fixtures do not establish alpha."""

import argparse
import json
from copy import deepcopy
from datetime import datetime
from email.utils import format_datetime

import pytest
import test_monthly_rolling as monthly
import test_rolling_news_integration as news_tests
from test_rolling_news_sources import receipt, rss

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research import rolling
from quant_rsi.research.discovery_factors import DiscoveryFactor
from quant_rsi.research.rolling_news_sources import collect_history, load_history
from quant_rsi.research.rolling_research import admit_decision, propose
from quant_rsi.research.rolling_windows import monthly_plan

retained = monthly.retained


def pool():
    history = news_tests.history()
    inputs, candidates = [], []
    for i in range(6):
        item = deepcopy(history["items"][0])
        item.update(
            id=f"topic{i}",
            title=f"Semiconductor topic{i} announcement",
            published_at=f"2026-03-{2 + i * 3:02d}T12:00:00+00:00",
        )
        history["items"].append(item)
        second = deepcopy(item)
        second.update(id=f"topic{i}_second", published_at=f"2026-04-{2 + i * 2:02d}T12:00:00+00:00")
        history["items"].append(second)
        recipe = deepcopy(news_tests.proposal()["news_inputs"][0])
        recipe.update(
            input_id=f"news.topic{i}", terms=[f"topic{i}"], window_days=7, evidence_ids=[item["id"]]
        )
        inputs.append(recipe)
        for lag in (0, 3):
            c = deepcopy(news_tests.proposal()["candidates"][0])
            c.update(
                name=f"topic{i}_lag{lag}",
                inputs=[{"name": "topic", "factor_id": recipe["input_id"]}],
                evidence_ids=[item["id"]],
            )
            nodes = [{"name": "source", "op": "leaf", "leaf": "topic"}]
            if lag:
                nodes.append({"name": "result", "op": "lag", "inputs": ["source"], "window": lag})
            c["expression"] = {"schema_version": 2, "nodes": nodes}
            candidates.append(c)
    decision = {
        "action": "replace",
        "basis": "news",
        "reason": "Joint evidence-linked pool",
        "candidates": candidates,
        "news_inputs": inputs,
    }
    return history, decision


def execute(retained_data, root, history, decision, limit=16):
    _, data, info, calendar = retained_data
    root.mkdir()
    cache = monthly.MemoryCache()
    folds = rolling.run_months(
        data,
        info,
        {},
        monthly_plan(data, calendar, "2026-07", "2026-07"),
        cache,
        root,
        model="ridge",
        llm=monthly.DecisionLLM([decision]),
        news_history=history,
        news_required=True,
        max_candidates=limit,
        max_news_inputs=12,
    )
    return folds[0], cache


def test_twelve_columns_reach_joint_fit_despite_invalid_pool_members(retained, tmp_path):
    history, decision = pool()
    invalid = deepcopy(decision["candidates"][0])
    invalid["name"] = "unsupported"
    invalid["expression"]["nodes"][0]["op"] = "unsupported_operator"
    decision["candidates"].append(invalid)
    decision["news_inputs"].append({"input_id": "news.broken"})
    fold, cache = execute(retained, tmp_path / "study", history, decision)
    assert fold["candidate_status"] == "fitted_with_rejections"
    assert fold["candidate_pool"] == {"proposed": 13, "accepted": 12, "limit": 16}
    fit = cache.calls["2026-07_adaptive_fit"]
    assert len(fit["factor_ids"]) == 14  # Two fixed features plus twelve candidates.
    assert all(len(row) == 14 for row in fit["x"])
    assert len(fold["candidate_news_inputs"]) == 6
    assert any(r["name"] == "unsupported" for r in fold["candidate_rejections"])
    assert any(r["name"] == "news.broken" for r in fold["candidate_rejections"])


def test_pool_pruning_and_predictions_ignore_future_news(retained, tmp_path):
    history, decision = pool()
    first, cache = execute(retained, tmp_path / "before", history, decision)
    late = deepcopy(history["items"][0])
    late.update(
        id="future_pool",
        title="topic0 topic1 topic2 topic3 topic4 topic5",
        published_at="2026-07-03T12:00:00+00:00",
    )
    history["items"].append(late)
    second, changed = execute(retained, tmp_path / "after", history, decision)
    assert first["prediction_sha256"] == second["prediction_sha256"]
    assert cache.calls["2026-07_adaptive_fit"] == changed.calls["2026-07_adaptive_fit"]


@pytest.mark.parametrize("change", ["retain", "add", "stale_parent", "invented_parent"])
def test_only_unchanged_incumbents_preserve_validated_ancestry(retained, tmp_path, change):
    _, data, info, calendar = retained
    history, pool_decision = pool()
    for article in history["items"]:
        if article["id"] == "topic0_second":
            article["published_at"] = "2026-05-15T12:00:00+00:00"
    initial = deepcopy(pool_decision)
    initial["candidates"] = initial["candidates"][:1]
    initial["news_inputs"] = initial["news_inputs"][:1]
    revised = deepcopy(initial)
    revised["candidates"] = [deepcopy(pool_decision["candidates"][1])]
    ancestor = DiscoveryFactor.model_validate_json(canonical(initial["candidates"][0])).recipe_ref
    revised["candidates"][0]["parent_recipe_refs"] = [ancestor]
    final_decision = {
        "action": "retain",
        "basis": "news",
        "reason": "Retain the admitted revised hypothesis",
        "candidates": [],
        "news_inputs": [],
    }
    if change != "retain":
        final_decision = deepcopy(revised)
        extra = deepcopy(revised["candidates"][0])
        extra["name"] = "extra_news_lag"
        extra["expression"]["nodes"][-1]["window"] = 7
        extra["parent_recipe_refs"] = [
            DiscoveryFactor.model_validate_json(canonical(revised["candidates"][0])).recipe_ref
        ]
        final_decision["candidates"].append(extra)
        if change in {"stale_parent", "invented_parent"}:
            # Even unchanged numerical recipes must not inherit trust after metadata edits.
            final_decision["candidates"][0]["description"] = "Modified business explanation"
            if change == "invented_parent":
                final_decision["candidates"][0]["parent_recipe_refs"] = ["f" * 64]
    cache = monthly.MemoryCache()
    folds = rolling.run_months(
        data,
        info,
        {},
        monthly_plan(data, calendar, "2026-07", "2026-09"),
        cache,
        tmp_path,
        model="ridge",
        llm=monthly.DecisionLLM([initial, revised, final_decision]),
        news_history=history,
        news_required=True,
    )
    rejected = change in {"stale_parent", "invented_parent"}
    assert [fold["candidate_status"] for fold in folds] == [
        "fitted",
        "fitted",
        "fitted_with_rejections" if rejected else "fitted",
    ]
    expected_names = ["extra_news_lag"] if rejected else ["topic0_lag3"]
    if change == "add":
        expected_names.append("extra_news_lag")
    assert [value["name"] for value in folds[2]["candidate_recipes"]] == expected_names
    if rejected:
        assert "unknown parent" in folds[2]["candidate_rejections"][0]["reason"]
    else:
        assert folds[2]["candidate_recipes"][0] == folds[1]["candidate_recipes"][0]
    assert "2026-09_adaptive_fit" in cache.calls
    assert folds[2]["fits"]["adaptive"]["factor_ids"][2:] == [
        f"discovery.{name}" for name in expected_names
    ]


def test_configured_limits_reach_provider_schema_and_local_admission():
    history, decision = pool()
    news = {"items": history["items"], "research_required": True}

    class Provider:
        def structured(self, prompt, schema):
            assert schema["properties"]["candidates"]["maxItems"] == 16
            assert schema["properties"]["news_inputs"]["maxItems"] == 12
            return json.dumps(decision)

    raw = propose(Provider(), {"max_candidates": 16, "max_news_inputs": 12})
    _, candidates = admit_decision(raw, [], news, 16, 12, screen_individually=True)
    assert len(candidates) == 12
    with pytest.raises(ValueError, match="budget"):
        admit_decision(raw, [], news, 2, 4, screen_individually=True)
    with pytest.raises(ValueError, match="budget"):
        admit_decision(
            {"text": canonical(decision).decode()}, [], news, 16, 4, screen_individually=True
        )


def test_cli_replays_identical_corpus_and_freezes_factor_limits(retained, tmp_path, monkeypatch):
    directory, _, _, _ = retained
    supplied, decision = pool()
    corpus = tmp_path / "original-corpus"
    rows = [
        (item["id"], item["title"], format_datetime(datetime.fromisoformat(item["published_at"])))
        for item in supplied["items"]
    ]
    frozen = collect_history(
        "SOXX",
        "2026-01-01",
        "2026-06-30",
        corpus,
        fetcher=lambda source, _: (rss(rows), receipt(source.url)),
    )
    # Provider RSS IDs are content identities, not the labels in our fixture.
    for value in decision["news_inputs"] + decision["candidates"]:
        value["evidence_ids"] = [
            "google-news-" + digest(canonical(x))[:32] for x in value["evidence_ids"]
        ]
    original_bytes = {
        p.relative_to(corpus): p.read_bytes() for p in corpus.rglob("*") if p.is_file()
    }
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
            "2026-07",
            "--model",
            "ridge",
            "--llm",
            "codex",
            "--news-research",
            "--news-history",
            str(corpus),
            "--max-candidates",
            "16",
            "--max-news-inputs",
            "12",
            "--output",
            str(tmp_path / "replayed"),
        ]
    )
    llm = monthly.DecisionLLM([decision])

    def no_fetch(*a, **kw):
        raise AssertionError("Replayed news must not refetch historical evidence")

    monkeypatch.setattr(rolling, "collect_history", no_fetch)
    monkeypatch.setattr("quant_rsi.market.news_feed.fetch", no_fetch)
    monkeypatch.setattr(rolling, "LLMHypothesizer", lambda *a, **kw: llm)
    result = rolling.run_rolling(args)
    assert result["status"] == "completed"
    assert load_history(args.output / "news_history", "SOXX", "2026-01-01", "2026-06-30") == frozen
    assert {
        p.relative_to(corpus): p.read_bytes() for p in corpus.rglob("*") if p.is_file()
    } == original_bytes
    config = result["configuration"]
    assert config["input_identity"]["history_sha256"] == digest(canonical(frozen))
    assert config["max_candidates"] == 16 and config["effective_max_news_inputs"] == 12
    fold = result["folds"][0]
    assert fold["candidate_pool"] == {"proposed": 12, "accepted": 12, "limit": 16}
    fit = fold["fits"]["adaptive"]
    assert len(fit["factor_ids"]) == 14
    assert set(fit["feature_usage"]) == set(fit["factor_ids"])
    assert all("standardized_coefficient" in usage for usage in fit["feature_usage"].values())
    html = (args.output / "report.html").read_text()
    assert "Actual model columns" in html and "Actual model feature usage" in html
    assert "discovery.topic0_lag0" in html and "standardized_coefficient" in html
    persisted = (args.output / "config.json").read_bytes()
    args.resume = True
    for field, changed in (("max_candidates", 15), ("max_news_inputs", 11)):
        original = getattr(args, field)
        setattr(args, field, changed)
        with pytest.raises(ValueError, match="configuration changed"):
            rolling.run_rolling(args)
        setattr(args, field, original)
    assert len(llm.contexts) == 1
    assert (args.output / "config.json").read_bytes() == persisted
