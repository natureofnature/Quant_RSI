"""Candidate-pool screening checks mechanics, not predictive usefulness."""

from copy import deepcopy

import numpy as np
import pytest

from quant_rsi.research.rolling_candidates import screen_candidates


@pytest.fixture
def inputs():
    times = [f"2026-06-{day:02d}T20:00:00Z" for day in range(1, 17)]
    articles = [
        {
            "id": identity,
            "title": title,
            "published_at": f"2026-06-{day:02d}T12:00:00Z",
            "received_at": "2026-11-04T12:00:00Z",
            "version_available_at": "2026-11-04T12:00:00Z",
        }
        for identity, title, day in (
            ("p1", "Chip export ban", 4),
            ("p2", "More chip export ban news", 8),
            ("d1", "Chip demand grows", 6),
            ("d2", "Chip demand grows again", 7),
            ("old", "Old headline", 2),
        )
    ]
    return {
        "columns": {"base": np.arange(16, dtype=float) ** 2, "price.close": np.arange(16) + 20.0},
        "definitions": [
            {"factor_id": "base", "unit": "dimensionless", "runnable": True},
            {"factor_id": "price.close", "unit": "price", "runnable": True},
        ],
        "baseline": ["base"],
        "news": {"items": articles},
        "times": times,
        "train": np.arange(4, 13),
        "news_required": True,
        "coverage": [{"start": "2026-06-01", "end": "2026-06-30", "status": "partial_results"}],
    }


def news_input(identity="news.policy", term="export ban", evidence="p1"):
    return {
        "input_id": identity,
        "hypothesis": "A lexical news activity hypothesis",
        "operation": "count",
        "terms": [term],
        "negative_terms": [],
        "window_days": 2,
        "evidence_ids": [evidence],
    }


def candidate(name="policy", identity="news.policy", evidence="p1"):
    return {
        "name": name,
        "description": "Use the numerical input",
        "inputs": [{"name": "source", "factor_id": identity}],
        "expression": {
            "schema_version": 2,
            "nodes": [{"name": "result", "op": "leaf", "leaf": "source"}],
        },
        "evidence_ids": [evidence] if evidence else [],
    }


def test_multiple_news_factors_survive_one_constant_candidate(inputs):
    candidates = [
        candidate(),
        candidate("old", "news.old", "old"),
        candidate("demand", "news.demand", "d1"),
    ]
    recipes = [
        news_input(),
        news_input("news.old", "old headline", "old"),
        news_input("news.demand", "demand grows", "d1"),
    ]
    original = deepcopy(candidates), deepcopy(recipes)
    result = screen_candidates(candidates, recipes, **inputs)
    assert [value["name"] for value in result["candidates"]] == ["policy", "demand"]
    assert result["ids"] == ["base", "discovery.policy", "discovery.demand"]
    assert result["matrix"].shape == (16, 3)
    np.testing.assert_array_equal(result["matrix"][:, 0], inputs["columns"]["base"])
    assert {value["input_id"] for value in result["news_inputs"]} == {"news.policy", "news.demand"}
    assert {value["input_id"] for value in result["diagnostics"] if value["used_by_model"]} == {
        "news.policy",
        "news.demand",
    }
    assert any(
        value["input_id"] == "news.old" and not value["used_by_model"]
        for value in result["diagnostics"]
    )
    assert any("no training variation" in value["reason"] for value in result["rejections"])
    assert (candidates, recipes) == original


def test_duplicate_aliases_keep_first_valid_column(inputs):
    result = screen_candidates(
        [candidate(), candidate("alias", "news.alias")],
        [news_input(), news_input("news.alias")],
        **inputs,
    )
    assert len(result["candidates"]) == 1
    assert list(result["news_columns"]) == ["news.policy"]
    assert any(
        "duplicates an existing model column" in value["reason"] for value in result["rejections"]
    )


def test_shared_identical_input_is_compiled_for_distinct_expressions(inputs):
    lagged = candidate("lagged")
    lagged["expression"]["nodes"] = [
        {"name": "source_node", "op": "leaf", "leaf": "source"},
        {"name": "result", "op": "lag", "inputs": ["source_node"], "window": 1},
    ]
    result = screen_candidates([candidate(), lagged], [news_input(), news_input()], **inputs)
    assert len(result["candidates"]) == 2
    assert len(result["news_inputs"]) == len(result["diagnostics"]) == 1
    assert not result["rejections"]


def test_conflicting_shared_input_rejects_only_its_dependents(inputs):
    result = screen_candidates(
        [candidate(), candidate("demand", "news.demand", "d1")],
        [news_input(), news_input(term="chip"), news_input("news.demand", "demand grows", "d1")],
        **inputs,
    )
    assert [value["name"] for value in result["candidates"]] == ["demand"]
    assert any("conflicting news input" in value["reason"] for value in result["rejections"])


def test_cancelled_news_is_rejected_but_other_candidate_survives(inputs):
    cancelled = candidate("cancelled")
    cancelled["inputs"].append({"name": "price", "factor_id": "price.close"})
    cancelled["expression"]["nodes"] = [
        {"name": "n", "op": "leaf", "leaf": "source"},
        {"name": "p", "op": "leaf", "leaf": "price"},
        {"name": "zero", "op": "subtract", "inputs": ["n", "n"]},
        {"name": "scaled", "op": "multiply", "inputs": ["zero", "p"]},
        {"name": "result", "op": "add", "inputs": ["scaled", "p"]},
    ]
    result = screen_candidates([cancelled, candidate()], [news_input()], **inputs)
    assert [value["name"] for value in result["candidates"]] == ["policy"]
    assert "does not numerically depend" in result["rejections"][0]["reason"]


def test_missing_input_or_evidence_does_not_discard_valid_candidate(inputs):
    result = screen_candidates(
        [
            candidate("missing", "news.absent"),
            candidate("bad_evidence", evidence="future"),
            candidate(),
        ],
        [news_input()],
        **inputs,
    )
    assert [value["name"] for value in result["candidates"]] == ["policy"]
    assert len(result["rejections"]) == 2
    assert "undeclared" in result["rejections"][0]["reason"]
    assert "cutoff-ineligible" in result["rejections"][1]["reason"]


def test_price_only_mode_and_empty_pool_preserve_baseline(inputs):
    kwargs = inputs | {"news_required": False, "news": None}
    price = candidate("price", "price.close", evidence=None)
    result = screen_candidates([price], [], **kwargs)
    assert result["ids"] == ["base", "discovery.price"]
    assert not result["news_inputs"] and not result["rejections"]
    empty = screen_candidates([], [], **kwargs)
    assert empty["ids"] == ["base"] and empty["matrix"].shape == (16, 1)
    rejected = screen_candidates([price], [], **inputs)
    assert not rejected["candidates"]
    assert "actual news input" in rejected["rejections"][0]["reason"]


def test_unavailable_required_rows_are_not_dropped(inputs):
    result = screen_candidates([candidate()], [news_input()], **(inputs | {"coverage": []}))
    assert not result["candidates"]
    assert result["matrix"].shape == (16, 1)
    assert "unavailable on fixed sample" in result["rejections"][0]["reason"]


def test_candidate_parent_and_evidence_checks_are_independent(inputs):
    unknown_parent = candidate("unknown_parent") | {"parent_recipe_refs": ["a" * 64]}
    no_evidence = candidate("no_evidence", evidence=None)
    valid = candidate() | {"parent_recipe_refs": ["b" * 64]}
    result = screen_candidates(
        [unknown_parent, no_evidence, valid],
        [news_input()],
        **inputs,
        allowed_parent_refs=["b" * 64],
        require_evidence=True,
    )
    assert [value["name"] for value in result["candidates"]] == ["policy"]
    assert "unknown parent" in result["rejections"][0]["reason"]
    assert "requires cutoff-eligible evidence" in result["rejections"][1]["reason"]


def test_malformed_candidate_and_news_declarations_do_not_discard_valid_pool(inputs):
    malformed_candidate = candidate("invalid_binding")
    malformed_candidate["inputs"] = [{"name": "source"}]
    malformed_news = news_input("news.invalid")
    malformed_news["window_days"] = 0
    unnamed_news = news_input()
    del unnamed_news["input_id"]
    result = screen_candidates(
        [malformed_candidate, candidate("invalid_source", "news.invalid"), candidate()],
        [malformed_news, unnamed_news, news_input()],
        **inputs,
    )
    assert [value["name"] for value in result["candidates"]] == ["policy"]
    assert result["matrix"].shape == (16, 2)
    assert result["news_inputs"] == [news_input()]
    rejected = {item["name"]: item for item in result["rejections"]}
    assert rejected["news.invalid"]["kind"] == "news_input"
    assert rejected["news_input_2"]["kind"] == "news_input"
    assert rejected["invalid_binding"]["kind"] == "candidate"
    assert "undeclared news input" in rejected["invalid_source"]["reason"]


def test_sixteen_news_columns_can_enter_one_unchanged_sample():
    times = [f"2026-06-{day:02d}T20:00:00Z" for day in range(1, 31)]
    recipes, proposals, articles = [], [], []
    for index in range(16):
        topic, identifier = f"topic{index:02d}", f"article{index:02d}"
        source_id = f"news.{topic}"
        recipes.append(news_input(source_id, topic, identifier) | {"window_days": 1})
        proposals.append(candidate(topic, source_id, identifier))
        articles.append(
            {
                "id": identifier,
                "title": topic,
                "published_at": f"2026-06-{index + 3:02d}T12:00:00Z",
                "received_at": "2026-11-04T12:00:00Z",
            }
        )
    result = screen_candidates(
        proposals,
        recipes,
        {"base": np.arange(30, dtype=float)},
        [{"factor_id": "base", "unit": "dimensionless"}],
        ["base"],
        {"items": articles},
        times,
        np.arange(2, 29),
        news_required=True,
        coverage=[{"start": "2026-06-01", "end": "2026-06-30", "status": "partial_results"}],
    )
    assert len(result["candidates"]) == len(result["news_inputs"]) == 16
    assert result["matrix"].shape == (30, 17)
    assert np.isfinite(result["matrix"][2:]).all()
    assert not result["rejections"]
