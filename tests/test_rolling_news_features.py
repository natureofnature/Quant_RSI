"""Offline headline counts verify mechanics, not sentiment or predictive quality."""

import json
from copy import deepcopy

import numpy as np
import pytest

from quant_rsi.research.discovery_factors import DiscoveryFactor, compute_factor
from quant_rsi.research.rolling_news_features import NewsInput, compile_news_inputs


def article(identifier, title, day, *, received="2026-11-04T12:00:00Z"):
    return {
        "id": identifier,
        "title": title,
        "url": f"https://example.test/{identifier}",
        "published_at": f"2026-06-{day:02d}T18:00:00Z",
        "received_at": received,
        "version_available_at": received,
    }


def recipe(**updates):
    return {
        "input_id": "news.policy_risk",
        "hypothesis": "Frequency of chip export-restriction headlines may describe policy risk",
        "operation": "count",
        "terms": ["export restrictions", "chip ban"],
        "negative_terms": [],
        "window_days": 3,
        "evidence_ids": ["a"],
    } | updates


def test_reconstructed_counts_are_causal_distinct_and_auditable():
    a = article("a", "Chip ban adds export restrictions", 2)
    b = article("b", "Chip ban concerns rise", 4)
    future = article("future", "Chip ban lifted", 8)
    original = deepcopy(a)
    times = [f"2026-06-{day:02d}T20:00:00Z" for day in range(1, 7)]
    cols, definitions, audit = compile_news_inputs([recipe()], [a, a, b, future], times)
    assert cols["news.policy_risk"].tolist() == [0, 1, 1, 2, 1, 1]
    assert audit[0]["row_evidence_ids"][3] == ["a", "b"]
    assert audit[0]["distinct_values"] == 3
    assert audit[0]["matched_articles"][0]["received_at"] == "2026-11-04T12:00:00Z"
    assert "zero is not evidence of no event" in audit[0]["coverage"]
    assert a == original
    assert definitions[0]["unit"] == "count"


def test_retained_counts_wait_for_actual_receipt_and_version():
    a = article("a", "Chip ban concern", 2, received="2026-06-03T18:00:00Z")
    a["version_available_at"] = "2026-06-04T18:00:00Z"
    times = [f"2026-06-{day:02d}T20:00:00Z" for day in (2, 3, 4)]
    cols, _, audit = compile_news_inputs([recipe()], [a], times, mode="retained")
    assert cols["news.policy_risk"].tolist() == [0, 0, 1]
    assert audit[0]["availability_basis"] == "publication_receipt_and_version"


def test_balance_is_literal_baskets_not_sentiment_and_matches_whole_words():
    rows = [
        article("a", "Chip demand grows", 2),
        article("b", "Chip demand falls", 3),
        article("c", "Demand grows but chip margin falls", 4),
        article("d", "Snowfalls in semiconductor city", 4),
    ]
    spec = recipe(operation="balance", terms=["grows"], negative_terms=["falls"])
    cols, _, audit = compile_news_inputs(
        [spec], rows, [f"2026-06-{d:02d}T20:00:00Z" for d in (2, 3, 4)]
    )
    assert cols[spec["input_id"]].tolist() == [1, 0, 0]
    assert audit[0]["row_evidence_ids"][-1] == ["a", "b", "c"]
    assert audit[0]["measurement"] == "retrieved_positive_minus_negative_headline_match_count"


def test_future_news_perturbation_cannot_change_earlier_rows():
    times = [f"2026-06-{d:02d}T20:00:00Z" for d in (2, 3, 4)]
    rows = [article("a", "Chip ban", 2), article("b", "Chip ban", 4)]
    before, _, _ = compile_news_inputs([recipe()], rows, times)
    rows[1]["title"] = "Unrelated sport result"
    after, _, _ = compile_news_inputs([recipe()], rows, times)
    np.testing.assert_equal(before["news.policy_risk"][:2], after["news.policy_risk"][:2])
    assert before["news.policy_risk"][-1] != after["news.policy_risk"][-1]


@pytest.mark.parametrize(
    "updates",
    [
        {"terms": ["chip ban", "CHIP BAN"]},
        {"operation": "balance", "negative_terms": []},
        {"negative_terms": ["rally"]},
        {"terms": ["   "]},
        {"window_days": 0},
        {"input_id": "price.fake_news"},
    ],
)
def test_recipe_rejects_ambiguous_baskets(updates):
    with pytest.raises(ValueError):
        NewsInput.model_validate_json(json.dumps(recipe(**updates)))


def test_evidence_requires_actual_matching_cutoff_eligible_headlines():
    end = ["2026-06-04T20:00:00Z"]
    with pytest.raises(ValueError, match="unknown or cutoff-ineligible"):
        compile_news_inputs([recipe()], [article("a", "Chip ban", 5)], end)
    with pytest.raises(ValueError, match="unknown or cutoff-ineligible"):
        compile_news_inputs([recipe()], [article("a", "Chip ban", 2)], end, mode="retained")
    with pytest.raises(ValueError, match="must match"):
        compile_news_inputs([recipe()], [article("a", "Unrelated sport result", 2)], end)
    with pytest.raises(ValueError, match="Duplicate news input"):
        compile_news_inputs([recipe(), recipe()], [article("a", "Chip ban", 2)], end)


def test_discovery_expression_executes_news_columns():
    times = [f"2026-06-{d:02d}T20:00:00Z" for d in (1, 2, 3)]
    columns, definitions, _ = compile_news_inputs([recipe()], [article("a", "Chip ban", 2)], times)
    factor = DiscoveryFactor.model_validate_json(
        json.dumps(
            {
                "name": "lagged_policy_risk",
                "description": "A lagged news observation, not a price proxy",
                "inputs": [{"name": "news", "factor_id": "news.policy_risk"}],
                "expression": {
                    "schema_version": 2,
                    "nodes": [
                        {"name": "source", "op": "leaf", "leaf": "news"},
                        {"name": "result", "op": "lag", "inputs": ["source"], "window": 1},
                    ],
                },
                "evidence_ids": ["a"],
            }
        )
    )
    values = compute_factor(factor.definition(definitions), columns, definitions)
    assert np.isnan(values[0])
    assert values[1:].tolist() == [0, 1]


def test_conflicting_article_versions_and_invalid_clocks_fail():
    a = article("a", "Chip ban", 2)
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        compile_news_inputs([recipe()], [a, a | {"title": "Rewritten"}], ["2026-06-04T20:00Z"])
    with pytest.raises(ValueError, match="strictly increasing"):
        compile_news_inputs([recipe()], [a], ["2026-06-04T20:00Z", "2026-06-03T20:00Z"])
    with pytest.raises(ValueError, match="timezone"):
        compile_news_inputs([recipe()], [a], ["2026-06-04"])


def test_history_boundary_and_unqueried_days_are_unavailable_not_zero():
    times = [f"2026-06-{d:02d}T20:00:00Z" for d in (2, 3, 4, 5)]
    coverage = [
        {"start": "2026-06-01", "end": "2026-06-04", "status": "partial_results"},
        {"start": "2026-06-05", "end": "2026-06-05", "status": "failed"},
    ]
    columns, _, audit = compile_news_inputs(
        [recipe(window_days=2)],
        [article("a", "Chip ban", 2)],
        times,
        history_start="2026-06-01T00:00:00Z",
        coverage=coverage,
    )
    np.testing.assert_equal(columns["news.policy_risk"], [np.nan, 1, 0, np.nan])
    assert audit[0]["values"] == [None, 1, 0, None]
    assert audit[0]["row_status"] == [
        "before_history_start",
        "observed_corpus",
        "observed_corpus",
        "missing_query_coverage",
    ]
    assert audit[0]["unavailable_rows"] == 2
    assert audit[0]["distinct_values"] == 2
    assert audit[0]["nonzero_rows"] == 1
    json.dumps(audit, allow_nan=False)


def test_successfully_queried_empty_interval_is_observed_corpus_zero():
    coverage = [
        {"start": "2026-06-01", "end": "2026-06-03", "status": "partial_results"},
        {"start": "2026-06-04", "end": "2026-06-30", "status": "empty_search_results"},
    ]
    columns, _, audit = compile_news_inputs(
        [recipe(window_days=2)],
        [article("a", "Chip ban", 2)],
        ["2026-06-06T20:00:00Z"],
        coverage=coverage,
    )
    assert columns["news.policy_risk"].tolist() == [0]
    assert audit[0]["row_status"] == ["observed_corpus"]
    assert audit[0]["query_coverage_checked"]


def test_unqueried_month_gap_remains_nan_despite_matches():
    columns, _, audit = compile_news_inputs(
        [recipe()],
        [article("a", "Chip ban", 2)],
        ["2026-06-04T20:00:00Z"],
        coverage=[{"start": "2026-05-01", "end": "2026-05-31", "status": "partial_results"}],
    )
    assert np.isnan(columns["news.policy_risk"][0])
    assert audit[0]["row_evidence_ids"] == [[]]
