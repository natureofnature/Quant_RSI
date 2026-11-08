"""Business provenance is checked before a fit without changing numerical identity."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research.discovery_contracts import guard_records
from quant_rsi.research.discovery_engine import DiscoveryExperimentRunner
from quant_rsi.research.discovery_factors import DiscoveryFactor


def setup_candidate():
    articles = {
        "payer": {"text": "Retailer funds orders from distributor with operating cash flow."},
        "supplier": {"text": "Distributor purchases beverages from producer."},
        "numbers": {"text": "Reported operating cash flow is 120 million USD."},
    }
    links = [
        {
            "from_entity": "PAY",
            "to_entity": "DIST",
            "relationship": "Retailer funds orders",
            "source_id": "payer",
            "excerpt": articles["payer"]["text"],
        },
        {
            "from_entity": "DIST",
            "to_entity": "PROD",
            "relationship": "Distributor buys beverages",
            "source_id": "supplier",
            "excerpt": articles["supplier"]["text"],
        },
    ]
    candidate = {
        "name": "payer_cashflow",
        "description": "A proposed downstream demand channel, not verified predictive value.",
        "inputs": [{"name": "cash", "factor_id": "acquired.payer_cashflow"}],
        "expression": {"schema_version": 2, "nodes": [{"name": "x", "op": "leaf", "leaf": "cash"}]},
        "evidence_ids": list(articles),
        "hypothesis": {
            "operating_mechanism": "Downstream funding may support future supplier orders.",
            "measurement_horizon": "Reported quarter, no YTD conversion",
            "forecast_channel": "Disclosed funding changes could revise near-term expectations.",
            "forecast_horizon_sessions": 5,
            "relationships": links,
        },
    }
    bundles = [
        {
            "request": {"symbol": "PAY", "input_id": "acquired.payer_cashflow"},
            "observations": [{"source_id": "numbers"}],
        }
    ]
    engine = object.__new__(DiscoveryExperimentRunner)
    engine.target_symbol = "PROD"
    engine.evidence_ids = set(articles)
    engine.evidence_articles = articles
    engine.model = SimpleNamespace(
        horizon=5,
        train=np.array([0, 1]),
        origins=np.array([2]),
        data=SimpleNamespace(dates=["2025-01-01", "2025-01-02", "2025-01-03"]),
    )
    return candidate, bundles, engine


def test_multihop_quarterly_measure_can_support_a_short_forecast_without_retargeting():
    candidate, bundles, engine = setup_candidate()
    calls = []
    engine.prepare = lambda *args: (calls.append(args) or {}, np.array([1.0, 2.0, 3.0]), None)
    assert engine.validate(candidate, bundles)["status"] == "ready"
    assert len(calls) == 1
    assert engine.model.horizon == 5


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("remove_hypothesis", "cross_entity_hypothesis_required"),
        ("no_channel", "forecast_channel_unspecified"),
        ("other_horizon", "hypothesis_forecast_horizon_mismatch"),
        ("missing_link", "cross_entity_relationship_missing"),
        ("unquoted_link", "relationship_quote_not_in_source"),
        ("blank_quote", "relationship_quote_too_short"),
        ("uncited_link", "relationship_source_not_cited_by_candidate"),
        ("uncited_measure", "acquired_measurement_source_not_cited"),
    ],
)
def test_incomplete_business_hypothesis_cannot_reach_numerical_preparation(change, reason):
    candidate, bundles, engine = setup_candidate()
    if change == "remove_hypothesis":
        candidate.pop("hypothesis")
    elif change == "no_channel":
        candidate["hypothesis"]["forecast_channel"] = None
    elif change == "other_horizon":
        candidate["hypothesis"]["forecast_horizon_sessions"] = 20
    elif change == "missing_link":
        candidate["hypothesis"]["relationships"].pop()
    elif change == "unquoted_link":
        candidate["hypothesis"]["relationships"][0]["excerpt"] = "Invented purchasing relationship."
    elif change == "blank_quote":
        candidate["hypothesis"]["relationships"][0]["excerpt"] = " \n " * 12
    elif change == "uncited_link":
        candidate["evidence_ids"].remove("payer")
    else:
        candidate["evidence_ids"].remove("numbers")
    engine.prepare = lambda *_: pytest.fail("Unqualified candidate reached numerical execution")
    assert engine.validate(candidate, bundles) == {"status": "needs_data", "reason": reason}


def test_future_relationship_quote_does_not_pass_retained_gate():
    candidate, bundles, engine = setup_candidate()
    records = [
        dict(
            row,
            id=identity,
            published_at="2026-01-01T00:00:00Z",
            received_at="2026-01-02T00:00:00Z",
            version_at="2026-01-02T00:00:00Z",
        )
        for identity, row in engine.evidence_articles.items()
    ]
    admitted, _ = guard_records(records, mode="retained", cutoff="2025-12-31T00:00:00Z")
    engine.evidence_articles = {r["id"]: r for r in admitted}
    engine.prepare = lambda *_: pytest.fail("Future relationship used for a historical hypothesis")
    assert engine.validate(candidate, bundles)["reason"] == "relationship_source_not_admitted"


def test_semantic_revision_keeps_recipe_but_changes_full_payload_identity():
    raw, _, _ = setup_candidate()
    legacy = deepcopy(raw)
    legacy.pop("hypothesis")
    revised = deepcopy(raw)
    revised["hypothesis"]["forecast_channel"] = "A different expectation-revision hypothesis."
    models = [DiscoveryFactor.model_validate_json(canonical(p)) for p in (legacy, raw, revised)]
    assert len({m.recipe_ref for m in models}) == 1
    assert len({digest(canonical(m.model_dump(mode="json"))) for m in models}) == 3
    assert "hypothesis" not in models[0].model_dump(mode="json")
