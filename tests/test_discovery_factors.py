"""Offline numerical and provenance checks; fixtures do not establish alpha."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.factors.expressions import (
    FactorExpression,
    SeriesFactorExpression,
    evaluate_expression,
    evaluate_series,
)
from quant_rsi.research.discovery_factors import (
    DiscoveryFactor,
    adapter_catalog,
    compute_factor,
    execute_survey,
    retrieve_candidates,
    survey_catalog,
)

ROOT = Path(__file__).resolve().parents[1]


def expression(op="rolling_mean", window=3):
    return SeriesFactorExpression.model_validate_json(
        json.dumps(
            {
                "schema_version": 2,
                "nodes": [
                    {"name": "a", "op": "leaf", "leaf": "source"},
                    {"name": "result", "op": op, "inputs": ["a"], "window": window},
                ],
            }
        )
    )


def proposal():
    return DiscoveryFactor.model_validate_json(
        json.dumps(
            {
                "name": "demand_deviation",
                "description": "Deviation of an acquired demand measure from its trailing mean.",
                "inputs": [{"name": "source", "factor_id": "acquired.demand"}],
                "expression": expression("rolling_zscore").model_dump(mode="json"),
                "evidence_ids": ["news-1"],
            }
        )
    )


def test_v1_serialization_and_identity_are_unchanged():
    payload = {
        "schema_version": 1,
        "nodes": [
            {"name": "a", "op": "leaf", "inputs": [], "leaf": "inventory"},
            {"name": "b", "op": "leaf", "inputs": [], "leaf": "revenue"},
            {"name": "gap", "op": "subtract", "inputs": ["a", "b"], "leaf": None},
        ],
    }
    legacy = FactorExpression.model_validate_json(json.dumps(payload))
    assert legacy.model_dump(mode="json") == payload
    assert legacy.ref == digest(
        canonical(["subtract", [["leaf", "inventory"], ["leaf", "revenue"]]])
    )
    assert evaluate_expression(legacy, {"inventory": 0.4, "revenue": 0.2}) == {
        "value": 0.2,
        "reasons": [],
    }
    with pytest.raises(ValidationError):
        FactorExpression.model_validate_json(expression().model_dump_json())


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("lag", [np.nan, np.nan, np.nan, 1, 2, 4]),
        ("rolling_mean", [np.nan, np.nan, 7 / 3, 14 / 3, 28 / 3, 56 / 3]),
        (
            "rolling_std",
            [
                np.nan,
                np.nan,
                np.std([1, 2, 4], ddof=1),
                np.std([2, 4, 8], ddof=1),
                np.std([4, 8, 16], ddof=1),
                np.std([8, 16, 32], ddof=1),
            ],
        ),
    ],
)
def test_temporal_parameter_semantics_and_no_future_access(op, expected):
    data = np.array([1.0, 2, 4, 8, 16, 32])
    result = evaluate_series(expression(op), {"source": data})
    np.testing.assert_allclose(result, expected, equal_nan=True)
    for split in range(1, len(data)):
        np.testing.assert_allclose(
            evaluate_series(expression(op), {"source": data[:split]}),
            result[:split],
            equal_nan=True,
        )
    changed = data.copy()
    changed[-1] = 1e20
    np.testing.assert_allclose(
        evaluate_series(expression(op), {"source": changed})[:-1], result[:-1], equal_nan=True
    )


def test_temporal_missingness_and_constant_zscore():
    data = np.array([1.0, 2, np.nan, 4, 5, 6])
    result = evaluate_series(expression(), {"source": data})
    assert np.isnan(result[:5]).all()
    assert result[-1] == 5
    assert np.isnan(evaluate_series(expression("rolling_zscore"), {"source": np.ones(8)})).all()
    with pytest.raises(ValueError, match="population"):
        evaluate_series(expression(), {"source": [1, 2], "another": [1]})
    with pytest.raises(ValueError, match="one-dimensional"):
        evaluate_series(expression(), {"source": [[1], [2]]})


@pytest.mark.parametrize("window", [0, -1, 253, True, 1.5])
def test_temporal_rejects_invalid_windows(window):
    with pytest.raises(ValidationError):
        expression(window=window)


def test_temporal_rejects_unknown_operator_unused_nodes_and_bad_arity():
    for op in ("eval", "future", "rolling_std"):
        with pytest.raises(ValidationError):
            expression(op, 1)
    payload = expression().model_dump(mode="json")
    payload["nodes"][-1]["inputs"] = ["a", "a"]
    with pytest.raises(ValidationError):
        SeriesFactorExpression.model_validate_json(json.dumps(payload))
    payload["nodes"][-1]["inputs"] = ["result"]
    with pytest.raises(ValidationError):
        SeriesFactorExpression.model_validate_json(json.dumps(payload))


def test_discovered_expression_binds_acquired_values_and_preserves_rows():
    candidate = proposal()
    definition = candidate.definition([])
    assert definition["missing_inputs"] == ["acquired.demand"]
    assert not definition["runnable"]
    definitions = [{"factor_id": "acquired.demand", "runnable": True, "unit": "tonnes"}]
    definition = candidate.definition(definitions)
    assert definition["unit"] == "dimensionless"
    values = np.array([1.0, 2, 4, 8, 16, 32])
    result = compute_factor(definition, {"acquired.demand": values}, definitions)
    assert result.shape == values.shape
    assert np.isnan(result[:2]).all()
    np.testing.assert_allclose(result[2:], [(4 - 7 / 3) / np.std([1, 2, 4], ddof=1)] * 4)
    assert definition["evidence_ids"] == ["news-1"]
    with pytest.raises(ValueError, match="needs input"):
        compute_factor(definition, {})


def test_recipe_identity_ignores_names_but_tracks_operator_and_input_changes():
    candidate = proposal()
    changed = candidate.model_dump(mode="json")
    changed["name"] = "renamed"
    changed["inputs"][0]["name"] = "alias"
    changed["expression"]["nodes"][0]["leaf"] = "alias"
    changed["expression"]["nodes"][0]["name"] = "renamed_leaf"
    changed["expression"]["nodes"][1]["inputs"] = ["renamed_leaf"]
    renamed = DiscoveryFactor.model_validate_json(json.dumps(changed))
    assert renamed.recipe_ref == candidate.recipe_ref
    changed["expression"]["nodes"][1]["window"] = 4
    assert (
        DiscoveryFactor.model_validate_json(json.dumps(changed)).recipe_ref != candidate.recipe_ref
    )
    changed = candidate.model_dump(mode="json")
    changed["inputs"][0]["factor_id"] = "acquired.supply"
    assert (
        DiscoveryFactor.model_validate_json(json.dumps(changed)).recipe_ref != candidate.recipe_ref
    )


def test_revision_parents_preserve_legacy_semantic_identity_and_numerical_definition():
    original = proposal()
    legacy = original.model_dump(mode="json", exclude={"parent_recipe_refs"})
    restored = DiscoveryFactor.model_validate_json(json.dumps(legacy))
    assert restored.parent_recipe_refs == ()
    assert "parent_recipe_refs" not in restored.definition([])

    revised = DiscoveryFactor.model_validate_json(
        json.dumps(legacy | {"parent_recipe_refs": ["a" * 64]})
    )
    assert revised.recipe_ref == restored.recipe_ref
    assert revised.parent_recipe_refs == ("a" * 64,)
    definition = revised.definition([])
    assert definition.pop("parent_recipe_refs") == ["a" * 64]
    assert definition == restored.definition([])


@pytest.mark.parametrize(
    "parents",
    [["a" * 63], ["A" * 64], [123], ["a" * 64, "a" * 64], [f"{i:064x}" for i in range(9)]],
)
def test_revision_parents_reject_malformed_duplicate_or_excessive_references(parents):
    payload = proposal().model_dump(mode="json") | {"parent_recipe_refs": parents}
    with pytest.raises(ValidationError):
        DiscoveryFactor.model_validate_json(json.dumps(payload))


def test_revision_parents_reject_self_reference():
    original = proposal()
    payload = original.model_dump(mode="json") | {"parent_recipe_refs": [original.recipe_ref]}
    with pytest.raises(ValidationError, match="own parent"):
        DiscoveryFactor.model_validate_json(json.dumps(payload))


def test_reference_only_sources_and_tampered_expressions_are_rejected():
    candidate = proposal()
    with pytest.raises(ValueError, match="Noncausal"):
        candidate.definition(
            [{"factor_id": "acquired.demand", "execution_status": "noncausal_reference_only"}]
        )
    definition = candidate.definition([{"factor_id": "acquired.demand", "runnable": True}])
    definition["recipe"]["expression"]["nodes"][-1]["window"] = 4
    with pytest.raises(ValueError, match="identity"):
        compute_factor(definition, {"acquired.demand": [1, 2, 3, 4]})


def test_binary_units_division_and_commutative_identity():
    payload = {
        "name": "relative_measure",
        "description": "Measured ratio",
        "inputs": [
            {"name": "a", "factor_id": "acquired.a"},
            {"name": "b", "factor_id": "acquired.b"},
        ],
        "expression": {
            "schema_version": 2,
            "nodes": [
                {"name": "a", "op": "leaf", "leaf": "a"},
                {"name": "b", "op": "leaf", "leaf": "b"},
                {"name": "c", "op": "divide", "inputs": ["a", "b"]},
            ],
        },
    }
    definitions = [
        {"factor_id": "acquired.a", "unit": "USD"},
        {"factor_id": "acquired.b", "unit": "tonnes"},
    ]
    candidate = DiscoveryFactor.model_validate_json(json.dumps(payload))
    values = compute_factor(
        candidate.definition(definitions), {"acquired.a": [2, 3], "acquired.b": [1, 0]}
    )
    np.testing.assert_allclose(values, [2, np.nan], equal_nan=True)
    payload["expression"]["nodes"][-1]["op"] = "add"
    candidate = DiscoveryFactor.model_validate_json(json.dumps(payload))
    with pytest.raises(ValueError, match="units"):
        candidate.definition(definitions)
    payload["expression"]["nodes"][-1]["inputs"].reverse()
    assert (
        candidate.recipe_ref == DiscoveryFactor.model_validate_json(json.dumps(payload)).recipe_ref
    )


def test_all_survey_records_have_provenance_and_explicit_capabilities():
    catalog = survey_catalog(ROOT, ["price.open", "price.close"])
    assert len(catalog) == 32
    assert len({item["survey_id"] for item in catalog}) == 32
    assert {item["survey_id"] for item in catalog if item["execution_status"] == "executable"} == {
        "qlib.roc",
        "qlib.ma",
        "qlib.kmid",
    }
    assert (
        len([item for item in catalog if item["execution_status"] == "noncausal_reference_only"])
        == 1
    )
    for item in catalog:
        assert len(item["provenance"]["revision"]) == 40
        assert item["provenance"]["license_artifacts"]
        assert item["source"]["url"]
        assert (ROOT / item["provenance"]["definition_file"]).is_file()
    hits = retrieve_candidates("earnings yield valuation", catalog, limit=3)
    assert len(hits) <= 3
    assert "quantdinger.earnings_yield" in {item["survey_id"] for item in hits}
    assert retrieve_candidates("unmatchedword", catalog) == []
    assert all(item["execution_status"] != "executable" for item in survey_catalog(ROOT))


def test_existing_and_new_survey_adapters_execute_without_upstream_imports():
    columns = {
        "price.open": np.array([10.0, 20, 0, 40]),
        "price.close": np.array([12.0, 10, 20, 80]),
    }
    definitions = adapter_catalog(ROOT, columns, window=2)
    assert len(definitions) == 4
    assert {item["factor_id"] for item in definitions if item["runnable"]} == {
        "survey.qlib.kmid",
        "survey.qlib.roc",
        "survey.qlib.ma",
    }
    np.testing.assert_allclose(
        execute_survey("qlib.kmid", columns), [0.2, -0.5, np.nan, 1], equal_nan=True
    )
    np.testing.assert_allclose(
        execute_survey("qlib.roc", columns, window=2),
        [np.nan, np.nan, 12 / 20, 10 / 80],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        execute_survey("qlib.ma", columns, window=2),
        [np.nan, 11 / 10, 15 / 20, 50 / 80],
        equal_nan=True,
    )
    with pytest.raises(ValueError, match="operator/adapter"):
        execute_survey("gs_quant.seasonal_trend", columns)
    with pytest.raises(ValueError, match="needs input"):
        execute_survey("qlib.kmid", {"price.close": [1, 2]})
