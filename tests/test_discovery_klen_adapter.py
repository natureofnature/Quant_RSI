"""Pinned KLEN arithmetic and admission tests; no claim of upstream runtime parity or alpha."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quant_rsi.core.data import MarketData
from quant_rsi.research.discovery_engine import DiscoveryExperimentRunner
from quant_rsi.research.discovery_factors import adapter_catalog, execute_survey, survey_catalog

ROOT = Path(__file__).resolve().parents[1]


def test_klen_matches_retained_hand_calculated_fixture_without_close():
    fixture = json.loads(
        (ROOT / "experimental/candidate_factors/fixtures/arithmetic.json").read_text()
    )
    case = next(case for case in fixture["cases"] if case["id"] == "bar_shape")
    inputs = {f"price.{key}": [case["inputs"][key]] for key in ("high", "low", "open")}
    np.testing.assert_allclose(execute_survey("qlib.klen", inputs), [case["expected"][1]])
    record = next(row for row in survey_catalog(ROOT, inputs) if row["survey_id"] == "qlib.klen")
    assert record["execution_status"] == "executable"
    assert record["provenance"]["revision"] == "54355232463878d2eebb91fe0ee5fa7fa1f5976c"
    assert record["formula_summary"] == "(high - low) / open"
    adapter = next(
        row for row in adapter_catalog(ROOT, inputs) if row["factor_id"] == "survey.qlib.klen"
    )
    assert adapter["runnable"] and adapter["recipe"]["window"] is None
    assert set(adapter["input_factor_ids"]) == set(inputs)


def test_klen_missing_high_low_remains_unavailable():
    inputs = {"price.open": [10], "price.close": [12]}
    record = next(row for row in survey_catalog(ROOT, inputs) if row["survey_id"] == "qlib.klen")
    assert record["execution_status"] == "needs_input"
    assert record["missing_inputs"] == ["price.high", "price.low"]
    adapter = next(
        row for row in adapter_catalog(ROOT, inputs) if row["factor_id"] == "survey.qlib.klen"
    )
    assert not adapter["runnable"] and adapter["availability_reason"] == "needs_input"
    with pytest.raises(ValueError, match="needs input"):
        execute_survey("qlib.klen", inputs)


def test_study_discovery_initializes_without_optional_high_low():
    opens = np.arange(100.0, 125.0).reshape(-1, 1)
    model = SimpleNamespace(
        data=MarketData(
            tuple(f"2026-01-{i:02d}" for i in range(1, 26)), ("KO",), opens, opens * 1.01
        ),
        library={"definitions": []},
        financial_features=None,
        external_features=None,
        peer_features=None,
        profile_features=None,
    )
    runner = DiscoveryExperimentRunner(model, SimpleNamespace(), [], lambda *args: None)
    klen = next(d for d in runner.definitions if d["factor_id"] == "survey.qlib.klen")
    assert not klen["runnable"]
    assert "survey.qlib.klen" not in runner.columns
    assert {"survey.qlib.kmid", "survey.qlib.roc", "survey.qlib.ma"} <= runner.columns.keys()


def test_klen_invalid_rows_are_missing_without_filling_or_dropping_rows():
    inputs = {
        "price.open": [10, 10, 0, -1, 10, 10, 10, 20, 5, 10, 10],
        "price.high": [15, 10, 15, 15, np.inf, 15, np.nan, 15, 15, 8, 15],
        "price.low": [9, 10, 9, 9, 9, -1, 9, 9, 9, 9, np.nan],
    }
    values = execute_survey("qlib.klen", inputs)
    np.testing.assert_allclose(values, [0.6, 0] + [np.nan] * 9, equal_nan=True)


@pytest.mark.parametrize("highs", [[15, 16], [[15]]])
def test_klen_requires_aligned_one_dimensional_inputs(highs):
    with pytest.raises(ValueError, match="aligned one-dimensional"):
        execute_survey("qlib.klen", {"price.open": [10], "price.high": highs, "price.low": [9]})


def test_klen_is_row_local_and_does_not_mutate_inputs():
    inputs = {
        "price.open": np.asarray([10.0, 20.0, 30.0]),
        "price.high": np.asarray([15.0, 24.0, 38.0]),
        "price.low": np.asarray([9.0, 17.0, 25.0]),
    }
    original = {key: values.copy() for key, values in inputs.items()}
    values = execute_survey("qlib.klen", inputs, window=1)
    for key in inputs:
        np.testing.assert_array_equal(inputs[key], original[key])
        inputs[key][-1] *= 100
    np.testing.assert_array_equal(execute_survey("qlib.klen", inputs, window=30)[:2], values[:2])
    np.testing.assert_allclose(values, [0.6, 0.35, 13 / 30])
