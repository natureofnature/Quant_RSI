"""Structural summaries use only admitted training values and perform no fitting."""

import json

import numpy as np
import pytest

from quant_rsi.research.rolling_diagnostics import (
    compare_forecast_diagnostics,
    diagnose_candidates,
    diagnose_forecast,
    summarize_forecast,
)


@pytest.fixture
def matrix():
    return np.asarray(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [1, 5, 2, 0],
            [2, 5, 4, 1],
            [3, 5, 6, 0],
            [4, 5, 8, 1],
            [1e10, -1e10, 1e10, -1e10],
        ]
    )


def test_training_structure_and_group_correlations(matrix):
    result = diagnose_candidates(matrix, [1, 2, 3, 4], ["base", "constant", "double", "sparse"], 2)
    assert result["training_samples"] == 4
    assert result["model_columns"] == 4
    assert result["baseline_columns"] == result["candidate_columns"] == 2
    assert result["standardized_rank"] == 2
    base, constant, double, sparse = result["columns"]
    assert base["distinct_values"] == 4 and base["nonzero_fraction"] == 1
    assert base["variance"] == pytest.approx(1.25)
    assert base["max_abs_baseline_correlation"] is None  # Self is excluded; other is constant.
    assert base["max_abs_candidate_correlation"] == pytest.approx(1)
    assert constant["constant"] and constant["variance"] == 0
    assert constant["max_abs_baseline_correlation"] is None
    assert constant["max_abs_candidate_correlation"] is None
    assert double["max_abs_baseline_correlation"] == pytest.approx(1)
    assert double["max_abs_candidate_correlation"] == pytest.approx(1 / np.sqrt(5))
    assert sparse["distinct_values"] == 2 and sparse["nonzero_rows"] == 2
    assert sparse["nonzero_fraction"] == 0.5
    assert result["high_correlation_pair_count"] == 1
    assert result["high_correlation_pairs"][0] == {
        "left": "base",
        "right": "double",
        "correlation": pytest.approx(1),
        "kind": "candidate_vs_baseline",
    }
    json.dumps(result, allow_nan=False)


def test_prediction_and_unadmitted_rows_cannot_change_diagnostics(matrix):
    original = matrix.copy()
    before = diagnose_candidates(matrix, [1, 2, 3, 4], ["b", "c", "d", "s"], 2)
    matrix[[0, -1]] = [[-1e50] * 4, [np.nan] * 4]
    after = diagnose_candidates(matrix, [1, 2, 3, 4], ["b", "c", "d", "s"], 2)
    assert before == after
    np.testing.assert_equal(matrix[1:5], original[1:5])
    unchanged = matrix.copy()
    diagnose_candidates(matrix, [1, 2, 3, 4], ["b", "c", "d", "s"], 2)
    np.testing.assert_equal(matrix, unchanged)


def test_constant_columns_and_single_training_row_have_no_correlations():
    for values in (np.ones((3, 2)), np.asarray([[0.0, 2.0]])):
        result = diagnose_candidates(values, np.arange(len(values)), ["one", "two"], 1)
        assert result["standardized_rank"] == 0
        assert result["high_correlation_pairs"] == []
        assert all(column["constant"] for column in result["columns"])
        assert all(column["max_abs_baseline_correlation"] is None for column in result["columns"])
        assert all(column["max_abs_candidate_correlation"] is None for column in result["columns"])
        json.dumps(result, allow_nan=False)


def test_strongest_pairs_are_bounded_and_negative_correlation_is_retained():
    base = np.arange(1, 9, dtype=float)
    matrix = np.column_stack([base, -base] + [base * i for i in range(2, 10)])
    ids = [f"factor_{i}" for i in range(matrix.shape[1])]
    first = diagnose_candidates(matrix, np.arange(len(base)), ids, 2)
    second = diagnose_candidates(matrix, np.arange(len(base)), ids, 2)
    assert first == second
    assert first["standardized_rank"] == 1
    assert first["high_correlation_pair_count"] == 45
    assert len(first["high_correlation_pairs"]) == 20
    assert first["omitted_correlation_pairs"] == 25
    assert any(pair["correlation"] < -0.99 for pair in first["high_correlation_pairs"])
    assert {pair["kind"] for pair in first["high_correlation_pairs"]} <= {
        "baseline_pair",
        "candidate_pair",
        "candidate_vs_baseline",
    }


def test_zero_candidates_and_zero_baselines_are_valid():
    values = np.asarray([[1.0], [2.0], [3.0]])
    for count in (0, 1):
        result = diagnose_candidates(values, [0, 1, 2], ["single"], count)
        assert result["candidate_columns"] == 1 - count
        assert result["columns"][0]["max_abs_baseline_correlation"] is None
        assert result["columns"][0]["max_abs_candidate_correlation"] is None


def test_large_finite_values_keep_rank_and_correlation_serializable():
    values = np.asarray([[1e308, 1e308], [-1e308, 1e308], [0, 1e308]])
    result = diagnose_candidates(values, [0, 1, 2], ["varying", "constant"], 1)
    assert result["standardized_rank"] == 1
    assert result["columns"][0]["variance"] is None
    assert not result["columns"][0]["constant"]
    assert result["columns"][1]["variance"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("train", [[], [1, 1], [-1], [6], [1.5], [[1]], [True]])
def test_invalid_training_indices_fail_without_silently_dropping_rows(matrix, train):
    with pytest.raises(ValueError, match="integer training-row"):
        diagnose_candidates(matrix, train, ["a", "b", "c", "d"], 2)


def test_nonfinite_training_rows_fail_and_configuration_is_validated(matrix):
    with pytest.raises(ValueError, match="finite values"):
        diagnose_candidates(matrix, [0, 1], ["a", "b", "c", "d"], 2)
    for ids, count in ((["a"] * 4, 2), ([[], "b", "c", "d"], 2), (["a", "b", "c", "d"], 5)):
        with pytest.raises(ValueError):
            diagnose_candidates(matrix, [1, 2], ids, count)
    with pytest.raises(ValueError, match="matching factor IDs"):
        diagnose_candidates(matrix, [1, 2], ["a"], 1)


def saved_fit(matrix, train, ids, coefficients, intercept=0.01):
    """A hand-authored linear state, without calling an estimator or fitting labels."""
    x = matrix[train]
    mean, scale = x.mean(axis=0), x.std(axis=0)
    constant = np.all(x == x[0], axis=0)
    mean[constant], scale[constant | (scale == 0)] = x[0, constant], 1.0
    return {
        "factor_ids": ids,
        "predicted_log_return": float(intercept + ((matrix[-1] - mean) / scale) @ coefficients),
        "state": {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "coefficients": coefficients,
            "intercept": intercept,
        },
    }


@pytest.fixture
def forecast_inputs():
    values = np.array([[np.nan] * 3, [1, 5, 2], [2, 5, 1], [3, 5, 4], [5, 5, 7]])
    train, ids = [1, 2, 3], ["momentum", "constant", "news"]
    return values, train, saved_fit(values, train, ids, [0.02, 0.0, -0.03])


def test_forecast_training_ranges_and_exact_saved_group_accounting(forecast_inputs):
    values, train, fit = forecast_inputs
    original, serialized = values.copy(), json.dumps(fit)
    result = diagnose_forecast(values, train, fit, 2)
    first, constant, news = result["columns"]
    assert first["training_min"] == 1 and first["training_max"] == 3
    assert first["prediction_z"] == pytest.approx(3 / np.sqrt(2 / 3))
    assert first["distinct_values"] == 3 and first["outside_training_range"]
    assert constant["prediction_z"] is None and constant["constant"]
    assert not constant["outside_training_range"]
    assert news["outside_training_range"] and result["outside_training_range_count"] == 2
    decomposition = result["decomposition"]
    assert decomposition["available"] and decomposition["units"] == "log_return"
    assert decomposition["intercept"] == 0.01
    assert decomposition["baseline_group"] == pytest.approx(first["prediction_z"] * 0.02)
    assert decomposition["candidate_group"] == pytest.approx(news["prediction_z"] * -0.03)
    assert decomposition["reconstructed_prediction"] == pytest.approx(fit["predicted_log_return"])
    assert abs(decomposition["reconstruction_error"]) < 1e-12
    json.dumps(result, allow_nan=False)
    np.testing.assert_equal(values, original)
    assert json.dumps(fit) == serialized


def test_forecast_unadmitted_values_and_training_order_do_not_change_ranges(forecast_inputs):
    values, train, fit = forecast_inputs
    before = diagnose_forecast(values, train, fit, 2)
    values[0] = [1e100, -1e100, np.nan]
    assert before == diagnose_forecast(values, train, fit, 2)
    after = diagnose_forecast(values, train[::-1], fit, 2)
    for left, right in zip(before["columns"], after["columns"], strict=True):
        assert left["prediction_z"] == pytest.approx(right["prediction_z"])
        right["prediction_z"] = left["prediction_z"]
    assert before == after


def test_forecast_permuted_baseline_columns_preserve_group_accounting(forecast_inputs):
    values, train, fit = forecast_inputs
    before = diagnose_forecast(values, train, fit, 2)
    order = [1, 0, 2]
    reordered = values[:, order]
    other_fit = saved_fit(reordered, train, [fit["factor_ids"][i] for i in order], [0, 0.02, -0.03])
    after = diagnose_forecast(reordered, train, other_fit, 2)
    assert after["decomposition"] == before["decomposition"]
    assert after["columns"] == [before["columns"][i] for i in order]


@pytest.mark.parametrize("state", [None, {"model_string": "retained LightGBM state"}, {}])
def test_missing_and_nonlinear_state_preserve_ranges_without_invented_contributions(
    forecast_inputs, state
):
    values, train, fit = forecast_inputs
    fit["state"] = state
    result = diagnose_forecast(values, train, fit, 2)
    assert result["outside_training_range_count"] == 2
    assert not result["decomposition"]["available"]
    assert "baseline_group" not in result["decomposition"]
    assert result["decomposition"]["predicted_log_return"] == fit["predicted_log_return"]
    json.dumps(result, allow_nan=False)


def test_constant_decimal_changed_prediction_has_no_empirical_z():
    values = np.array([[0.1], [0.1], [0.1], [0.2]])
    fit = saved_fit(values, [0, 1, 2], ["constant"], [0.0])
    result = diagnose_forecast(values, [0, 1, 2], fit, 1)
    assert result["columns"][0]["constant"]
    assert result["columns"][0]["prediction_z"] is None
    assert result["columns"][0]["outside_training_range"]
    assert result["decomposition"]["baseline_group"] == 0


def test_forecast_summary_is_bounded_in_factor_order_without_mutation():
    values = np.array([np.arange(12), np.arange(12) + 1, np.arange(12) + 3], dtype=float)
    fit = saved_fit(values, [0, 1], [f"factor_{i}" for i in range(12)], [0.01] * 12)
    diagnostic = diagnose_forecast(values, [0, 1], fit, 4)
    original = json.dumps(diagnostic)
    summary = summarize_forecast(diagnostic)
    assert summary["outside_training_range_count"] == 12
    assert summary["omitted_outside_training_range_columns"] == 4
    assert [c["factor_id"] for c in summary["outside_training_range_columns"]] == fit["factor_ids"][
        :8
    ]
    assert not any("error" in k or "label" in k for k in summary if k != "decomposition")
    summary["outside_training_range_columns"][0]["prediction_value"] = -1
    summary["decomposition"]["intercept"] = -1
    assert json.dumps(diagnostic) == original


def test_paired_accounting_separates_refitted_baseline_from_added_columns(forecast_inputs):
    values, train, fit = forecast_inputs
    fixed_fit = saved_fit(values[:, :2], train, fit["factor_ids"][:2], [0.01, 0.0], 0.02)
    fixed = diagnose_forecast(values[:, :2], train, fixed_fit, 2)
    adaptive = diagnose_forecast(values, train, fit, 2)
    fixed["comparison_identity"] = adaptive["comparison_identity"] = "bound_rows_labels_inputs"
    result = compare_forecast_diagnostics(fixed, adaptive)
    assert result["available"] and result["units"] == "log_return"
    assert result["intercept_change"] == pytest.approx(-0.01)
    assert result["baseline_refit_change"] == pytest.approx(
        adaptive["decomposition"]["baseline_group"] / 2
    )
    assert sum(
        result[k] for k in ("intercept_change", "baseline_refit_change", "candidate_group")
    ) == (pytest.approx(result["predicted_log_return_change"]))
    for alteration in ("identity", "baseline", "state", "missing_identity"):
        changed = json.loads(json.dumps(adaptive))
        if alteration == "identity":
            changed["comparison_identity"] = "different_labels_or_rows"
        elif alteration == "missing_identity":
            changed.pop("comparison_identity")
        elif alteration == "baseline":
            changed["columns"][0]["factor_id"] = "another_factor"
        else:
            changed["decomposition"]["available"] = False
        assert not compare_forecast_diagnostics(fixed, changed)["available"]


def test_forecast_rejects_nonfinite_prediction_and_mismatched_saved_state(forecast_inputs):
    values, train, fit = forecast_inputs
    fit["predicted_log_return"] += 0.1
    with pytest.raises(ValueError, match="reconstruct"):
        diagnose_forecast(values, train, fit, 2)
    fit["state"] = None
    values[-1, 0] = np.nan
    with pytest.raises(ValueError, match="finite prediction"):
        diagnose_forecast(values, train, fit, 2)
