"""Structural summaries use only admitted training values and perform no fitting."""

import json

import numpy as np
import pytest

from quant_rsi.research.rolling_diagnostics import diagnose_candidates


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
