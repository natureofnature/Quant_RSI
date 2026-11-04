"""Separate factor-pool increment from absolute skill on the same mature target."""

from copy import deepcopy

import pytest

from quant_rsi.research.rolling import _feedback


def fold():
    return {
        "month": "2024-07",
        "target_session": "2024-07-31",
        "target_close_at": "2024-07-31T20:00:00+00:00",
        "outcome": "observed",
        "prediction_sha256": "frozen-prediction",
        "decision": {"action": "replace", "reason": "candidate pool"},
        "candidate_status": "fitted",
        "fits": {"adaptive": {"factor_ids": ["price", "news"]}},
        "forecasts": {
            name: {"reference_price": 100.0, "observed_price": 102.0, "predicted_price": value}
            for name, value in {
                "fixed": 112.0,
                "adaptive": 110.0,
                "unchanged_price": 100.0,
                "training_label_mean": 108.0,
            }.items()
        },
    }


def test_increment_can_improve_fixed_while_losing_to_simple_controls():
    original = fold()
    snapshot = deepcopy(original)
    feedback = _feedback([original], original["target_close_at"])[0]
    assert original == snapshot
    assert feedback["prediction_sha256"] == "frozen-prediction"
    assert feedback["absolute_return_errors_pp"] == {
        "fixed": 10.0,
        "adaptive": 8.0,
        "unchanged_price": 2.0,
        "training_label_mean": 6.0,
    }
    assert feedback["adaptive_loss_deltas_pp"] == {
        "fixed": -2.0,
        "unchanged_price": 6.0,
        "training_label_mean": 2.0,
    }
    assert feedback["errors"]["unchanged_price"] == -2.0
    for row in original["forecasts"].values():
        for field in ("reference_price", "observed_price", "predicted_price"):
            row[field] *= 3
    scaled = _feedback([original], original["target_close_at"])[0]
    assert scaled["absolute_return_errors_pp"] == feedback["absolute_return_errors_pp"]
    assert scaled["adaptive_loss_deltas_pp"] == feedback["adaptive_loss_deltas_pp"]


@pytest.mark.parametrize("outcome", ["observed", "not_observed"])
def test_unmatured_or_unobserved_targets_never_enter_comparison(outcome):
    value = fold()
    value["outcome"] = outcome
    assert _feedback([value], "2024-07-31T19:59:59+00:00") == []
    if outcome == "not_observed":
        assert _feedback([value], value["target_close_at"]) == []
