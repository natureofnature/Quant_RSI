"""Saved-model diagnostics remain readable without changing forecast charts."""

import numpy as np
from test_rolling_report import fixture

from quant_rsi.research.rolling_diagnostics import (
    compare_forecast_diagnostics,
    diagnose_forecast,
)
from quant_rsi.research.rolling_report import render_rolling


def diagnostic():
    return diagnose_forecast(
        np.asarray([[1.0], [3.0], [5.0]]),
        [0, 1],
        {
            "factor_ids": ["<unsafe-factor>"],
            "predicted_log_return": 0.32,
            "state": {"mean": [2.0], "scale": [1.0], "coefficients": [0.1], "intercept": 0.02},
        },
        1,
    ) | {"comparison_identity": "same-rows-inputs-labels"}


def test_report_shows_ranges_and_log_return_accounting_without_losing_charts():
    result = fixture()
    old = render_rolling(result)
    item = diagnostic()
    fold = result["folds"][0]
    fold["forecast_diagnostics"] = {"fixed": item, "adaptive": item}
    fold["paired_forecast_decomposition"] = compare_forecast_diagnostics(item, item)
    html = render_rolling(result)
    assert "<details><summary>Forecast input ranges" in html
    assert "log-return percentage points" in html
    assert "Baseline coefficient refit" in html
    assert "&lt;unsafe-factor&gt;" in html and "<unsafe-factor>" not in html
    assert "32.0000" in html  # 0.32 log return becomes 32 log-return pp, not exp(.32)-1.
    assert html.count("<svg") == old.count("<svg") > 0
    assert "Forecast input ranges" not in old


def test_report_handles_unavailable_nonlinear_accounting_and_constant_z():
    result = fixture()
    item = diagnose_forecast(
        np.ones((3, 1)),
        [0, 1],
        {"factor_ids": ["constant"], "predicted_log_return": 0.01, "state": {}},
        1,
    )
    result["folds"][0]["forecast_diagnostics"] = {"fixed": item}
    result["folds"][0]["paired_forecast_decomposition"] = compare_forecast_diagnostics(item, item)
    html = render_rolling(result)
    assert "Linear saved state unavailable" in html
    assert "Matching training rows" in html
    assert "Constant columns have no defined z-score" in html
    assert "<td>nan</td>" not in html and "<td>inf</td>" not in html
