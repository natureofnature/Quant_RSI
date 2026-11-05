"""Synthetic fixed-session reports retain controls, pending gaps and source wording."""

import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from test_rolling_report import VisibleText, fixture

from quant_rsi.research.rolling_diagnostics import diagnose_forecast
from quant_rsi.research.rolling_report import render_rolling


def session_fixture():
    result = fixture()
    result.update(
        kind="SessionRollingStudy",
        notice="Synthetic report fixture only; no predictive-performance evidence.",
        configuration={"horizon_sessions": 3},
    )
    for fold, start, origin, target in zip(
        result["folds"],
        ("2026-01-01", "2026-01-07", "2026-01-10"),
        ("2026-06-30", "2026-07-06", "2026-07-09"),
        ("2026-07-06", "2026-07-09", "2026-07-14"),
        strict=True,
    ):
        fold.pop("month")
        fold.update(
            period_id=target,
            period_kind="trading_sessions",
            origin_session=origin,
            target_session=target,
            horizon_sessions=3,
            training_window={"start": start, "end": origin},
            training_targets=["2026-02-20", "2026-06-03"],
        )
        for row in fold["forecasts"].values():
            row.update(origin_session=origin, target_session=target)
        fold["fits"] = {"fixed": {"factor_ids": ["momentum_10"], "feature_usage": {}}}
    for account in result["accounts"]["fixed"].values():
        account["metrics"]["risk_statistics"]["daily_return_count"] = 6
        account["valuations"][-1]["session_id"] = "2026-07-09"
    return result


def test_session_report_identifies_periods_and_preserves_controls_accounts_and_charts():
    html = render_rolling(session_fixture())
    text = " ".join(VisibleText(html).parts)
    for term in (
        "3-session rolling research",
        "Observed period forecasts",
        "Periods",
        "Fixed baseline",
        "Adaptive research",
        "Unchanged price",
        "Training-label mean",
        "Buy-and-hold",
        "Sharpe",
        "0.7500",
        "Commission",
        "1.2500",
        "Slippage",
        "2.5000",
        "continuous account",
        "next-open fills",
        "daily net NAV returns",
        "2026-07-06",
        "Synthetic report fixture only",
    ):
        assert term in text
    assert "<title>3-session rolling research</title>" in html
    assert "<th>Month</th>" not in html and "<th>Months</th>" not in html
    assert "Monthly rolling" not in html and "monthly Sharpes" not in html
    assert html.count("<svg") == 2
    charts = re.findall(r"<svg.*?</svg>", html, re.S)
    forecast = ET.fromstring(charts[0])
    assert forecast.find("g[@data-series='Fixed baseline']") is not None
    assert forecast.find("g[@data-series='Unchanged price']") is not None
    assert ET.fromstring(charts[1]).find("g[@data-series='Buy-and-hold']") is not None


def test_pending_target_never_becomes_an_observed_return():
    result = session_fixture()
    for row in result["folds"][-1]["forecasts"].values():
        row["observed_price"] = 777
    html = render_rolling(result)
    chart = ET.fromstring(re.findall(r"<svg.*?</svg>", html, re.S)[0])
    actual = chart.find("g[@data-series='Observed return']")
    assert [point.get("data-session") for point in actual.findall("circle")] == [
        "2026-07-06",
        "2026-07-09",
    ]
    assert "777" not in html
    pending = html.split("<h2>Pending forecasts", 1)[1].split("</section>", 1)[0]
    assert "2026-07-14" in pending and "<td>—</td>" in pending


def test_source_wording_and_business_news_sections_are_not_rewritten():
    result = session_fixture()
    fold = result["folds"][0]
    fold["candidate_attempts"] = [{"stage": "proposal", "status": "accepted"}]
    fold["candidate_pool"] = {"limit": 16, "proposed": 1, "accepted": 1}
    fold["acquired_inputs"] = {
        "notice": "Monthly business metrics remain source wording.",
        "coverage": [
            {
                "input_id": "acquired.revenue",
                "measurement": "Monthly revenue",
                "unit": "USD",
                "training_available": 60,
                "training_samples": 60,
                "distinct_training_values": 3,
                "training_disclosures": 3,
                "prediction_available": True,
            }
        ],
    }
    result["measurement_usage"] = {"counts": {"extract": 1}}
    fold["news_coverage"] = "partial_search_results"
    fold["news_evidence"] = {
        "research_required": True,
        "information_mode": "reconstructed",
        "items": [
            {
                "id": "headline-1",
                "title": "Monthly market update <unsafe>",
                "published_at": "2026-06-30T12:00:00Z",
                "received_at": "2026-11-05T12:00:00Z",
            }
        ],
    }
    fold["forecast_diagnostics"] = {
        "fixed": diagnose_forecast(
            np.ones((3, 1)),
            [0, 1],
            {"factor_ids": ["constant"], "predicted_log_return": 0.01, "state": {}},
            1,
        )
    }
    html = render_rolling(result)
    assert "Monthly business metrics remain source wording." in html
    assert "Monthly revenue" in html and "Monthly market update &lt;unsafe&gt;" in html
    assert "<unsafe>" not in html
    assert "to per-period decision calls" in html
    assert "per-period models used executable news inputs" in html
    assert "Acquired business measurements" in html and "Forecast input ranges" in html
    assert "Candidate proposal and one revision" in html and "Candidate pool screening" in html
    assert "<th>Month</th>" not in html


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_report_uses_the_configured_session_horizon(horizon):
    result = session_fixture()
    result["configuration"]["horizon_sessions"] = horizon
    assert f"{horizon}-session rolling research" in render_rolling(result)


def test_empty_failed_session_report_keeps_failure_and_no_invented_accuracy():
    html = render_rolling(
        {
            "kind": "SessionRollingStudy",
            "symbol": "SOXX",
            "folds": [],
            "configuration": {"horizon_sessions": 3},
            "status": "failed",
            "failure": "No complete training history <detail>",
        }
    )
    assert "3-session rolling research" in html and "Incomplete run" in html
    assert "&lt;detail&gt;" in html
    metrics = html.split("<h2>Observed forecast metrics", 1)[1].split("</section>", 1)[0]
    assert "<tbody></tbody>" in metrics
    assert "unavailable — no recorded observations to plot" in html
