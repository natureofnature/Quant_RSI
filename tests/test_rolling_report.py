"""Visible rolling reports keep pending forecasts out of observed outcome charts."""

import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from html.parser import HTMLParser

from quant_rsi.research.rolling_report import render_rolling


def fixture():
    folds = []
    for month, origin, target, observed in (
        ("2026-07", "2026-06-30", "2026-07-31", 105),
        ("2026-08", "2026-07-31", "2026-08-31", 103),
        ("2026-09", "2026-08-31", "2026-09-30", None),
    ):
        forecasts = {}
        for arm, predicted in (
            ("fixed", 106),
            ("adaptive", 104),
            ("unchanged_price", 100),
            ("training_label_mean", 102),
        ):
            forecasts[arm] = {
                "origin_session": origin,
                "target_session": target,
                "reference_price": 100,
                "predicted_price": predicted,
                "predicted_log_return": 0.01,
            }
            if observed is not None:
                forecasts[arm]["observed_price"] = observed
        folds.append(
            {
                "month": month,
                "training_window": {"start": "2026-01-01", "end": origin},
                "training_origins": ["2026-02-17", "2026-05-29"],
                "training_targets": ["2026-03-17", "2026-06-30"],
                "origin_session": origin,
                "target_session": target,
                "horizon_sessions": 22,
                "forecasts": forecasts,
                "outcome": "observed" if observed else "not_observed",
                "decision": {"action": "retain", "reason": "No historical news available"},
                "candidate_status": "baseline_retained",
                "fits": {"fixed": 1},
            }
        )
    account = {
        "metrics": {
            "net_return": "0.03",
            "sharpe_ratio": 0.75,
            "max_drawdown": "-0.02",
            "fees": {"commission": "1.25", "slippage": "2.50", "tax": "0.00"},
            "risk_statistics": {"daily_return_count": 42, "status": "available"},
        },
        "valuations": [
            {"session_id": "2026-07-01", "nav": "10000"},
            {"session_id": "2026-08-31", "nav": "10300"},
        ],
    }
    return {
        "symbol": "SOXX",
        "notice": "Historical reconstruction, not independent validation",
        "configuration": {},
        "folds": folds,
        "metrics": {
            "fixed": {
                "sample_count": 2,
                "mae_price": 2.5,
                "rmse_price": 2.6,
                "mae_return_pp": 2.5,
                "direction_hit_rate": 1,
            }
        },
        "accounts": {"fixed": {"strategy": account, "buy_and_hold": deepcopy(account)}},
        "price_basis_notice": "Compare prices on a consistent share basis.",
    }


class VisibleText(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.parts, self.hidden = [], False
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden = True

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = False

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def test_visible_report_contains_windows_baselines_costs_and_decisions():
    html = render_rolling(fixture())
    text = " ".join(VisibleText(html).parts)
    for term in (
        "2026-01-01",
        "2026-02-17",
        "2026-05-29",
        "2026-06-30",
        "Fixed baseline",
        "Adaptive research",
        "Unchanged price",
        "Training-label mean",
        "Pending forecasts",
        "No historical news available",
        "Buy-and-hold",
        "Sharpe",
        "0.7500",
        "Commission",
        "1.2500",
        "Slippage",
        "2.5000",
        "continuous account",
        "consistent share basis",
    ):
        assert term in text
    assert html.count("<svg") == 2
    assert "average of monthly Sharpes" in text


def test_pending_actual_values_are_ignored_and_shared_axis_keeps_future_gap():
    result = fixture()
    for row in result["folds"][-1]["forecasts"].values():
        row["observed_price"] = 777
    html = render_rolling(result)
    chart = ET.fromstring(re.findall(r"<svg.*?</svg>", html, re.S)[0])
    actual = chart.find("g[@data-series='Observed return']")
    assert [point.get("data-session") for point in actual.findall("circle")] == [
        "2026-07-31",
        "2026-08-31",
    ]
    predicted = chart.find("g[@data-series='Fixed baseline']")
    assert [point.get("cx") for point in predicted.findall("circle")] == [
        "90.00",
        "525.00",
        "960.00",
    ]
    assert [point.get("cx") for point in actual.findall("circle")] == ["90.00", "525.00"]
    assert "777" not in html
    pending = html.split("<h2>Pending forecasts", 1)[1].split("</section>", 1)[0]
    assert "<td>—</td>" in pending
    assert "Observed month" not in pending


def test_no_adaptive_or_account_failure_is_displayed_without_fabricated_metrics():
    result = fixture()
    result["folds"] = [result["folds"][-1]]
    for fold in result["folds"]:
        fold["forecasts"].pop("adaptive")
    result["accounts"] = {}
    result.update(status="incomplete", failures=[{"month": "2026-08", "reason": "fit failed"}])
    html = render_rolling(result)
    assert "fit failed" in html and "incomplete" in html
    assert "unavailable — no recorded observations to plot" in html
    assert "data-series='Adaptive research'" not in html
    metrics = html.split("<h2>Observed forecast metrics", 1)[1].split("</section>", 1)[0]
    assert "<tbody></tbody>" in metrics


def test_report_escapes_labels_and_does_not_emit_untrusted_html():
    result = fixture()
    result["symbol"] = "<script>alert(1)</script>"
    result["price_basis_notice"] = "<b>basis</b>"
    result["folds"][0]["decision"] = {"reason": '<img src=x onerror="alert(1)">'}
    html = render_rolling(result)
    assert "<script>" not in html and "<img" not in html and "<b>basis" not in html
    assert "&lt;script&gt;" in html and "&lt;img" in html and "&lt;b&gt;basis" in html
