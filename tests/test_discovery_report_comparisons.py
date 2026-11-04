"""Saved discovery results keep their visible baselines and separate chart populations."""

import copy
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

import pytest

from quant_rsi.research.discovery_report import render_discovery


def metrics(mae):
    return {
        "mae_price": mae,
        "rmse_price": mae + 0.25,
        "mape_percent": 1.25,
        "direction_hit_rate": 2 / 3,
        "sample_count": 3,
    }


def account(dates, values):
    return {
        "metrics": {
            "net_return": "0.02",
            "sharpe_ratio": 0.75,
            "annualized_volatility": 0.2,
            "max_drawdown": "-0.01",
            "fees": {"commission": "1.25", "slippage": "2.50", "tax": "0.00"},
            "turnover": "0.8",
            "risk_statistics": {
                "daily_return_count": 2,
                "status": "available",
                "annualization_sessions": 252,
                "annual_risk_free_rate": 0.0,
                "standard_deviation_ddof": 1,
            },
        },
        "valuations": [
            {"session_id": date, "nav": str(value), "status": "complete", "reasons": []}
            for date, value in zip(dates, values, strict=True)
        ],
    }


def experiment(round_number=1, *, month="09", offset=0):
    origins = [f"2026-{month}-{day}" for day in ("09", "10", "11")]
    targets = [f"2026-{month}-{day}" for day in ("16", "17", "18")]
    rows = [
        {
            "origin_session": origin,
            "target_session": target,
            "reference_price": 100 + index + offset,
            "predicted_price": 101 + index + offset + round_number / 10,
            "observed_price": 102 + index + offset,
            "predicted_log_return": 0.01,
        }
        for index, (origin, target) in enumerate(zip(origins, targets, strict=True))
    ]
    baselines = {}
    for name, shift, error in (("unchanged_price", 0, 0.5), ("training_label_mean", 0.5, 1.25)):
        baseline_rows = copy.deepcopy(rows)
        for row in baseline_rows:
            row["predicted_price"] = row["reference_price"] + shift
        baselines[name] = {"rows": baseline_rows, "metrics": metrics(error)}
    prediction = {
        "development": {
            "rows": rows,
            "metrics": metrics(1.75),
            "baselines": baselines,
            "currency": "USD",
            "first_evaluation_session": origins[0],
            "last_evaluation_session": origins[-1],
        },
        "strategy": {
            "period": {"first_session": origins[0], "last_session": origins[-1]},
            "strategy": account(origins, [10000, 10010 + round_number, 10200 + round_number]),
            "buy_and_hold": account(origins, [10000, 10020, 10300]),
        },
    }
    return {
        "ref": f"{month}-fit-{round_number}",
        "spec": {"kind": "baseline" if round_number == 1 else "candidate"},
        "result": {
            "round": round_number,
            "fit_identity": f"{month}-model-{round_number}",
            "prediction": prediction,
        },
    }


def group(*, candidates=0, month="09", offset=0):
    baseline = experiment(month=month, offset=offset)
    return {
        "baseline": baseline,
        "incumbent": baseline,
        "candidates": [
            experiment(number, month=month, offset=offset) for number in range(2, candidates + 2)
        ],
    }


def result(groups):
    return {
        "symbol": "KO",
        "configuration": {"question": "Compare retained experimental results"},
        "notice": "Retrospective development only; not independent validation.",
        "study": {"conclusion": "Retain the recorded incumbent"},
        "discovery": {
            "counts": {"fits": sum(1 + len(value["candidates"]) for value in groups.values())},
            "groups": groups,
            "selected": None,
            "memory": [],
        },
    }


class Text(HTMLParser):
    def __init__(self, document):
        super().__init__()
        self.values = []
        self.groups = []
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "data-comparison-group" in attrs:
            self.groups.append(attrs["data-comparison-group"])

    def handle_data(self, data):
        self.values.append(data)


def visible_report(value):
    markdown, html = render_discovery(value)
    # Assertions must prove the visible report, not merely find data in its JSON dump.
    visible = re.split(r"<details[^>]*>\s*<summary[^>]*>Complete evidence JSON", html)[0]
    return markdown, visible, " ".join(Text(visible).values)


def charts(document):
    return [ET.fromstring(svg) for svg in re.findall(r"<svg\b.*?</svg>", document, re.S)]


def chart(document, label):
    title, key = label.split(" · ", 1)
    labels = {label, title + " · " + key[:12]}
    return next(svg for svg in charts(document) if svg.get("aria-label") in labels)


def series(svg, label):
    return [element for element in svg.iter() if element.get("data-series") == label]


def geometry(svg, label):
    roots = series(svg, label)
    elements = []
    for root in roots:
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1] in {"polyline", "path", "circle"}:
                if element not in elements:
                    elements.append(element)
    return elements


def coordinates(elements):
    points = []
    for element in elements:
        if "points" in element.attrib:
            points.extend(
                tuple(map(float, value.split(","))) for value in element.get("points").split()
            )
        elif element.tag.rsplit("}", 1)[-1] == "circle":
            points.append((float(element.get("cx")), float(element.get("cy"))))
        elif "d" in element.attrib:
            points.extend(
                (float(x), float(y))
                for x, y in re.findall(r"[ML]\s*([\d.+-]+)[,\s]+([\d.+-]+)", element.get("d"))
            )
    return set(points)


def test_baseline_only_result_still_shows_simple_controls_accounts_and_charts():
    markdown, html, text = visible_report(result({"matched-september": group()}))
    for label in (
        "Prediction comparison",
        "Strategy comparison",
        "Fixed-factor baseline",
        "Unchanged price",
        "Training-label mean",
        "Buy-and-hold",
        "Sharpe",
    ):
        assert label in text
        assert label in markdown
    assert "No candidate completed training" in text
    assert "1.75" in text and "1.25" in text
    assert "not independent validation" in text
    assert "drawdown" in text.lower()
    assert "commission" in text.lower() and "slippage" in text.lower()
    assert chart(html, "Development predictions · matched-september") is not None
    assert chart(html, "Hypothetical account NAV · matched-september") is not None


def test_many_fits_keep_every_series_and_share_session_coordinates():
    value = group(candidates=4)
    # Missing the first retained prediction must not stretch this fit back to the first date.
    value["candidates"][0]["result"]["prediction"]["development"]["rows"].pop(0)
    _, html, text = visible_report(result({"same-population": value}))
    prediction = chart(html, "Development predictions · same-population")
    nav = chart(html, "Hypothetical account NAV · same-population")
    for number in range(1, 6):
        label = ("Fixed-factor baseline" if number == 1 else "Candidate") + f" · Fit {number}"
        assert label in text
        assert geometry(prediction, label)
        assert geometry(nav, label + " strategy")
    assert geometry(nav, "Buy-and-hold")
    baseline_points = [
        x for x, _ in coordinates(geometry(prediction, "Fixed-factor baseline · Fit 1"))
    ]
    candidate_points = [x for x, _ in coordinates(geometry(prediction, "Candidate · Fit 2"))]
    assert baseline_points and candidate_points
    assert min(candidate_points) > min(baseline_points)
    assert max(candidate_points) == max(baseline_points)


def test_multiple_groups_never_share_dates_or_chart_populations():
    _, html, _ = visible_report(
        result({"september": group(), "october": group(month="10", offset=100)})
    )
    assert Text(html).groups == ["september", "october"]
    for key, month, other in (("september", "09", "10"), ("october", "10", "09")):
        for title in ("Development predictions", "Hypothetical account NAV"):
            svg = chart(html, title + " · " + key)
            text = " ".join(svg.itertext())
            assert f"2026-{month}-" in text
            assert f"2026-{other}-" not in text


def test_missing_retained_series_keeps_metrics_and_discloses_unavailable_charts():
    value = group()
    prediction = value["baseline"]["result"]["prediction"]
    prediction["development"]["rows"] = []
    for baseline in prediction["development"]["baselines"].values():
        baseline.pop("rows")
    for name in ("strategy", "buy_and_hold"):
        prediction["strategy"][name]["valuations"] = []
    _, html, text = visible_report(result({"missing-series": value}))
    assert "Prediction comparison" in text and "1.75" in text
    assert "Buy-and-hold" in text
    assert "unavailable" in text.lower() or "no retained" in text.lower()
    assert not charts(html)


@pytest.mark.parametrize("kind", ["observation", "nav"])
def test_missing_middle_observation_is_a_gap_not_a_joined_or_zero_filled_curve(kind):
    value = group()
    prediction = value["baseline"]["result"]["prediction"]
    if kind == "observation":
        prediction["development"]["rows"][1]["observed_price"] = None
    else:
        middle = prediction["strategy"]["strategy"]["valuations"][1]
        middle.update(nav=None, status="unpriced", reasons=["missing_quote"])
    _, html, _ = visible_report(result({"missing-outcome": value}))
    title = "Development predictions" if kind == "observation" else "Hypothetical account NAV"
    svg = chart(html, title + " · missing-outcome")
    if kind == "observation":
        labels = {element.get("data-series", "") for element in svg.iter()}
        label = next(label for label in labels if label.startswith("Observed"))
    else:
        label = "Fixed-factor baseline · Fit 1 strategy"
    shapes = geometry(svg, label)
    assert shapes
    assert len(coordinates(shapes)) == 2
    # Three retained sessions with an unobserved middle outcome have no connecting edge.
    assert all(len(element.get("points", "").split()) <= 1 for element in shapes)
    assert all("L" not in element.get("d", "").upper() for element in shapes)
    assert all("nan" not in str(element.attrib).lower() for element in shapes)


def test_group_labels_are_escaped_in_chart_accessibility_and_visible_report():
    hostile = 'group"><script>window.bad=true</script>'
    _, html, _ = visible_report(result({hostile: group()}))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert chart(html, "Development predictions · " + hostile) is not None
