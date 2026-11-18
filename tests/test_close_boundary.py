"""D3 boundary cases at a real session close (#159 Phase 1 acceptance).

Decision instant = session close; information is usable iff available strictly before it.
2026-10-23 is a NYSE session closing at 16:00 EDT = 20:00 UTC.
"""

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_acquired_alignment_rules import column, observation
from test_study_news import RECIPE
from test_study_news import snapshot as news_snapshot

from quant_rsi.market import daily_data
from quant_rsi.market.market_contracts import timestamp, utc_text
from quant_rsi.research.study_news import NewsHistory
from quant_rsi.research.study_windows import retained_news

CLOSE = "2026-10-23T20:00:00+00:00"


def at(seconds):
    return utc_text(timestamp(CLOSE) + timedelta(seconds=seconds))


@pytest.mark.parametrize(("received", "visible"), [(-1, True), (0, False)])
def test_headline_received_at_the_close_belongs_to_the_next_decision(received, visible):
    snapshot = news_snapshot(received=at(received))
    assert bool(retained_news([snapshot], CLOSE, "KO", 10)["items"]) is visible
    result = NewsHistory([snapshot]).evaluate(RECIPE, "KO", CLOSE)
    if visible:
        assert result["value"] == 1
    else:
        assert result["value"] is None
        assert result["reasons"] == ["no_snapshot_received_by_cutoff"]


def test_item_received_at_the_close_is_excluded_even_from_an_earlier_snapshot():
    snapshot = news_snapshot(received=at(-60))
    snapshot["items"][0]["received_at"] = CLOSE
    assert retained_news([snapshot], CLOSE, "KO", 10)["items"] == []


def test_max_age_boundary_at_minus_one_exact_and_plus_one_second():
    row = observation(published="2025-01-04T12:00:00Z", effective="2025-01-04T12:00:00Z")
    limit = datetime(2025, 4, 14, 12, tzinfo=UTC)  # exactly 100 days after publication
    times = [utc_text(limit + timedelta(seconds=s)) for s in (-1, 0, 1)]
    assert column([row], times) == [3.0, 3.0, None]


def test_close_readiness_boundary():
    close = timestamp(CLOSE)
    buffer = timedelta(seconds=daily_data.CLOSE_BUFFER_SECONDS)
    assert not daily_data.close_ready(close, close + buffer - timedelta(seconds=1))
    assert daily_data.close_ready(close, close + buffer)


CONSUMERS = (
    "market/daily_data.py",
    "forecast/forecast_context_data.py",
    "forecast/enhanced_forecast_workflow.py",
)


@pytest.mark.parametrize("module", CONSUMERS)
def test_every_daily_close_consumer_uses_the_one_readiness_rule(module):
    # Equality by construction: each consumer calls close_ready and none adds its own
    # close + timedelta(minutes=...) literal.
    source = (Path(daily_data.__file__).resolve().parents[1] / module).read_text()
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", getattr(node.func, "id", None)) == "close_ready"
    ]
    literals = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "timedelta"
        and any(k.arg == "minutes" for k in node.keywords)
    ]
    assert calls and not literals
