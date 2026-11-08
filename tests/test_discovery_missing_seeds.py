"""Unconfigured discovery seeds stay empty; registered source errors remain errors."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
from test_research_study import command

from quant_rsi.cli import main
from quant_rsi.market import news_feed
from quant_rsi.research import study, study_discovery
from quant_rsi.research.demo import load_catalog
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.demo_news import issuer_news
from quant_rsi.research.demo_related import profile_related


@pytest.mark.parametrize("prices", ["synthetic", "daily"])
def test_tsm_active_zero_fit_reaches_decision_without_inventing_seed_data(
    tmp_path, monkeypatch, prices
):
    root = tmp_path / "study"
    args = command(
        root,
        "--llm",
        "codex",
        "--research-mode",
        "discovery",
        "--news-research",
        "--related",
        "profile",
        "--news",
        "issuer-rss",
        "--prices",
        prices,
        "--max-fits",
        "0",
        "--horizon-sessions",
        "3",
        "--no-ui",
    )
    args[1] = "TSM"
    contexts = []

    def forbidden(*args, **kwargs):
        pytest.fail("Empty discovery seeds must not fetch RSS, fit, or fabricate news history")

    monkeypatch.setattr(news_feed, "fetch", forbidden)
    monkeypatch.setattr(study, "simulated_history", forbidden)
    monkeypatch.setattr(study, "retain_snapshot", forbidden)
    monkeypatch.setattr(LightGBMDemo, "predict", forbidden)
    # Script the native decision boundary; no Codex session or public service is called.
    monkeypatch.setattr(study, "LLMHypothesizer", lambda *a, **kw: object())
    monkeypatch.setattr(study_discovery, "LLMHypothesizer", lambda *a, **kw: object())

    def decide(transport, context):
        contexts.append(deepcopy(context))
        return {"action": "stop", "reason": "No source-backed candidate yet", "candidates": []}

    monkeypatch.setattr(study_discovery, "discovery_decision", decide)
    if prices == "daily":
        # Exercise daily orchestration with clearly constructed, offline price inputs.
        fixture = LightGBMDemo(load_catalog(args[args.index("--catalog") + 1]), horizon_sessions=3)
        data = replace(fixture.data, instruments=("TSM",))
        metadata = {
            "horizon_sessions": 3,
            "session_close_times": [data.timestamp(i, "close") for i in range(len(data.dates))],
            "fixture": True,
        }
        monkeypatch.setattr(study, "load_daily_prices", lambda *a, **kw: (data, metadata))

    assert main(args) == 0
    assert len(contexts) == 1
    context = contexts[0]
    assert context["symbol"] == "TSM" and context["evidence"] == []
    assert context["related"]["status"] == "not_configured"
    assert context["related"]["stocks"] == context["related"]["relationships"] == []
    assert context["news_seed"]["status"] == "not_configured"
    assert context["news_seed"]["coverage"] == [
        {"symbol": "TSM", "status": "not_configured", "selected_items": 0}
    ]
    assert context["source_limits"]["search"] > 0
    news = json.loads((root / "checkpoints" / "issuer_news.json").read_text())["output"]
    assert news["items"] == [] and news["status"] == "not_configured"
    assert "received_at" not in news and "source_url" not in news
    assert json.loads((root / "news_history.json").read_text()) == []
    assert not (root / "news_archive").exists()
    result = json.loads((root / "result.json").read_text())
    assert result["discovery"]["counts"]["fits"] == 0


@pytest.mark.parametrize(
    "mode, active", [("discovery", False), ("legacy", True), ("incremental", True)]
)
def test_missing_seed_opt_in_does_not_change_other_study_modes(tmp_path, mode, active):
    args = command(
        tmp_path / "study",
        "--llm",
        "simulated",
        "--related",
        "profile",
        "--research-mode",
        mode,
        "--no-ui",
    )
    args[1] = "TSM"
    if active:
        args.append("--news-research")
    assert main(args) == 1
    error = json.loads((tmp_path / "study" / "failure.json").read_text())
    assert "No relationship profile for TSM" in str(error)


def test_missing_seed_helpers_keep_strict_defaults():
    with pytest.raises(ValueError, match="No relationship profile"):
        profile_related("TSM")
    with pytest.raises(ValueError, match="No registered issuer RSS"):
        issuer_news({"symbol": "TSM"})


def test_configured_rss_failure_remains_an_error_with_missing_seed_opt_in(monkeypatch):
    monkeypatch.setattr(
        news_feed, "fetch", lambda *a: (b"", {"error": "fixture transport failure"})
    )
    with pytest.raises(ValueError, match="Issuer RSS retrieval failed: fixture transport failure"):
        issuer_news({"symbol": "KO"}, allow_missing=True)
