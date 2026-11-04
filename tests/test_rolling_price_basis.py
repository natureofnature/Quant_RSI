"""Reviewed split-basis wiring with synthetic quotes and no real fitting/providers."""

import argparse
import json
import shutil
from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest
import test_daily_data as daily_tests
import test_monthly_rolling as monthly
import test_reviewed_price_basis as basis_tests

from quant_rsi.core.artifacts import canonical
from quant_rsi.market import daily_data
from quant_rsi.research import rolling
from quant_rsi.research.demo_prices import read_daily_history


@pytest.fixture
def reviewed_monthly(tmp_path, monkeypatch):
    monkeypatch.setattr(basis_tests, "chart", lambda: daily_tests.chart("2026-11-03"))
    monkeypatch.setattr(
        basis_tests,
        "install",
        lambda mp, payload: daily_tests.install(mp, payload, now="2026-11-04T12:00:00Z"),
    )
    monkeypatch.setattr(
        basis_tests,
        "collect_daily",
        lambda symbol, path: daily_data.collect_daily(
            symbol,
            path,
            historical_sessions=209,
            future_sessions=20,
            as_of=datetime.fromisoformat("2026-10-31T23:59:59-04:00"),
        ),
    )
    return basis_tests.reviewed_fixture(tmp_path, monkeypatch)


@pytest.fixture
def stub_fits(monkeypatch):
    calls = []

    def fit(matrix, labels, train, model, factor_ids):
        calls.append({"x": matrix[train].copy(), "y": labels.copy(), "factor_ids": factor_ids})
        return {
            "predicted_log_return": 0.01,
            "factor_ids": factor_ids,
            "training_samples": len(train),
            "feature_usage": {},
            "state": {"kind": "synthetic_fixture_no_model_fitted"},
        }

    monkeypatch.setattr(rolling, "fit_forecast", fit)
    return calls


def arguments(fixture, output, *, last="2026-07", llm="none", extra=()):
    parser = argparse.ArgumentParser()
    rolling.add_arguments(parser.add_subparsers(dest="command"))
    return parser.parse_args(
        [
            "research-rolling",
            "SOXL",
            "--prices-dir",
            str(fixture[0]),
            "--price-basis-review",
            str(fixture[1]),
            "--first-month",
            "2026-07",
            "--last-month",
            last,
            "--model",
            "ridge",
            "--llm",
            llm,
            "--output",
            str(output),
            *extra,
        ]
    )


def test_reviewed_reader_preserves_original_quotes_and_default_split_refusal(reviewed_monthly):
    prices, review, _, events, _, _ = reviewed_monthly
    before = {p.name: p.read_bytes() for p in prices.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="Unresolved corporate action"):
        read_daily_history("SOXL", prices)
    data, info = read_daily_history("SOXL", prices, price_basis_review=review)
    quotes = {
        (e.kind, e.session_id): float(e.price)
        for e in events
        if e.kind in {"OPEN_QUOTE", "CLOSE_QUOTE"}
    }
    assert len(data.dates) == 209
    np.testing.assert_array_equal(
        data.opens[:, 0], [quotes["OPEN_QUOTE", day] for day in data.dates]
    )
    np.testing.assert_array_equal(
        data.closes[:, 0], [quotes["CLOSE_QUOTE", day] for day in data.dates]
    )
    assert info["price_basis_review"]["review"]["transform"] == "identity"
    assert info["price_basis_review"]["original_resolved_events"]
    assert {p.name: p.read_bytes() for p in prices.iterdir() if p.is_file()} == before
    with pytest.raises(ValueError, match="Unresolved corporate action"):
        read_daily_history("SOXL", prices)


def test_tampered_review_source_cannot_authorize_daily_reader(reviewed_monthly):
    prices, review, *_ = reviewed_monthly
    source = review / "issuer_action.txt"
    source.write_bytes(source.read_bytes() + b" changed")
    with pytest.raises(ValueError, match="source artifact hash"):
        read_daily_history("SOXL", prices, price_basis_review=review)


def test_cli_retains_review_and_resumes_without_external_source_or_repeat_fits(
    reviewed_monthly, tmp_path, monkeypatch, stub_fits
):
    original_review = {p.name: p.read_bytes() for p in reviewed_monthly[1].iterdir() if p.is_file()}
    args = arguments(reviewed_monthly, tmp_path / "study", last="2026-08", llm="codex")
    llm = monthly.DecisionLLM(
        [
            {"action": "retain", "basis": "price_only", "reason": "Fixture", "candidates": []},
            RuntimeError("fixture second-month interruption"),
        ]
    )
    monkeypatch.setattr(rolling, "LLMHypothesizer", lambda *a, **kw: llm)
    with pytest.raises(RuntimeError, match="second-month interruption"):
        rolling.run_rolling(args)
    retained = args.output / "price_basis_review"
    assert {p.name: p.read_bytes() for p in retained.iterdir() if p.is_file()} == original_review
    prediction = (args.output / "2026-07/prediction.json").read_bytes()
    config = json.loads((args.output / "config.json").read_bytes())
    assert config["price_basis_review"] == str(reviewed_monthly[1])
    assert "price_basis_review" in str(config["input_identity"])
    assert len(stub_fits) == 2
    shutil.rmtree(reviewed_monthly[1])
    args.resume = True
    source = retained / "human_review.txt"
    old = source.read_bytes()
    source.write_bytes(old + b" modified")
    with pytest.raises(ValueError, match="source artifact hash"):
        rolling.run_rolling(args)
    source.write_bytes(old)
    manifest = retained / "review.json"
    original_manifest = manifest.read_bytes()
    changed = json.loads(original_manifest)
    changed["reviewer"] = "Another internally valid synthetic review"
    manifest.write_bytes(canonical(changed))
    with pytest.raises(ValueError, match="configuration changed"):
        rolling.run_rolling(args)
    manifest.write_bytes(original_manifest)
    assert len(stub_fits) == 2
    result = rolling.run_rolling(args)
    assert result["status"] == "completed"
    assert len(stub_fits) == 2
    assert (args.output / "2026-07/prediction.json").read_bytes() == prediction
    assert (
        result["configuration"]["input_identity"]["target"]["price_basis_review"]["review"][
            "transform"
        ]
        == "identity"
    )
    assert "not PIT" in (args.output / "report.html").read_text()


def test_review_after_first_origin_is_rejected_before_provider_or_fit(
    reviewed_monthly, tmp_path, monkeypatch, stub_fits
):
    review = deepcopy(reviewed_monthly[-1])
    review["qualification_cutoff"] = "2026-07-01"
    (reviewed_monthly[1] / "review.json").write_bytes(canonical(review))
    args = arguments(reviewed_monthly, tmp_path / "late", llm="codex")
    monkeypatch.setattr(
        rolling, "LLMHypothesizer", lambda *a, **kw: pytest.fail("late review reached provider")
    )
    with pytest.raises(ValueError, match="cutoff|origin"):
        rolling.run_rolling(args)
    assert stub_fits == []


def test_reviewed_open_close_cannot_enable_unreviewed_ohlcv(
    reviewed_monthly, tmp_path, monkeypatch, stub_fits
):
    args = arguments(reviewed_monthly, tmp_path / "ohlcv", extra=["--ohlcv-features"])
    monkeypatch.setattr(
        rolling, "load_ohlcv_history", lambda *a, **kw: pytest.fail("unreviewed OHLCV loaded")
    )
    with pytest.raises(ValueError, match="ohlcv|OHLCV"):
        rolling.run_rolling(args)
    assert stub_fits == []


def test_target_review_does_not_flow_to_related_instrument(
    reviewed_monthly, tmp_path, monkeypatch, stub_fits
):
    args = arguments(
        reviewed_monthly, tmp_path / "related", extra=["--related-prices-dir", str(tmp_path)]
    )
    monkeypatch.setattr(rolling, "profile_related", lambda symbol: {"stocks": ["OTHER"]})
    reader = rolling.read_daily_history
    calls = []

    def read(symbol, directory, **kwargs):
        calls.append((symbol, kwargs))
        if symbol == "OTHER":
            assert kwargs == {}
            raise ValueError("Unresolved corporate action in related instrument")
        return reader(symbol, directory, **kwargs)

    monkeypatch.setattr(rolling, "read_daily_history", read)
    with pytest.raises(ValueError, match="related instrument"):
        rolling.run_rolling(args)
    assert [row[0] for row in calls] == ["SOXL", "OTHER"]
    assert stub_fits == []
