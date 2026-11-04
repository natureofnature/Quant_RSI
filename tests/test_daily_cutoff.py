"""Historical price truncation keeps actual receipts and calendar-bound targets."""

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pytest
from test_daily_data import EASTERN, chart, install, read_events

from quant_rsi.market import daily_data
from quant_rsi.research.demo_prices import load_daily_prices


def test_asof_excludes_later_quotes_and_actions_without_backdating_receipts(tmp_path, monkeypatch):
    data = chart("2026-11-03")
    events = data["chart"]["result"][0]["events"]
    events["splits"]["split"]["date"] = int(datetime(2026, 11, 2, tzinfo=EASTERN).timestamp())
    events["dividends"]["future-cash"] = {
        "amount": 0.5,
        "date": int(datetime(2026, 11, 2, tzinfo=EASTERN).timestamp()),
    }
    now = "2026-11-04T12:00:00+00:00"
    raw, _ = install(monkeypatch, data, now=now)
    cutoff = datetime.fromisoformat("2026-10-31T23:59:59-04:00")
    result = daily_data.collect_daily(
        "SOXL", tmp_path, historical_sessions=209, future_sessions=20, as_of=cutoff
    )
    rows = read_events(result)
    assert all(datetime.fromisoformat(row["event_at"]) <= cutoff for row in rows)
    assert len(result["cash_actions"]) == 1
    assert not any('"event_kind": "splits"' in row["action"] for row in rows)
    receipt = json.loads((tmp_path / "collection_receipt.json").read_bytes())
    assert receipt["completed_at"] == now
    assert all(item["received_at"] == now for item in receipt["downloads"])
    assert (tmp_path / "yahoo_chart_raw.json").read_bytes() == raw
    prices, info = load_daily_prices("SOXL", tmp_path, horizon_sessions=20, historical_sessions=209)
    assert (len(prices.dates), prices.dates[0], prices.dates[-1]) == (
        209,
        "2026-01-02",
        "2026-10-30",
    )
    assert info["latest_target_session"] == "2026-11-30"
    assert info["price_cutoff"] == cutoff.isoformat()
    assert receipt["unsupported_action_event_ids"] == []


def test_asof_still_rejects_missing_completed_session(tmp_path, monkeypatch):
    install(monkeypatch, chart("2026-10-29"), now="2026-11-04T12:00:00+00:00")
    with pytest.raises(ValueError, match="Latest completed exchange session is absent"):
        daily_data.collect_daily(
            "SOXL", tmp_path, as_of=datetime.fromisoformat("2026-10-31T23:59:59-04:00")
        )


def test_distinct_cutoffs_bind_distinct_source_versions(tmp_path, monkeypatch):
    install(monkeypatch, chart("2026-11-03"), now="2026-11-04T12:00:00+00:00")
    results = [
        daily_data.collect_daily(
            "SOXL",
            tmp_path / str(day),
            as_of=datetime.fromisoformat(f"2026-{day}T23:59:59-04:00"),
        )
        for day in ("10-31", "11-01")
    ]
    assert results[0]["origin_session_id"] == results[1]["origin_session_id"]
    assert results[0]["contract"]["source_id"] != results[1]["contract"]["source_id"]


@pytest.mark.parametrize(
    "cutoff",
    ["2026-10-31", datetime(2026, 10, 31), datetime.fromisoformat("2026-11-05T00:00:00+00:00")],
)
def test_invalid_cutoff_fails_before_collection(tmp_path, monkeypatch, cutoff):
    _, calls = install(monkeypatch, now="2026-11-04T12:00:00+00:00")
    with pytest.raises(ValueError, match="as_of"):
        daily_data.collect_daily("SOXL", tmp_path, as_of=cutoff)
    assert calls == []
    assert not list(tmp_path.iterdir())


def test_soxx_example_freezes_all_peer_inputs_and_rejects_wrong_boundary(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples/soxx_november_study.py"
    spec = importlib.util.spec_from_file_location("soxx_november_study", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    original = daily_data.collect_daily

    def collect(symbol, directory, **kwargs):
        data = chart("2026-11-03")
        data["chart"]["result"][0]["meta"]["symbol"] = symbol
        data["chart"]["result"][0]["events"].pop("splits")
        install(monkeypatch, data, now="2026-11-04T12:00:00+00:00")
        return original(symbol, directory, **kwargs)

    monkeypatch.setattr(example, "collect_daily", collect)
    directory = example.prepare_inputs(tmp_path / "inputs")
    assert {p.name for p in directory.iterdir()} == {"SOXX", "XSD", "SPY"}
    command = example.study_command(directory, tmp_path / "study")
    from quant_rsi.cli import main
    from quant_rsi.research import study

    captured = []

    class CapturedArguments(Exception):
        pass

    def capture(args):
        captured.append(args)
        raise CapturedArguments

    monkeypatch.setattr(study, "run_study", capture)
    with pytest.raises(CapturedArguments):
        main(command[3:])
    args = captured[0]
    assert args.information_mode == "retained"
    assert args.terminal_sessions == 0 and args.horizon_sessions == 20
    assert args.research_cutoff == example.CUTOFF
    assert not args.news_research and args.no_history
    receipt_path = directory / "SPY/collection_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["price_cutoff"] = "2026-11-01T23:59:59-04:00"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="SPY collection does not match"):
        example.prepare_inputs(directory)

    receipt["price_cutoff"] = example.CUTOFF
    receipt["completed_at"] = "2026-11-05T12:00:00+00:00"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="preserved pre-split collections"):
        example.prepare_inputs(directory)

    result = {
        "summary": {
            "latest_forecast": {
                "origin_session": "2026-10-30",
                "target_session": "2026-11-30",
                "reference_price": 450.0,
                "predicted_price": 495.0,
                "training_information_cutoff": "2026-08-31",
                "outcome": "not_observed",
            }
        }
    }
    interpretation = example.forecast_interpretation(result)
    assert interpretation["predicted_price_return"] == pytest.approx(0.1)
    assert interpretation["indicative_post_split_price"] == 165
    assert result["summary"]["latest_forecast"]["predicted_price"] == 495
    output = tmp_path / "study"
    output.mkdir()
    (output / "result.json").write_text(json.dumps(result))
    example.write_forecast_summary(output)
    html = (output / "november_forecast.html").read_text()
    assert "$495.00" in html and "$165.00" in html
    assert 'href="report.html"' in html
    assert example.forecast_interpretation({"summary": {}})["status"] == "unavailable"
    result["summary"]["latest_forecast"]["origin_session"] = "2026-11-03"
    with pytest.raises(ValueError, match="Forecast does not match"):
        example.forecast_interpretation(result)
