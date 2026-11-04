"""Official exceptional closure admission; synthetic quotes never establish market returns."""

import json
from datetime import datetime, time

import pytest
from test_daily_data import EASTERN, Response
from test_study_horizon import CLOSURE_2025, long_collection

from quant_rsi.market import daily_data


@pytest.mark.parametrize("nasdaq", [False, True])
def test_january_ninth_is_closed_with_retained_exchange_specific_notice(
    tmp_path, monkeypatch, nasdaq
):
    snapshot = long_collection(
        tmp_path,
        monkeypatch,
        "2025-10-31",
        20,
        archived=True,
        nasdaq=nasdaq,
        historical_sessions=209,
    )
    assert len(snapshot["dates"]) == 209
    assert snapshot["dates"][0] == "2025-01-02"
    assert snapshot["dates"][-1] == "2025-10-31"
    assert "2025-01-09" not in snapshot["dates"]
    assert "2025-01-08" in snapshot["dates"] and "2025-01-10" in snapshot["dates"]
    evidence = json.loads((tmp_path / "calendar_evidence.json").read_bytes())
    assert "2025-01-09" in evidence["holidays"]["2025"]
    notice = evidence["historical_sources"][-1]
    url = daily_data.NASDAQ_2025_CLOSURE_URL if nasdaq else daily_data.NYSE_2025_CLOSURE_URL
    assert notice == {
        "year": 2025,
        "source_url": url,
        "source_ref": daily_data._sha(CLOSURE_2025[nasdaq]),
        "method": "published_exceptional_closure_notice",
        "closed_sessions": ["2025-01-09"],
    }
    receipt = json.loads((tmp_path / "collection_receipt.json").read_bytes())
    download = next(row for row in receipt["downloads"] if row["url"] == url)
    assert download["sha256"] == notice["source_ref"]
    assert download["http_status"] == 200
    assert receipt["result"]["evidence_paths"][notice["source_ref"]] == download["path"]
    assert receipt["result"]["calendar_evidence_ref"] == daily_data._sha(
        (tmp_path / "calendar_evidence.json").read_bytes()
    )


@pytest.mark.parametrize("nasdaq", [False, True])
@pytest.mark.parametrize("problem", ["missing_open_day", "invented_closed_day"])
def test_other_missing_bars_and_invented_january_ninth_still_fail(
    tmp_path, monkeypatch, nasdaq, problem
):
    good = tmp_path / "complete"
    long_collection(
        good, monkeypatch, "2025-10-31", 20, archived=True, nasdaq=nasdaq, historical_sessions=209
    )
    payload = json.loads((good / "yahoo_chart_raw.json").read_bytes())
    result = payload["chart"]["result"][0]
    stamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    if problem == "missing_open_day":
        index = next(
            i
            for i, value in enumerate(stamps)
            if datetime.fromtimestamp(value, EASTERN).date().isoformat() == "2025-01-08"
        )
        stamps.pop(index)
        for values in quote.values():
            values.pop(index)
        expected = "Missing completed exchange sessions"
    else:
        opening = datetime.combine(datetime(2025, 1, 9).date(), time(9, 30), EASTERN)
        index = next(i for i, value in enumerate(stamps) if value > opening.timestamp())
        stamps.insert(index, int(opening.timestamp()))
        for values in quote.values():
            values.insert(index, 60)
        expected = "Daily timestamp differs from the official registered opening"
    previous = daily_data.urlopen

    def source(request, timeout):
        if request.full_url.startswith(daily_data.CHART_URL):
            return Response(json.dumps(payload).encode(), request.full_url)
        return previous(request, timeout)

    monkeypatch.setattr(daily_data, "urlopen", source)
    with pytest.raises(ValueError, match=expected):
        daily_data.collect_daily(
            "KO", tmp_path / "invalid", historical_sessions=208, future_sessions=20
        )
    receipt = json.loads((tmp_path / "invalid/collection_receipt.json").read_bytes())
    assert receipt["status"] == "failed"


@pytest.mark.parametrize("nasdaq", [False, True])
def test_closure_notice_requires_matching_exchange_date_and_reason(nasdaq):
    assert daily_data._exceptional_2025_closure(CLOSURE_2025[nasdaq], nasdaq=nasdaq) == {
        datetime(2025, 1, 9).date()
    }
    for changed in (
        CLOSURE_2025[not nasdaq],
        CLOSURE_2025[nasdaq].replace(b"January 9, 2025", b"January 10, 2025"),
        CLOSURE_2025[nasdaq].replace(b"Jimmy Carter", b"Another reason"),
    ):
        with pytest.raises(ValueError, match="differs from reviewed"):
            daily_data._exceptional_2025_closure(changed, nasdaq=nasdaq)
