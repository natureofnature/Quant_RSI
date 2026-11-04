"""Offline historical bounds/calendar regressions; prices are engineering fixtures."""

import json
from datetime import date, datetime, time, timedelta
from urllib.parse import parse_qs, urlsplit

import pytest
from test_daily_data import CALENDAR, EASTERN, NASDAQ_CALENDAR, Response, chart

from quant_rsi.market import daily_data

HOLIDAYS = {
    2024: (
        "01-01",
        "01-15",
        "02-19",
        "03-29",
        "05-27",
        "06-19",
        "07-04",
        "09-02",
        "11-28",
        "12-25",
    ),
    2025: (
        "01-01",
        "01-20",
        "02-17",
        "04-18",
        "05-26",
        "06-19",
        "07-04",
        "09-01",
        "11-27",
        "12-25",
    ),
}


def archive(year):
    days = [date.fromisoformat(f"{year}-{day}") for day in HOLIDAYS[year]]
    return (
        f"NYSE Group Announces {year}, {year + 1} and {year + 2} "
        "Holiday and Early Closings Calendar"
        "<table><tr><td></td><td></td></tr>"
        f"<tr><th>HOLIDAY</th><th>{year}</th></tr>"
        + "".join(f"<tr><td>Holiday</td><td>{day:%A, %B %d}</td></tr>" for day in days)
        + "<tr><td></td><td></td></tr></table>"
        + f"<p>Each market will close early at 1:00 p.m. on July 3, {year}, "
        f"November {29 if year == 2024 else 28}, {year} and December 24, {year}.</p>"
    ).encode()


def install_history(monkeypatch, *, nasdaq=False, missing=None, session_days=None):
    holidays = {date.fromisoformat(f"2024-{day}") for day in HOLIDAYS[2024]}
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(366)]
    days = [day for day in days if day.weekday() < 5 and day not in holidays]
    assert len(days) == 252
    days = [day for day in days if day.isoformat() != missing]
    if session_days is not None:
        days = session_days
    data = chart()
    result = data["chart"]["result"][0]
    result["meta"].update(symbol="SOXX", exchangeName="NMS" if nasdaq else "PCX")
    result["events"] = {}
    result["timestamp"] = [
        int(datetime.combine(day, time(9, 30), EASTERN).timestamp()) for day in days
    ]
    result["indicators"]["quote"][0] = {
        "open": [100.0 + i for i in range(len(days))],
        "close": [101.0 + i for i in range(len(days))],
    }
    pdfs = {year: f"offline reviewed PDF {year}".encode() for year in HOLIDAYS}
    monkeypatch.setattr(daily_data, "NASDAQ_2024_CALENDAR_SHA256", daily_data._sha(pdfs[2024]))
    monkeypatch.setattr(daily_data, "NASDAQ_PAST_CALENDAR_SHA256", daily_data._sha(pdfs[2025]))
    scope = (
        "Nasdaq U.S. equities and options markets will be closed"
        if nasdaq
        else "The exchange will close all NYSE Group equity and options markets"
    )
    notice = (
        f"<p>{scope} on Thursday, January 9, 2025. "
        "National Day of Mourning for President Jimmy Carter.</p>"
    ).encode()
    bodies = {
        daily_data.CALENDAR_URL: CALENDAR,
        daily_data.NASDAQ_CALENDAR_URL: NASDAQ_CALENDAR,
        daily_data.NYSE_2024_CALENDAR_URL: archive(2024),
        daily_data.NYSE_PAST_CALENDAR_URL: archive(2025),
        daily_data.NASDAQ_2024_CALENDAR_URL: pdfs[2024],
        daily_data.NASDAQ_PAST_CALENDAR_URL: pdfs[2025],
        daily_data.NASDAQ_2025_CLOSURE_URL: notice,
        daily_data.NYSE_2025_CLOSURE_URL: notice,
    }
    calls = []

    def source(request, timeout):
        calls.append(request.full_url)
        body = (
            json.dumps(data).encode()
            if request.full_url.startswith(daily_data.CHART_URL)
            else bodies[request.full_url]
        )
        return Response(body, request.full_url)

    monkeypatch.setattr(daily_data, "urlopen", source)
    monkeypatch.setattr(
        daily_data, "_now", lambda: datetime.fromisoformat("2026-11-04T12:00:00+00:00")
    )
    return calls


@pytest.mark.parametrize("nasdaq", [False, True])
def test_2024_collection_requests_cutoff_window_and_retains_official_calendars(
    tmp_path, monkeypatch, nasdaq
):
    calls = install_history(monkeypatch, nasdaq=nasdaq)
    cutoff = datetime.fromisoformat("2024-12-31T23:59:59.750000-05:00")
    result = daily_data.collect_daily(
        "SOXX", tmp_path, historical_sessions=252, future_sessions=10, as_of=cutoff
    )
    query = parse_qs(urlsplit(calls[0]).query)
    assert "range" not in query
    assert int(query["period2"][0]) == int(cutoff.timestamp())
    assert int(query["period1"][0]) == int(cutoff.timestamp()) - 732 * 86400
    assert query["interval"] == ["1d"]
    assert result["origin_session_id"] == "2024-12-31"
    sessions = result["contract"]["calendar"]["sessions"]
    assert len(sessions) == 262
    assert sessions[0]["session_id"] == "2024-01-02"
    assert sessions[251]["session_id"] == "2024-12-31"
    assert "2025-01-09" not in {s["session_id"] for s in sessions}
    evidence = json.loads((tmp_path / "calendar_evidence.json").read_bytes())
    assert [s["year"] for s in evidence["historical_sources"]] == [2024, 2025, 2025]
    receipt = json.loads((tmp_path / "collection_receipt.json").read_bytes())
    for source in evidence["historical_sources"]:
        download = next(d for d in receipt["downloads"] if d["url"] == source["source_url"])
        assert download["sha256"] == source["source_ref"]
        assert download["received_at"].startswith("2026-11-04")
        assert source["source_ref"] in result["evidence_paths"]
    assert "2024-11-29" in evidence["early_closes_1300"]


@pytest.mark.parametrize("nasdaq", [False, True])
def test_missing_2024_open_session_is_not_invented(tmp_path, monkeypatch, nasdaq):
    install_history(monkeypatch, nasdaq=nasdaq, missing="2024-08-14")
    with pytest.raises(ValueError, match="Missing completed exchange sessions"):
        daily_data.collect_daily(
            "SOXX",
            tmp_path,
            historical_sessions=250,
            as_of=datetime.fromisoformat("2024-12-31T23:59:59-05:00"),
        )


def test_2024_pdf_hash_and_archive_heading_fail_closed():
    with pytest.raises(ValueError, match="differs from the reviewed source PDF"):
        daily_data._past_calendar(b"changed PDF", nasdaq=True, year=2024)
    with pytest.raises(ValueError, match="heading is absent"):
        daily_data._past_calendar(archive(2025), nasdaq=False, year=2024)


def test_minimal_history_uses_one_year_bounded_request(tmp_path, monkeypatch):
    calls = install_history(monkeypatch)
    cutoff = datetime.fromisoformat("2024-12-31T23:59:59-05:00")
    daily_data.collect_daily("SOXX", tmp_path, as_of=cutoff)
    query = parse_qs(urlsplit(calls[0]).query)
    assert int(query["period2"][0]) - int(query["period1"][0]) == 366 * 86400


@pytest.mark.parametrize("use_cutoff", [False, True])
@pytest.mark.parametrize("nasdaq", [False, True])
def test_partial_bar_does_not_hide_needed_prior_year_calendar(
    tmp_path, monkeypatch, use_cutoff, nasdaq
):
    closed = {date.fromisoformat(f"2025-{day}") for day in HOLIDAYS[2025]} | {date(2025, 1, 9)}
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(200)]
    days = [day for day in days if day.weekday() < 5 and day not in closed][:120]
    days = [date(2024, 12, 31)] + days
    install_history(monkeypatch, nasdaq=nasdaq, session_days=days)
    partial_time = datetime.combine(days[-1], time(10, 0), EASTERN)
    if not use_cutoff:
        monkeypatch.setattr(daily_data, "_now", lambda: partial_time)
    result = daily_data.collect_daily("SOXX", tmp_path, as_of=partial_time if use_cutoff else None)
    sessions = result["contract"]["calendar"]["sessions"]
    assert sessions[0]["session_id"] == "2024-12-31"
    assert sessions[119]["session_id"] == days[-2].isoformat()
    assert result["origin_session_id"] == days[-2].isoformat()
    evidence = json.loads((tmp_path / "calendar_evidence.json").read_bytes())
    assert evidence["historical_sources"][0]["year"] == 2024
