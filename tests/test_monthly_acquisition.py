"""Explicit monthly evidence admission; offline fixtures establish no predictive value."""

import calendar
import json
from datetime import date

import pytest
from pydantic import ValidationError

from quant_rsi.core.artifacts import canonical
from quant_rsi.research.discovery_sources import (
    DataRequest,
    DiscoverySources,
    NumericalExtraction,
    _date_in_excerpt,
    align_input_bundle,
    validate_extraction,
)
from quant_rsi.research.rolling_acquired import _replay


def fixture(month="2026-05", *, monthly=True):
    year, number = map(int, month.split("-"))
    start = date(year, number, 1)
    end = date(year, number, calendar.monthrange(year, number)[1])
    label = start.strftime("%B %Y")
    request = {
        "input_id": "acquired.monthly_revenue_growth",
        "symbol": "SOXX",
        "measurement": "Realized current-month revenue year-over-year growth",
        "unit": "%",
        "query": "Monthly issuer disclosure",
        "source_urls": ["https://example.com/monthly-release"],
        "max_age_days": 31,
    }
    if monthly:
        request["period_granularity"] = "calendar_month"
    value = f"Revenue for {label}: YoY increase 30.1%."
    article = {
        "id": "monthly-source",
        "url": request["source_urls"][0],
        "status": "available",
        "received_at": "2026-11-05T12:00:00Z",
        "version_available_at": "2026-11-05T12:00:00Z",
        "text": f"Published June 10, 2026. {value}",
    }
    extracted = {
        "publication_date": "2026-06-10",
        "publication_excerpt": "Published June 10, 2026.",
        "measurement": request["measurement"],
        "observations": [
            {
                "period_start": start.isoformat(),
                "period_end": end.isoformat(),
                "value": 30.1,
                "unit": "%",
                "excerpt": value,
                "period_excerpt": label,
            }
        ],
        "limitations": [],
    }
    return request, article, extracted


def validate(request, article, extracted, *, mode="retrospective"):
    return validate_extraction(
        DataRequest.model_validate_json(canonical(request)),
        article,
        NumericalExtraction.model_validate_json(canonical(extracted)),
        extracted_at="2026-11-05T13:00:00Z",
        information_mode=mode,
        research_cutoff="2026-06-30T20:00:00Z",
    )


@pytest.mark.parametrize("month", ["2026-05", "2026-02", "2024-02", "2025-12"])
def test_opt_in_requires_complete_natural_month(month):
    request, article, extracted = fixture(month)
    observations, gaps = validate(request, article, extracted)
    assert not gaps
    assert len(observations) == 1
    assert observations[0]["period_end"] == extracted["observations"][0]["period_end"]
    assert observations[0]["published_at"] == "2026-06-11T12:00:00+00:00"
    assert observations[0]["received_at"] == "2026-11-05T12:00:00.000000+00:00"


def test_default_request_identity_and_exact_date_contract_remain_unchanged():
    request, article, extracted = fixture(monthly=False)
    parsed = DataRequest.model_validate_json(canonical(request))
    assert parsed.model_dump(mode="json") == request
    explicit_null = DataRequest.model_validate_json(
        canonical(request | {"period_granularity": None})
    )
    assert explicit_null.model_dump(mode="json") == request
    assert validate(request, article, extracted) == ([], ["period_end_not_in_excerpt"])
    opted_in = DataRequest.model_validate_json(
        canonical(request | {"period_granularity": "calendar_month"})
    )
    assert canonical(opted_in.model_dump(mode="json")) != canonical(request)


@pytest.mark.parametrize(
    "changes",
    [
        {"period_start": None},
        {"period_start": "2026-05-02"},
        {"period_end": "2026-05-30"},
        {"period_start": "2026-04-01", "period_end": "2026-04-30"},
        {"period_start": "2025-05-01", "period_end": "2025-05-31"},
        {"period_excerpt": "May 2026 that is not quoted"},
        {"period_start": "2026-07-01", "period_end": "2026-07-31"},
        {"target_quarter": "2026-Q2"},
    ],
)
def test_month_opt_in_rejects_incomplete_mismatched_or_future_periods(changes):
    request, article, extracted = fixture()
    extracted["observations"][0].update(changes)
    observations, gaps = validate(request, article, extracted)
    assert observations == []
    assert gaps


def test_month_quote_cannot_remove_fiscal_context():
    request, article, extracted = fixture()
    article["text"] = article["text"].replace("Revenue for May 2026", "Revenue for fiscal May 2026")
    extracted["observations"][0]["excerpt"] = "Revenue for fiscal May 2026: YoY increase 30.1%."
    assert validate(request, article, extracted) == ([], ["unverified_calendar_month_period"])


def test_calendar_month_and_forecast_are_mutually_exclusive():
    request, _, _ = fixture()
    request["forecast"] = {
        "product": "DRAM",
        "statistic": "lower",
        "basis": "quarter-on-quarter",
        "pairing_group": None,
    }
    with pytest.raises(ValidationError, match="cannot also be forecasts"):
        DataRequest.model_validate_json(canonical(request))


@pytest.mark.parametrize(
    "period",
    [
        "first half of May 2026",
        "first 15 days of May 2026",
        "last 10 trading days of May 2026",
        "May 2026 (first half)",
        "January through May 2026",
        "January 2026 to May 2026",
        "year-to-date through May 2026",
        "YTD May 2026",
    ],
)
def test_month_quote_cannot_hide_explicit_partial_or_cumulative_scope(period):
    request, article, extracted = fixture()
    value = f"Revenue for {period}: YoY increase 30.1%."
    article["text"] = f"Published June 10, 2026. {value}"
    extracted["observations"][0]["excerpt"] = value
    assert validate(request, article, extracted) == ([], ["unverified_calendar_month_period"])


def test_heading_cannot_override_partial_scope_of_the_actual_value():
    request, article, extracted = fixture()
    value = "Revenue for first half of May 2026: YoY increase 30.1%."
    article["text"] = f"Published June 10, 2026. May 2026 update. {value}"
    extracted["observations"][0]["excerpt"] = value
    assert validate(request, article, extracted) == ([], ["unverified_calendar_month_period"])


def test_independent_cumulative_row_does_not_disqualify_a_monthly_value():
    request, article, extracted = fixture()
    article["text"] += " Separate cumulative revenue for January through May 2026: 20.0%."
    observations, gaps = validate(request, article, extracted)
    assert len(observations) == 1
    assert not gaps


@pytest.mark.parametrize("quoted", ["Jan. 9, 2026", "9 Jan. 2026", "JAN. 09 2026"])
def test_abbreviated_month_punctuation_preserves_exact_publication_day(quoted):
    assert _date_in_excerpt("2026-01-09", quoted)
    assert not _date_in_excerpt("2026-01-08", quoted)
    assert not _date_in_excerpt("2026-01-09", "Jan. 2026")
    assert not _date_in_excerpt("2026-01-09", "NotJan. 9, 2026")


def test_month_name_cannot_substitute_for_publication_day():
    request, article, extracted = fixture()
    extracted["publication_excerpt"] = "May 2026"
    assert validate(request, article, extracted) == ([], ["publication_date_not_in_excerpt"])


def test_monthly_admission_preserves_receipt_cutoff_and_existing_age_limit():
    request, article, extracted = fixture()
    assert validate(request, article, extracted, mode="retained") == (
        [],
        ["extraction_after_research_cutoff"],
    )
    observations, _ = validate(request, article, extracted)
    aligned = align_input_bundle(
        DataRequest.model_validate_json(canonical(request)),
        observations,
        [
            "2026-05-31T20:00:00Z",
            "2026-06-10T20:00:00Z",
            "2026-06-11T20:00:00Z",
            "2026-07-13T20:00:00Z",
        ],
    )
    assert aligned["columns"][request["input_id"]] == [None, None, 30.1, None]


def test_monthly_request_runs_through_native_acquisition_and_archive_replay(monkeypatch):
    request, article, extracted = fixture()

    class OfflineExtractor:
        def structured(self, prompt, schema):
            payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
            assert payload["request"]["period_granularity"] == "calendar_month"
            return json.dumps(extracted)

    monkeypatch.setattr("quant_rsi.research.discovery_sources._now", lambda: "2026-11-05T13:00:00Z")
    sources = DiscoverySources(OfflineExtractor(), retained_articles=[article])
    result = sources.acquire(
        request, prediction_times=["2026-06-10T20:00:00Z", "2026-06-11T20:00:00Z"]
    )
    assert result["columns"][request["input_id"]] == [None, 30.1]
    assert result["gaps"] == []
    assert sources.counts == {"search": 0, "read": 0, "extract": 1, "analysis": 0}
    assert _replay(result) == result
