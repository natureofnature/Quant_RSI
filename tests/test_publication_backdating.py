"""An acquired value cannot be dated earlier than the evidence of its own publication."""

import json

import pytest
from test_discovery_sources import article, extraction, request
from test_forecast_acquisition import (
    checked,
    forecast_article,
    forecast_extraction,
    forecast_request,
)

from quant_rsi.research.discovery_sources import (
    DataRequest,
    NumericalExtraction,
    validate_extraction,
)


def validate(source, result):
    return validate_extraction(
        DataRequest.model_validate_json(json.dumps(request())),
        source,
        NumericalExtraction.model_validate_json(json.dumps(result)),
        extracted_at="2026-11-04T10:00:00Z",
        information_mode="retrospective",
        research_cutoff=None,
    )


def quarter_release(publication_date, excerpt, period_excerpt=None):
    """A release dated April 25 that also quotes its own quarter-end date."""
    text = (
        "April 25, 2024 -- KO reported results for the quarter ended March 31, 2024. "
        "Unit case volume grew 3.0 percent for period ending 2024-03-31."
    )
    source = article() | {"text": text}
    source.pop("published_at")
    result = extraction() | {"publication_date": publication_date, "publication_excerpt": excerpt}
    result["observations"][0] |= {
        "period_end": "2024-03-31",
        "excerpt": "Unit case volume grew 3.0 percent for period ending 2024-03-31.",
    }
    if period_excerpt is not None:
        result["observations"][0]["period_excerpt"] = period_excerpt
    return source, result


def test_quarter_end_date_cannot_be_claimed_as_publication_date():
    source, result = quarter_release("2024-03-31", "quarter ended March 31, 2024")
    rows, reasons = validate(source, result)
    assert not rows
    assert reasons == ["invalid_period"]


def test_period_description_cannot_serve_as_publication_evidence():
    source, result = quarter_release(
        "2024-03-31", "quarter ended March 31, 2024", "the quarter ended March 31, 2024"
    )
    assert validate(source, result) == ([], ["publication_excerpt_is_period_text"])


def test_release_dateline_is_still_accepted():
    source, result = quarter_release("2024-04-25", "April 25, 2024")
    rows, reasons = validate(source, result)
    assert len(rows) == 1 and not reasons
    assert rows[0]["effective_available_at"] == "2024-04-26T12:00:00+00:00"


@pytest.mark.parametrize(("period_end", "admitted"), [("2025-01-03", False), ("2025-01-02", True)])
def test_realized_period_must_end_before_its_publication_day(period_end, admitted):
    result = extraction()
    result["observations"][0]["period_end"] = period_end
    source = article()
    source["text"] = source["text"].replace("2024-12-31", period_end)
    result["observations"][0]["excerpt"] = result["observations"][0]["excerpt"].replace(
        "2024-12-31", period_end
    )
    rows, reasons = validate(source, result)
    assert bool(rows) is admitted
    assert reasons == ([] if admitted else ["invalid_period"])


def test_claimed_publication_cannot_precede_source_metadata():
    source = article() | {"published_at": "2025-01-10T12:00:00Z"}
    rows, reasons = validate(source, extraction())
    assert (rows, reasons) == ([], ["publication_before_source_metadata"])


def test_metadata_floor_tolerates_one_day_of_timezone_difference():
    source = article() | {"published_at": "2025-01-04T02:00:00Z"}
    rows, reasons = validate(source, extraction())
    assert len(rows) == 1 and not reasons


def test_forecast_vintage_cannot_be_backdated_to_an_earlier_quoted_date():
    source = forecast_article(2)
    source |= {
        "published_at": "2026-02-02T09:00:00Z",
        "text": "Earlier guidance dated 5 January 2026 was lower. " + source["text"],
    }
    result = forecast_extraction(2) | {
        "publication_date": "2026-01-05",
        "publication_excerpt": "5 January 2026",
    }
    assert checked(forecast_request(), source, result) == (
        [],
        ["publication_before_source_metadata"],
    )
    assert checked(forecast_request(), source, forecast_extraction(2))[0]
