"""publication_date is a calendar date; the official time comes from source fields.

Regression from the 2026-11-18 TSM preparation run: the provider returned the MOPS
timestamp "2025-10-16T15:06:48+08:00" as publication_date and both guidance extractions
were rejected with the generic invalid_source_dates.
"""

import json
import re

from test_mops_extraction_boundaries import records, validate

from quant_rsi.research.discovery_sources import (
    CALENDAR_DATE,
    NumericalExtraction,
    ProviderNumericalExtraction,
    strict_schema,
)


def test_timestamp_in_publication_date_has_a_specific_reason():
    request, article, extraction = records()
    extraction["publication_date"] = "2026-10-09T14:30:00+08:00"
    assert validate(request, article, extraction) == (
        [],
        ["publication_date_not_calendar_date"],
    )


def test_calendar_date_from_the_same_source_is_admitted():
    request, article, extraction = records()
    observations, reasons = validate(request, article, extraction)
    assert not reasons and len(observations) == 1
    assert observations[0]["published_at"] == "2026-10-09T14:30:00+08:00"


def test_provider_schema_requires_a_calendar_date_but_stored_records_stay_readable():
    schema = strict_schema(ProviderNumericalExtraction)["properties"]["publication_date"]
    patterns = [branch.get("pattern") for branch in schema["anyOf"] if "pattern" in branch]
    assert patterns == [CALENDAR_DATE]
    assert re.fullmatch(CALENDAR_DATE, "2025-10-16")
    assert not re.fullmatch(CALENDAR_DATE, "2025-10-16T15:06:48+08:00")
    # Archives keep earlier rejected answers; they must still parse for replay.
    stored = {
        "publication_date": "2025-10-16T15:06:48+08:00",
        "publication_excerpt": "發言日期: 114/10/16",
        "measurement": "m",
        "observations": [],
        "limitations": [],
    }
    assert NumericalExtraction.model_validate_json(json.dumps(stored))
