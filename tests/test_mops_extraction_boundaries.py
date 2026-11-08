"""Offline MOPS literal boundaries; synthetic figures are not market evidence."""

import json
from copy import deepcopy

import pytest

from quant_rsi.core.artifacts import digest
from quant_rsi.research import discovery_sources as sources
from quant_rsi.research.mops_source import DETAIL_URL

ISSUER = {
    "symbol": "TSM",
    "provider": "mops",
    "company_id": "2330",
    "company_name": "Offline issuer fixture",
    "identity_source": "https://example.com/fixture-identity",
}


def records(body="2026年9月營收新台幣51元。", *, quote="新台幣51元", value=51, unit="TWD"):
    text = "發言日期: 115/10/09\n發言時間: 14:30:00\n說明:\n" + body
    article = {
        "id": "source-mops-fixture",
        "url": DETAIL_URL,
        "provider": "mops",
        "status": "available",
        "text": text,
        "content_sha256": digest(text.encode()),
        "publication_date_raw": "115/10/09",
        "publication_time_raw": "14:30:00",
        "published_at": "2026-10-09T14:30:00+08:00",
        "received_at": "2026-10-10T12:00:00Z",
        "version_available_at": "2026-10-10T12:00:00Z",
        "company_id": "2330",
        "issuer_symbol": "TSM",
        "issuer_mapping": deepcopy(ISSUER),
    }
    request = {
        "input_id": "acquired.monthly_measurement",
        "symbol": "TSM",
        "measurement": "Literal reported monthly measurement",
        "unit": unit,
        "query": "Offline source only",
        "source_ids": [article["id"]],
        "period_granularity": "calendar_month",
    }
    extraction = {
        "publication_date": "2026-10-09",
        "publication_excerpt": "發言日期: 115/10/09",
        "measurement": request["measurement"],
        "observations": [
            {
                "period_start": "2026-09-01",
                "period_end": "2026-09-30",
                "value": value,
                "unit": unit,
                "excerpt": quote,
                "period_excerpt": "2026年9月",
            }
        ],
        "limitations": ["Synthetic fixture; no actual issuer observation."],
    }
    return request, article, extraction


def validate(
    request,
    article,
    extraction,
    *,
    mode="retrospective",
    cutoff=None,
    extracted_at="2026-10-10T12:01:00Z",
):
    return sources.validate_extraction(
        sources.DataRequest.model_validate_json(json.dumps(request)),
        article,
        sources.NumericalExtraction.model_validate_json(json.dumps(extraction)),
        extracted_at=extracted_at,
        information_mode=mode,
        research_cutoff=cutoff,
    )


@pytest.mark.parametrize(
    "quote,value,unit",
    [
        ("新台幣5118億5700萬元", 5118, "TWD"),
        ("新台幣5118億5700萬元", 5700, "TWD thousand"),
        ("新台幣5118億5700萬元", 5118.57, "TWD hundred million"),
        ("新台幣5118億元5700萬元", 5118, "TWD hundred million"),
        ("年增12.3%", 12.3, "TWD"),
        ("新台幣51元", 51, "percent"),
        ("新台幣51百萬元", 51, "TWD"),
        ("新台幣51元", 51, "TWD million"),
        ("2026年9月營收新台幣51元", 2026, "TWD"),
        ("人民幣51元", 51, "TWD"),
        ("新台幣51億元", 51, "TWD billion"),
    ],
)
def test_mops_rejects_compound_conversions_and_unbound_units(quote, value, unit):
    observations, reasons = validate(
        *records("2026年9月 " + quote + "。", quote=quote, value=value, unit=unit)
    )
    assert observations == []
    assert reasons == ["unverified_literal_value_or_unit"]


@pytest.mark.parametrize(
    "amount,quote,value,unit",
    [
        ("新台幣5億元9000萬元", "新台幣5億元", 5, "TWD hundred million"),
        ("新台幣5億元 9000萬元", "新台幣5億元 9000萬元", 5, "TWD hundred million"),
        ("新台幣5億元\n9000萬元", "新台幣5億元", 5, "TWD hundred million"),
        ("新台幣5億元又9000萬", "新台幣5億元", 5, "TWD hundred million"),
        ("新台幣5億元，9000萬元", "新台幣5億元", 5, "TWD hundred million"),
        ("年增12.3%", "2.3%", 2.3, "percent"),
        ("年增-12.3%", "12.3%", 12.3, "percent"),
    ],
)
def test_mops_cannot_truncate_original_amount_boundaries(amount, quote, value, unit):
    observations, reasons = validate(
        *records("2026年9月營收" + amount + "。", quote=quote, value=value, unit=unit)
    )
    assert observations == []
    assert reasons == ["unverified_literal_value_or_unit"]


def test_mops_complete_amount_before_an_independent_value_is_eligible():
    observations, reasons = validate(
        *records(
            "2026年9月營收新台幣5億元，營業利益新台幣9000萬元。",
            quote="新台幣5億元",
            value=5,
            unit="TWD hundred million",
        )
    )
    assert reasons == []
    assert observations[0]["value"] == 5


@pytest.mark.parametrize(
    "body",
    [
        "2026年9月營收報告\n累計營收\n新台幣51元。",
        "2026年1月至9月營收新台幣51元。\n2026年9月營收報告",
        "2026年1月至2026年9月營收新台幣51元。",
        "2026年9月上半月營收新台幣51元。",
        "2026年9月下半月營收新台幣51元。",
        "2026年9月上旬營收新台幣51元。",
        "2026年9月中旬營收新台幣51元。",
        "2026年9月下旬營收新台幣51元。",
        "2026年9月營收新台幣51元（累計）。",
        "2026年9月營收新台幣51元（前九個月）。",
        "2026年9月營收新台幣51元（九個月合計）。",
        "2026年9月月初至今營收新台幣51元。",
    ],
)
def test_mops_rejects_ytd_and_partial_month_values(body):
    observations, reasons = validate(*records(body))
    assert observations == []
    assert reasons == ["unverified_calendar_month_period"]


@pytest.mark.parametrize(
    "quote,value,unit",
    [
        ("新台幣51元", 51, "TWD"),
        ("新臺幣1,234.5元", 1234.5, "TWD"),
        ("NT$51千元", 51, "TWD thousand"),
        ("新台幣51仟元", 51, "TWD thousand"),
        ("TWD 51百萬元", 51, "TWD million"),
        ("新台幣5,118.57億元", 5118.57, "TWD hundred million"),
        ("年增12.3%", 12.3, "percent"),
        ("年增12.3％", 12.3, "%"),
        ("年增−1.2%", -1.2, "percent"),
    ],
)
def test_mops_preserves_literal_scale_without_conversion(quote, value, unit):
    request, article, extraction = records(
        "2026年9月 " + quote + "。", quote=quote, value=value, unit=unit
    )
    observations, reasons = validate(request, article, extraction)
    assert reasons == []
    assert len(observations) == 1
    assert observations[0]["value"] == value
    assert observations[0]["unit"] == unit
    assert observations[0]["source_id"] == article["id"]
    assert observations[0]["source_sha256"] == article["content_sha256"]


@pytest.mark.parametrize(
    "body",
    [
        "累計營收新台幣99元。2026年9月營收新台幣51元。",
        "累計營收新台幣99元\n2026年9月營收新台幣51元",
        "2026年9月營收新台幣51元\n累計營收新台幣99元",
        "2026年9月營收新台幣51元，累計營收新台幣99元。",
    ],
)
def test_independent_cumulative_amount_does_not_reject_monthly_value(body):
    observations, reasons = validate(*records(body))
    assert not reasons
    assert observations[0]["value"] == 51


def test_roc_publication_clock_aligns_at_taipei_instant_not_utc_day():
    request, article, extraction = records()
    article["text"] = article["text"].replace("14:30:00", "00:30:00")
    article["publication_time_raw"] = "00:30:00"
    article["published_at"] = "2026-10-08T16:30:00Z"
    observations, reasons = validate(request, article, extraction)
    assert not reasons
    assert observations[0]["published_at"] == "2026-10-09T00:30:00+08:00"
    bundle = sources.align_input_bundle(
        request, observations, ["2026-10-08T16:29:59Z", "2026-10-08T16:30:00Z"]
    )
    assert bundle["columns"][request["input_id"]] == [None, 51]


@pytest.mark.parametrize(
    "field,value",
    [
        ("publication_date_raw", "115/02/30"),
        ("publication_time_raw", "25:30:00"),
        ("published_at", "2026-10-09T14:30:00Z"),
        ("received_at", "2026-10-09T06:29:59Z"),
    ],
)
def test_invalid_or_inconsistent_official_dates_are_not_admitted(field, value):
    request, article, extraction = records()
    article[field] = value
    observations, reasons = validate(request, article, extraction)
    assert not observations
    assert reasons


def test_roc_month_label_remains_explicit_gap_without_supported_period_parser():
    request, article, extraction = records("115年9月營收新台幣51元。")
    extraction["observations"][0]["period_excerpt"] = "115年9月"
    observations, reasons = validate(request, article, extraction)
    assert not observations
    assert reasons == ["unverified_calendar_month_period"]


@pytest.mark.parametrize("latest", ["received_at", "version_available_at", "extracted_at"])
def test_retained_availability_uses_latest_receipt_version_or_extraction(latest):
    request, article, extraction = records()
    extracted_at = "2026-10-10T12:01:00Z"
    later = "2026-10-11T12:00:00Z"
    if latest == "extracted_at":
        extracted_at = later
    else:
        article[latest] = later
    options = {"mode": "retained", "extracted_at": extracted_at}
    observations, reasons = validate(
        request, article, extraction, cutoff="2026-10-11T11:59:59Z", **options
    )
    assert observations == []
    assert reasons == ["extraction_after_research_cutoff"]
    observations, reasons = validate(request, article, extraction, cutoff=later, **options)
    assert not reasons
    assert observations[0]["effective_available_at"] == "2026-10-11T12:00:00+00:00"
    assert observations[0]["availability_basis"] == "observed_receipts"


@pytest.mark.parametrize(
    "quote,value,unit,available",
    [
        ("新台幣51百萬元", 51, "TWD million", True),
        ("新台幣5118億5700萬元", 5118, "TWD", False),
    ],
)
def test_exact_source_acquisition_only_aligns_verified_literal_units(
    monkeypatch, quote, value, unit, available
):
    request, article, extraction = records(
        "2026年9月 " + quote + "。", quote=quote, value=value, unit=unit
    )

    class Transport:
        def structured(self, prompt, schema):
            return json.dumps(extraction)

        def search(self, *args):
            pytest.fail("Exact retained source must not search")

    monkeypatch.setattr(sources, "_now", lambda: "2026-10-10T12:01:00Z")
    provider = sources.DiscoverySources(
        Transport(),
        retained_articles=[article],
        disclosure_profiles=[ISSUER],
        reader=lambda _: pytest.fail("Exact retained source must not fetch"),
    )
    bundle = provider.acquire(
        request, prediction_times=["2026-10-09T06:29:59Z", "2026-10-09T06:30:00Z"]
    )
    assert bundle["columns"][request["input_id"]] == [None, value if available else None]
    assert bundle["status"] == ("available" if available else "needs_data")
    assert provider.counts == {"search": 0, "read": 0, "extract": 1, "analysis": 0}
    if not available:
        assert bundle["gaps"] == ["unverified_literal_value_or_unit"]
