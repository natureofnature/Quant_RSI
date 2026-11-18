"""MOPS USD amounts and Chinese calendar-quarter guidance (#159 Phase 2).

Measured on #159: the TSM re-run listed all four quarterly releases, but its guidance
(合併營收預計介於322億美元到334億美元之間, under 2025年第四季) could not be acquired. The
verifier knew only TWD and percent units and English quarter labels. Synthetic figures
here are not market evidence.
"""

import json

import pytest
from test_mops_extraction_boundaries import ISSUER, records, validate

from quant_rsi.core.artifacts import digest
from quant_rsi.research import discovery_sources as sources

GUIDANCE = "合併營收預計介於322億美元到334億美元之間"
OUTLOOK = "台積公司2026年第四季的業績展望如下："


@pytest.mark.parametrize(
    "quote,value,unit",
    [
        ("營收預計介於322億美元到334億美元之間", 322, "USD hundred million"),
        ("營收預計介於322億美元到334億美元之間", 334, "USD hundred million"),
        ("資本支出預估將介於520億美元到560億美元之間", 520, "USD hundred million"),
        ("每單位為2.92美元", 2.92, "USD"),
        ("營收51百萬美元", 51, "USD million"),
    ],
)
def test_usd_suffix_amounts_are_literal_units(quote, value, unit):
    observations, reasons = validate(
        *records("2026年9月 " + quote + "。", quote=quote, value=value, unit=unit)
    )
    assert reasons == []
    assert observations[0]["value"] == value
    assert observations[0]["unit"] == unit


@pytest.mark.parametrize(
    "amount,quote,value,unit",
    [
        # A quote cut from a compound amount must not pass as a whole amount.
        ("1兆1,341億美元", "1,341億美元", 1341, "USD hundred million"),
        ("1兆1,341億美元", "1兆1,341億美元", 1341, "USD hundred million"),
        ("1,341億美元", "341億美元", 341, "USD hundred million"),
        ("5118億5700萬美元", "5700萬美元", 5700, "USD"),
        ("5118億5700萬美元", "5118億5700萬美元", 5118, "USD hundred million"),
        # Scale and currency must both match the requested unit.
        ("322億美元", "322億美元", 322, "USD"),
        ("322億美元", "322億美元", 322, "USD million"),
        ("322億美元", "322億美元", 322, "TWD hundred million"),
        ("新台幣51億元", "新台幣51億元", 51, "USD hundred million"),
        # The currency is named in the clause, not next to the amount: still a gap.
        ("若以美元計算，營收為331.0億", "營收為331.0億", 331.0, "USD hundred million"),
    ],
)
def test_usd_amounts_keep_compound_scale_and_currency_boundaries(amount, quote, value, unit):
    observations, reasons = validate(
        *records("2026年9月營收" + amount + "。", quote=quote, value=value, unit=unit)
    )
    assert observations == []
    assert reasons == ["unverified_literal_value_or_unit"]


def guidance(*, outlook=OUTLOOK, quarter="2026-Q4", start="2026-10-01", end="2026-12-31"):
    request, article, extraction = records(
        outlook + "\n" + GUIDANCE + "；", quote=GUIDANCE, value=322, unit="USD hundred million"
    )
    del request["period_granularity"]
    request |= {
        "input_id": "acquired.revenue_guidance_lower",
        "measurement": "Literal lower bound of next-quarter revenue guidance",
        "forecast": {
            "product": "consolidated_revenue",
            "statistic": "lower",
            "basis": "Management quarterly revenue guidance, USD presentation",
            "pairing_group": "revenue_guidance",
        },
    }
    extraction["measurement"] = request["measurement"]
    extraction["observations"][0] |= {
        "period_start": start,
        "period_end": end,
        "target_quarter": quarter,
        "period_excerpt": outlook.rstrip("："),
    }
    return request, article, extraction


def test_mops_guidance_under_a_gregorian_chinese_quarter_label_is_admitted():
    observations, reasons = validate(*guidance())
    assert reasons == []
    assert observations[0]["value"] == 322
    assert observations[0]["published_at"] == "2026-10-09T14:30:00+08:00"


@pytest.mark.parametrize(
    "options",
    [
        {"outlook": "台積公司115年第四季的業績展望如下："},  # ROC year: unsupported
        {"outlook": "台積公司2026年第三季的業績展望如下："},  # label is another quarter
        {"outlook": "台積公司12026年第四季的業績展望如下："},  # not a year boundary
        {"outlook": "台積公司2026會計年度2026年第四季的業績展望如下："},  # fiscal context
        {"start": "2026-10-01", "end": "2026-12-30"},  # not the calendar bounds
    ],
)
def test_chinese_quarter_label_requires_its_exact_calendar_quarter(options):
    observations, reasons = validate(*guidance(**options))
    assert observations == []
    assert reasons == ["unverified_forecast_target_quarter"]


def test_chinese_quarter_label_is_only_read_for_mops_sources():
    request, article, extraction = guidance()
    article["provider"] = "web"
    article["text"] = "2026-10-09\n" + article["text"]
    article["content_sha256"] = digest(article["text"].encode())
    extraction["publication_excerpt"] = "2026-10-09"
    _, reasons = sources.validate_extraction(
        sources.DataRequest.model_validate_json(json.dumps(request)),
        article,
        sources.NumericalExtraction.model_validate_json(json.dumps(extraction)),
        extracted_at="2026-10-10T12:01:00Z",
        information_mode="retrospective",
        research_cutoff=None,
    )
    assert "unverified_forecast_target_quarter" in reasons


def provider_for(article, extraction, prompts):
    class Transport:
        def structured(self, prompt, schema):
            prompts.append(prompt)
            return json.dumps(extraction)

        def search(self, *args):
            pytest.fail("Exact retained source must not search")

    return sources.DiscoverySources(
        Transport(),
        retained_articles=[article],
        disclosure_profiles=[ISSUER],
        reader=lambda _: pytest.fail("Exact retained source must not fetch"),
    )


def test_unsupported_unit_is_a_gap_without_an_extraction_call():
    request, article, extraction = guidance()
    request["unit"] = "USD 100 million"  # the unit the 2026-11-18 run requested
    prompts = []
    provider = provider_for(article, extraction, prompts)
    bundle = provider.acquire(request, prediction_times=["2026-10-12T20:00:00Z"])
    assert prompts == []
    assert provider.counts["extract"] == 0
    assert bundle["status"] == "needs_data"
    [gap] = bundle["gaps"]
    assert "unsupported_mops_unit: 'USD 100 million'" in gap
    assert "USD hundred million" in gap


def test_supported_usd_unit_is_extracted_and_aligned(monkeypatch):
    monkeypatch.setattr(sources, "_now", lambda: "2026-10-10T12:01:00Z")
    request, article, extraction = guidance()
    prompts = []
    provider = provider_for(article, extraction, prompts)
    bundle = provider.acquire(
        request, prediction_times=["2026-10-09T06:30:00Z", "2026-10-09T06:30:01Z"]
    )
    assert provider.counts["extract"] == 1
    assert bundle["columns"][request["input_id"]] == [None, 322]
    assert "USD hundred million" in prompts[0] and "億美元" in prompts[0]


def test_decision_prompt_names_every_supported_mops_unit():
    seen = []

    class Transport:
        def structured(self, prompt, schema):
            seen.append(prompt)
            return json.dumps(
                {"action": "stop", "reason": "Done", "candidates": [], "data_requests": []}
                | {"query": None}
            )

    sources.discovery_decision(Transport(), {"symbol": "TSM"})
    for unit in sources.MOPS_LITERAL_UNITS:
        assert unit in seen[0]
    assert "data_requests.unit must be exactly one of these names" in seen[0]
