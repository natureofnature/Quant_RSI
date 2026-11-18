"""A bounded month range lists each month but is charged as one search (#159 Phase 2).

Measured blocker on #159: monthly MOPS listings with a 2-search budget could not reach the
quarterly releases a 252-session window needs.
"""

from email.message import Message

import pytest
from test_discovery_disclosures import DisclosureLLM, PostArchive, source
from test_invalid_llm_answers import Cache

from quant_rsi.core.artifacts import canonical
from quant_rsi.research import article_reader, mops_source
from quant_rsi.research.discovery_sources import DataRequest, DisclosureQuery

SINGLE = {"provider": "mops", "year": 2026, "month": 10}
RANGE = SINGLE | {"year": 2026, "month": 8, "end_year": 2026, "end_month": 10}


def survey(query):
    return {"symbol": "TSM", "query": "Investigate operating demand", "disclosure_query": query}


def listed_months(archive):
    return [
        (int(body["year"]) + 1911, int(body["month"]))
        for url, body in archive.calls
        if url == mops_source.LIST_URL
    ]


def selection_documents(llm):
    return [call["documents"] for call in llm.calls if "documents" in call]


@pytest.fixture
def archive(monkeypatch):
    value = PostArchive()
    monkeypatch.setattr(article_reader, "_fetch", value)
    return value


def test_single_month_query_keeps_its_stored_form():
    assert canonical(DisclosureQuery.model_validate(SINGLE).model_dump(mode="json")) == canonical(
        SINGLE
    )
    request = {
        "input_id": "acquired.demand",
        "symbol": "TSM",
        "measurement": "Reported operating demand",
        "unit": "percent",
        "query": "Find the declared operating demand",
        "disclosure_query": SINGLE,
    }
    assert DataRequest.model_validate(request).model_dump(mode="json")["disclosure_query"] == SINGLE


@pytest.mark.parametrize(
    "query",
    [
        SINGLE | {"end_year": 2026},  # end_month missing
        RANGE | {"end_year": 2027, "end_month": 8},  # 13 months
        RANGE | {"end_year": 2026, "end_month": 7},  # ends before it starts
    ],
)
def test_range_must_be_complete_ordered_and_bounded(query):
    with pytest.raises(ValueError):
        DisclosureQuery.model_validate(query)


def test_range_lists_every_month_once_and_charges_one_search(tmp_path, archive):
    llm = DisclosureLLM()
    engine = source(tmp_path, llm)
    result = engine.survey(survey(RANGE))
    assert listed_months(archive) == [(2026, 8), (2026, 9), (2026, 10)]
    assert engine.counts["search"] == 1
    # Every listed month's documents are offered to one selection call.
    assert [len(documents) for documents in selection_documents(llm)] == [6]
    assert len(result["articles"]) == 1
    assert "listings" in result["search"] and len(result["search"]["listings"]) == 3


def test_overlapping_range_reuses_listed_months(tmp_path, archive):
    cache = Cache()
    engine = source(tmp_path, call=cache)
    engine.survey(survey(SINGLE))
    engine.survey(survey(RANGE | {"year": 2026, "month": 9}))
    assert listed_months(archive) == [(2026, 10), (2026, 9)]
    assert engine.counts["search"] == 2

    replay = source(tmp_path, call=cache)
    replay.survey(survey(RANGE | {"year": 2026, "month": 9}))
    assert listed_months(archive) == [(2026, 10), (2026, 9)]  # nothing fetched again
    assert replay.counts["search"] == 0


def test_range_still_respects_the_search_allowance(tmp_path, archive):
    engine = source(tmp_path, max_searches=1)
    engine.survey(survey(RANGE))
    result = engine.survey(survey(RANGE | {"year": 2026, "month": 5, "end_month": 6}))
    assert listed_months(archive) == [(2026, 8), (2026, 9), (2026, 10)]
    assert any("allowance exhausted" in gap for gap in result["gaps"])


def test_one_unavailable_month_does_not_block_the_others(tmp_path, monkeypatch):
    class FlakyArchive(PostArchive):
        def __call__(self, url, *, json_body):
            if url == mops_source.LIST_URL and json_body["month"] == "09":
                self.calls.append((url, json_body))
                headers = Message()
                headers["Content-Type"] = "application/json"
                return 503, headers, b""
            return super().__call__(url, json_body=json_body)

    archive = FlakyArchive()
    monkeypatch.setattr(article_reader, "_fetch", archive)
    llm = DisclosureLLM()
    result = source(tmp_path, llm).survey(survey(RANGE))
    assert listed_months(archive) == [(2026, 8), (2026, 9), (2026, 10)]
    assert [len(documents) for documents in selection_documents(llm)] == [4]
    unavailable = result["search"]["unavailable_months"]
    assert [item["reason"] for item in unavailable] == ["http_503"]
