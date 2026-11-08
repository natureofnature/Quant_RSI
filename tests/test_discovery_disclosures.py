"""Official POST documents stay distinct through survey, extraction and checkpoint replay."""

import json
from copy import deepcopy
from datetime import date, timedelta
from email.message import Message

import pytest
from test_discovery_loop import cache_at

from quant_rsi.core.artifacts import canonical
from quant_rsi.research import article_reader, mops_source
from quant_rsi.research.discovery_sources import DataRequest, DiscoverySources

PROFILE = {
    "symbol": "TSM",
    "provider": "mops",
    "company_id": "2330",
    "company_name": "Fixture foundry",
    "identity_source": "https://example.com/issuer/TSM",
}
QUERY = {"provider": "mops", "year": 2026, "month": 10}


class PostArchive:
    """Two fictional records share the real provider shape, never a real network call."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, *, json_body):
        self.calls.append((url, deepcopy(json_body)))
        company = json_body["companyId"]
        if url == mops_source.LIST_URL:
            raw_day = f"{json_body['year']}/{json_body['month']}/03"
            titles = ["公司代號", "公司名稱", "發言日期", "發言時間", "主旨", "詳細資料"]
            rows = [
                [
                    company,
                    "Fixture foundry",
                    raw_day,
                    "13:00:00",
                    f"Fixture disclosure {serial}",
                    {
                        "apiName": "t05st01_detail",
                        "parameters": {
                            "marketKind": "sii",
                            "enterDate": raw_day.replace("/", ""),
                            "serialNumber": str(serial),
                            "companyId": company,
                        },
                    },
                ]
                for serial in (1, 2)
            ]
        else:
            assert url == mops_source.DETAIL_URL
            raw = json_body["enterDate"]
            raw_day = f"{raw[:-4]}/{raw[-4:-2]}/{raw[-2:]}"
            year, month = int(raw[:-4]) + 1911, int(raw[-4:-2])
            period = date(year, month, 1) - timedelta(days=1)
            serial = int(json_body["serialNumber"])
            value = 99.0 if serial == 1 else 12.0
            body = f"Reported operating demand {value} percent for period ending {period}."
            titles = ["序號", "發言日期", "發言時間", "主旨", "說明"]
            rows = [[str(serial), raw_day, "13:00:00", f"Fixture disclosure {serial}", body]]
        payload = {
            "code": 200,
            "result": {
                "companyId": company,
                "titles": [{"main": t, "sub": []} for t in titles],
                "data": rows,
            },
        }
        headers = Message()
        headers["Content-Type"] = "application/json"
        return 200, headers, canonical(payload)


class DisclosureLLM:
    mode, model = "fixture", "scripted-disclosure-fixture"

    def __init__(self, selected=(2,), fail_review=False):
        self.selected, self.fail_review = selected, fail_review
        self.calls = []
        self.schemas = []

    def search(self, *args, **kwargs):
        pytest.fail("Official archive queries and exact source IDs must not invoke web search")

    def structured(self, prompt, schema):
        data = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
        self.calls.append(deepcopy(data))
        self.schemas.append(deepcopy(schema))
        if "documents" in data:
            by_serial = {
                int(d["selector"]["serialNumber"]): d["document_ref"] for d in data["documents"]
            }
            return json.dumps(
                {
                    "document_refs": [by_serial.get(n, "unknown-document") for n in self.selected],
                    "reason": "Fixture chooses the demand document, not the first row",
                }
            )
        if "article" in data:
            article = data["article"]
            published = date.fromisoformat(article["published_at"][:10])
            period = published.replace(day=1) - timedelta(days=1)
            serial = int(article["source_request"]["serialNumber"])
            value = 99.0 if serial == 1 else 12.0
            return json.dumps(
                {
                    "publication_date": published.isoformat(),
                    "publication_excerpt": "發言日期: " + article["publication_date_raw"],
                    "measurement": data["request"]["measurement"],
                    "observations": [
                        {
                            "period_start": None,
                            "period_end": period.isoformat(),
                            "value": value,
                            "unit": "percent",
                            "excerpt": (
                                f"Reported operating demand {value} percent "
                                f"for period ending {period}."
                            ),
                        }
                    ],
                    "limitations": ["Offline fixture; no predictive quality evidence"],
                }
            )
        assert "articles" in data
        if self.fail_review:
            raise RuntimeError("survey review interrupted")
        return json.dumps(
            {"findings": [], "gaps": ["Fixture body only; hypothesis remains untested"]}
        )


def source(tmp_path, llm=None, **kwargs):
    return DiscoverySources(
        llm or DisclosureLLM(),
        disclosure_profiles=[PROFILE],
        disclosure_root=tmp_path / "raw",
        **kwargs,
    )


def survey_request():
    return {
        "symbol": "TSM",
        "query": "Investigate reported operating demand",
        "disclosure_query": QUERY,
    }


def measurement(**extra):
    return {
        "input_id": "acquired.demand",
        "symbol": "TSM",
        "measurement": "Reported operating demand",
        "unit": "percent",
        "query": "Find the declared operating demand",
        "max_age_days": 185,
        **extra,
    }


@pytest.fixture
def archive(monkeypatch):
    value = PostArchive()
    monkeypatch.setattr(article_reader, "_fetch", value)
    return value


def test_llm_selects_second_document_and_exact_source_id_does_not_mix_same_url(tmp_path, archive):
    engine = source(tmp_path, DisclosureLLM(selected=(2, 1)))
    result = engine.survey(survey_request())
    assert [r[1]["serialNumber"] for r in archive.calls[1:]] == ["2", "1"]
    articles = result["articles"]
    assert len({a["url"] for a in articles}) == 1
    assert len({a["id"] for a in articles}) == len({a["document_ref"] for a in articles}) == 2
    selected = next(a for a in articles if a["source_request"]["serialNumber"] == "2")
    acquired = engine.acquire(
        measurement(source_ids=[selected["id"]]), prediction_times=["2026-10-05T21:00:00Z"]
    )
    assert acquired["status"] == "available"
    assert [o["value"] for o in acquired["observations"]] == [12.0]
    assert {o["source_id"] for o in acquired["observations"]} == {selected["id"]}
    assert len(archive.calls) == 3 and engine.counts["extract"] == 1
    assert all(a["id"] == selected["id"] for a in acquired["articles"])


def test_unknown_id_wrong_issuer_and_shared_post_url_never_trigger_fallback_search(
    tmp_path, archive
):
    engine = source(tmp_path)
    selected = engine.survey(survey_request())["articles"][0]
    requests = [
        measurement(source_ids=["unknown-source"]),
        measurement(source_ids=[selected["id"]], symbol="KO"),
        measurement(source_urls=[mops_source.DETAIL_URL]),
        measurement(source_urls=[mops_source.DETAIL_URL + "?companyId=2330"]),
        measurement(source_urls=[mops_source.LIST_URL + "#announcement"]),
    ]
    # Source URLs are not usable identities for a structured POST disclosure.
    for request in requests:
        before = len(archive.calls)
        result = engine.acquire(request, prediction_times=["2026-10-05T21:00:00Z"])
        assert result["status"] == "needs_data" and result["observations"] == []
        assert len(archive.calls) == before
    assert engine.counts["extract"] == 0


def test_structured_search_link_is_not_read_as_get_but_normal_pages_are(tmp_path, archive):
    engine = source(tmp_path)
    assert not mops_source.is_post_endpoint("https://[broken")
    rejected = engine._read(mops_source.DETAIL_URL + "?companyId=2330")
    assert rejected["reason"] == "structured_disclosure_requires_source_ids"
    assert rejected["fetch_attempted"] is False
    assert engine.counts["read"] == 0 and archive.calls == []
    calls = []

    def reader(url):
        calls.append(url)
        return {"url": url, "status": "available", "text": "Ordinary article body"}

    engine.reader = reader
    url = "https://mops.twse.com.tw/ordinary-article.html"
    assert engine._read(url)["status"] == "available"
    assert calls == [url] and engine.counts["read"] == 1


@pytest.mark.parametrize(
    "selectors",
    [
        {"source_ids": ["source-1"], "source_urls": ["https://example.com/body"]},
        {"source_ids": ["source-1"], "disclosure_query": QUERY},
        {"source_urls": ["https://example.com/body"], "disclosure_query": QUERY},
    ],
)
def test_source_selectors_are_mutually_exclusive(selectors):
    with pytest.raises(ValueError, match="Choose source_urls, source_ids or disclosure_query"):
        DataRequest.model_validate_json(json.dumps(measurement(**selectors)))


def test_official_archive_acquisition_can_request_another_month(tmp_path, archive):
    engine = source(tmp_path)
    engine.survey(survey_request())
    result = engine.acquire(
        measurement(disclosure_query=QUERY | {"month": 9}),
        prediction_times=["2026-09-04T21:00:00Z"],
    )
    assert result["status"] == "available"
    assert [o["period_end"] for o in result["observations"]] == ["2026-08-31"]
    assert [body["month"] for url, body in archive.calls if url == mops_source.LIST_URL] == [
        "10",
        "09",
    ]


@pytest.mark.parametrize(
    "searches,reads,selected,reason",
    [
        (0, 1, (2,), "Discovery search allowance exhausted"),
        (1, 0, (2,), "Discovery read allowance exhausted"),
        (1, 1, (1, 2), "invalid_disclosure_selection"),
        (1, 1, (99,), "invalid_disclosure_selection"),
    ],
)
def test_source_budgets_and_llm_selection_are_enforced(
    tmp_path, archive, searches, reads, selected, reason
):
    engine = source(
        tmp_path, DisclosureLLM(selected=selected), max_searches=searches, max_reads=reads
    )
    result = engine.survey(survey_request())
    assert result["search"]["reason"] == reason and result["articles"] == []
    assert engine.counts["read"] == 0
    assert len(archive.calls) == searches
    assert engine.counts["analysis"] == int(bool(searches and reads))


def test_resume_reuses_official_list_selection_and_details_after_review_failure(tmp_path, archive):
    options = {
        "call": cache_at(tmp_path).call,
        "usage_path": tmp_path / "usage.json",
        "max_searches": 1,
        "max_reads": 2,
    }
    engine = source(tmp_path, DisclosureLLM(selected=(2, 1), fail_review=True), **options)
    with pytest.raises(RuntimeError, match="survey review interrupted"):
        engine.survey(survey_request(), context={"operation_id": "survey-1"})
    resumed = source(tmp_path, DisclosureLLM(selected=(2, 1)), **options)
    result = resumed.survey(survey_request(), context={"operation_id": "survey-1"})
    assert len(archive.calls) == 3 and len(result["articles"]) == 2
    assert len(resumed.llm.calls) == 1 and "articles" in resumed.llm.calls[0]
    assert resumed.counts["search"] == 1 and resumed.counts["read"] == 2
    assert resumed.counts["analysis"] == 3  # One selection, failed review, retried review.


def test_retained_mode_does_not_collect_and_rejects_future_snapshot(tmp_path, archive):
    selected = source(tmp_path).survey(survey_request())["articles"][0]
    future = selected | {
        "received_at": "2099-01-01T00:00:00Z",
        "version_available_at": "2099-01-01T00:00:00Z",
    }
    before = len(archive.calls)
    retained = source(
        tmp_path,
        retained_articles=[future],
        information_mode="retained",
        research_cutoff="2098-12-01T00:00:00Z",
    )
    result = retained.survey(survey_request())
    assert result["search"]["status"] == "retained_search_disabled"
    assert (
        result["articles"] == []
        and result["excluded"][0]["reason"] == "source_after_research_cutoff"
    )
    acquired = retained.acquire(
        measurement(source_ids=[future["id"]]), prediction_times=["2026-10-05T21:00:00Z"]
    )
    assert acquired["status"] == "needs_data" and acquired["observations"] == []
    assert retained.llm.calls == [] and len(archive.calls) == before
