"""Offline genuine provider boundary calls; fixtures establish no investment result."""

import json

import pytest

from quant_rsi.research.discovery_sources import (
    DataRequest,
    DiscoverySources,
    NumericalExtraction,
    align_input_bundle,
    discovery_decision,
    eligible_articles,
    simulated_discovery_sources,
    strict_schema,
    validate_extraction,
)


def article(number=1):
    return {
        "id": f"article-{number}",
        "url": f"https://example.com/release-{number}",
        "status": "available",
        "published_at": f"2025-0{number}-03T12:00:00Z",
        "received_at": "2026-11-04T09:00:00Z",
        "version_available_at": "2026-11-04T09:00:00Z",
        "text": (
            f"Published January 3, 2025. KO disclosed demand {number + 2}.0 percent "
            "for period ending 2024-12-31."
        ),
        "content_sha256": str(number) * 64,
    }


def request():
    return {
        "input_id": "acquired.volume_growth",
        "symbol": "KO",
        "measurement": "Reported unit case volume growth, percent",
        "unit": "percent",
        "query": "KO quarterly volume disclosure",
        "source_urls": ["https://example.com/release-1"],
        "max_age_days": 100,
    }


def extraction():
    return {
        "publication_date": "2025-01-03",
        "publication_excerpt": "Published January 3, 2025.",
        "measurement": "Reported unit case volume growth, percent",
        "observations": [
            {
                "period_start": None,
                "period_end": "2024-12-31",
                "value": 3.0,
                "unit": "percent",
                "excerpt": "KO disclosed demand 3.0 percent for period ending 2024-12-31.",
            }
        ],
        "limitations": ["Offline extraction fixture, not real disclosure."],
    }


class Transport:
    def __init__(self, extracted=None):
        self.calls = []
        self.extracted = extracted or extraction()

    def structured(self, prompt, schema):
        payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
        self.calls.append(payload)
        if "request" in payload:
            return json.dumps(self.extracted)
        return json.dumps(
            {
                "findings": [
                    {
                        "mechanism": "Demand may differ from reported revenue growth.",
                        "evidence_ids": [payload["articles"][0]["id"]],
                        "measurements": ["Volume growth"],
                        "uncertainty": "Not a return-prediction claim.",
                    }
                ],
                "gaps": [],
            }
        )

    def search(self, prompt, schema):
        self.calls.append({"search": prompt})
        return {
            "text": json.dumps(
                {
                    "query": "KO volume growth",
                    "rationale": "Read primary disclosure",
                    "results": [
                        {
                            "title": "Fixture release",
                            "url": "https://example.com/release-1",
                            "summary": "Potential disclosure; not verified.",
                        }
                    ],
                }
            ),
            "search_actions": [{"type": "search", "query": "KO volume growth"}],
        }


def test_missing_input_acquisition_calls_reader_and_extractor_and_preserves_lineage():
    transport = Transport()
    reads = []

    def reader(url):
        reads.append(url)
        return article()

    source = DiscoverySources(transport, reader=reader)
    times = ["2025-01-03T21:00:00Z", "2025-01-06T21:00:00Z", "2025-06-01T21:00:00Z"]
    bundle = source.acquire(request(), prediction_times=times)
    assert reads == request()["source_urls"]
    assert bundle["columns"]["acquired.volume_growth"] == [None, 3.0, None]
    observation = bundle["observations"][0]
    assert observation["received_at"].startswith("2026-11-04")
    assert observation["availability_basis"] == "reconstructed_public_disclosure_date"
    assert (
        bundle["input_evidence"]["acquired.volume_growth"][1]["observation_id"]
        == observation["observation_id"]
    )
    assert (
        bundle["input_evidence"]["acquired.volume_growth"][1]["effective_available_at"]
        == "2025-01-04T12:00:00+00:00"
    )
    assert bundle["distinct_observations"] == bundle["distinct_values"] == 1
    assert source.counts == {"search": 0, "read": 1, "extract": 1, "analysis": 0}
    json.dumps(bundle, allow_nan=False)


@pytest.mark.parametrize("change", ["received_at", "version_available_at", "published_at"])
def test_retained_ineligible_article_is_never_supplied_to_llm(change):
    retained = article()
    for key in ("received_at", "version_available_at", "published_at"):
        retained[key] = "2025-01-03T12:00:00Z"
    retained[change] = "2026-01-01T12:00:00Z"
    transport = Transport()
    source = DiscoverySources(
        transport,
        information_mode="retained",
        research_cutoff="2025-06-01T00:00:00Z",
        retained_articles=[retained],
        reader=lambda _: pytest.fail("No live read in retained mode"),
    )
    survey = source.survey({"query": "KO volume", "symbol": "KO"})
    bundle = source.acquire(request(), prediction_times=["2025-01-06T21:00:00Z"])
    assert not transport.calls
    assert survey["findings"] == []
    assert bundle["status"] == "needs_data"
    assert bundle["excluded"][0]["reason"] == "source_after_research_cutoff"


def test_retained_new_extraction_does_not_backdate_information(monkeypatch):
    retained = article()
    retained.update(received_at="2025-01-04T13:00:00Z", version_available_at="2025-01-04T13:00:00Z")
    monkeypatch.setattr("quant_rsi.research.discovery_sources._now", lambda: "2026-11-04T12:00:00Z")
    source = DiscoverySources(
        Transport(),
        information_mode="retained",
        research_cutoff="2025-06-01T00:00:00Z",
        retained_articles=[retained],
    )
    bundle = source.acquire(request(), prediction_times=["2025-01-06T21:00:00Z"])
    assert bundle["status"] == "needs_data"
    assert bundle["gaps"] == ["extraction_after_research_cutoff"]


@pytest.mark.parametrize("kind", ["value", "unit", "excerpt", "publication", "period"])
def test_extraction_rejects_unverifiable_numeric_or_temporal_claims(kind):
    result = extraction()
    if kind == "value":
        result["observations"][0]["value"] = 3000.0
    elif kind == "unit":
        result["observations"][0]["unit"] = "USD"
    elif kind == "excerpt":
        result["observations"][0]["excerpt"] = "A made-up 3.0 percent observation."
    elif kind == "publication":
        result["publication_date"] = "2024-01-03"
    else:
        result["observations"][0]["period_end"] = "2027-01-01"
    rows, reasons = validate_extraction(
        DataRequest.model_validate_json(json.dumps(request())),
        article(),
        NumericalExtraction.model_validate_json(json.dumps(result)),
        extracted_at="2026-11-04T10:00:00Z",
        information_mode="retrospective",
        research_cutoff=None,
    )
    assert not rows and reasons


def test_acquisition_unknown_urls_are_searched_before_reading():
    value = request() | {"source_urls": []}
    source = DiscoverySources(Transport(), reader=lambda _: article())
    result = source.acquire(value, prediction_times=["2025-01-06T21:00:00Z"])
    assert result["covered_rows"] == 1
    assert source.counts["search"] == source.counts["read"] == source.counts["extract"] == 1


def test_acquisition_conflicting_simultaneous_measurements_do_not_pick_arbitrarily():
    source = DiscoverySources(Transport(), reader=lambda _: article())
    result = source.acquire(request(), prediction_times=["2025-01-06T21:00:00Z"])
    first = result["observations"][0]
    second = first | {"observation_id": "conflicting", "value": 5.0}
    bundle = align_input_bundle(request(), [first, second], ["2025-01-06T21:00:00Z"])
    assert bundle["columns"]["acquired.volume_growth"] == [None]
    assert bundle["conflicts"][0]["reason"] == "conflicting_measurements"


def test_bounded_operations_and_unknown_evidence_are_reported():
    source = DiscoverySources(Transport(), max_searches=0)
    result = source.acquire(request() | {"source_urls": []}, prediction_times=[])
    assert result["status"] == "needs_data"
    assert result["gaps"] == ["Discovery search allowance exhausted"]
    assert source.counts["search"] == 0
    eligible, gaps = eligible_articles(
        [{"id": "missing", "text": "secret"}],
        information_mode="retained",
        research_cutoff="2025-01-01T00:00:00Z",
    )
    assert not eligible and gaps[0]["reason"] == "missing_retained_timestamp"


def test_decision_can_propose_absent_input_and_temporal_expression():
    from quant_rsi.research.discovery_factors import DiscoveryFactor

    candidate = {
        "name": "demand_lag",
        "description": "Test delayed demand effect",
        "inputs": [{"name": "demand", "factor_id": "acquired.volume_growth"}],
        "evidence_ids": ["article-1"],
        "expression": {
            "schema_version": 2,
            "nodes": [
                {"name": "raw", "op": "leaf", "inputs": [], "leaf": "demand", "window": None},
                {"name": "delayed", "op": "lag", "inputs": ["raw"], "leaf": None, "window": 2},
            ],
        },
    }

    class DecisionTransport:
        def structured(self, prompt, schema):
            assert "catalog is reference knowledge" in prompt
            assert "additionalProperties" in schema
            return json.dumps(
                {
                    "action": "acquire",
                    "reason": "New missing demand input",
                    "candidates": [candidate],
                    "data_requests": [request()],
                    "query": None,
                }
            )

    output = discovery_decision(DecisionTransport(), {"symbol": "KO", "memory": []})
    assert output["action"] == "acquire"
    assert (
        DiscoveryFactor.model_validate_json(json.dumps(output["candidates"][0])).name
        == "demand_lag"
    )
    schema = strict_schema(DataRequest)
    assert set(schema["required"]) == set(schema["properties"])


def test_survey_reads_actual_pages_and_validates_evidence_references():
    source = DiscoverySources(Transport(), reader=lambda _: article())
    survey = source.survey({"query": "KO demand", "symbol": "KO"})
    assert len(survey["findings"]) == len(survey["articles"]) == 1
    assert survey["findings"][0]["evidence_ids"] == [survey["articles"][0]["id"]]
    assert source.counts["analysis"] == 1


def test_simulated_documents_exercise_shared_acquisition_and_time_alignment():
    from datetime import UTC, datetime, timedelta

    times = [
        (datetime(2025, 2, 1, 21, tzinfo=UTC) + timedelta(days=i)).isoformat() for i in range(120)
    ]
    source = simulated_discovery_sources(times)
    survey = source.survey({"query": "Operating demand", "symbol": "KO"})
    assert survey["findings"][0]["uncertainty"].startswith("Synthetic")
    bundle = source.acquire(
        request() | {"source_urls": [], "measurement": "Reported operating demand"},
        prediction_times=times,
    )
    assert bundle["covered_rows"] == len(times)
    assert bundle["distinct_values"] >= 4
    assert all(a["provider"] == "simulated_discovery_fixture" for a in bundle["articles"])
    assert bundle["counts"]["extract"] >= 4


def test_new_requested_sources_are_not_hidden_by_existing_context_limit():
    old = article() | {"url": "https://example.com/old", "id": "old"}
    source = DiscoverySources(
        Transport(), retained_articles=[old], reader=lambda _: article(), max_reads=1
    )
    bundle = source.acquire(request(), prediction_times=["2025-01-06T21:00:00Z"])
    assert bundle["covered_rows"] == 1
    assert bundle["articles"][0]["url"] == request()["source_urls"][0]


def test_extraction_receipt_and_policy_binding_survive_cached_replay(monkeypatch):
    saved = {}

    def cache(name, inputs, operation):
        if name not in saved:
            saved[name] = operation()
        return saved[name]

    times = ["2025-01-06T21:00:00Z"]
    monkeypatch.setattr("quant_rsi.research.discovery_sources._now", lambda: "2026-11-04T10:00:00Z")
    first = DiscoverySources(
        Transport(), reader=lambda _: article(), call=cache, policy={"identity": "p1"}
    )
    old = first.acquire(request(), prediction_times=times)
    monkeypatch.setattr("quant_rsi.research.discovery_sources._now", lambda: "2026-11-05T10:00:00Z")
    transport = Transport()
    resumed = DiscoverySources(
        transport,
        reader=lambda _: pytest.fail("Read must be reused"),
        call=cache,
        policy={"identity": "p1"},
    )
    new = resumed.acquire(request(), prediction_times=times)
    assert not transport.calls
    assert new["identity"] == old["identity"]
    assert new["observations"][0]["extracted_at"] == "2026-11-04T10:00:00Z"
    changed = DiscoverySources(
        transport, reader=lambda _: article(), call=cache, policy={"identity": "p2"}
    )
    changed.acquire(request(), prediction_times=times)
    assert transport.calls[0]["policy"] == {"identity": "p2"}


def test_usage_restores_after_outer_operation_replay_and_enforces_remaining_budget():
    times = ["2025-01-06T21:00:00Z"]
    first = DiscoverySources(Transport(), reader=lambda _: article(), max_extractions=1)
    result = first.acquire(request(), prediction_times=times)
    assert {row["kind"] for row in result["operations"]} == {"read", "extract"}
    resumed = DiscoverySources(Transport(), retained_articles=[article()], max_extractions=1)
    resumed.restore_usage([result, result])
    assert resumed.counts == first.counts
    assert resumed.operations == first.operations
    after = resumed.acquire(request(), prediction_times=times)
    assert after["status"] == "needs_data"
    assert after["gaps"] == ["Extraction allowance exhausted"]
    assert after["counts"]["extract"] == 1


def test_disabled_read_returns_gap_without_reading_even_with_explicit_url():
    source = DiscoverySources(
        Transport(), reader=lambda _: pytest.fail("Read forbidden"), max_reads=0
    )
    result = source.acquire(request(), prediction_times=["2025-01-06T21:00:00Z"])
    assert result["status"] == "needs_data"
    assert "Discovery read allowance exhausted" in result["gaps"]


@pytest.mark.parametrize("quoted", ["07-28-2026", "7-28-2026", "07/28/2026"])
def test_literal_publication_accepts_unambiguous_numeric_us_dates(quoted):
    item = article()
    item["text"] = item["text"].replace("Published January 3, 2025.", quoted)
    result = extraction() | {"publication_date": "2026-07-28", "publication_excerpt": quoted}
    rows, reasons = validate_extraction(
        DataRequest.model_validate_json(json.dumps(request())),
        item,
        NumericalExtraction.model_validate_json(json.dumps(result)),
        extracted_at="2026-11-04T10:00:00Z",
        information_mode="retrospective",
        research_cutoff=None,
    )
    assert len(rows) == 1 and not reasons


@pytest.mark.parametrize("quoted", ["07-08-2026", "07-28-20260", "17-28-2026"])
def test_numeric_date_support_does_not_accept_ambiguity_or_partial_tokens(quoted):
    from quant_rsi.research.discovery_sources import _date_in_excerpt

    assert not _date_in_excerpt("2026-07-28", quoted)
    assert not _date_in_excerpt("2026-07-08", quoted)


def test_targeted_acquisition_fetches_existing_headline_body_and_skips_other_headlines():
    headlines = [
        article(n) | {"text": "An unrelated headline", "title": "An unrelated headline"}
        for n in range(1, 7)
    ]
    reads = []
    source = DiscoverySources(
        Transport(),
        retained_articles=headlines,
        reader=lambda url: reads.append(url) or article(),
        max_extractions=1,
    )
    result = source.acquire(
        request() | {"source_urls": []}, prediction_times=["2025-01-06T21:00:00Z"]
    )
    assert reads == ["https://example.com/release-1"]
    assert result["covered_rows"] == 1
    assert result["counts"]["extract"] == 1
    assert len(result["excluded"]) == 6
    assert {x["reason"] for x in result["excluded"]} == {"headline_only_no_numerical_source_body"}
    assert len(result["articles"]) == 1


def test_retained_headline_does_not_fetch_or_extract_missing_body():
    item = article() | {"text": "Headline only", "title": "Headline only"}
    transport = Transport()
    source = DiscoverySources(
        transport,
        retained_articles=[item],
        information_mode="retained",
        research_cutoff="2026-11-05T00:00:00Z",
        reader=lambda _: pytest.fail("Retained mode must not fetch a body"),
    )
    result = source.acquire(request(), prediction_times=["2025-01-06T21:00:00Z"])
    assert result["status"] == "needs_data"
    assert result["excluded"][0]["reason"] == "headline_only_no_numerical_source_body"
    assert not transport.calls


@pytest.mark.parametrize("kind", ["search", "extract"])
def test_failed_source_dispatch_keeps_its_charge_after_resume(tmp_path, kind):
    class BrokenTransport(Transport):
        def search(self, prompt, schema):
            if kind == "search":
                raise RuntimeError("Paid search failed")
            return super().search(prompt, schema)

        def structured(self, prompt, schema):
            if kind == "extract":
                raise RuntimeError("Paid extraction failed")
            return super().structured(prompt, schema)

    usage_path = tmp_path / "usage.json"
    params = {
        "reader": lambda _: article(),
        "usage_path": usage_path,
        "max_searches": 1,
        "max_extractions": 1,
    }
    value = request() | ({"source_urls": []} if kind == "search" else {})
    first = DiscoverySources(BrokenTransport(), **params)
    with pytest.raises(RuntimeError, match="Paid"):
        first.acquire(value, prediction_times=["2025-01-06T21:00:00Z"])
    saved = json.loads(usage_path.read_text())["usage"]
    assert saved["counts"][kind] == 1
    assert saved["operations"][-1]["status"] == "failed"
    transport = Transport()
    resumed = DiscoverySources(transport, **params)
    result = resumed.acquire(value, prediction_times=["2025-01-06T21:00:00Z"])
    assert result["status"] == "needs_data"
    assert result["counts"][kind] == 1
    assert not transport.calls


def test_source_cache_replay_neither_spends_nor_requires_remaining_allowance(tmp_path):
    saved = {}

    def cache(name, inputs, operation):
        if name not in saved:
            saved[name] = operation()
        return saved[name]

    params = {
        "call": cache,
        "usage_path": tmp_path / "usage.json",
        "max_reads": 1,
        "max_extractions": 1,
    }
    first = DiscoverySources(Transport(), reader=lambda _: article(), **params)
    times = ["2025-01-06T21:00:00Z"]
    previous = first.acquire(request(), prediction_times=times)
    transport = Transport()
    resumed = DiscoverySources(transport, reader=lambda _: pytest.fail("Must reuse body"), **params)
    result = resumed.acquire(request(), prediction_times=times)
    assert result["identity"] == previous["identity"]
    assert result["counts"] == previous["counts"]
    assert result["operations"] == previous["operations"]
    assert not transport.calls
