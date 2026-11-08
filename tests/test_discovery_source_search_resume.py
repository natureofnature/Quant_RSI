"""An interrupted source operation replays its frozen search, not a new paid query."""

import json
from copy import deepcopy

import pytest
from test_discovery_loop import cache_at
from test_discovery_source_access import DENIED, OTHER, Transport, failed
from test_discovery_sources import article, extraction, request

from quant_rsi.research.discovery_sources import DiscoverySources


@pytest.mark.parametrize("search_limit", [1, 2])
@pytest.mark.parametrize("interruption", ["analysis", "read"])
def test_same_survey_replays_search_and_completed_reads_after_interruption(
    tmp_path, search_limit, interruption
):
    cache = cache_at(tmp_path)
    physical_reads = []
    fail_read = interruption == "read"

    def reader(url):
        nonlocal fail_read
        physical_reads.append(url)
        if url == OTHER and fail_read:
            fail_read = False
            raise RuntimeError("read interrupted")
        return failed(url) if url == DENIED else article() | {"url": url}

    options = dict(
        call=cache.call,
        usage_path=tmp_path / "usage.json",
        reader=reader,
        max_searches=search_limit,
        max_reads=3,
    )
    source = DiscoverySources(
        Transport([[DENIED, OTHER]], fail_analysis=interruption == "analysis"), **options
    )
    query = {"query": "same operating query", "symbol": "TSM"}
    with pytest.raises(RuntimeError, match=interruption + " interrupted"):
        source.survey(query)
    resumed = DiscoverySources(Transport(), **options)
    result = resumed.survey(query)
    assert resumed.llm.search_inputs == []
    assert len(resumed.llm.analysis_inputs) == 1
    assert {a["url"] for a in result["evidence"]} == {OTHER}
    assert result["search"]["avoid_hosts"] == []  # Original search context is replayed.
    assert resumed.counts["search"] == 1
    assert resumed.counts["read"] == (2 if interruption == "analysis" else 3)
    assert physical_reads == (
        [DENIED, OTHER] if interruption == "analysis" else [DENIED, OTHER, OTHER]
    )
    assert result["access_receipts"][0]["reason"] == "http_403"
    assert not any(r.get("reason") == "skipped_denied_host" for r in result["access_receipts"])


def test_completed_survey_uses_new_feedback_even_after_stale_started_receipt_restore(tmp_path):
    cache = cache_at(tmp_path)
    options = dict(
        call=cache.call,
        usage_path=tmp_path / "usage.json",
        reader=lambda url: failed(url) if url == DENIED else article() | {"url": url},
        max_searches=2,
        max_reads=2,
    )
    source = DiscoverySources(Transport([[DENIED, OTHER]], fail_analysis=True), **options)
    query = {"query": "same operating query", "symbol": "TSM"}
    with pytest.raises(RuntimeError):
        source.survey(query)
    old_started = deepcopy(next(op for op in source.operations if op["kind"] == "source_context"))
    restored = DiscoverySources(Transport([[OTHER]]), **options)
    restored.survey(query)
    assert restored.llm.search_inputs == []
    restored.restore_usage([{"operations": [old_started]}])
    restored.survey(query)
    assert len(restored.llm.search_inputs) == 1
    assert restored.llm.search_inputs[0]["avoid_hosts"] == ["denied.test"]
    assert restored.counts["search"] == 2 and restored.counts["read"] == 2
    contexts = [op for op in restored.operations if op["kind"] == "source_context"]
    assert max(op["attempt"] for op in contexts) == 2


def test_acquire_replays_frozen_search_and_body_after_extraction_failure(tmp_path):
    class ExtractionTransport(Transport):
        def structured(self, prompt, schema):
            if self.fail_analysis:
                raise RuntimeError("extraction interrupted")
            return json.dumps(extraction())

    reads = []
    options = dict(
        call=cache_at(tmp_path).call,
        usage_path=tmp_path / "usage.json",
        reader=lambda url: reads.append(url) or article() | {"url": url},
        max_searches=1,
        max_reads=1,
        max_extractions=2,
    )
    source = DiscoverySources(ExtractionTransport([[OTHER]], fail_analysis=True), **options)
    query = request() | {"source_urls": []}
    with pytest.raises(RuntimeError, match="extraction interrupted"):
        source.acquire(query, prediction_times=["2025-01-06T21:00:00Z"])
    resumed = DiscoverySources(ExtractionTransport(), **options)
    result = resumed.acquire(query, prediction_times=["2025-01-06T21:00:00Z"])
    assert result["status"] == "available" and len(result["observations"]) == 1
    assert resumed.llm.search_inputs == [] and reads == [OTHER]
    assert resumed.counts["search"] == resumed.counts["read"] == 1
    assert resumed.counts["extract"] == 2  # Failed extraction was not refunded.


def test_interrupted_read_is_not_refunded_when_restored_budget_is_exhausted(tmp_path):
    def reader(url):
        raise RuntimeError("read interrupted")

    options = dict(
        call=cache_at(tmp_path).call,
        usage_path=tmp_path / "usage.json",
        max_searches=1,
        max_reads=1,
    )
    source = DiscoverySources(Transport([[OTHER]]), reader=reader, **options)
    query = {"query": "same operating query", "symbol": "TSM"}
    with pytest.raises(RuntimeError, match="read interrupted"):
        source.survey(query)
    resumed = DiscoverySources(
        Transport(),
        reader=lambda _: pytest.fail("Exhausted read allowance must not retry"),
        **options,
    )
    result = resumed.survey(query)
    assert result["evidence"] == []
    assert resumed.llm.search_inputs == []
    assert resumed.counts["read"] == resumed.counts["search"] == 1


def test_source_context_identity_is_checked_before_reusing_feedback(tmp_path):
    source = DiscoverySources(
        Transport([[DENIED, OTHER]], fail_analysis=True),
        reader=lambda url: failed(url) if url == DENIED else article() | {"url": url},
        call=cache_at(tmp_path).call,
    )
    query = {"query": "same operating query", "symbol": "TSM"}
    with pytest.raises(RuntimeError):
        source.survey(query)
    frozen = next(op for op in source.operations if op["kind"] == "source_context")
    frozen["feedback"]["avoid_hosts"].append("forged.test")
    with pytest.raises(ValueError, match="Frozen discovery search context changed"):
        source.survey(query)
    assert len(source.llm.search_inputs) == 1


def test_cached_body_on_later_denied_host_is_replayed_without_network(tmp_path):
    good, denied = "https://same.test/good", "https://same.test/denied"
    reads = []
    options = dict(
        call=cache_at(tmp_path).call,
        usage_path=tmp_path / "usage.json",
        max_searches=1,
        max_reads=2,
    )

    def reader(url):
        reads.append(url)
        return article() | {"url": url} if url == good else failed(url)

    source = DiscoverySources(
        Transport([[good, denied]], fail_analysis=True), reader=reader, **options
    )
    query = {"query": "same operating query", "symbol": "TSM"}
    with pytest.raises(RuntimeError):
        source.survey(query)
    resumed = DiscoverySources(
        Transport(), reader=lambda _: pytest.fail("Denied host must not be read again"), **options
    )
    result = resumed.survey(query)
    assert {a["url"] for a in result["evidence"]} == {good}
    assert reads == [good, denied] and resumed.llm.search_inputs == []
    assert resumed.counts["read"] == 2


def test_missing_completed_body_checkpoint_does_not_fetch_or_spend_read_budget(tmp_path):
    source = DiscoverySources(Transport(), call=cache_at(tmp_path).call)
    source.restore_usage(
        [
            {
                "access_receipts": [
                    {"url": "https://same.test/good", "status": "available"},
                    failed("https://same.test/denied"),
                ]
            }
        ]
    )
    with pytest.raises(ValueError, match="Completed discovery read checkpoint is unavailable"):
        source._read("https://same.test/good")
    assert source.counts["read"] == 0


def test_explicit_outer_operation_identity_cannot_be_rebound_to_another_request(tmp_path):
    source = DiscoverySources(Transport([[]]), call=cache_at(tmp_path).call)
    source.survey({"query": "original"}, context={"operation_id": "outer-survey-1"})
    with pytest.raises(ValueError, match="Discovery source operation identity was rebound"):
        source.survey({"query": "different"}, context={"operation_id": "outer-survey-1"})
    assert source.counts["search"] == 1
