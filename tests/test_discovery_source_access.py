"""Access failures inform bounded searches; they never become research evidence."""

import json
from copy import deepcopy
from email.message import Message

import pytest
from test_discovery_loop import cache_at
from test_discovery_sources import article, request

from quant_rsi.research import article_reader
from quant_rsi.research.discovery_loop import run_discovery
from quant_rsi.research.discovery_sources import DiscoverySources, SourceAllowanceExhausted

DENIED = "https://denied.test/release"
OTHER = "https://other.test/release"


def failed(url=DENIED, reason="http_403"):
    return {
        "url": url,
        "final_url": url,
        "status": "unavailable",
        "reason": reason,
        "received_at": "2026-11-08T12:00:00Z",
        "text": "",
        "fetch_attempted": True,
        "content_sha256": None,
    }


class Transport:
    def __init__(self, searches=(), fail_analysis=False):
        self.searches = list(searches)
        self.search_inputs = []
        self.analysis_inputs = []
        self.fail_analysis = fail_analysis

    def search(self, prompt, schema):
        data = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
        self.search_inputs.append(data)
        urls = self.searches.pop(0)
        return {
            "text": json.dumps(
                {
                    "query": data["query"],
                    "rationale": "Offline access recovery fixture",
                    "results": [
                        {"url": url, "title": "Disclosure", "summary": "Fixture"} for url in urls
                    ],
                }
            )
        }

    def structured(self, prompt, schema):
        self.analysis_inputs.append(json.loads(prompt.split("INPUT_JSON:\n", 1)[1]))
        if self.fail_analysis:
            raise RuntimeError("analysis interrupted")
        return json.dumps({"findings": [], "gaps": ["No measured hypothesis in this fixture"]})


@pytest.mark.parametrize("reason", ["http_401", "http_403", "http_429", "non_public_address"])
def test_search_feedback_and_host_gate_choose_other_source_without_more_budget(reason):
    transport = Transport([[DENIED], [DENIED + "-again", OTHER]])
    reads = []

    def read(url):
        reads.append(url)
        return failed(url, reason) if url == DENIED else article() | {"url": url}

    sources = DiscoverySources(transport, reader=read, max_reads=2)
    first = sources.survey({"query": "same operating question", "symbol": "TSM"})
    assert first["articles"] == first["evidence"] == []
    assert first["access_receipts"][0]["reason"] == reason
    assert "text" not in first["access_receipts"][0]
    second = sources.survey({"query": "same operating question", "symbol": "TSM"})
    assert reads == [DENIED, OTHER]
    assert sources.counts["read"] == sources.counts["search"] == 2
    feedback = transport.search_inputs[1]
    assert feedback["avoid_hosts"] == ["denied.test"]
    assert feedback["previous_articles"][0]["reason"] == reason
    skipped = second["access_receipts"][1]
    assert skipped["reason"] == "skipped_denied_host" and skipped["fetch_attempted"] is False
    assert skipped["http_requests"] == 0 and skipped["received_at"] is None
    assert {a["url"] for a in second["evidence"]} == {OTHER}
    assert {a["url"] for a in transport.analysis_inputs[0]["articles"]} == {OTHER}


def test_failed_reads_survive_interrupted_analysis_and_new_instance(tmp_path):
    usage = tmp_path / "source_usage.json"
    source = DiscoverySources(
        Transport([[DENIED, OTHER]], fail_analysis=True),
        reader=lambda url: failed(url) if url == DENIED else article() | {"url": url},
        usage_path=usage,
        max_reads=2,
    )
    with pytest.raises(RuntimeError, match="analysis interrupted"):
        source.survey({"query": "operating news", "symbol": "TSM"})
    persisted = json.loads(usage.read_text())["usage"]["access_receipts"]
    assert persisted[0]["reason"] == "http_403" and len(persisted) == 2
    resumed = DiscoverySources(
        Transport([[]]),
        reader=lambda _: pytest.fail("No remaining read allowance"),
        usage_path=usage,
        max_reads=2,
    )
    result = resumed.survey({"query": "alternative sources", "symbol": "TSM"})
    assert resumed.llm.search_inputs[0]["avoid_hosts"] == ["denied.test"]
    assert resumed._read(DENIED + "/next")["reason"] == "skipped_denied_host"
    assert resumed.counts["read"] == 2
    with pytest.raises(SourceAllowanceExhausted):
        resumed._read("https://third.test/release")
    resumed.restore_state([result, result])
    assert resumed.access_receipts[:2] == persisted
    assert resumed.articles == []


def test_search_cache_identity_includes_feedback_and_replay_keeps_it_stable():
    saved, inputs = {}, {}

    def cache(name, payload, operation):
        if name not in saved:
            inputs[name] = deepcopy(payload)
            saved[name] = operation()
        return deepcopy(saved[name])

    transport = Transport([[DENIED], [OTHER]])
    source = DiscoverySources(transport, reader=failed, call=cache)
    source._search("same query", "TSM")
    source._read(DENIED)
    source._search("same query", "TSM")
    searches = [p["input"] for name, p in inputs.items() if name.startswith("discovery_search_")]
    assert len(searches) == 2
    assert searches[0]["avoid_hosts"] == [] and searches[1]["avoid_hosts"] == ["denied.test"]
    restored = DiscoverySources(Transport(), call=cache)
    restored.restore_state([{"access_receipts": source.access_receipts, "counts": source.counts}])
    assert restored._search("same query", "TSM")["results"][0]["url"] == OTHER
    assert restored.llm.search_inputs == [] and restored.counts["search"] == 2


def test_native_reader_enforces_denied_host_again_after_redirect(monkeypatch):
    fetches = []

    def fetch(url):
        fetches.append(url)
        if url == DENIED:
            failure = article_reader._Unavailable("http_403")
            failure.http_requests = 1
            raise failure
        headers = Message()
        headers["Location"] = DENIED
        return 302, headers, b""

    monkeypatch.setattr(article_reader, "_fetch", fetch)
    sources = DiscoverySources(Transport())
    sources._read(DENIED)
    result = sources._read(OTHER)
    assert fetches == [DENIED, OTHER]
    assert result["reason"] == "skipped_denied_host" and result["http_requests"] == 1
    assert result["fetch_attempted"] is True and result["received_at"] is not None
    assert sources.access_receipts[-1]["final_url"] == DENIED


@pytest.mark.parametrize("kind", ["survey", "acquire"])
def test_outer_loop_retains_access_metadata_on_resume_but_not_as_evidence(tmp_path, kind):
    source = DiscoverySources(Transport([[DENIED]]), reader=failed)
    calls = []

    def provider(value):
        calls.append(value)
        return (
            source.survey(value)
            if kind == "survey"
            else source.acquire(request() | {"source_urls": [DENIED]}, prediction_times=[])
        )

    def decide(context):
        if context["iteration"] == 1:
            return {"action": kind, "query": "news", "data_requests": [request()]}
        assert context["memory"][-1]["result"]["access_receipts"][0]["reason"] == "http_403"
        assert context["evidence"] == []
        return {"action": "stop"}

    options = dict(
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=decide,
        run_experiment=lambda _: pytest.fail("No experiment expected"),
        survey=provider,
        acquire=provider,
        fit_budget=0,
        decision_budget=2,
    )
    first = run_discovery({"symbol": "TSM"}, **options)
    restored = DiscoverySources(Transport())
    restored.restore_state([row["result"] for row in first["memory"] if row["kind"] == kind])
    assert restored._read(DENIED + "/again")["reason"] == "skipped_denied_host"
    second = run_discovery({"symbol": "TSM"}, **options)
    assert len(calls) == 1
    assert second["memory"] == first["memory"]


def test_generic_failure_does_not_blacklist_host_and_legacy_restore_keeps_denial():
    source = DiscoverySources(Transport(), reader=lambda url: failed(url, "request_failed"))
    source._read(DENIED)
    source._read(DENIED + "-other")
    assert source.counts["read"] == 2
    source.restore_state([{"articles": [failed()], "counts": {"read": 2}}])
    assert source._read(DENIED + "-third")["reason"] == "skipped_denied_host"
    assert source.counts["read"] == 2


def test_denied_host_gate_still_validates_url_and_does_not_retain_credentials():
    source = DiscoverySources(Transport(), reader=failed)
    source._read(DENIED)
    result = source._read("https://user:secret@denied.test/another")
    assert result["reason"] == "unsafe_url"
    assert result["url"] is None and result["fetch_attempted"] is False
    assert result["http_requests"] == 0 and source.counts["read"] == 1
    assert "secret" not in json.dumps(source.access_receipts)
