"""A completed but invalid LLM answer is a recorded gap that replays, not a permanent failure."""

import json

from test_discovery_disclosures import (
    DisclosureLLM,
    PostArchive,
    source,
    survey_request,
)
from test_discovery_sources import Transport, article, request

from quant_rsi.research import article_reader
from quant_rsi.research.discovery_sources import DiscoverySources

TIMES = ["2025-01-06T21:00:00Z"]


class Cache:
    """The same receipt contract as StudyCache.call: a completed output is replayed."""

    def __init__(self):
        self.saved = {}

    def __call__(self, name, inputs, operation):
        if name not in self.saved:
            self.saved[name] = operation()
        return json.loads(json.dumps(self.saved[name]))


class InvalidAnswers(Transport):
    def __init__(self, invalid, **kwargs):
        super().__init__(**kwargs)
        self.invalid = invalid

    def structured(self, prompt, schema):
        payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
        kind = "extract" if "request" in payload else "analysis"
        if kind == self.invalid:
            self.calls.append(payload)
            return "not a JSON answer"
        return super().structured(prompt, schema)

    def search(self, prompt, schema):
        if self.invalid == "search":
            self.calls.append({"search": prompt})
            return {"text": '{"query": "missing required fields"}', "search_actions": []}
        return super().search(prompt, schema)


def run_twice(invalid, operation):
    """Run, then resume on the same cache; the replay must not call the provider again."""
    cache = Cache()
    first = InvalidAnswers(invalid)
    result = operation(DiscoverySources(first, reader=lambda _: article(), call=cache))
    replayed = InvalidAnswers(invalid)
    again = operation(DiscoverySources(replayed, reader=lambda _: article(), call=cache))
    assert again["gaps"] == result["gaps"]
    assert not replayed.calls
    return result


def test_invalid_survey_analysis_is_a_gap_and_replays():
    result = run_twice(
        "analysis", lambda engine: engine.survey({"query": "KO demand", "symbol": "KO"})
    )
    assert result["findings"] == []
    assert "Invalid analysis response" in result["gaps"]


def test_invalid_extraction_is_a_gap_and_replays():
    result = run_twice("extract", lambda engine: engine.acquire(request(), prediction_times=TIMES))
    assert result["status"] == "needs_data"
    assert [gap for gap in result["gaps"] if gap.endswith(": invalid extraction response")]


def test_invalid_search_response_is_a_gap_and_replays():
    result = run_twice(
        "search",
        lambda engine: engine.acquire(request() | {"source_urls": []}, prediction_times=TIMES),
    )
    assert result["status"] == "needs_data"
    assert "Invalid search response" in result["gaps"]


def test_finding_without_relationships_citing_unknown_source_is_rejected_not_raised():
    class UnknownEvidence(Transport):
        def structured(self, prompt, schema):
            payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
            if "request" in payload:
                return super().structured(prompt, schema)
            self.calls.append(payload)
            finding = {
                "mechanism": "Demand may differ from reported revenue growth.",
                "evidence_ids": ["source-hallucinated"],
                "measurements": ["Volume growth"],
                "uncertainty": "Not a return-prediction claim.",
            }
            return json.dumps({"findings": [finding], "gaps": []})

    engine = DiscoverySources(UnknownEvidence(), reader=lambda _: article())
    result = engine.survey({"query": "KO demand", "symbol": "KO"})
    assert result["findings"] == []
    assert "Rejected finding: finding_cites_unavailable_source" in result["gaps"]


def test_invalid_disclosure_selection_is_a_gap_and_replays(tmp_path, monkeypatch):
    monkeypatch.setattr(article_reader, "_fetch", PostArchive())

    class InvalidSelection(DisclosureLLM):
        def structured(self, prompt, schema):
            data = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
            if "documents" in data:
                self.calls.append(data)
                return json.dumps({"document_refs": "not-a-list"})
            return super().structured(prompt, schema)

    cache = Cache()
    first = InvalidSelection()
    result = source(tmp_path, first, call=cache).survey(survey_request())
    assert "invalid_disclosure_selection" in result["gaps"]
    replayed = InvalidSelection()
    again = source(tmp_path, replayed, call=cache).survey(survey_request())
    assert again["gaps"] == result["gaps"]
    assert not [call for call in replayed.calls if "documents" in call]
