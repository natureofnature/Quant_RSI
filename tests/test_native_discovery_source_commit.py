"""Native source identities survive failure while committing the outer loop receipt."""

import json
from copy import deepcopy

import pytest
from test_discovery_source_access import DENIED, OTHER, Transport, failed
from test_discovery_sources import article, extraction, request
from test_study_discovery import arguments

from quant_rsi.cli import main
from quant_rsi.research import study_discovery, study_resume
from quant_rsi.research.discovery_sources import DiscoverySources


@pytest.mark.parametrize("new_iteration", [False, True])
def test_native_outer_commit_failure_replays_original_output_without_any_source_calls(
    tmp_path, monkeypatch, new_iteration
):
    instances, reads, outputs, contexts = [], [], [], []

    def factory(times, **kwargs):
        searches = [[DENIED, OTHER]] if not instances else [[OTHER]] if new_iteration else []

        def reader(url):
            reads.append(url)
            return failed(url) if url == DENIED else article() | {"url": url}

        source = DiscoverySources(
            Transport(searches),
            reader=reader,
            max_searches=2 if new_iteration else 1,
            max_reads=2,
            **kwargs,
        )
        actual_survey = source.survey
        index = len(instances)
        outputs.append([])
        contexts.append([])

        def tracked_survey(value, *, context=None):
            contexts[index].append(deepcopy(context))
            result = actual_survey(value, context=context)
            outputs[index].append(deepcopy(result))
            return result

        source.survey = tracked_survey
        instances.append(source)
        return source

    def decide(context):
        if context["iteration"] <= (2 if new_iteration else 1):
            return {"action": "survey", "query": "same operating question", "reason": "Survey"}
        return {"action": "stop"}

    actual_write = study_resume.atomic_write

    def interrupted_write(path, data):
        if path.name == "discovery_survey_1_0.json":
            raise RuntimeError("outer survey receipt write interrupted")
        actual_write(path, data)

    monkeypatch.setattr(study_discovery, "simulated_discovery_sources", factory)
    monkeypatch.setattr(study_discovery, "simulated_discovery_decision", decide)
    monkeypatch.setattr(study_resume, "atomic_write", interrupted_write)
    root = tmp_path / "study"
    with pytest.raises(RuntimeError, match="outer survey receipt write interrupted"):
        main(arguments(root, "--max-fits", "0"))
    assert len(outputs[0]) == 1
    assert {a["url"] for a in outputs[0][0]["articles"]} == {OTHER}
    assert not (root / "checkpoints" / "discovery_survey_1_0.json").exists()
    monkeypatch.setattr(study_resume, "atomic_write", actual_write)
    assert main(["research-resume", str(root)]) == 0
    assert outputs[1][0] == outputs[0][0]
    assert contexts[1][0]["operation_id"] == contexts[0][0]["operation_id"]
    assert reads == [DENIED, OTHER]
    assert instances[1].llm.analysis_inputs == []
    assert len(instances[1].llm.search_inputs) == int(new_iteration)
    if new_iteration:
        assert contexts[1][1]["operation_id"] != contexts[1][0]["operation_id"]
        assert instances[1].llm.search_inputs[0]["avoid_hosts"] == ["denied.test"]
    discovery = json.loads((root / "result.json").read_text())["discovery"]
    surveyed = [m["result"] for m in discovery["memory"] if m["kind"] == "survey"]
    assert surveyed[0]["articles"] == outputs[0][0]["articles"]
    assert discovery["counts"]["fits"] == 0


def test_duplicate_native_batch_requests_share_inner_operation_but_consume_outer_slots(
    tmp_path, monkeypatch
):
    instances, reads = [], []

    class ExtractionTransport(Transport):
        def structured(self, prompt, schema):
            return json.dumps(extraction())

    def factory(times, **kwargs):
        source = DiscoverySources(
            ExtractionTransport([[OTHER]]),
            reader=lambda url: reads.append(url) or article() | {"url": url},
            max_searches=1,
            max_reads=1,
            max_extractions=1,
            **kwargs,
        )
        instances.append(source)
        return source

    def decide(context):
        query = request() | {"source_urls": []}
        return (
            {"action": "acquire", "data_requests": [query, deepcopy(query)]}
            if context["iteration"] == 1
            else {"action": "stop"}
        )

    monkeypatch.setattr(study_discovery, "simulated_discovery_sources", factory)
    monkeypatch.setattr(study_discovery, "simulated_discovery_decision", decide)
    root = tmp_path / "study"
    assert main(arguments(root, "--max-fits", "0")) == 0
    discovery = json.loads((root / "result.json").read_text())["discovery"]
    acquired = [m["result"] for m in discovery["memory"] if m["kind"] == "acquire"]
    assert len(acquired) == discovery["counts"]["acquisitions"] == 2
    assert acquired[0] == acquired[1]
    assert reads == [OTHER]
    assert all(instances[0].counts[k] == 1 for k in ("search", "read", "extract"))
    contexts = [op for op in instances[0].operations if op["kind"] == "source_context"]
    assert len(contexts) == 1
