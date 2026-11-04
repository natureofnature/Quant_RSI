"""Resume source evidence through the same discovery CLI without paid calls."""

import json
from copy import deepcopy

import pytest
from test_study_discovery import arguments

from quant_rsi.cli import main
from quant_rsi.research import study_discovery
from quant_rsi.research.discovery_sources import DiscoverySources


def test_resume_restores_surveyed_bodies_before_retrying_failed_extraction(tmp_path, monkeypatch):
    original_factory = study_discovery.simulated_discovery_sources
    original_decision = study_discovery.simulated_discovery_decision
    instances, reads, extractions = [], [], []
    fail = True

    def factory(times, **kwargs):
        source = original_factory(times, **kwargs)
        source.articles = []  # Sources must actually be found and read by the survey.
        source.limits.update(search=1, read=3, extract=6)
        original_reader, original_structured = source.reader, source.llm.structured

        def read(url):
            reads.append(url)
            return original_reader(url)

        def structured(prompt, schema):
            payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
            if "request" in payload:
                extractions.append(payload["article"])
                if fail:
                    raise ValueError("Injected numerical extraction outage")
            return original_structured(prompt, schema)

        source.reader, source.llm.structured = read, structured
        instances.append(source)
        return source

    def decide(context):
        if context["iteration"] == 1:
            return {"action": "survey", "query": "Find dated operating demand", "reason": "Survey"}
        if context["iteration"] == 2:
            return original_decision(context)
        return {"action": "stop", "reason": "Inspect acquired evidence"}

    monkeypatch.setattr(study_discovery, "simulated_discovery_sources", factory)
    monkeypatch.setattr(study_discovery, "simulated_discovery_decision", decide)
    root = tmp_path / "study"
    assert main(arguments(root)) == 1
    surveyed = json.loads((root / "checkpoints/discovery_survey_1_0.json").read_text())["output"]
    original_articles = surveyed["articles"]
    assert len(original_articles) == len(reads) == 3
    assert len(extractions) == 1
    assert instances[0].counts == {"search": 1, "read": 3, "extract": 1, "analysis": 1}
    fail = False
    assert main(["research-resume", str(root)]) == 0
    result = json.loads((root / "result.json").read_text())["discovery"]
    acquisition = next(row["result"] for row in result["memory"] if row["kind"] == "acquire")
    assert len(extractions) == 4, "Resume lost surveyed bodies and silently skipped extraction"
    assert len(reads) == 3, "Resume fetched already retained article bodies again"
    assert acquisition["articles"] == original_articles
    assert acquisition["status"] == "available"
    assert len(acquisition["observations"]) == 3
    assert result["source_counts"] == {"search": 1, "read": 3, "extract": 4, "analysis": 1}
    assert extractions[1:] == original_articles


def snapshot():
    return {
        "id": "retained-disclosure",
        "url": "https://example.com/disclosure",
        "text": "Published 2025-01-03. Demand was 3 percent for period ending 2024-12-31.",
        "status": "available",
        "published_at": "2025-01-03T12:00:00Z",
        "received_at": "2025-01-04T13:00:00Z",
        "version_available_at": "2025-01-04T13:00:00Z",
        "content_sha256": "retained-content-fingerprint",
    }


def test_source_restoration_is_idempotent_preserves_provenance_and_copies_receipt():
    article = snapshot()
    receipt = {"articles": [article], "counts": {"search": 2, "read": 3, "extract": 1}}
    source = DiscoverySources(object())
    report = source.restore_state([receipt, receipt])
    assert report == {"restored_article_ids": [article["id"]], "excluded": []}
    assert source.articles == [article]
    assert source.counts == {"search": 2, "read": 3, "extract": 1, "analysis": 0}
    assert source.restore_state([receipt])["restored_article_ids"] == []
    article["text"] = "Mutated caller copy"
    assert source.articles == [snapshot()]


@pytest.mark.parametrize("field", ["published_at", "received_at", "version_available_at"])
def test_source_restoration_rechecks_retained_visibility_without_live_fallback(field):
    eligible = snapshot()
    late = deepcopy(eligible) | {"id": "late", field: "2026-01-01T00:00:00Z"}
    missing = {k: v for k, v in eligible.items() if k != field} | {"id": "missing"}
    source = DiscoverySources(
        object(),
        information_mode="retained",
        research_cutoff="2025-02-01T00:00:00Z",
        reader=lambda _: pytest.fail("Restoration must not redownload evidence"),
    )
    report = source.restore_state([{"articles": [eligible, late, missing], "counts": {"read": 3}}])
    assert source.articles == [eligible]
    assert {row["id"] for row in report["excluded"]} == {"late", "missing"}
    assert source.counts["read"] == 3


@pytest.mark.parametrize("field", ["text", "received_at", "content_sha256"])
def test_conflicting_source_identity_cannot_replace_original_evidence(field):
    original = snapshot()
    source = DiscoverySources(object(), retained_articles=[original])
    with pytest.raises(ValueError, match="conflicting content"):
        source.restore_state([{"articles": [original | {field: "different"}]}])
    assert source.articles == [original]
