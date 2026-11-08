"""Native zero-fit workflow uses actual disclosure routing and feature computation offline."""

import json
from copy import deepcopy

import pytest
from test_discovery_disclosures import PROFILE, DisclosureLLM, PostArchive, measurement
from test_discovery_preparation import forbid_outcomes
from test_research_study import command

from quant_rsi.cli import main
from quant_rsi.research import article_reader, mops_source, study, study_discovery, study_resume


class NativeDisclosureLLM(DisclosureLLM):
    def __init__(self):
        super().__init__()
        self.contexts = []
        self.decision_schemas = []

    def structured(self, prompt, schema):
        context = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
        if "operators" not in context:
            return super().structured(prompt, schema)
        context = context["context"]
        self.contexts.append(deepcopy(context))
        self.decision_schemas.append(deepcopy(schema))
        action = {
            "action": "stop",
            "reason": "Fixture has demonstrated partial preparation without prediction claims",
            "candidates": [],
            "data_requests": [],
            "query": None,
        }
        evidence = [a for a in context["evidence"] if a.get("provider") == "mops"]
        if not evidence:
            return json.dumps(
                action
                | {
                    "action": "survey",
                    "reason": "Read the official archive for the required operating period",
                    "query": "Reported operating demand",
                    "survey_symbol": "TSM",
                    "disclosure_query": {"provider": "mops", "year": 2020, "month": 3},
                }
            )
        candidate = {
            "name": "reported_demand_mean",
            "description": "Fixture hypothesis: a smoothed quoted operating measurement",
            "inputs": [{"name": "demand", "factor_id": "acquired.demand"}],
            "expression": {
                "schema_version": 2,
                "nodes": [
                    {
                        "name": "demand",
                        "op": "leaf",
                        "inputs": [],
                        "leaf": "demand",
                        "window": None,
                    },
                    {
                        "name": "smooth",
                        "op": "rolling_mean",
                        "inputs": ["demand"],
                        "leaf": None,
                        "window": 3,
                    },
                ],
            },
            "evidence_ids": [evidence[0]["id"]],
        }
        preparations = [m for m in context["memory"] if m.get("kind") == "preparation"]
        if not preparations or (context["input_bundles"] and len(preparations) == 1):
            action.update(
                action="test", candidates=[candidate], reason="Compute the declared input"
            )
        elif not context["input_bundles"]:
            action.update(
                action="acquire",
                data_requests=[measurement(source_ids=[evidence[0]["id"]])],
                reason="Missing candidate input requires a quote from the exact selected document",
            )
        return json.dumps(action)


@pytest.mark.parametrize("interrupt_outer_commit", [False, True])
def test_native_disclosure_survey_missing_input_acquisition_and_partial_column(
    tmp_path, monkeypatch, interrupt_outer_commit
):
    forbid_outcomes(monkeypatch)
    transport, archive = NativeDisclosureLLM(), PostArchive()
    monkeypatch.setattr(article_reader, "_fetch", archive)
    monkeypatch.setattr(study, "LLMHypothesizer", lambda *args, **kwargs: transport)
    monkeypatch.setattr(study_discovery, "LLMHypothesizer", lambda *args, **kwargs: transport)
    profiles = tmp_path / "profiles"
    issuer = profiles / "US" / "TSM"
    issuer.mkdir(parents=True)
    (issuer / "disclosures.json").write_text(json.dumps(PROFILE))
    root = tmp_path / "study"
    args = command(
        root,
        "--llm",
        "codex",
        "--research-mode",
        "discovery",
        "--news-research",
        "--max-fits",
        "0",
        "--horizon-sessions",
        "3",
        "--rounds",
        "6",
        "--no-ui",
        "--no-history",
        "--no-related-prices",
        "--factor-root",
        str(profiles),
    )
    args[1] = "TSM"
    if interrupt_outer_commit:
        original_write = study_resume.atomic_write

        def interrupted_write(path, data):
            if path.name == "discovery_survey_1_0.json":
                raise RuntimeError("outer disclosure survey commit interrupted")
            original_write(path, data)

        monkeypatch.setattr(study_resume, "atomic_write", interrupted_write)
        with pytest.raises(RuntimeError, match="outer disclosure survey commit interrupted"):
            main(args)
        assert len(archive.calls) == 2
        assert len(transport.calls) == 2  # Selection and review completed before outer commit.
        original_survey = next(c for c in transport.calls if "articles" in c)
        original_article = original_survey["articles"][0]
        monkeypatch.setattr(study_resume, "atomic_write", original_write)
        assert main(["research-resume", str(root)]) == 0
    else:
        assert main(args) == 0
    result = json.loads((root / "result.json").read_text())
    discovery = result["discovery"]
    assert discovery["mode"] == "preparation_only" and discovery["counts"]["fits"] == 0
    assert not list((root / "checkpoints").glob("fit_*.json"))
    assert "terminal_evaluation" not in result
    assert all(c["symbol"] == "TSM" for c in transport.contexts)
    assert transport.contexts[0]["disclosure_sources"] == [PROFILE]
    schema = transport.decision_schemas[0]
    assert "disclosure_query" in schema["properties"]
    assert "source_ids" in schema["$defs"]["DataRequest"]["properties"]
    memory = discovery["memory"]
    survey = next(m for m in memory if m["kind"] == "survey")
    assert survey["request"]["disclosure_query"] == {"provider": "mops", "year": 2020, "month": 3}
    article = next(a for a in survey["result"]["evidence"] if a.get("provider") == "mops")
    assert article["source_request"]["serialNumber"] == "2"
    if interrupt_outer_commit:
        assert article == original_article
    assert sum("documents" in call for call in transport.calls) == 1
    assert sum("articles" in call for call in transport.calls) == 1
    acquired = next(m for m in memory if m["kind"] == "acquire")["result"]
    assert acquired["request"]["source_ids"] == [article["id"]]
    assert acquired["observations"][0]["value"] == 12.0
    records = [m for m in memory if m["kind"] == "preparation"]
    assert len(records) == 2
    assert records[0]["result"]["status"] == "needs_data"
    assert records[0]["candidate"] == records[1]["candidate"]
    prepared = records[1]["result"]["preparation"]
    coverage = prepared["coverage"]
    assert 0 < coverage["computed_rows"] < coverage["required_rows"]
    assert {r["value"] for r in prepared["rows"]} == {None, 12.0}
    assert {r["partition"] for r in prepared["rows"]} == {"train", "development"}
    assert prepared["source_ids"] == [article["id"]]
    assert [url for url, _ in archive.calls] == [mops_source.LIST_URL, mops_source.DETAIL_URL]
    assert discovery["source_counts"]["search"] == 1
    assert discovery["source_counts"]["read"] == 1
    assert discovery["source_counts"]["extract"] == 1
