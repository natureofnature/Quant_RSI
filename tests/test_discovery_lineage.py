"""Explicit revision edges must refer to candidates actually visible to the decision."""

import json
from copy import deepcopy

import pytest

from quant_rsi.research.discovery_factors import DiscoveryFactor
from quant_rsi.research.discovery_loop import run_discovery
from quant_rsi.research.discovery_report import render_discovery
from quant_rsi.research.discovery_sources import discovery_decision
from quant_rsi.research.study_resume import StudyCache


def candidate(name, window, parents=()):
    return {
        "name": name,
        "description": "A lagged disclosed demand hypothesis, not a measured predictive gain.",
        "inputs": [{"name": "demand", "factor_id": "acquired.demand"}],
        "evidence_ids": ["news-1"],
        "parent_recipe_refs": list(parents),
        "expression": {
            "schema_version": 2,
            "nodes": [
                {"name": "raw", "op": "leaf", "leaf": "demand"},
                {"name": "lagged", "op": "lag", "inputs": ["raw"], "window": window},
            ],
        },
    }


def recipe_ref(value):
    return DiscoveryFactor.model_validate_json(json.dumps(value)).recipe_ref


def decision(action, candidates):
    return {
        "action": action,
        "reason": "Use previously observed candidate evidence to choose the next investigation.",
        "candidates": deepcopy(candidates),
        "data_requests": [],
        "query": None,
    }


def cache_at(root):
    cache = object.__new__(StudyCache)
    cache.root = root / "cache"
    cache.root.mkdir(exist_ok=True)
    cache.progress = None
    return cache


def execute(root, context, actions, *, acquisition=None):
    seen, fitted = [], []
    actions = iter(actions)

    def decide(current):
        seen.append(deepcopy(current))
        return next(actions)

    def fit(spec):
        fitted.append(deepcopy(spec))
        return {"status": "completed", "fit_count": 1, "metric": 2.0}

    result = run_discovery(
        context,
        root=root,
        cache=cache_at(root),
        decide=decide,
        run_experiment=fit,
        acquire=acquisition,
        decision_budget=3,
        fit_budget=4,
        validate_candidate=lambda value, _: {"status": "ready", "recipe_ref": recipe_ref(value)},
    )
    return result, seen, fitted


def test_precise_parent_and_duplicate_replacement_edge_survive_without_extra_fit(tmp_path):
    first, other = candidate("first", 2), candidate("other", 4)
    revised = candidate("revised", 6, [recipe_ref(first)])
    duplicate = candidate("same_expression_other_parent", 6, [recipe_ref(other)])
    result, seen, fitted = execute(
        tmp_path,
        {},
        [
            decision("test", [first, other]),
            decision("revise", [revised]),
            decision("revise", [duplicate]),
        ],
    )
    assert seen[0]["visible_candidates"] == []
    visible = {item["recipe_ref"]: item for item in seen[1]["visible_candidates"]}
    assert set(visible) == {recipe_ref(first), recipe_ref(other)}
    assert all(item["memory_refs"] for item in visible.values())
    assert all(
        set(item["memory_refs"]) <= set(seen[1]["consumed_memory_refs"])
        for item in visible.values()
    )
    assert [spec["candidate"]["name"] for spec in fitted if spec["kind"] == "candidate"] == [
        "first",
        "other",
        "revised",
    ]
    assert result["counts"]["fits"] == 4
    assert result["counts"]["duplicates"] == 1
    interpretations = [item for item in result["memory"] if item["kind"] == "interpretation"]
    left, right = [item["candidate_lineage"][0] for item in interpretations[1:]]
    assert left["recipe_ref"] == right["recipe_ref"] == recipe_ref(revised)
    assert left["parent_recipe_refs"] == [recipe_ref(first)]
    assert right["parent_recipe_refs"] == [recipe_ref(other)]
    assert left["status"] == right["status"] == "accepted"
    persisted = json.loads((tmp_path / "discovery" / "result.json").read_text())
    assert persisted["memory"] == result["memory"]


@pytest.mark.parametrize("source", ["unknown", "future", "terminal", "rejected"])
def test_invisible_parent_cannot_authorize_acquisition_or_candidate_fit(tmp_path, source):
    parent = candidate("hidden", 2)
    prior = {
        "kind": "interpretation",
        "ref": "prior-record",
        "created_at": "2020-01-01T00:00:00Z",
        "decision": decision("test", [parent]),
    }
    if source == "future":
        prior["created_at"] = "2021-01-01T00:00:00Z"
    elif source == "terminal":
        prior["partition"] = "terminal"
    elif source == "rejected":
        prior["candidate_lineage"] = [
            {
                "recipe_ref": recipe_ref(parent),
                "parent_recipe_refs": [],
                "status": "rejected",
                "reason": "invalid_candidate_lineage",
            }
        ]
    context = {
        "information_mode": "retained",
        "research_cutoff": "2020-02-01T00:00:00Z",
        "prior_memory": [] if source == "unknown" else [prior],
        # Caller-supplied indexes cannot invent visibility independently of admitted Memory.
        "visible_candidates": [
            {"recipe_ref": recipe_ref(parent), "name": "forged", "memory_refs": []}
        ],
    }
    action = decision("acquire", [candidate("child", 5, [recipe_ref(parent)])])
    action["data_requests"] = [{"input_id": "acquired.demand"}]
    result, seen, fitted = execute(
        tmp_path,
        context,
        [action, decision("stop", [])],
        acquisition=lambda _: pytest.fail("rejected lineage reached acquisition"),
    )
    assert seen[0]["visible_candidates"] == []
    assert all(spec["kind"] == "baseline" for spec in fitted)
    assert result["counts"]["acquisitions"] == 0
    interpretation = next(item for item in result["memory"] if item["kind"] == "interpretation")
    assert interpretation["candidate_lineage"][0]["status"] == "rejected"
    assert any(item.get("reason") == "invalid_candidate_lineage" for item in result["memory"])


def test_same_batch_forward_parent_and_empty_revision_are_rejected(tmp_path):
    first = candidate("not_yet_visible", 2)
    child = candidate("child", 3, [recipe_ref(first)])
    result, _, fitted = execute(
        tmp_path,
        {},
        [
            decision("test", [first, child]),
            decision("revise", [candidate("unlinked", 7)]),
            decision("stop", []),
        ],
    )
    assert all(spec["kind"] == "baseline" for spec in fitted)
    diagnostics = [
        item for item in result["memory"] if item.get("reason") == "invalid_candidate_lineage"
    ]
    assert len(diagnostics) == 2


def test_legacy_report_does_not_invent_a_parent_or_hide_new_explicit_edges():
    parent = candidate("legacy", 2)
    parent.pop("parent_recipe_refs")
    revised = candidate("revised", 3, [recipe_ref(parent)])
    result = {
        "symbol": "KO",
        "notice": "Offline lineage fixture",
        "configuration": {"question": "Investigate demand"},
        "study": {"conclusion": "Fixture"},
        "discovery": {
            "counts": {},
            "groups": {},
            "selected": None,
            "memory": [
                {"kind": "interpretation", "decision": decision("test", [parent])},
                {
                    "kind": "interpretation",
                    "decision": decision("revise", [revised]),
                    "candidate_lineage": [
                        {
                            "recipe_ref": recipe_ref(revised),
                            "parent_recipe_refs": [recipe_ref(parent)],
                            "status": "accepted",
                            "reason": None,
                        }
                    ],
                },
            ],
        },
    }
    markdown, html = render_discovery(result)
    for document in [markdown, html]:
        assert "legacy" in document.lower() and "unspecified" in document.lower()
        assert recipe_ref(revised) in document and recipe_ref(parent) in document
        assert "accepted" in document


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "empty",
        "self",
        "duplicate",
        "malformed",
        "unknown",
        "candidate_shape",
        "parent_string",
        "parent_number",
        "parent_null",
    ],
)
def test_invalid_provider_revision_becomes_feedback_without_transport_retry(tmp_path, invalid):
    value = candidate("revision", 3)
    if invalid == "missing":
        value.pop("parent_recipe_refs")
    elif invalid == "self":
        value["parent_recipe_refs"] = [recipe_ref(value)]
    elif invalid == "duplicate":
        value["parent_recipe_refs"] = ["a" * 64, "a" * 64]
    elif invalid == "malformed":
        value["parent_recipe_refs"] = ["not-a-recipe-id"]
    elif invalid == "unknown":
        value["parent_recipe_refs"] = ["a" * 64]
    elif invalid == "candidate_shape":
        value = {"name": "broken", "parent_recipe_refs": [7]}
    elif invalid == "parent_string":
        value["parent_recipe_refs"] = "a" * 64
    elif invalid == "parent_number":
        value["parent_recipe_refs"] = 7
    elif invalid == "parent_null":
        value["parent_recipe_refs"] = None
    contexts, fitted = [], []

    class Transport:
        def structured(self, prompt, schema):
            assert "parent_recipe_refs" in prompt
            payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
            current = payload["context"]
            contexts.append(current)
            assert current["visible_candidates"] == []
            factor_schema = schema["$defs"]["DiscoveryFactor"]
            assert "parent_recipe_refs" in factor_schema["required"]
            if len(contexts) == 2:
                assert any(
                    item.get("reason") == "invalid_candidate_lineage" for item in current["memory"]
                )
                return json.dumps(decision("stop", []))
            assert len(contexts) == 1
            return json.dumps(decision("revise", [value]))

    def fit(spec):
        fitted.append(spec)
        assert spec["kind"] == "baseline"
        return {"status": "completed", "fit_count": 1, "metric": 2.0}

    result = run_discovery(
        {},
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=lambda context: discovery_decision(Transport(), context),
        run_experiment=fit,
        decision_budget=2,
    )
    assert len(contexts) == 2 and len(fitted) == 1
    assert result["counts"]["decisions"] == 2
    assert result["counts"]["trained"] == 0
    assert result["memory"][0]["candidate_lineage"][0]["status"] == "rejected"
    assert not list((tmp_path / "discovery" / "failures").glob("*.json"))
    saved = json.loads((tmp_path / "discovery" / "result.json").read_text())
    assert saved["memory"] == result["memory"]
    report = {
        "symbol": "KO",
        "notice": "Offline invalid-provider fixture",
        "configuration": {"question": "Investigate demand"},
        "study": {"conclusion": "Invalid proposal; retained baseline"},
        "discovery": saved,
    }
    markdown, html = render_discovery(report)
    for document in [markdown, html]:
        assert "rejected" in document
        assert "invalid_candidate_lineage" in document
        assert value["name"] in document


def test_provider_transport_error_remains_a_failure_not_a_candidate_diagnostic(tmp_path):
    calls = []

    class Transport:
        def structured(self, prompt, schema):
            calls.append(prompt)
            raise RuntimeError("offline provider unavailable")

    with pytest.raises(RuntimeError, match="offline provider unavailable"):
        run_discovery(
            {},
            root=tmp_path,
            cache=cache_at(tmp_path),
            decide=lambda context: discovery_decision(Transport(), context),
            run_experiment=lambda _: pytest.fail("transport failure must not fabricate a fit"),
        )
    assert len(calls) == 1
    failure = json.loads((tmp_path / "discovery" / "failures" / "decision_1.json").read_text())
    assert failure["type"] == "RuntimeError"
    assert failure["message"] == "offline provider unavailable"
    assert not (tmp_path / "discovery" / "result.json").exists()
