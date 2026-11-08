"""Outer decisions use persisted facts; fixtures do not establish research quality."""

import json
from copy import deepcopy

import pytest

from quant_rsi.research.discovery_contracts import (
    guard_numerical,
    guard_records,
    patch_policy,
    research_policy,
)
from quant_rsi.research.discovery_loop import run_discovery
from quant_rsi.research.study_resume import StudyCache


def cache_at(tmp_path):
    # The same real operation receipt implementation; identity setup belongs to run_study.
    cache = object.__new__(StudyCache)
    cache.root = tmp_path / "cache"
    cache.root.mkdir(exist_ok=True)
    cache.progress = None
    return cache


def runner(spec):
    return {
        "status": "completed",
        "fit_count": 1,
        "metric": 2 if spec["kind"] == "baseline" else spec["candidate"]["metric"],
    }


@pytest.mark.parametrize("kind,counter", [("survey", "surveys"), ("acquire", "acquisitions")])
@pytest.mark.parametrize("limit", [0, 1])
def test_budget_refusal_is_feedback_before_next_decision_and_survives_resume(
    tmp_path, kind, counter, limit
):
    calls, seen = [], []
    interrupted = False
    request = {"input_id": "acquired.demand"}

    def provider(value):
        calls.append(value)
        return {"status": "needs_data", "gaps": ["unavailable source"]}

    def decide(context):
        nonlocal interrupted
        seen.append(context)
        iteration = context["iteration"]
        used = min(iteration - 1, limit)
        assert context["action_budgets"][counter] == {
            "limit": limit,
            "used": used,
            "remaining": limit - used,
        }
        if iteration <= limit + 1:
            return {"action": kind, "query": "retry source", "data_requests": [request]}
        diagnostic = context["memory"][-1]
        assert diagnostic["kind"] == "diagnostic"
        assert diagnostic["status"] == "not_executed"
        assert diagnostic["reason"] == counter + "_budget_exhausted"
        assert diagnostic["operation"] == kind
        assert diagnostic["action_budgets"][counter]["remaining"] == 0
        assert diagnostic["ref"] in context["consumed_memory_refs"]
        path = tmp_path / f"discovery/memory/budget_{limit + 1}_0.json"
        assert json.loads(path.read_text()) == diagnostic
        assert diagnostic["requests"] == (
            [request] if kind == "acquire" else [{"query": "retry source", "reason": ""}]
        )
        if not interrupted:
            interrupted = True
            raise RuntimeError("interrupt after receiving refusal")
        return {"action": "stop"}

    options = dict(
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=decide,
        run_experiment=lambda _: pytest.fail("zero-fit preparation must not fit"),
        survey=provider,
        acquire=provider,
        survey_budget=limit,
        acquisition_budget=limit,
        fit_budget=0,
        decision_budget=limit + 2,
    )
    context = {"source_limits": {"read": 6, "search": 2, "extract": 12}}
    with pytest.raises(RuntimeError, match="interrupt after receiving refusal"):
        run_discovery(context, **options)
    diagnostic_path = tmp_path / f"discovery/memory/budget_{limit + 1}_0.json"
    original = diagnostic_path.read_bytes()
    result = run_discovery(context, **options)
    assert len(calls) == limit
    assert result["counts"][counter] == limit
    assert result["counts"]["decisions"] == limit + 2
    assert result["stop_reason"] == "stop"
    assert diagnostic_path.read_bytes() == original
    assert seen[-1] == seen[-2]
    assert any(m.get("status") == "not_executed" for m in result["memory"])


def test_partial_acquisition_batch_records_all_undispatched_requests(tmp_path):
    requests = [{"input_id": f"acquired.input_{n}"} for n in range(3)]
    calls = []
    result = run_discovery(
        {},
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=lambda _: {"action": "acquire", "data_requests": requests},
        acquire=lambda request: calls.append(request) or {"status": "needs_data"},
        run_experiment=lambda _: pytest.fail("preparation must not fit"),
        decision_budget=1,
        acquisition_budget=1,
        fit_budget=0,
    )
    assert calls == requests[:1]
    diagnostic = result["memory"][-1]
    assert diagnostic["status"] == "not_executed"
    assert diagnostic["requests"] == requests[1:]
    assert diagnostic["action_budgets"]["acquisitions"] == {
        "limit": 1,
        "used": 1,
        "remaining": 0,
    }


def test_feedback_is_on_disk_before_next_decision_and_resume(tmp_path):
    actions = [
        {"action": "test", "candidates": [{"name": "one", "metric": 3}]},
        {"action": "revise", "candidates": [{"name": "two", "metric": 1}]},
    ]
    calls = []

    def decide(context):
        calls.append(context)
        if len(calls) == 2:
            facts = [r for r in context["memory"] if r["kind"] == "experiment"]
            assert [r["result"]["metric"] for r in facts] == [2, 3]
            saved = json.loads((tmp_path / "discovery/memory/candidate_1_0.json").read_text())
            assert saved["ref"] in context["consumed_memory_refs"]
            assert context["policy"]["settings"]["proposal"]
        return deepcopy(actions[len(calls) - 1])

    cache = cache_at(tmp_path)
    result = run_discovery(
        {}, root=tmp_path, cache=cache, decide=decide, run_experiment=runner, decision_budget=2
    )
    assert result["selected"]["result"]["metric"] == 1
    assert result["counts"]["fits"] == 3
    replay = run_discovery(
        {},
        root=tmp_path,
        cache=cache,
        decide=lambda _: pytest.fail("completed decision was repeated"),
        run_experiment=runner,
        decision_budget=2,
    )
    assert result == replay


def test_new_inputs_require_new_baseline_and_no_global_ranking(tmp_path):
    actions = iter(
        [
            {"action": "test", "candidates": [{"name": "first", "metric": 1}]},
            {"action": "acquire", "data_requests": [{"field": "new"}]},
            {"action": "test", "candidates": [{"name": "second", "metric": 0.1}]},
        ]
    )
    fitted = []

    def fit(spec):
        fitted.append(spec)
        return runner(spec)

    result = run_discovery(
        {},
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=lambda _: next(actions),
        run_experiment=fit,
        acquire=lambda _: {"status": "completed", "values": [1, 2]},
        fit_budget=4,
    )
    assert [s["kind"] for s in fitted] == ["baseline", "candidate", "baseline", "candidate"]
    assert len(result["groups"]) == 2
    assert result["selected"] is None
    assert result["counts"]["fits"] == 4


def test_no_partial_linked_pair_when_budget_insufficient(tmp_path):
    actions = iter(
        [
            {"action": "test", "candidates": [{"name": "first", "metric": 1}]},
            {"action": "acquire", "data_requests": [{"field": "new"}]},
            {"action": "test", "candidates": [{"name": "second", "metric": 0.1}]},
        ]
    )
    result = run_discovery(
        {},
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=lambda _: next(actions),
        run_experiment=runner,
        acquire=lambda _: {"status": "completed", "values": [1, 2]},
    )
    assert result["counts"]["fits"] == 2
    assert result["stop_reason"] == "fit_budget_exhausted"


def test_zero_fit_diagnostic_does_not_exhaust_research(tmp_path):
    actions = iter(
        [
            {"action": "test", "candidates": [{"name": "missing"}]},
            {"action": "test", "candidates": [{"name": "ready", "metric": 1}]},
        ]
    )
    result = run_discovery(
        {},
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=lambda _: next(actions),
        run_experiment=runner,
        decision_budget=2,
        validate_candidate=lambda c, _: {
            "status": "needs_data" if c["name"] == "missing" else "pass"
        },
    )
    assert result["counts"]["fits"] == 2
    assert result["memory"][1]["result"]["status"] == "needs_data"


def test_temporal_guard_excludes_source_versions_and_terminal_contents():
    valid = {
        "id": "a",
        "published_at": "2020-01-01T00:00:00Z",
        "received_at": "2020-01-01T00:00:00Z",
        "version_at": "2020-01-01T00:00:00Z",
    }
    admitted, report = guard_records(
        [
            valid,
            valid | {"id": "b", "version_at": "2021-01-01T00:00:00Z"},
            {"id": "c", "text": "must not enter diagnostics"},
            {"id": "d", "partition": "terminal", "text": "held out secret"},
        ],
        mode="retained",
        cutoff="2020-02-01T00:00:00Z",
    )
    assert admitted == [valid]
    assert report["excluded"] == ["b", "c", "d"]
    assert "secret" not in json.dumps(report)
    assert "must not" not in json.dumps(report)


def test_numerical_gate_checks_each_origin_and_training_labels():
    report = guard_numerical(
        prediction_times=["2020-03-01T00:00:00Z", "2020-03-02T00:00:00Z"],
        available_times=["2020-03-02T00:00:00Z", "2020-03-02T00:00:00Z"],
        label_times=["2020-04-01T00:00:00Z"],
        fit_time="2020-03-03T00:00:00Z",
        preprocessing_partition="development",
    )
    assert report["status"] == "reject"
    assert report["rules"] == [
        "unavailable_row:0",
        "immature_training_label",
        "preprocessing_outside_training",
    ]


def test_policy_patch_changes_consumed_configuration_and_rejects_authority(tmp_path):
    original = research_policy()
    patch = {
        "base_identity": original["identity"],
        "changes": {"proposal": "Try demand mechanisms."},
        "reason": "prior supply mechanism had insufficient data",
        "evidence_refs": ["feedback"],
    }
    revised = patch_policy(original, patch)
    assert revised["parent"] == original["identity"]
    assert original["settings"]["proposal"] != revised["settings"]["proposal"]
    seen = []
    run_discovery(
        {},
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=lambda context: seen.append(context["policy"]) or {"action": "stop"},
        run_experiment=runner,
        policy=revised,
    )
    assert seen == [revised]
    with pytest.raises(ValueError, match="protected"):
        patch_policy(original, patch | {"changes": {"fit_budget": 100}})


def test_missing_input_still_reports_original_baseline(tmp_path):
    result = run_discovery(
        {},
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=lambda _: {"action": "test", "candidates": [{"name": "absent"}]},
        run_experiment=runner,
        decision_budget=1,
        validate_candidate=lambda *_: {"status": "needs_data"},
    )
    assert result["counts"]["fits"] == 1
    assert result["selected"]["spec"]["kind"] == "baseline"
    assert result["selected"]["spec"]["input_bundles"] == []


def test_unavailable_metric_retains_baseline(tmp_path):
    def fit(spec):
        return {
            "status": "completed",
            "fit_count": 1,
            "metric": None if spec["kind"] == "baseline" else 2,
        }

    result = run_discovery(
        {},
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=lambda _: {"action": "test", "candidates": [{"name": "a"}]},
        run_experiment=fit,
        decision_budget=1,
    )
    assert result["selected"]["spec"]["kind"] == "baseline"


def test_normalized_duplicate_does_not_refit(tmp_path):
    actions = iter(
        [
            {"action": "test", "candidates": [{"name": "a", "metric": 1}]},
            {"action": "test", "candidates": [{"name": "renamed", "metric": 1}]},
        ]
    )
    result = run_discovery(
        {},
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=lambda _: next(actions),
        run_experiment=runner,
        decision_budget=2,
        validate_candidate=lambda *_: {"status": "ready", "recipe_ref": "same"},
    )
    assert result["counts"]["duplicates"] == 1
    assert result["counts"]["fits"] == 2


def test_prior_memory_is_guarded_and_supplied_to_decisions(tmp_path):
    seen = []
    context = {
        "information_mode": "retained",
        "research_cutoff": "2020-02-01T00:00:00Z",
        "prior_memory": [
            {"ref": "old", "created_at": "2020-01-01T00:00:00Z"},
            {"ref": "future", "created_at": "2021-01-01T00:00:00Z"},
            {"ref": "holdout", "partition": "terminal"},
        ],
    }
    run_discovery(
        context,
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=lambda c: seen.append(c) or {"action": "stop"},
        run_experiment=runner,
    )
    assert seen[0]["consumed_memory_refs"] == ["old"]
    assert seen[0]["prior_memory"] == [context["prior_memory"][0]]


def test_failed_fit_record_survives_before_error(tmp_path):
    def failed(spec):
        raise RuntimeError("worker exited after fit started")

    with pytest.raises(RuntimeError):
        run_discovery(
            {},
            root=tmp_path,
            cache=cache_at(tmp_path),
            decide=lambda _: {"action": "test", "candidates": [{"name": "a"}]},
            run_experiment=failed,
        )
    record = json.loads((tmp_path / "discovery/memory/baseline_1_0_failure.json").read_text())
    assert record["fit_count"] == 1
    assert record["kind"] == "experiment_failure"


def test_context_restore_hook_runs_on_cached_decisions(tmp_path):
    seen = []
    cache = cache_at(tmp_path)
    arguments = dict(
        root=tmp_path,
        cache=cache,
        decide=lambda _: {"action": "stop"},
        run_experiment=runner,
        on_context=lambda context: seen.append(context["iteration"]),
    )
    run_discovery({}, **arguments)
    run_discovery({}, **arguments)
    assert seen == [1, 1]


def test_survey_keeps_accounting_and_only_eligible_findings(tmp_path):
    actions = iter([{"action": "survey", "query": "demand"}, {"action": "stop"}])
    article = {
        "id": "allowed",
        "published_at": "2020-01-01T00:00:00Z",
        "received_at": "2020-01-01T00:00:00Z",
        "version_available_at": "2020-01-01T00:00:00Z",
    }
    response = {
        "evidence": [article, article | {"id": "late", "received_at": "2021-01-01T00:00:00Z"}],
        "counts": {"searches": 0, "reads": 2},
        "operations": ["read-one", "read-two"],
        "findings": [
            {"evidence_ids": ["allowed"], "mechanism": "usable"},
            {"evidence_ids": ["late"], "mechanism": "secret"},
        ],
    }
    result = run_discovery(
        {"information_mode": "retained", "research_cutoff": "2020-02-01T00:00:00Z"},
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=lambda _: next(actions),
        run_experiment=runner,
        survey=lambda _: response,
        decision_budget=2,
    )
    record = next(m["result"] for m in result["memory"] if m["kind"] == "survey")
    assert record["counts"]["reads"] == 2
    assert record["operations"] == response["operations"]
    assert len(record["findings"]) == 1
    assert "secret" not in json.dumps(record)


def test_reacquisition_replaces_current_bundle_without_mutating_past_experiment(tmp_path):
    actions = iter(
        [
            {"action": "acquire", "data_requests": [{"input_id": "acquired.demand"}]},
            {"action": "test", "candidates": [{"name": "one", "metric": 1}]},
            {"action": "acquire", "data_requests": [{"input_id": "acquired.demand"}]},
            {"action": "test", "candidates": [{"name": "two", "metric": 0.5}]},
        ]
    )
    versions = iter([1, 2])
    result = run_discovery(
        {},
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=lambda _: next(actions),
        run_experiment=runner,
        acquire=lambda request: {
            "status": "available",
            "request": request,
            "version": next(versions),
        },
        decision_budget=4,
        fit_budget=4,
    )
    groups = list(result["groups"].values())
    assert len(groups) == 2
    assert groups[0]["baseline"]["spec"]["input_bundles"][0]["version"] == 1
    assert groups[1]["baseline"]["spec"]["input_bundles"][0]["version"] == 2
    assert len(groups[1]["baseline"]["spec"]["input_bundles"]) == 1
    assert len([r for r in result["memory"] if r["kind"] == "acquire"]) == 2


def test_acquisition_candidates_count_as_discovery_without_blocking_later_fit(tmp_path):
    candidate = {"name": "demand", "metric": 1}
    actions = iter(
        [
            {
                "action": "acquire",
                "candidates": [candidate],
                "data_requests": [{"input_id": "acquired.demand"}],
            },
            {"action": "test", "candidates": [candidate]},
            {"action": "stop", "candidates": [{"name": "pending", "metric": 3}]},
        ]
    )
    result = run_discovery(
        {},
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=lambda _: next(actions),
        run_experiment=runner,
        acquire=lambda request: {"status": "available", "request": request},
    )
    assert result["counts"]["proposed"] == 3
    assert result["counts"]["distinct_candidates"] == 2
    assert result["counts"]["proposal_duplicates"] == 1
    assert result["counts"]["duplicates"] == 0
    assert (
        result["counts"]["screened"]
        == result["counts"]["executable"]
        == result["counts"]["trained"]
        == 1
    )
    assert result["counts"]["fits"] == 2


def test_acquisition_only_run_retains_discovered_candidate_count(tmp_path):
    actions = iter(
        [
            {
                "action": "acquire",
                "candidates": [{"name": "missing", "metric": 1}],
                "data_requests": [{"input_id": "acquired.missing"}],
            },
            {"action": "stop", "candidates": []},
        ]
    )
    result = run_discovery(
        {},
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=lambda _: next(actions),
        run_experiment=runner,
        acquire=lambda _: {"status": "needs_data"},
    )
    assert result["counts"]["proposed"] == result["counts"]["distinct_candidates"] == 1
    assert result["counts"]["trained"] == 0
    assert result["counts"]["fits"] == 1  # baseline remains a usable result


def test_failed_fit_charges_survive_resume_without_changing_cached_decision(tmp_path):
    calls = []
    decisions = []

    def fit(spec):
        calls.append(spec["kind"])
        if len(calls) == 1:
            raise RuntimeError("Worker failed after fit started")
        return runner(spec)

    def decide(context):
        decisions.append(context)
        return {"action": "test", "candidates": [{"name": "candidate", "metric": 1}]}

    arguments = dict(
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=decide,
        run_experiment=fit,
        fit_budget=2,
        decision_budget=1,
    )
    with pytest.raises(RuntimeError):
        run_discovery({}, **arguments)
    result = run_discovery({}, **arguments)
    assert len(decisions) == 1  # originally frozen decision replayed
    assert len(calls) == 2  # failed baseline + successful baseline; no budget for candidate
    assert result["counts"]["fits"] == 2
    assert result["counts"]["failed_fits"] == 1
    assert result["counts"]["successful_fits"] == 1
    assert len(result["fit_failures"]) == 1
    assert result["selected"]["spec"]["kind"] == "baseline"


def test_json_candidate_renaming_uses_normalized_discovery_identity(tmp_path):
    candidate = {
        "name": "one",
        "description": "Original",
        "inputs": [{"name": "price", "factor_id": "price.close"}],
        "expression": {
            "schema_version": 2,
            "nodes": [{"name": "x", "op": "leaf", "leaf": "price", "inputs": [], "window": None}],
        },
        "evidence_ids": ["news"],
    }
    actions = iter(
        [
            {"action": "acquire", "candidates": [candidate], "data_requests": [{"field": "x"}]},
            {
                "action": "stop",
                "candidates": [candidate | {"name": "two", "description": "Renamed"}],
            },
        ]
    )
    result = run_discovery(
        {},
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=lambda _: next(actions),
        run_experiment=runner,
        acquire=lambda _: {"status": "needs_data"},
        decision_budget=2,
    )
    assert result["counts"]["proposed"] == 2
    assert result["counts"]["distinct_candidates"] == 1
    assert result["counts"]["proposal_duplicates"] == 1
