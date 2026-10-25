"""Pure contract examples; these hand-built traces are never real training evidence."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_rsi import changes, evidence, researcher
from quant_rsi.artifacts import ArtifactStore, canonical, decode, digest
from quant_rsi.multigen_changes import extend_capsule
from quant_rsi.researcher import ResearchPlan
from quant_rsi.task_definition import domain_ref, task_definition
from quant_rsi.trajectory_multigen import next_stop
from quant_rsi.trajectory_registration import registration_path, root_execution
from quant_rsi.trajectory_state import TrajectoryPlan


def ref(value):
    return digest(canonical(value))


def registered_plan(**kwargs):
    value = decode(Path("examples/research_development.json").read_bytes())
    value.update(
        schema_version=2,
        neighbor_domain=dict(
            kind="RegisteredNeighborDomain",
            schema_version=1,
            operators=["momentum", "mean_return"],
            window_min=1,
            window_max=12,
            initial_window_max=6,
            cap_step=1,
            max_cap_advances=2,
        ),
        trajectory_registration=dict(
            kind="TrajectoryRegistration",
            schema_version=1,
            workspace="/tmp/pure-contract",
            source_directory="/tmp/pure-contract/source",
            worker_attempts=40,
            registered_worker_seconds=800,
            max_proposal_stages=2,
            max_generations=2,
        ),
    )
    value["policy"]["window_max"] = 6
    for key, item in kwargs.items():
        value[key] = item
    return ResearchPlan.model_validate_json(canonical(value))


def unit_evidence(plan, scores):
    base = ref(researcher.package_definition(plan))
    task = ref(task_definition(plan, "data", ref(plan.development.model_dump(mode="json"))))
    checkpoint = researcher.initial_state("pure-unit", base, ref("memory0"))
    entries, parents, requests, attempts = [], {}, [], []
    for index, score in enumerate(scores):
        parent_ref = ref(checkpoint)
        parents[parent_ref] = checkpoint
        observation = dict(
            kind="ResearchObservation",
            visibility="development",
            campaign_id="pure-unit",
            task_ref=task,
            ordinal=index,
            request_ref=ref(["request", index]),
            factor=researcher.next_factor(plan, checkpoint) if score is not None else None,
            score=score,
            status="completed" if score is not None else "proposal_failed",
            eligible=score is not None,
            result_ref=ref(["result", index]) if score is not None else None,
            proposal_ref=ref(["proposal", index]) if score is not None else None,
        )
        requests.append(
            dict(
                ordinal=index,
                request_ref=observation["request_ref"],
                parent_ref=parent_ref,
                proposal_ref=observation["proposal_ref"],
                observation_ref=ref(observation),
            )
        )
        if score is not None:
            attempts.append(
                dict(
                    ordinal=index,
                    proposal_ref=observation["proposal_ref"],
                    result_ref=observation["result_ref"],
                    status="completed",
                    worker_seconds=20,
                )
            )
        entries.append(
            dict(
                observation_ref=ref(observation),
                observation=observation,
                observation_event_seq=index + 1,
                parent_checkpoint_ref=parent_ref,
                parent_incumbent=checkpoint["incumbent"],
            )
        )
        checkpoint = researcher.advance(
            plan, checkpoint, observation, ref(["memory", index]), parent_ref
        )
    capsule = dict(
        kind="DevelopmentEvidence",
        source_campaign_id="pure-unit",
        task_ref=task,
        source_manifest_digest=ref("manifest"),
        source_events_digest=ref("events"),
        feedback_watermark=len(entries),
        entries=entries,
        historical_attempts=len(entries),
        historical_worker_slots=sum(a["worker_seconds"] for a in attempts),
    )
    auth = dict(
        profile="registered_cap_v2",
        source_receipt_ref=ref("receipt"),
        source_head=dict(selection_ref=ref("selection")),
        expected_revision=0,
        source_cas_ref=None,
        development_data_ref="data",
    )
    capsule = extend_capsule(
        capsule, plan, ref(checkpoint), checkpoint, parents, auth, requests, attempts
    )
    return capsule


def propose(plan, capsule):
    request = changes.request_definition(
        "unit-request",
        ref(researcher.package_definition(plan)),
        ref(changes.proposer_definition(plan)),
        ref(capsule),
        capsule,
    )
    return changes.expected_change(ref(request), request, capsule), request


def test_domain_is_task_identity_and_authority_registration_is_separately_bound():
    plan = registered_plan()
    old = ResearchPlan.model_validate_json(Path("examples/research_development.json").read_bytes())
    assert task_definition(old, "data", "template") == dict(
        data_ref="data", development_template_ref="template"
    )
    original = task_definition(plan, "data", "template")
    altered = plan.model_copy(
        update={"neighbor_domain": plan.neighbor_domain.model_copy(update={"window_max": 13})}
    )
    assert task_definition(altered, "data", "template") != original
    altered = plan.model_copy(
        update={
            "trajectory_registration": plan.trajectory_registration.model_copy(
                update={"worker_attempts": 41}
            )
        }
    )
    assert task_definition(altered, "data", "template") == original
    assert ref(altered.model_dump(mode="json")) != ref(plan.model_dump(mode="json"))
    missing = plan.model_dump(mode="json")
    missing.pop("neighbor_domain")
    with pytest.raises(ValueError, match="explicit registered"):
        ResearchPlan.model_validate_json(canonical(missing))


def test_final_boundary_patch_counts_baseline_and_new_full_timeout():
    plan = registered_plan()
    response, _ = propose(plan, unit_evidence(plan, [-2.0, -1.0, -3.0]))
    assert response["outcome"] == "patch"
    patch = response["patch"]
    assert patch["value"] == 7 and patch["path"] == "policy.window_max"
    assert patch["reset_opportunity"]["required_attempts"] == 3
    assert patch["reset_opportunity"]["required_registered_seconds"] == 60
    assert patch["reset_opportunity"]["new_factor"] == dict(op="momentum", window=7)
    assert len(patch["reset_opportunity"]["prefix_observation_refs"]) == 2


@pytest.mark.parametrize("scores", [[-2.0, -3.0], [-3.0, -2.0, -1.0]])
def test_unsupported_or_superseded_boundary_is_no_patch(scores):
    plan = registered_plan()
    response, _ = propose(plan, unit_evidence(plan, scores))
    assert response["outcome"] == "no_patch"
    assert response["reason"] == "unsupported_final_boundary_feedback"


def test_no_static_reset_capacity_is_no_patch_not_reused_source_balance():
    plan = registered_plan()
    plan = plan.model_copy(
        update={
            "development": plan.development.model_copy(
                update={"budget": plan.development.budget.model_copy(update={"max_attempts": 2})}
            )
        }
    )
    response, _ = propose(plan, unit_evidence(plan, [-2.0, -1.0]))
    assert response["outcome"] == "no_patch"
    assert response["reason"] == "insufficient_reset_prefix_opportunity"


@pytest.mark.parametrize(
    "attempts,seconds,expected", [(5, 100, "patch"), (4, 100, "no_patch"), (5, 99, "no_patch")]
)
def test_reset_prefix_counts_unfunded_request_and_full_new_timeouts(attempts, seconds, expected):
    value = registered_plan().model_dump(mode="json")
    value["development"]["proposals"][0]["factor"]["window"] = 4
    value["development"]["budget"].update(max_attempts=attempts, max_worker_seconds=seconds)
    plan = ResearchPlan.model_validate_json(canonical(value))
    capsule = unit_evidence(plan, [-4.0, None, -3.0, -2.0])
    assert len(capsule["source_requests"]) == 4
    assert len(capsule["source_attempts"]) == 3
    assert capsule["historical_worker_slots"] == 60
    response, _ = propose(plan, capsule)
    assert response["outcome"] == expected
    if expected == "patch":
        opportunity = response["patch"]["reset_opportunity"]
        assert opportunity["required_attempts"] == 5
        assert opportunity["required_registered_seconds"] == 100
        assert len(opportunity["prefix_request_refs"]) == 4
        assert opportunity["prefix_attempts"][1] is None


def test_strict_threshold_and_stale_final_selection():
    plan = registered_plan()
    plan = plan.model_copy(
        update={"policy": plan.policy.model_copy(update={"min_improvement": 1.0})}
    )
    response, _ = propose(plan, unit_evidence(plan, [-2.0, -1.0]))
    assert response["outcome"] == "no_patch"
    plan = registered_plan()
    capsule = unit_evidence(plan, [-3.0, -2.0, -1.0])
    capsule["final_selection"] = capsule["entries"][-1]["parent_incumbent"]
    with pytest.raises(ValueError, match="final checkpoint"):
        propose(plan, capsule)


def test_two_typed_lineage_edges_and_missing_ancestor(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    plan = registered_plan()
    store.put_json(plan.neighbor_domain.model_dump(mode="json"))
    store.put_json(plan.trajectory_registration.model_dump(mode="json"))
    ancestors = []
    for generation, scores in enumerate([[-3.0, -2.0], [-4.0, -3.0, -2.0]]):
        capsule = unit_evidence(plan, scores)
        response, request = propose(plan, capsule)
        assert response["outcome"] == "patch"
        ancestors.append(store.put_json(researcher.package_definition(plan)))
        for value in [capsule, changes.proposer_definition(plan), request, response["patch"]]:
            store.put_json(value)
        plan = changes.experimental_plan(plan, ref(response["patch"]), ref(request), request)
        changes.validate_lineage(plan, store)
        assert plan.policy.window_max == 7 + generation
    closure = changes.lineage_artifact_refs(plan, store)
    assert set(ancestors) <= closure and domain_ref(plan) in closure
    (store.root / ancestors[0]).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        changes.validate_lineage(plan, store)


def test_c5_rejects_new_family_without_rejecting_legacy_lineage():
    plan = registered_plan()
    with pytest.raises(ValueError, match="unsupported"):
        evidence.require_legacy_import(
            dict(plans={"parent": plan.model_dump(mode="json")}, artifacts={})
        )

    old = decode(Path("examples/research_development.json").read_bytes())
    old["package_lineage"] = dict(
        base_package_ref="base",
        patch_ref="patch",
        proposer_ref="p",
        request_ref="request",
        evidence_ref="evidence",
        state_policy="reset",
    )
    evidence.require_legacy_import(dict(plans={"candidate": old}, artifacts={}))
    bad = copy.deepcopy(old)
    with pytest.raises(ValueError, match="patch/proposer"):
        evidence.require_legacy_import(
            dict(
                plans={"candidate": bad},
                artifacts={
                    "p": dict(kind="PolicyPatch", schema_version=2, path="policy.window_max")
                },
            )
        )


def test_registered_caps_global_bounds_and_capacity_are_distinct(tmp_path):
    plan = registered_plan()
    store = ArtifactStore(tmp_path / "artifacts")

    def authority(
        current, *, generation=0, stages=(), max_generations=2, max_proposal_stages=2, attempts=40
    ):
        receipt = store.put_json(
            dict(research_plan_ref=store.put_json(current.model_dump(mode="json")))
        )
        return SimpleNamespace(
            store=store,
            state=dict(
                head=dict(research_receipt_ref=receipt), generation=generation, stages=list(stages)
            ),
            program=dict(
                initial_costs=dict(worker_attempts=4, registered_worker_seconds=80),
                input=dict(
                    max_generations=max_generations,
                    max_proposal_stages=max_proposal_stages,
                    budget=dict(worker_attempts=attempts, registered_worker_seconds=800),
                ),
            ),
        )

    assert next_stop(authority(plan)) is None
    assert next_stop(authority(plan, attempts=19)) == "capacity_exhausted"
    assert next_stop(authority(plan, generation=1, max_generations=1)) == "generation_limit"
    stage = dict(capacity=dict(worker_attempts=16, registered_worker_seconds=320))
    assert next_stop(authority(plan, stages=[stage], attempts=35)) == "capacity_exhausted"
    assert next_stop(authority(plan, stages=[stage], max_proposal_stages=1)) == (
        "proposal_stage_limit"
    )
    # Pure limit checks: build typed child lineages separately; never call these real receipts.
    lineage = dict(
        base_package_ref="a",
        patch_ref="b",
        proposer_ref="c",
        request_ref="d",
        evidence_ref="e",
        state_policy="reset",
    )
    value = plan.model_dump(mode="json")
    value["policy"]["window_max"] = 8
    value["package_lineage"] = lineage
    capped = ResearchPlan.model_validate_json(canonical(value))
    assert next_stop(authority(capped, generation=2, stages=[stage, stage])) == (
        "registered_policy_cap_exhausted"
    )
    value["neighbor_domain"]["window_max"] = 8
    bounded = ResearchPlan.model_validate_json(canonical(value))
    assert next_stop(authority(bounded, generation=2, stages=[stage, stage])) == (
        "global_window_bound_reached"
    )


def test_root_registration_recovers_same_identity_and_handoff_blocks_dispatch(
    tmp_path, monkeypatch
):
    from quant_rsi import research
    from quant_rsi.artifacts import atomic_write

    value = registered_plan().model_dump(mode="json")
    value["trajectory_registration"].update(
        workspace=str(tmp_path), source_directory=str(tmp_path / "source")
    )
    plan = ResearchPlan.model_validate_json(canonical(value))
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(tmp_path))
    with pytest.raises(RuntimeError, match="before root creation"):
        with root_execution(plan, tmp_path / "source", None, None, create=True) as (identity, fd):
            with root_execution(plan, tmp_path / "source", identity, fd, create=False) as nested:
                assert nested == (identity, fd)
            raise RuntimeError("unit interruption before root creation")
    registered = decode(registration_path().read_bytes())
    assert not (tmp_path / "source").exists()
    with root_execution(plan, tmp_path / "source", None, None, create=True) as (reused, _):
        assert reused == registered["identity"]["campaign_id"]
    # Initializing the C2 journal performs no worker training.
    assert research.create_research(plan, tmp_path / "source", initialize_only=True) == {
        "initialized": True
    }
    manifest = decode((tmp_path / "source/manifest.json").read_bytes())
    assert manifest["campaign_id"] == reused
    with pytest.raises(ValueError, match="another source/workspace"):
        research.create_research(plan, tmp_path / "alternate", initialize_only=True)
    assert not (tmp_path / "alternate").exists()
    # This hand-built state is a guard unit negative, not an actual handoff demonstration.
    atomic_write(registration_path(), canonical(registered | dict(state="HANDOFF", handoff={})))
    before = {str(p): digest(p.read_bytes()) for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="handed off"):
        research.resume_research(tmp_path / "source", steps=1)
    with pytest.raises(ValueError, match="handed off"):
        research.create_research(plan, tmp_path / "source", initialize_only=True)
    after = {str(p): digest(p.read_bytes()) for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
    other = tmp_path / "other-workspace"
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(other))
    with pytest.raises(ValueError, match="another source/workspace"):
        research.create_research(plan, tmp_path / "source", initialize_only=True)
    assert not other.exists()


@pytest.mark.parametrize(
    "relative", [".quant_rsi/trajectory", ".quant_rsi/trajectory/source", ".quant_rsi", "."]
)
def test_registration_rejects_overlapping_source_before_creation(tmp_path, relative):
    value = registered_plan().model_dump(mode="json")
    value["trajectory_registration"].update(
        workspace=str(tmp_path), source_directory=str((tmp_path / relative).resolve())
    )
    with pytest.raises(ValueError, match="disjoint"):
        ResearchPlan.model_validate_json(canonical(value))
    assert list(tmp_path.iterdir()) == []


def test_invalid_initialization_path_does_not_irreversibly_handoff(tmp_path, monkeypatch):
    from quant_rsi.trajectory import init_trajectory

    value = registered_plan().model_dump(mode="json")
    value["trajectory_registration"].update(
        workspace=str(tmp_path), source_directory=str(tmp_path / "source")
    )
    plan = ResearchPlan.model_validate_json(canonical(value))
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(tmp_path))
    with root_execution(plan, tmp_path / "source", None, None, create=True):
        pass
    before = registration_path().read_bytes()
    authority = tmp_path / ".quant_rsi/trajectory"
    authority.mkdir()
    (authority / "foreign.txt").write_text("unrelated material")
    inputs = TrajectoryPlan.model_validate_json(
        canonical(
            dict(
                schema_version=2,
                name="invalid-path-unit",
                source_directory=str(tmp_path / "source"),
                source_checkpoint_ref="0" * 64,
                state_policy="reset",
                max_proposal_stages=2,
                max_generations=2,
                budget=dict(worker_attempts=40, registered_worker_seconds=800),
            )
        )
    )
    with pytest.raises(ValueError, match="without its initialization authority"):
        init_trajectory(inputs)
    assert registration_path().read_bytes() == before
    assert not (tmp_path / ".quant_rsi/trajectory_initialization.json").exists()
    with root_execution(plan, tmp_path / "source", None, None, create=True):
        pass
