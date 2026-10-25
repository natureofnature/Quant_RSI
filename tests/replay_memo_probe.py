"""Clean-interpreter no-fit integration probe, using synthetic metadata, not worker evidence.

The fixed three-feature model comes from the original engineering seed17 baseline;
its SHA is b905718901e0dcc86b57bd8177e8ce153752938a6f0010bb1b83c3682166fc42.
It is only an inference fixture here and is never represented as newly fitted.
"""

import copy
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
from market_helpers import account_rules, contract_and_events, write_events
from replay_memo_helpers import unit_ledger

from quant_rsi import controller, market_data, replay_memo, trajectory_delegation
from quant_rsi import local_research as local
from quant_rsi.artifacts import ArtifactStore, canonical, digest
from quant_rsi.market_contracts import timestamp, utc_text
from quant_rsi.schemas import DevelopmentPlan, FactorSpec
from quant_rsi.trajectory_state import TrajectoryLedger, lock_path, locked_trajectory


def warm_authority_negatives(workspace, scope):
    """Actual FD/current pending guards on a synthetic zero-capacity chain, no mocks."""
    root, anchor = unit_ledger(workspace)
    child = root / "children" / "unit"
    context = dict(
        kind="TrajectoryC4Context",
        authority_root=str(root),
        authority_id=anchor["authority_id"],
        authorization_ref="a" * 64,
    )
    rejected = []

    def reject(label, fd, *, current=context, steps=1, message="actual owner lock"):
        try:
            trajectory_delegation.authorize_package(child, fd, context=current, steps=steps)
        except ValueError as error:
            assert message in str(error), (label, error)
            rejected.append(label)
        else:
            raise AssertionError(f"fresh authority guard skipped: {label}")

    before_stats = scope.report()
    assert all(pool.values and pool.stats["hits"] for pool in scope.pools.values())
    with locked_trajectory() as (_, fd), TrajectoryLedger(root, owner_fd=fd) as ledger:
        reject("missing owner FD", None)
        closed_fd = os.dup(fd)
        os.close(closed_fd)
        reject("closed owner FD", closed_fd)
        with lock_path(root).open("r+") as independent:
            reject("independent same-inode non-owner FD", independent.fileno())
        reject("no current pending", fd, message="active parent operation")

        def reserve(state):
            state["stages"] = [
                dict(
                    ordinal=0,
                    authorization_ref="b" * 64,
                    capacity=dict(worker_attempts=0, registered_worker_seconds=0),
                    consumed=dict(worker_attempts=0, registered_worker_seconds=0),
                )
            ]
            state["pending"] = dict(
                kind="package_round",
                authorization_ref="b" * 64,
                steps=1,
                stage=0,
                child_campaign_id="unit-only",
            )

        ledger.update("unit_zero_capacity_pending", reserve)
        before = (root / "control.sqlite").read_bytes()
        reject("old authorization", fd, message="active parent operation")
        reject(
            "wrong pending step count",
            fd,
            current=context | {"authorization_ref": "b" * 64},
            steps=2,
            message="active parent operation",
        )
        assert (root / "control.sqlite").read_bytes() == before
    assert scope.report() == before_stats
    return rejected


def frozen(value):
    if isinstance(value, np.ndarray):
        return dict(shape=value.shape, dtype=str(value.dtype), bytes=value.tobytes().hex())
    if isinstance(value, tuple):
        return dict(tuple_items=[frozen(item) for item in value])
    if isinstance(value, list):
        return [frozen(item) for item in value]
    if isinstance(value, dict):
        return {key: frozen(item) for key, item in value.items()}
    return value


def main(workspace):
    os.environ["QUANT_RSI_WORKSPACE"] = str(workspace)
    contract, events, evidence = contract_and_events(count=24, instruments=("A", "B"))
    evidence_path = workspace / "availability.txt"
    evidence_path.write_bytes(evidence)
    report = market_data.ingest(
        contract,
        write_events(workspace / "small.csv", contract, events),
        request_id="no-fit-cache-fixture",
        evidence_file=evidence_path,
    )
    value = json.loads(
        (Path(__file__).parents[1] / "examples/research_development.json").read_bytes()
    )
    value = value["development"]
    value["data"] = dict(
        source="local_event_snapshot",
        sessions=24,
        instruments=2,
        train_cutoff=15,
        seed=7,
        snapshot_ref=report["snapshot_ref"],
        decision_delay_seconds=60,
    )
    value["portfolio"] = account_rules().model_dump(mode="json")
    value["min_development_sessions"] = 3
    value["proposals"][0]["factor"] = dict(op="momentum", window=2)
    plan = DevelopmentPlan.model_validate_json(canonical(value))
    data = local.LocalResearchData.from_json(local.export_local_data(plan.data))
    data_before = canonical(data.to_json())
    factor = FactorSpec(op="momentum", window=2)
    model = Path(__file__).with_name("fixtures").joinpath("replay_booster.txt").read_text()
    assert (
        digest(model.encode()) == "b905718901e0dcc86b57bd8177e8ce153752938a6f0010bb1b83c3682166fc42"
    )
    violations, allowed_children = [], []

    def forbidden(*args, **kwargs):
        violations.append("fit_or_business_dispatch")
        raise AssertionError("no-fit integration probe attempted fitting/business dispatch")

    lgb.train = forbidden
    lgb.LGBMModel.fit = forbidden
    child_code = (
        "import json; from quant_rsi import replay_memo; "
        "assert replay_memo._ACTIVE.get() is None; "
        "assert replay_memo.calculate('training_view', lambda: 17, ()) == 17; "
        "print(json.dumps({'active_scope':None,'uncached_value':17,'module':replay_memo.__file__}))"
    )
    child_command = [sys.executable, "-c", child_code]
    original_popen = subprocess.Popen

    def guarded_popen(command, *args, **kwargs):
        if command != child_command:
            return forbidden(command, *args, **kwargs)
        allowed_children.append("explicit no-fit cold-interpreter probe")
        return original_popen(command, *args, **kwargs)

    subprocess.Popen = guarded_popen
    assert replay_memo.warnings_allow_reuse(), "probe requires an actually standard clean receiver"
    baseline, snapshots = {}, []
    scope = replay_memo.CalculationScope(controller.runtime_identity())
    store = ArtifactStore(workspace / "synthetic_result_cas")
    with scope:
        for index in range(3):
            training = local.training_view(data, plan, factor)
            predictions = local.development_predictions(data, plan, factor, model)
            accounting = local.evaluate_local(data, plan, predictions)
            for name, result in [
                ("training_view", training),
                ("development_predictions", predictions),
                ("evaluate_local", accounting),
            ]:
                encoded = canonical(frozen(result))
                if index == 0:
                    baseline[name] = encoded
                else:
                    assert encoded == baseline[name]
            snapshots.append(scope.report())
            if index == 0:
                training[0][0, 0] = 987654
                training[2]["qualifications"].append("caller mutation")
                training[3]["rows"][0]["feature"]["source_refs"].append("caller mutation")
                predictions[0]["rows"][0]["score"] = 987654
                accounting["account"]["cash"] = "987654"
        assert canonical(data.to_json()) == data_before
        assert all(pool.stats["hits"] >= 1 for pool in scope.pools.values())
        assert all(
            pool.max_entries == 64 and pool.max_bytes == 32 * 1024 * 1024
            for pool in scope.pools.values()
        )

        # A synthetic component envelope tests current checks, not fitting provenance/authority.
        metadata = copy.deepcopy(training[2])
        started = timestamp(metadata["training_view_asof"]) + timedelta(days=1)
        completed = utc_text(started + timedelta(seconds=1))
        binding = dict(proposal=dict(factor=factor.model_dump(mode="json")))
        job = dict(binding=binding, job_digest=digest(canonical(binding)))
        model_ref = store.put(model.encode())
        metadata.update(
            job_digest=job["job_digest"],
            model_digest=model_ref,
            actual_fit_started_at=utc_text(started),
            actual_fit_completed_at=completed,
            model_effective_at=completed,
        )
        bundle = dict(
            job_ref=store.put_json(job),
            model_ref=model_ref,
            worker_metadata_ref=store.put_json(metadata),
            predictions_ref=store.put_json(predictions),
            report_ref=store.put_json(accounting),
            log_ref=store.put(b""),
        )
        assert controller.replay_result(store, bundle, job, plan, data) == accounting
        rejected = []
        for bad, expected in [
            (metadata | {"actual_fit_completed_at": "1990-01-01T00:00:00Z"}, "fit/effective time"),
            (metadata | {"training_rows": metadata["training_rows"] + 1}, "preprocessing metadata"),
        ]:
            changed = bundle | {"worker_metadata_ref": store.put_json(bad)}
            try:
                controller.replay_result(store, changed, job, plan, data)
            except ValueError as error:
                assert expected in str(error)
                rejected.append(expected)
            else:
                raise AssertionError("current worker metadata check was skipped")
        wrong_predictions = copy.deepcopy(predictions)
        wrong_predictions[0]["rows"][0]["score"] = 987654
        changed = bundle | {"predictions_ref": store.put_json(wrong_predictions)}
        try:
            controller.replay_result(store, changed, job, plan, data)
        except ValueError as error:
            assert "predictions differ" in str(error)
            rejected.append("predictions differ")
        else:
            raise AssertionError("current prediction comparison was skipped")
        model_path = workspace / "synthetic_result_cas" / model_ref
        for missing in (False, True):
            stats = scope.report()
            try:
                if missing:
                    model_path.unlink()
                else:
                    model_path.write_bytes(b"private unit corruption")
                try:
                    controller.replay_result(store, bundle, job, plan, data)
                except (ValueError, FileNotFoundError):
                    rejected.append("missing CAS" if missing else "corrupt CAS")
                else:
                    raise AssertionError("fresh CAS validation was skipped")
            finally:
                model_path.write_bytes(model.encode())
            assert scope.report() == stats
        before = scope.report()
        child = subprocess.run(child_command, capture_output=True, text=True, check=True)
        child_value = json.loads(child.stdout)
        assert child_value["active_scope"] is None and child_value["uncached_value"] == 17
        assert scope.report() == before
        authority_rejections = warm_authority_negatives(workspace, scope)
        final_stats = scope.report()
    assert replay_memo._ACTIVE.get() is None and scope.closed
    assert all(not pool.values and pool.bytes == 0 for pool in scope.pools.values())
    for name, pool in scope.pools.items():
        assert pool.report()["peak_entries"] == final_stats[name]["peak_entries"] > 0
        assert pool.report()["peak_payload_key_bytes"] > 0
    assert (
        canonical(data.to_json()) == data_before and not violations and len(allowed_children) == 1
    )
    result = dict(
        status="PASS_NO_FIT_SYNTHETIC_COMPONENT_AND_EXEC_BOUNDARY",
        model_fixture_ref=model_ref,
        math_output_refs={k: digest(v) for k, v in baseline.items()},
        passes=snapshots,
        final_stats=final_stats,
        current_rejections=rejected,
        actual_warm_authority_rejections=authority_rejections,
        allowed_no_fit_children=allowed_children,
        child=child_value,
        violations=violations,
        source_module=local.__file__,
        memo_module=replay_memo.__file__,
    )
    (workspace / "clean_probe_result.json").write_bytes(canonical(result))
    print(
        "PASS: exact outputs/isolation, fresh metadata/CAS, one cold no-fit exec; "
        "no fits/business dispatch"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
