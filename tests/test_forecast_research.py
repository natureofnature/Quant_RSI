from pathlib import Path

import pytest
from market_helpers import contract_and_events, write_events

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.execution import baseline
from quant_rsi.forecast import forecast_requests
from quant_rsi.market import market_data
from quant_rsi.research import package_experiment, research
from quant_rsi.research.researcher import ResearchPlan


@pytest.fixture
def forecast_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(tmp_path))
    contract, events, evidence = contract_and_events(count=70)
    evidence_path = tmp_path / "availability.txt"
    evidence_path.write_bytes(evidence)
    imported = market_data.ingest(
        contract,
        write_events(tmp_path / "market.csv", contract, events),
        request_id="forecast-fixture",
        evidence_file=evidence_path,
    )
    with market_data.market_context() as (registry, _):
        units = registry.store.put(b"Engineering quote units")
        calendar = registry.store.put(b"Engineering session calendar")
    value = ResearchPlan.model_validate_json(
        Path("examples/research_development.json").read_bytes()
    ).model_dump(mode="json")
    dev = value["development"]
    dev.pop("portfolio")
    dev.pop("annualization")
    dev.update(
        plan_type="forecast_development",
        primary_metric="negative_mae_pp",
        evaluation_stride=5,
        min_development_sessions=3,
        data=dict(
            source="local_event_snapshot",
            sessions=70,
            instruments=1,
            train_cutoff=40,
            seed=7,
            snapshot_ref=imported["snapshot_ref"],
            decision_delay_seconds=60,
        ),
        objective=dict(
            schema_version=1,
            kind="five_session_raw_close",
            horizon_sessions=5,
            target="log_close_ratio",
            currency="TEST",
            price_unit="raw_currency_per_share",
            price_unit_claim="explicit_assumption",
            price_unit_evidence_ref=units,
            calendar_evidence_ref=calendar,
            notices=[],
            unknown_action_policy="affected_or_unknown_interval_abstain",
            prior_exposure="previously_exposed_historical_exploration",
        ),
    )
    dev["budget"].update(max_attempts=2, max_worker_seconds=40)
    value["policy"].update(window_min=5, window_max=5)
    return ResearchPlan.model_validate_json(canonical(value))


def test_forecast_c2_seal_replay_and_legacy_boundary(forecast_plan, tmp_path, monkeypatch):
    root = tmp_path / "research"
    result = research.create_research(forecast_plan, root, steps=1)
    assert result["kernel_attempts"] == 1
    assert result["results"][0]["observation"]["status"] == "completed", result["results"]
    assert result["incumbent"] is not None
    assert research.inspect_research(root, replay=True)["verified_experiments"] == 1
    with pytest.raises(ValueError, match="forecast"):
        baseline.seal_baseline(root)
    with pytest.raises(ValueError, match="C4/C5"):
        package_experiment.evidence_capsule(root)
    sealed = baseline.seal_forecast(root)
    assert baseline.seal_forecast(root) == sealed
    with market_data.market_context() as (registry, _):
        bundle, _, _, _ = baseline.read_forecast_bundle(registry, sealed["bundle_ref"])
        assert bundle["objective_ref"] == digest(canonical(bundle["objective"]))
        with pytest.raises(ValueError, match="registration identity"):
            baseline.read_bundle(registry, sealed["bundle_ref"])

    first = forecast_requests.issue_forecast(
        sealed["bundle_ref"],
        forecast_plan.development.data.snapshot_ref,
        "s64",
        request_id="persistent-forecast",
    )
    assert first["status"] == "late"
    assert first["forecast"]["target_session"] == "s69"
    assert first["forecast"]["rows"][0]["forecast_price"] > 0
    assert (
        forecast_requests.issue_forecast(
            sealed["bundle_ref"],
            forecast_plan.development.data.snapshot_ref,
            "s64",
            request_id="persistent-forecast",
        )
        == first
    )
    with pytest.raises(ValueError, match="different inputs"):
        forecast_requests.issue_forecast(
            sealed["bundle_ref"],
            forecast_plan.development.data.snapshot_ref,
            "s63",
            request_id="persistent-forecast",
        )

    original_compute = forecast_requests.compute_forecast
    with monkeypatch.context() as failed:

        def interrupted(*args, **kwargs):
            raise RuntimeError("injected inference interruption")

        failed.setattr(forecast_requests, "compute_forecast", interrupted)
        with pytest.raises(RuntimeError, match="injected"):
            forecast_requests.issue_forecast(
                sealed["bundle_ref"],
                forecast_plan.development.data.snapshot_ref,
                "s64",
                request_id="recover-before-result",
            )
    pending = forecast_requests.inspect_forecast("recover-before-result")
    assert pending["status"] == "pending" and len(pending["failures"]) == 1
    recovered = forecast_requests.issue_forecast(
        sealed["bundle_ref"],
        forecast_plan.development.data.snapshot_ref,
        "s64",
        request_id="recover-before-result",
    )
    assert recovered["forecast"]["issued_at"] == pending["request"]["issued_at"]
    assert len(recovered["failures"]) == 1
    assert forecast_requests.compute_forecast is original_compute

    calls = 0
    real_clock = forecast_requests.market_contracts.trusted_now
    with monkeypatch.context() as failed:

        def interrupted_confirmation():
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("injected post-commit interruption")
            return real_clock()

        failed.setattr(forecast_requests.market_contracts, "trusted_now", interrupted_confirmation)
        with pytest.raises(RuntimeError, match="post-commit"):
            forecast_requests.issue_forecast(
                sealed["bundle_ref"],
                forecast_plan.development.data.snapshot_ref,
                "s64",
                request_id="recover-after-result",
            )
    unconfirmed = forecast_requests.inspect_forecast("recover-after-result", replay=True)
    assert unconfirmed["status"] == "unconfirmed"
    confirmed = forecast_requests.issue_forecast(
        sealed["bundle_ref"],
        forecast_plan.development.data.snapshot_ref,
        "s64",
        request_id="recover-after-result",
    )
    assert confirmed["result_ref"] == unconfirmed["result_ref"]
    assert confirmed["generated_at"] == unconfirmed["generated_at"]
    assert confirmed["status"] == "late"

    from quant_rsi.core.artifacts import ArtifactStore

    def forbidden(*args, **kwargs):
        pytest.fail("read-only forecast replay attempted artifact publication")

    monkeypatch.setattr(ArtifactStore, "put", forbidden)
    assert forecast_requests.inspect_forecast("persistent-forecast", replay=True) == first


def test_mean_benchmark_uses_the_model_training_mask(forecast_plan):
    import math

    from quant_rsi.core.schemas import FactorSpec
    from quant_rsi.evaluation.local_research import LocalResearchData, export_local_data
    from quant_rsi.forecast.forecast_research import training_mean_baseline, training_view

    data = LocalResearchData.from_json(export_local_data(forecast_plan.development.data))
    factor = FactorSpec(op="momentum", window=5)
    _, labels, metadata, view = training_view(data, forecast_plan.development, factor)
    mean, evidence = training_mean_baseline(data, forecast_plan.development, factor)
    assert evidence["label_count"] == len(labels) == 31
    assert evidence["training_view_ref"] == metadata["training_view_ref"]
    assert mean == math.fsum(labels) / len(labels)
    assert evidence["eligible_grid"] == [
        dict(session_id=row["session_id"], instrument=row["instrument"])
        for row in view["rows"]
        if row["training_eligible"]
    ]
    assert evidence["eligible_grid"][0]["session_id"] == "s5"
    assert evidence["eligible_grid"][-1]["session_id"] == "s35"


@pytest.mark.parametrize("op", ["momentum", "mean_return"])
def test_training_batch_matches_fresh_views_and_hashes_snapshot_once(
    forecast_plan, monkeypatch, op
):
    import numpy as np

    from quant_rsi.core.schemas import FactorSpec
    from quant_rsi.evaluation.local_research import LocalResearchData, export_local_data
    from quant_rsi.forecast import forecast_research
    from quant_rsi.market import asof

    data = LocalResearchData.from_json(export_local_data(forecast_plan.development.data))
    factor = FactorSpec(op=op, window=5)

    def individually(snapshot, contract, decisions, factor, objective):
        return [
            forecast_research.forecast_feature_view(
                snapshot, contract, session, as_of, factor, objective
            )
            for session, as_of in decisions
        ]

    with monkeypatch.context() as reference:
        reference.setattr(forecast_research, "forecast_feature_views", individually)
        reference.setattr(
            forecast_research,
            "mature_forecast_labels",
            lambda snapshot, contract, requests, cutoff, objective: [
                forecast_research.mature_forecast_label(
                    snapshot, contract, origin, instrument, cutoff, objective
                )
                for origin, instrument in requests
            ],
        )
        expected = forecast_research.training_view(data, forecast_plan.development, factor)

    snapshot_encodings = []

    def encode(value):
        if value is data.snapshot:
            snapshot_encodings.append(value["receipt_watermark"])
        return canonical(value)

    monkeypatch.setattr(asof, "canonical", encode)
    from quant_rsi.forecast import forecast_kernel

    project = forecast_kernel.visible_revisions
    label_projections = []

    def project_labels(*args, **kwargs):
        label_projections.append(args[2])
        return project(*args, **kwargs)

    monkeypatch.setattr(forecast_kernel, "visible_revisions", project_labels)
    actual = forecast_research.training_view(data, forecast_plan.development, factor)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert canonical(actual[2:]) == canonical(expected[2:])
    assert snapshot_encodings == [data.snapshot["receipt_watermark"]]
    assert label_projections == [data.decision_at(forecast_plan.development.data.train_cutoff)]
