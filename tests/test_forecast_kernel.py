import json
import math
from copy import deepcopy

import pytest
from market_helpers import contract_and_events, write_events

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.core.schemas import FactorSpec
from quant_rsi.forecast.forecast_contracts import ForecastObjective
from quant_rsi.forecast.forecast_kernel import (
    forecast_errors,
    forecast_feature_view,
    forecast_feature_views,
    mature_forecast_label,
    mature_forecast_labels,
    target_session,
)
from quant_rsi.market import market_contracts, market_data
from quant_rsi.market.asof import make_feature_view


def objective(notices=()):
    return ForecastObjective.model_validate_json(
        canonical(
            dict(
                schema_version=1,
                kind="five_session_raw_close",
                horizon_sessions=5,
                target="log_close_ratio",
                currency="TEST",
                price_unit="raw_currency_per_share",
                price_unit_claim="explicit_assumption",
                price_unit_evidence_ref=digest(b"oracle quote units"),
                calendar_evidence_ref=digest(b"oracle session calendar"),
                notices=list(notices),
                unknown_action_policy="affected_or_unknown_interval_abstain",
                prior_exposure="previously_exposed_historical_exploration",
            )
        )
    )


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: "2026-10-26T09:00:00+00:00")

    def prepare(*, notice=None, prices=None, instruments=("A",)):
        contract, events, evidence = contract_and_events(count=10, instruments=instruments)
        for row in events:
            if row["kind"] == "CLOSE_QUOTE" and row["session_id"] in (prices or {}):
                row["price"] = prices[row["session_id"]]
        if notice is not None:
            events.append(
                dict(
                    event_id="A.action",
                    revision=0,
                    kind="CORPORATE_ACTION_NOTICE",
                    instrument="A",
                    session_id=None,
                    event_at=contract.calendar.sessions[3].open_at.isoformat(),
                    source_available_at=contract.calendar.sessions[1].close_at.isoformat(),
                    price=None,
                    status=None,
                    action=json.dumps(notice),
                    quality="valid",
                )
            )
        proof = tmp_path / "availability.txt"
        proof.write_bytes(evidence)
        path = write_events(tmp_path / "quotes.csv", contract, events)
        report = market_data.ingest(contract, path, request_id="fixture", evidence_file=proof)
        snapshot, contract = market_data.load_snapshot(report["snapshot_ref"])
        return contract, snapshot

    return prepare


def cash_notice():
    return {
        "schema_version": 1,
        "action_kind": "cash_dividend",
        "amount": "1",
        "currency": "TEST",
        "effective_at": "2020-10-22T09:00:00+00:00",
    }


def interpreted(snapshot):
    row = next(r for r in snapshot["events"] if r["event"]["kind"] == "CORPORATE_ACTION_NOTICE")
    return dict(
        schema_version=1,
        notice_ref=row["payload_ref"],
        action_kind="cash_dividend",
        effective_at=cash_notice()["effective_at"],
        currency="TEST",
        amount="1",
        unit="currency_per_share",
        source_format="cash_dividend_v1",
        evidence_ref=digest(b"explicit artificial cash oracle"),
    )


def test_percentage_point_score_is_not_log_or_dollar_error():
    result = forecast_errors([math.log(1.05), math.log(0.95)], [math.log(1.1), math.log(0.9)])
    assert result["mae_pp"] == pytest.approx(5)
    assert result["rmse_pp"] == pytest.approx(5)
    assert result["negative_mae_pp"] == pytest.approx(-5)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 1000.0, -1000.0])
def test_physically_invalid_output_is_not_clipped(invalid):
    with pytest.raises(ValueError):
        forecast_errors([invalid], [0.0])


def test_target_uses_fifth_registered_session_and_exact_maturity(source):
    contract, snapshot = source()
    spec = objective()
    assert target_session(contract, "s2", spec).session_id == "s7"
    before = mature_forecast_label(snapshot, contract, "s2", "A", "2020-10-26T15:59:59Z", spec)
    at = mature_forecast_label(snapshot, contract, "s2", "A", "2020-10-26T16:00:00Z", spec)
    assert before["status"] == "immature"
    assert at["status"] == "mature"
    assert at["target_session"] == "s7"
    assert target_session(contract, "s5", spec) is None


@pytest.mark.parametrize("visibility", [{}, {"basis": "observed_receipts"}, {"watermark": 0}])
def test_forecast_batch_preserves_decision_visibility_and_currency_guard(source, visibility):
    contract, snapshot = source(notice=cash_notice())
    spec = objective([interpreted(snapshot)])
    factor = FactorSpec(op="momentum", window=2)
    decisions = [("s2", "2020-10-21T16:00:00Z"), ("s4", "2020-10-23T16:00:00Z")]
    expected = [
        forecast_feature_view(snapshot, contract, session, as_of, factor, spec, **visibility)
        for session, as_of in decisions
    ]
    batch = forecast_feature_views(snapshot, contract, decisions, factor, spec, **visibility)
    assert canonical(batch) == canonical(expected)
    with pytest.raises(ValueError, match="currency/basis"):
        forecast_feature_views(
            snapshot, contract.model_copy(update={"currency": "USD"}), decisions, factor, spec
        )


def test_missing_target_quote_does_not_roll_to_next_session(source):
    contract, snapshot = source()
    snapshot = deepcopy(snapshot)
    snapshot["events"] = [
        row
        for row in snapshot["events"]
        if not (row["event"]["kind"] == "CLOSE_QUOTE" and row["event"]["session_id"] == "s7")
    ]
    label = mature_forecast_label(
        snapshot, contract, "s2", "A", "2020-10-28T16:00:00Z", objective()
    )
    assert label["target_session"] == "s7"
    assert label["status"] == "immature"
    assert label["reason"] == "target_or_anchor_quote_not_known"


def test_cash_does_not_enter_raw_price_label_or_unblock_legacy_account_features(source):
    contract, snapshot = source(notice=cash_notice(), prices={"s2": "100", "s7": "99"})
    spec = objective([interpreted(snapshot)])
    label = mature_forecast_label(snapshot, contract, "s2", "A", "2020-10-26T16:00:00Z", spec)
    assert label["status"] == "mature"
    assert math.expm1(label["value"]) == pytest.approx(-0.01)
    factor = FactorSpec(op="momentum", window=2)
    args = (snapshot, contract, "s4", "2020-10-23T16:00:00Z", factor)
    assert make_feature_view(*args)["rows"][0]["status"] == "unresolved"
    assert forecast_feature_view(*args, spec)["rows"][0]["status"] == "eligible"


def test_known_future_notice_blocks_target_but_future_receipt_does_not_leak(source):
    contract, snapshot = source(notice={"action_kind": "split", "ratio": "2:1"})
    label = mature_forecast_label(
        snapshot, contract, "s2", "A", "2020-10-26T16:00:00Z", objective()
    )
    assert label["status"] == "ineligible"
    assert label["reason"] == "forecast_action_effect_or_type_unknown"
    late = deepcopy(snapshot)
    item = next(r for r in late["events"] if r["event"]["kind"] == "CORPORATE_ACTION_NOTICE")
    item["event"]["source_available_at"] = "2020-10-28T16:00:00+00:00"
    item["payload_ref"] = digest(canonical(item["event"]))
    label = mature_forecast_label(late, contract, "s2", "A", "2020-10-26T16:00:00Z", objective())
    assert label["status"] == "mature"


def test_old_watermark_and_observed_receipts_cannot_use_historical_download(source):
    contract, snapshot = source()
    at = "2020-10-26T16:00:00Z"
    assert (
        mature_forecast_label(snapshot, contract, "s2", "A", at, objective(), watermark=0)["status"]
        == "immature"
    )
    assert (
        mature_forecast_label(
            snapshot, contract, "s2", "A", at, objective(), basis="observed_receipts"
        )["status"]
        == "immature"
    )


@pytest.mark.parametrize("visibility", [{}, {"basis": "observed_receipts"}, {"watermark": 0}])
def test_label_batch_matches_individual_labels_and_projects_once(source, monkeypatch, visibility):
    from quant_rsi.forecast import forecast_kernel

    contract, snapshot = source(
        instruments=("A", "B"), notice={"action_kind": "split", "ratio": "2:1"}
    )
    requests = [
        (session, instrument) for session in ("s0", "s2", "s4", "s5") for instrument in ("A", "B")
    ]
    cutoff, spec = "2020-10-26T16:00:00Z", objective()
    expected = [
        mature_forecast_label(snapshot, contract, origin, instrument, cutoff, spec, **visibility)
        for origin, instrument in requests
    ]
    project = forecast_kernel.visible_revisions
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return project(*args, **kwargs)

    monkeypatch.setattr(forecast_kernel, "visible_revisions", counted)
    actual = mature_forecast_labels(snapshot, contract, requests, cutoff, spec, **visibility)
    assert canonical(actual) == canonical(expected)
    assert len(calls) == 1


def test_label_batch_preserves_lazy_visibility_and_instrument_rejection(source, monkeypatch):
    from quant_rsi.forecast import forecast_kernel

    contract, snapshot = source()
    cutoff, spec = "2020-10-26T15:59:59Z", objective()

    def forbidden(*args, **kwargs):
        raise AssertionError("an immature or invalid label should not project events")

    monkeypatch.setattr(forecast_kernel, "visible_revisions", forbidden)
    assert mature_forecast_labels(snapshot, contract, [], cutoff, spec) == []
    labels = mature_forecast_labels(snapshot, contract, [("s2", "A"), ("s5", "A")], cutoff, spec)
    assert [row["reason"] for row in labels] == [
        "target_close_not_elapsed",
        "future_label_calendar_absent",
    ]
    with pytest.raises(ValueError, match="outside the registered grid"):
        mature_forecast_labels(snapshot, contract, [("s2", "missing")], cutoff, spec)


def test_label_projection_is_not_reused_across_snapshot_or_visibility_calls(source):
    contract, snapshot = source()
    requests, cutoff, spec = [("s2", "A")], "2020-10-26T16:00:00Z", objective()
    original = mature_forecast_labels(snapshot, contract, requests, cutoff, spec)
    changed = deepcopy(snapshot)
    quote = next(
        row
        for row in changed["events"]
        if row["event"]["kind"] == "CLOSE_QUOTE" and row["event"]["session_id"] == "s7"
    )
    quote["event"]["price"] = "200"
    quote["payload_ref"] = digest(canonical(quote["event"]))
    revised = mature_forecast_labels(changed, contract, requests, cutoff, spec)
    assert revised[0]["value"] != original[0]["value"]
    assert mature_forecast_labels(snapshot, contract, requests, cutoff, spec) == original
    assert (
        mature_forecast_labels(snapshot, contract, requests, cutoff, spec, watermark=0)[0]["status"]
        == "immature"
    )


def test_cash_currency_and_source_payload_cannot_be_reinterpreted(source):
    contract, snapshot = source(notice=cash_notice())
    row = interpreted(snapshot)
    row["currency"] = "USD"
    with pytest.raises(ValueError):
        objective([row])
    row["currency"], row["amount"] = "TEST", "2"
    label = mature_forecast_label(
        snapshot, contract, "s2", "A", "2020-10-26T16:00:00Z", objective([row])
    )
    assert label["reason"] == "forecast_cash_interpretation_mismatch"


def test_non_cash_annotation_cannot_move_original_split_outside_target(source):
    contract, snapshot = source(notice={"action_kind": "split", "ratio": "2:1"})
    row = interpreted(snapshot)
    row.update(
        action_kind="split",
        source_format="unsupported",
        amount=None,
        currency=None,
        unit=None,
        effective_at="2019-01-01T00:00:00Z",
    )
    result = mature_forecast_label(
        snapshot, contract, "s2", "A", "2020-10-26T16:00:00Z", objective([row])
    )
    assert result["status"] == "ineligible"
    assert result["reason"] == "forecast_unsupported_action_effect_unverified"
