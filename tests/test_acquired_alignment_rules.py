"""Acquired alignment rules are versioned: D3 for new alignments, recorded rules on replay."""

import json

import pytest
from test_discovery_sources import request
from test_forecast_acquisition import article, checked, extraction
from test_rolling_acquired import archive, save  # noqa: F401  (fixture)

from quant_rsi.core.artifacts import canonical
from quant_rsi.research.discovery_sources import (
    CURRENT_ALIGNMENT_RULES,
    DataRequest,
    align_input_bundle,
)
from quant_rsi.research.rolling_acquired import load_acquired_history


def observation(*, published, effective, value=3.0, period_end="2024-12-31"):
    return {
        "observation_id": f"obs-{published}-{effective}",
        "period_end": period_end,
        "value": value,
        "published_at": published,
        "effective_available_at": effective,
    }


def column(observations, times, **kwargs):
    bundle = align_input_bundle(request(), observations, times, **kwargs)
    return bundle["columns"]["acquired.volume_growth"]


def test_current_rules_are_recorded_and_identity_is_pinned():
    rows, reasons = checked(request(), article(), extraction())
    assert reasons == []
    times = ["2025-01-03T21:00:00Z", "2025-01-06T21:00:00Z"]
    current = align_input_bundle(DataRequest.model_validate_json(canonical(request())), rows, times)
    assert current["availability_rules"] == CURRENT_ALIGNMENT_RULES
    assert current["identity"] == "3549966c4cc7ba0843788e5f9be9add32928354b4ea5fdf05e33379648e9a4ce"


def test_value_exactly_at_the_decision_instant_is_not_yet_usable():
    row = observation(published="2025-01-04T12:00:00Z", effective="2025-01-04T12:00:00Z")
    times = ["2025-01-04T11:59:59Z", "2025-01-04T12:00:00Z", "2025-01-04T12:00:01Z"]
    assert column([row], times) == [None, None, 3.0]
    assert column([row], times, rules=None) == [None, 3.0, 3.0]  # legacy: inclusive


def test_max_age_is_an_exact_duration_from_publication():
    row = observation(published="2025-01-04T12:00:00Z", effective="2025-01-04T12:00:00Z")
    limit = "2025-04-14T12:00:00Z"  # exactly 100 days after publication
    times = [limit, "2025-04-14T12:00:01Z", "2025-04-15T11:59:59Z"]
    assert column([row], times) == [3.0, None, None]
    assert column([row], times, rules=None) == [3.0, 3.0, 3.0]  # legacy floored whole days


def test_refetching_an_old_release_does_not_renew_its_freshness():
    # Retained mode: published in January, first received in November.
    row = observation(published="2025-01-04T12:00:00Z", effective="2025-11-04T09:00:00Z")
    times = ["2025-11-05T21:00:00Z"]
    assert column([row], times) == [None]
    assert column([row], times, rules=None) == [3.0]  # legacy anchored on receipt


def test_legacy_and_current_archives_replay_and_unknown_rules_are_rejected(
    archive,  # noqa: F811
    tmp_path,
):
    assert load_acquired_history([archive])  # created now: current rules

    body = json.loads(archive.read_bytes())["output"]
    request_id = body["request"]["input_id"]
    times = [row["prediction_time"] for row in body["input_evidence"][request_id]]
    legacy = align_input_bundle(body["request"], body["observations"], times, rules=None)
    assert "availability_rules" not in legacy
    legacy_body = {k: v for k, v in body.items() if k != "availability_rules"} | legacy
    assert load_acquired_history([save(tmp_path / "legacy.json", legacy_body)])

    unknown = body | {"availability_rules": {"at_instant": "inclusive_v0"}}
    with pytest.raises(ValueError):
        load_acquired_history([save(tmp_path / "unknown.json", unknown)])


def test_unknown_rules_are_rejected_for_new_alignments():
    with pytest.raises(ValueError, match="Unknown acquired availability rules"):
        align_input_bundle(request(), [], [], rules={"at_instant": "inclusive_v0"})
