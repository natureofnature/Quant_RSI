"""Sector map, sector-aware baseline, tier coverage and knowledge labels (#157, #159).

Synthetic prices only: invented peer prices are not market evidence.
"""

import json
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_study_discovery import arguments

from quant_rsi.cli import main
from quant_rsi.research import study_peers
from quant_rsi.research.discovery_report import _period_label
from quant_rsi.research.sector_map import (
    BASELINE_KIND,
    SectorMap,
    comparator_baseline_ids,
    decision_sector_context,
    load_sector_map,
    post_period,
    tier_coverage,
)
from quant_rsi.research.study_incremental import freeze_plan
from quant_rsi.research.study_knowledge import knowledge_references

INSTRUMENTS = Path(__file__).resolve().parents[1] / "instruments"
MAP = INSTRUMENTS / "shared" / "sectors" / "semiconductors.json"


def test_semiconductor_map_validates_and_is_selected_by_its_members():
    for symbol in ("TSM", "SOXX"):
        sector = load_sector_map(symbol, INSTRUMENTS)
        assert sector["sector_id"] == "semiconductors"
        assert sector["path"] == "shared/sectors/semiconductors.json"
    assert load_sector_map("KO", INSTRUMENTS) is None
    tsm = load_sector_map("TSM", INSTRUMENTS)
    assert comparator_baseline_ids(tsm, "TSM") == [
        "peer.SOXX.peer_return_10",
        "peer.SPY.peer_return_10",
    ]
    # A comparator is never the target itself.
    assert comparator_baseline_ids(tsm, "SOXX") == ["peer.SPY.peer_return_10"]


def test_tsm_relationship_profile_configures_both_comparators():
    profile = json.loads((INSTRUMENTS / "US" / "TSM" / "relationships.json").read_text())
    tsm = load_sector_map("TSM", INSTRUMENTS)
    configured = {item["symbol"] for item in profile["relationships"]}
    assert {tsm["sector_comparator"]["symbol"], tsm["market_comparator"]["symbol"]} <= configured


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m["members"].append(m["members"][0]),
        lambda m: m["market_comparator"].update(symbol=m["sector_comparator"]["symbol"]),
        lambda m: m["tiers"].append(deepcopy(m["tiers"][0])),
        lambda m: m["tiers"][0].update(tier="suppliers"),
    ],
)
def test_invalid_maps_are_refused(change):
    body = json.loads(MAP.read_text())
    change(body)
    with pytest.raises(ValueError):
        SectorMap.model_validate(body)


def test_a_symbol_in_two_maps_is_ambiguous(tmp_path):
    directory = tmp_path / "shared" / "sectors"
    directory.mkdir(parents=True)
    shutil.copy(MAP, directory / "a.json")
    body = json.loads(MAP.read_text()) | {"sector_id": "other"}
    (directory / "b.json").write_text(json.dumps(body))
    with pytest.raises(ValueError, match="more than one sector map"):
        load_sector_map("TSM", tmp_path)


def model(definitions, sector=None, symbol="TSM"):
    return SimpleNamespace(
        library={"definitions": deepcopy(definitions)},
        data=SimpleNamespace(instruments=(symbol,)),
        sector_map=sector,
    )


def peer(symbol, op="peer_return", runnable=True):
    return {
        "factor_id": f"peer.{symbol}.{op}_10",
        "runnable": runnable,
        "recipe": {"op": op, "window": 10, "symbol": symbol},
    }


def test_sector_aware_baseline_uses_both_comparators_and_records_its_kind():
    sector = load_sector_map("TSM", INSTRUMENTS)
    definitions = [peer("SOXX"), peer("SPY"), peer("SOXX", "relative_return")]
    plan = freeze_plan(model(definitions, sector))
    assert plan["baseline_ids"] == [
        "research.momentum_10",
        "research.log_return_volatility_10",
        "peer.SOXX.peer_return_10",
        "peer.SPY.peer_return_10",
    ]
    assert plan["baseline_kind"] == BASELINE_KIND and plan["sector_ref"] == sector["ref"]


def test_non_runnable_comparator_fails_instead_of_leaving_the_baseline():
    sector = load_sector_map("TSM", INSTRUMENTS)
    with pytest.raises(ValueError, match="peer.SPY.peer_return_10"):
        freeze_plan(model([peer("SOXX"), peer("SPY", runnable=False)], sector))
    with pytest.raises(ValueError, match="peer.SPY.peer_return_10"):
        freeze_plan(model([peer("SOXX")], sector))


def test_without_a_sector_map_the_baseline_is_unchanged():
    definitions = [peer("SPY", "relative_return"), peer("XSD", "relative_return")]
    plan = freeze_plan(model(definitions))
    assert plan["baseline_ids"][2:] == ["peer.SPY.relative_return_10"]
    assert "baseline_kind" not in plan and "sector_ref" not in plan


def test_comparator_return_never_reads_past_the_origin_close():
    target = np.linspace(10, 20, 40)
    comparator = np.linspace(50, 60, 40)
    origin = 25
    before = study_peers.peer_feature(target, comparator, "peer_return", 10)[origin]
    later = comparator.copy()
    later[origin + 1 :] *= 3  # the label window and beyond
    assert study_peers.peer_feature(target, later, "peer_return", 10)[origin] == before


def test_tier_coverage_distinguishes_measured_evidenced_and_gap():
    sector = decision_sector_context(load_sector_map("TSM", INSTRUMENTS), "TSM")
    bundles = [{"status": "available", "request": {"symbol": "TSM"}}]
    evidence = [{"title": "Microsoft raises capital expenditure", "text": ""}]
    memory = [
        {
            "kind": "acquire",
            "result": {"status": "available", "request": {"symbol": "NVDA"}},
        },
        {"kind": "survey", "result": {"findings": [{"mechanism": "OpenAI funding round"}]}},
    ]
    coverage = tier_coverage(sector, "TSM", memory, bundles, evidence)
    rows = {row["tier"]: row for row in coverage["tiers"]}
    assert rows["issuer"]["counts"] == {"measured": 1, "evidenced": 0, "gap": 0}
    statuses = {e["name"]: e["status"] for row in coverage["tiers"] for e in row["entities"]}
    assert statuses["NVIDIA"] == "measured"
    assert statuses["Microsoft"] == "evidenced"
    assert statuses["OpenAI"] == "evidenced"
    assert statuses["Alphabet"] == "gap"
    # Whole-word matching: "AMD" inside another word is not a mention.
    assert tier_coverage(sector, "TSM", [], [], [{"title": "GAMDA"}])["tiers"][1]["counts"][
        "gap"
    ] == len(rows["direct_customers"]["entities"])


def test_reference_reviewed_after_the_first_decision_is_post_period():
    sector = load_sector_map("TSM", INSTRUMENTS)
    assert post_period(sector["reviewed_on"], "2025-12-31T16:00:00-05:00")
    assert not post_period(sector["reviewed_on"], "2026-11-19T16:00:00-05:00")
    rows = knowledge_references(
        sector=sector,
        related={"reviewed_on": "2026-11-04", "stocks": ["SOXX"], "relationships": []},
        prior_memory=[{"kind": "experiment", "projection_of": "abc", "prior_study": "s1"}],
        first_decision_time="2025-12-31T16:00:00-05:00",
    )
    assert [(r["source"], r["post_period_knowledge"]) for r in rows] == [
        ("sector_map", True),
        ("relationship_profile", True),
        ("prior_study", None),  # undated: unknown, never in period
    ]


def test_prior_study_records_are_labelled_by_their_creation_time():
    first_decision = "2025-12-31T21:00:00+00:00"
    rows = knowledge_references(
        sector=None,
        related=None,
        prior_memory=[
            {"kind": "experiment", "ref": "late", "created_at": "2026-11-18T12:00:00+00:00"},
            {"kind": "interpretation", "ref": "early", "created_at": "2025-12-01T12:00:00+00:00"},
            {"kind": "experiment", "ref": "at-decision", "created_at": first_decision},
            {"kind": "experiment", "ref": "legacy"},
        ],
        first_decision_time=first_decision,
    )
    assert [(r["ref"], r["kind"], r["post_period_knowledge"]) for r in rows] == [
        ("late", "result", True),
        ("early", "interpretation", False),
        ("at-decision", "result", True),  # D3.1: usable only strictly before the instant
        ("legacy", "result", None),
    ]
    # The report renders the three states; undated is never shown as in period.
    assert [_period_label(r["post_period_knowledge"]) for r in rows] == [
        "post_period_knowledge",
        "in period",
        "post_period_knowledge",
        "unknown (undated)",
    ]


def test_a_date_only_review_is_usable_from_the_next_day():
    assert post_period("2025-12-31", "2025-12-31T21:00:00+00:00")
    assert not post_period("2025-12-30", "2025-12-31T21:00:00+00:00")
    assert post_period(None, "2025-12-31T21:00:00+00:00") is None
    assert post_period("not a date", "2025-12-31T21:00:00+00:00") is None


def test_tsm_study_compares_candidates_against_the_sector_baseline_on_one_sample(tmp_path):
    root = tmp_path / "study"
    args = arguments(root, "--related", "profile", "--factor-root", str(INSTRUMENTS))
    args[1] = "TSM"
    assert main(args) == 0
    result = json.loads((root / "result.json").read_text())
    discovery = result["discovery"]
    assert discovery["baseline"]["kind"] == BASELINE_KIND
    assert discovery["baseline"]["factor_ids"][2:] == [
        "peer.SOXX.peer_return_10",
        "peer.SPY.peer_return_10",
    ]
    attempts = result["iterations"]["attempts"]
    assert len(attempts) >= 2  # the baseline and at least one candidate were fitted
    windows = result["study"]["research_windows"]
    for attempt in attempts:
        prediction = attempt["prediction"]
        assert prediction["training_rows"] == len(windows["partitions"]["train"]["origin_sessions"])
        assert [r["origin_session"] for r in prediction["development"]["rows"]] == windows[
            "partitions"
        ]["development"]["origin_sessions"]
        assert set(discovery["baseline"]["factor_ids"]) <= set(attempt["factor_ids"])
    context = json.loads((root / "discovery_context.json").read_text())
    assert context["sector_map"]["comparators"] == ["SOXX", "SPY"]
    assert [r["source"] for r in context["knowledge_references"]][:2] == [
        "sector_map",
        "relationship_profile",
    ]
    assert all(r["post_period_knowledge"] for r in context["knowledge_references"][:2])
    assert discovery["evaluated_attempts"]["this_run_evaluated_attempts"] == len(attempts)
    assert discovery["sector_coverage"]["tiers"][0]["tier"] == "issuer"
    report = (root / "report.md").read_text()
    for heading in (
        "## Baseline (frozen before any candidate)",
        "## Sector map coverage by tier (sector_tier_coverage_v1)",
        "## Retrieved knowledge references",
        "## Evaluated attempts (cumulative, same target and horizon)",
    ):
        assert heading in report
    assert "post_period_knowledge" in report


def test_decision_prompt_puts_the_issuer_tier_before_other_tiers():
    from quant_rsi.research.discovery_sources import discovery_decision

    prompts = []

    class Transport:
        def structured(self, prompt, schema):
            prompts.append(prompt)
            return json.dumps(
                {"action": "stop", "reason": "Done", "candidates": [], "data_requests": []}
                | {"query": None}
            )

    discovery_decision(Transport(), {"symbol": "TSM"})
    issuer = prompts[0].index("measurable disclosures of the prediction target itself")
    assert issuer < prompts[0].index("then spend remaining allowances on uncovered tiers")
