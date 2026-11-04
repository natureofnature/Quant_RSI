"""Paired diagnostics keep pre-run plans, failures and uncertainty on disk."""

import json

import pytest

from examples.discovery_acceptance import assess_pairs, paired_contexts, run_pairs


def news_plan():
    return {
        "kind": "news_ablation",
        "max_pairs": 2,
        "criteria": "An evidence-linked acquisition request for the removed operating mechanism.",
        "provider": {"mode": "codex", "model": None, "timeout": 180},
        "catalog_provenance": "pre_evidence",
        "removed_evidence_ids": ["demand"],
        "context": {
            "symbol": "KO",
            "question": "What changed?",
            "policy": {"identity": "fixed"},
            "source_limits": {"searches": 2, "reads": 3, "extractions": 4},
            "requested_data_period": {"start": "2026-01-01", "end": "2026-09-01"},
            "iteration": 1,
            "catalog": [],
            "evidence": [
                {"id": "demand", "text": "Unit sales decreased."},
                {"id": "other", "text": "The company issued a statement."},
            ],
            "memory": [{"ref": "derived", "text": "Unit sales decreased."}],
            "summary": "Unit sales decreased.",
            "cache": {"story": "Unit sales decreased."},
        },
    }


def test_plan_and_cleaned_inputs_precede_any_call_and_all_pairs_retained(tmp_path):
    root = tmp_path / "paired"
    seen = []

    def decide(context):
        assert (root / "plan.json").exists()
        assert context["memory"] == []
        assert "summary" not in context and "cache" not in context
        for field in ("source_limits", "requested_data_period", "iteration"):
            assert context[field] == news_plan()["context"][field]
        seen.append(context)
        return {
            "action": "acquire"
            if any(e["id"] == "demand" for e in context["evidence"])
            else "stop",
            "reason": "Fixture decision; no empirical influence claim",
        }

    outcomes = run_pairs(news_plan(), root, decide)
    assert len(seen) == 4
    assert len(outcomes) == 2
    assert seen[0] == seen[3] and seen[1] == seen[2]
    report = json.loads((root / "results.json").read_text())
    assert report["status"] == "inconclusive"
    assert len(list(root.glob("pair-*-input.json"))) == 4
    with pytest.raises(FileExistsError):
        run_pairs(news_plan(), root, decide)


def test_failed_arms_are_preserved_without_extra_attempts(tmp_path):
    count = 0

    def decide(_):
        nonlocal count
        count += 1
        raise ValueError("provider unavailable")

    outcomes = run_pairs(news_plan(), tmp_path / "failed", decide)
    assert count == 4
    assert all(arm["status"] == "failed" for row in outcomes for arm in row["arms"].values())


def test_counterfactual_branches_are_separate_synthetic_history():
    original = {"memory": [{"ref": "h1", "candidate": "fixed"}], "evidence": [{"id": "news"}]}
    plan = {
        "kind": "counterfactual_feedback",
        "max_pairs": 1,
        "context": original,
        "provider": {"mode": "codex"},
        "criteria": "Follow-up action responds to feedback.",
        "feedback_consistency_note": "Two synthetic internally consistent metrics and predictions.",
        "feedback_branches": [{"synthetic": True, "metric": 1}, {"synthetic": True, "metric": 3}],
    }
    left, right, _ = paired_contexts(plan)
    assert original["memory"] == [{"ref": "h1", "candidate": "fixed"}]
    assert left["evidence"] == right["evidence"]
    assert left["memory"][-1]["synthetic"] and right["memory"][-1]["synthetic"]
    assert left["memory"][-1]["ref"] != right["memory"][-1]["ref"]
    assert "not a measured result" in left["diagnostic_notice"]


def test_assessment_needs_exact_plan_actions_and_evidence(tmp_path):
    root = tmp_path / "assessment"
    run_pairs(news_plan(), root, lambda _: {"action": "stop"})
    identity = json.loads((root / "plan.json").read_text())["identity"]
    record = {
        "plan_identity": identity,
        "reviewer": "fixture reviewer",
        "status": "pass",
        "rationale": "Different outputs alone are insufficient.",
        "artifact_refs": ["pair-0-original.json", "pair-0-intervention.json"],
    }
    with pytest.raises(ValueError, match="observed actions"):
        assess_pairs(root, record)
    result = assess_pairs(root, record | {"status": "inconclusive"})
    assert result["statistical_claim"] == "none"
    with pytest.raises(ValueError, match="exact frozen plan"):
        assess_pairs(root, record | {"plan_identity": "other"})
