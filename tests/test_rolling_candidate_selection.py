"""Select unchanged pool members without creating false self-parent revisions."""

import json
from copy import deepcopy

import numpy as np
import pytest
import test_monthly_rolling as monthly
import test_multi_factor_rolling as multi
import test_rolling_news_integration as news_fixtures

from quant_rsi.core.artifacts import canonical
from quant_rsi.research import rolling
from quant_rsi.research.discovery_factors import DiscoveryFactor
from quant_rsi.research.rolling_research import admit_decision
from quant_rsi.research.rolling_windows import monthly_plan

retained = monthly.retained


def recipe_ref(candidate):
    return DiscoveryFactor.model_validate_json(canonical(candidate)).recipe_ref


def selection(candidates):
    return {
        "action": "select",
        "basis": "news",
        "reason": "Keep an unchanged subset after checking training coverage and redundancy",
        "candidates": [],
        "news_inputs": [],
        "selected_recipe_refs": [recipe_ref(candidate) for candidate in candidates],
    }


def sixteen_candidate_pool():
    history, proposal = multi.pool()
    for index in (6, 7):
        for suffix, when in (
            ("", f"2026-03-{2 + index * 3:02d}"),
            ("_second", f"2026-04-{2 + index * 2:02d}"),
        ):
            article = deepcopy(history["items"][0])
            article.update(
                id=f"topic{index}{suffix}",
                title=f"Semiconductor topic{index} announcement",
                published_at=when + "T12:00:00+00:00",
            )
            history["items"].append(article)
        recipe = deepcopy(proposal["news_inputs"][0])
        recipe.update(
            input_id=f"news.topic{index}", terms=[f"topic{index}"], evidence_ids=[f"topic{index}"]
        )
        proposal["news_inputs"].append(recipe)
        for lag, original in zip((0, 3), proposal["candidates"][:2], strict=True):
            candidate = deepcopy(original)
            candidate.update(
                name=f"topic{index}_lag{lag}",
                evidence_ids=[f"topic{index}"],
                inputs=[{"name": "topic", "factor_id": f"news.topic{index}"}],
            )
            proposal["candidates"].append(candidate)
    assert len(proposal["candidates"]) == 16
    return history, proposal


def execute(retained_data, root, history, decisions, *, revision=False, last="2026-07", limit=16):
    _, data, info, calendar = retained_data
    root.mkdir()
    cache, llm = monthly.MemoryCache(), monthly.DecisionLLM(decisions)
    folds = rolling.run_months(
        data,
        info,
        {},
        monthly_plan(data, calendar, "2026-07", last),
        cache,
        root,
        model="ridge",
        llm=llm,
        news_history=history,
        news_required=True,
        max_candidates=limit,
        max_news_inputs=16,
        candidate_revision=revision,
    )
    return folds, cache, llm


def test_same_cutoff_selects_twelve_of_sixteen_without_rewriting_or_extra_fits(retained, tmp_path):
    history, proposal = sixteen_candidate_pool()
    selected = selection(list(reversed(proposal["candidates"][:12])))
    folds, cache, llm = execute(
        retained, tmp_path / "select", history, [proposal, selected], revision=True
    )
    fold = folds[0]
    assert len(llm.contexts) == 2
    preview = llm.contexts[1]["incumbent"]
    assert len(preview) == 16
    assert fold["candidate_status"] == "fitted"
    assert fold["decision"]["action"] == "select"
    assert fold["candidate_recipes"] == preview[:12]
    assert [recipe_ref(c) for c in fold["candidate_recipes"]] == list(
        reversed(selected["selected_recipe_refs"])
    )
    assert all(candidate["parent_recipe_refs"] == [] for candidate in fold["candidate_recipes"])
    assert fold["candidate_news_inputs"] == llm.contexts[1]["incumbent_news_inputs"][:6]
    assert len(fold["candidate_news_inputs"]) == 6
    assert [a["candidate_pool"]["accepted"] for a in fold["candidate_attempts"]] == [16, 12]
    assert fold["candidate_rejections"] == []
    assert [name for name in cache.calls if name.endswith("_fit")] == [
        "2026-07_fixed_fit",
        "2026-07_adaptive_fit",
    ]
    fitted = cache.calls["2026-07_adaptive_fit"]
    assert fitted["factor_ids"][2:] == ["discovery." + c["name"] for c in preview[:12]]
    assert np.asarray(fitted["x"]).shape[1] == 14
    assert np.isfinite(fitted["x"]).all()
    assert set(fold["news_training"]) == {f"news.topic{i}" for i in range(6)}


def test_monthly_select_keeps_validated_child_lineage_and_only_needed_inputs(retained, tmp_path):
    history, pool = multi.pool()
    for article in history["items"]:
        if article["id"].endswith("_second"):
            index = int(article["id"][5:].split("_", 1)[0])
            article["published_at"] = f"2026-05-{15 + index * 2:02d}T12:00:00+00:00"
    initial = deepcopy(pool)
    initial["candidates"] = [deepcopy(pool["candidates"][i]) for i in (0, 2, 4)]
    initial["news_inputs"] = deepcopy(pool["news_inputs"][:3])
    revised = deepcopy(initial)
    child = deepcopy(pool["candidates"][1])
    child["parent_recipe_refs"] = [recipe_ref(initial["candidates"][0])]
    revised["candidates"][0] = child
    selected = selection([revised["candidates"][2], revised["candidates"][0]])
    folds, cache, llm = execute(
        retained,
        tmp_path / "monthly",
        history,
        [initial, revised, selected],
        last="2026-09",
        limit=3,
    )
    assert len(llm.contexts) == 3
    assert [f["candidate_status"] for f in folds] == ["fitted"] * 3
    assert folds[2]["candidate_recipes"] == [
        folds[1]["candidate_recipes"][0],
        folds[1]["candidate_recipes"][2],
    ]
    assert folds[2]["candidate_recipes"][0]["parent_recipe_refs"] == child["parent_recipe_refs"]
    assert [r["input_id"] for r in folds[2]["candidate_news_inputs"]] == [
        "news.topic0",
        "news.topic2",
    ]
    assert cache.calls["2026-09_adaptive_fit"]["factor_ids"][2:] == [
        "discovery.topic0_lag3",
        "discovery.topic2_lag0",
    ]
    assert all("candidate_attempts" not in f for f in folds)


@pytest.mark.parametrize("screen_individually", [False, True])
@pytest.mark.parametrize(
    "problem",
    ["unknown", "duplicate", "empty", "no_incumbent", "over_budget", "candidates", "news_inputs"],
)
def test_invalid_selection_is_rejected(problem, screen_individually):
    history, proposal = multi.pool()
    incumbent = proposal["candidates"][:3]
    value = selection(incumbent[:1])
    limit = 3
    if problem == "unknown":
        value["selected_recipe_refs"] = ["f" * 64]
    elif problem == "duplicate":
        value["selected_recipe_refs"] *= 2
    elif problem == "empty":
        value["selected_recipe_refs"] = []
    elif problem == "no_incumbent":
        incumbent = []
    elif problem == "over_budget":
        value = selection(incumbent)
        limit = 2
    elif problem == "candidates":
        value["candidates"] = proposal["candidates"][:1]
    else:
        value["news_inputs"] = proposal["news_inputs"][:1]
    with pytest.raises(ValueError):
        admit_decision(
            {"text": json.dumps(value)},
            incumbent,
            history,
            limit,
            screen_individually=screen_individually,
        )


@pytest.mark.parametrize("action", ["retain", "withdraw", "replace"])
def test_selection_refs_are_forbidden_on_other_actions(action):
    history, proposal = multi.pool()
    value = deepcopy(proposal) if action == "replace" else news_fixtures.retain_news()
    value["action"] = action
    value["selected_recipe_refs"] = [recipe_ref(proposal["candidates"][0])]
    with pytest.raises(ValueError):
        admit_decision(
            {"text": json.dumps(value)}, proposal["candidates"], history, max_candidates=16
        )


def test_self_parent_replacement_remains_rejected_not_sanitized(retained, tmp_path):
    proposal = news_fixtures.proposal()
    bad = deepcopy(proposal)
    bad["reason"] = "This old replacement payload is still an invalid self-parent"
    bad["candidates"][0]["parent_recipe_refs"] = [recipe_ref(proposal["candidates"][0])]
    folds, cache, _ = execute(
        retained, tmp_path / "invalid", news_fixtures.history(), [proposal, bad], revision=True
    )
    fold = folds[0]
    assert fold["candidate_status"] == "rejected"
    assert "own parent" in fold["candidate_error"]
    assert "2026-07_adaptive_fit" not in cache.calls
    assert fold["forecasts"]["adaptive"] == fold["forecasts"]["fixed"]
    assert fold["candidate_attempts"][1]["candidate_rejections"]


def test_old_decision_payloads_still_work_without_selection_field():
    history = news_fixtures.history()
    original = news_fixtures.proposal()
    decision, incumbent = admit_decision({"text": json.dumps(original)}, [], history)
    assert decision["action"] == "replace"
    assert incumbent[0]["name"] == "export_pressure"
    assert decision["selected_recipe_refs"] == []
    kept, same = admit_decision(
        {"text": json.dumps(news_fixtures.retain_news())}, incumbent, history
    )
    assert kept["action"] == "retain" and same == incumbent
    withdrawn = news_fixtures.retain_news() | {"action": "withdraw"}
    _, empty = admit_decision({"text": json.dumps(withdrawn)}, incumbent, history)
    assert empty == []
