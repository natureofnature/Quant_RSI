"""Offline two-call candidate review; synthetic Ridge/news fixtures do not prove alpha."""

import argparse
import json
from copy import deepcopy

import numpy as np
import pytest
import test_monthly_rolling as monthly
import test_rolling_news_integration as news_fixtures

from quant_rsi.core.artifacts import canonical
from quant_rsi.core.data import MarketData
from quant_rsi.research import rolling
from quant_rsi.research.discovery_factors import DiscoveryFactor
from quant_rsi.research.rolling_windows import monthly_plan
from quant_rsi.research.study_resume import StudyCache

retained = monthly.retained


def revised_proposal(*, parent=False):
    value = news_fixtures.proposal()
    value["reason"] = "Shorten the evidence-linked counting window after structural review"
    value["news_inputs"][0]["window_days"] = 10
    value["candidates"][0]["name"] = "revised_export_pressure"
    if parent:
        value["candidates"][0]["expression"]["nodes"] = [
            {"name": "source", "op": "leaf", "leaf": "pressure"},
            {"name": "result", "op": "lag", "inputs": ["source"], "window": 3},
        ]
        value["candidates"][0]["parent_recipe_refs"] = [
            DiscoveryFactor.model_validate_json(
                canonical(news_fixtures.proposal()["candidates"][0])
            ).recipe_ref
        ]
    return value


def execute(
    retained_data,
    root,
    decisions,
    *,
    cache=None,
    llm=None,
    data=None,
    news=None,
    last="2026-07",
    candidate_revision=True,
):
    _, original, info, calendar = retained_data
    data = original if data is None else data
    root.mkdir(exist_ok=True)
    cache = monthly.MemoryCache() if cache is None else cache
    llm = monthly.DecisionLLM(decisions) if llm is None else llm
    folds = rolling.run_months(
        data,
        info,
        {},
        monthly_plan(data, calendar, "2026-07", last),
        cache,
        root,
        model="ridge",
        llm=llm,
        news_history=news_fixtures.history() if news is None else news,
        news_mode="reconstructed",
        news_required=True,
        candidate_revision=candidate_revision,
    )
    return folds, cache, llm


def test_revision_uses_training_diagnostics_then_fits_only_final_pool(
    retained, tmp_path, monkeypatch
):
    root = tmp_path / "study"
    cache = monthly.MemoryCache()
    first = news_fixtures.proposal()
    invalid = deepcopy(first["candidates"][0])
    invalid["name"] = "invalid_operator"
    invalid["expression"]["nodes"][0]["op"] = "unsupported"
    first["candidates"].append(invalid)
    fitted = []
    native_fit, native_settle = rolling.fit_forecast, rolling.settle

    def fit(*args, **kwargs):
        fitted.append(list(args[-1]))
        return native_fit(*args, **kwargs)

    class InspectLLM(monthly.DecisionLLM):
        def structured(self, prompt, schema):
            assert len(fitted) == 1  # Only the fixed control exists at either decision.
            assert "2026-07_adaptive_fit" not in cache.calls
            return super().structured(prompt, schema)

    def settle(fold, data):
        frozen = json.loads((root / "2026-07/prediction.json").read_bytes())
        assert len(frozen["candidate_attempts"]) == 2
        assert "outcome" not in frozen
        assert all("observed_price" not in value for value in frozen["forecasts"].values())
        assert (
            json.loads((root / "2026-07/candidate_attempts.json").read_bytes())
            == frozen["candidate_attempts"]
        )
        return native_settle(fold, data)

    monkeypatch.setattr(rolling, "fit_forecast", fit)
    monkeypatch.setattr(rolling, "settle", settle)
    llm = InspectLLM([first, revised_proposal()])
    folds, _, _ = execute(retained, root, [], cache=cache, llm=llm)
    assert len(llm.contexts) == 2
    assert "candidate_review" not in llm.contexts[0]
    review = llm.contexts[1]["candidate_review"]
    assert review["candidate_rejections"]
    diagnostics = review["diagnostics"]
    assert diagnostics["training_samples"] == len(folds[0]["training_origins"])
    assert diagnostics["candidate_columns"] == 1
    assert diagnostics["baseline_columns"] == 2
    assert {r["factor_id"] for r in diagnostics["columns"]} == set(fitted[0]) | {
        "discovery.export_pressure"
    }
    assert llm.contexts[1]["mature_feedback"] == []
    assert llm.contexts[1]["incumbent"][0]["name"] == "export_pressure"
    assert len(fitted) == 2
    assert fitted[1][-1] == "discovery.revised_export_pressure"
    assert "discovery.export_pressure" not in fitted[1]
    assert [key for key in cache.calls if key.endswith("_fit")] == [
        "2026-07_fixed_fit",
        "2026-07_adaptive_fit",
    ]
    assert [key for key in cache.calls if key.endswith("_decision")] == [
        "2026-07_decision",
        "2026-07_revision_decision",
    ]
    assert folds[0]["candidate_recipes"][0]["name"] == "revised_export_pressure"
    adaptive = np.asarray(cache.calls["2026-07_adaptive_fit"]["x"])
    assert adaptive.shape[1] == 3
    assert np.isfinite(adaptive).all()
    assert np.unique(adaptive[:, -1]).size > 1


def test_revision_retain_reuses_first_preview_pool(retained, tmp_path):
    folds, cache, llm = execute(
        retained, tmp_path / "retain", [news_fixtures.proposal(), news_fixtures.retain_news()]
    )
    assert len(llm.contexts) == 2
    assert llm.contexts[1]["incumbent"][0]["name"] == "export_pressure"
    assert folds[0]["candidate_status"] == "fitted"
    assert folds[0]["candidate_recipes"] == llm.contexts[1]["incumbent"]
    assert folds[0]["candidate_news_inputs"] == llm.contexts[1]["incumbent_news_inputs"]
    assert cache.calls["2026-07_adaptive_fit"]["factor_ids"][-1] == "discovery.export_pressure"


def test_replacement_can_reference_first_preview_recipe(retained, tmp_path):
    folds, _, llm = execute(
        retained, tmp_path / "lineage", [news_fixtures.proposal(), revised_proposal(parent=True)]
    )
    fold = folds[0]
    assert fold["candidate_status"] == "fitted"
    assert fold["candidate_recipes"][0]["parent_recipe_refs"] == [
        llm.contexts[1]["incumbent_recipe_refs"]["export_pressure"]
    ]


def test_malformed_first_decision_can_be_repaired_once(retained, tmp_path):
    folds, cache, llm = execute(
        retained, tmp_path / "repair", [{"broken": True}, revised_proposal()]
    )
    assert len(llm.contexts) == 2
    assert llm.contexts[1]["incumbent"] == []
    assert llm.contexts[1]["candidate_review"]["error"]
    assert folds[0]["candidate_status"] == "fitted"
    assert len(folds[0]["candidate_attempts"]) == 2
    assert len([key for key in cache.calls if key.endswith("_fit")]) == 2


def test_invalid_final_decision_falls_back_to_fixed_not_first_preview(retained, tmp_path):
    folds, cache, llm = execute(
        retained, tmp_path / "rejected", [news_fixtures.proposal(), {"broken": True}]
    )
    fold = folds[0]
    assert len(llm.contexts) == 2
    assert fold["candidate_status"] == "rejected"
    assert fold["forecasts"]["adaptive"] == fold["forecasts"]["fixed"]
    assert fold["fits"]["adaptive"]["reused_arm"] == "fixed"
    assert "2026-07_adaptive_fit" not in cache.calls
    assert len(fold["candidate_attempts"]) == 2
    assert "revision_decision" in fold["decision"]["response_checkpoint"]


def test_invalid_first_decision_preserves_prior_month_incumbent_for_repair(retained, tmp_path):
    folds, _, llm = execute(
        retained,
        tmp_path / "previous",
        [
            news_fixtures.proposal(),
            news_fixtures.retain_news(),
            {"broken": True},
            news_fixtures.retain_news(),
        ],
        last="2026-08",
    )
    assert len(llm.contexts) == 4
    assert llm.contexts[3]["incumbent"] == folds[0]["candidate_recipes"]
    assert folds[1]["candidate_recipes"] == folds[0]["candidate_recipes"]
    assert folds[1]["candidate_status"] == "fitted"


def test_future_prices_and_headlines_cannot_change_either_decision_or_frozen_prediction(
    retained, tmp_path
):
    decisions = [news_fixtures.proposal(), revised_proposal()]
    first, _, llm1 = execute(retained, tmp_path / "original", deepcopy(decisions))
    original = retained[1]
    future = np.array([day > "2026-06-30" for day in original.dates])
    opens, closes = original.opens.copy(), original.closes.copy()
    opens[future] *= 1.7
    closes[future] *= 1.7
    altered_data = MarketData(original.dates, original.instruments, opens, closes)
    news = news_fixtures.history()
    for item in news["items"]:
        if item["published_at"] > "2026-06-30T20:00:00+00:00":
            item["title"] = item["text"] = "Unrelated future report"
    second, _, llm2 = execute(
        retained, tmp_path / "changed", deepcopy(decisions), data=altered_data, news=news
    )
    assert llm1.contexts == llm2.contexts
    assert (tmp_path / "original/2026-07/prediction.json").read_bytes() == (
        tmp_path / "changed/2026-07/prediction.json"
    ).read_bytes()
    assert (
        first[0]["forecasts"]["fixed"]["observed_price"]
        != second[0]["forecasts"]["fixed"]["observed_price"]
    )


@pytest.mark.parametrize("provider_error", [RuntimeError, ValueError])
def test_revision_provider_failure_preserves_attempt_and_resume_reuses_first_call_and_fit(
    retained, tmp_path, monkeypatch, provider_error
):
    root = tmp_path / "resume"
    root.mkdir()
    config = {
        "test": "candidate_revision",
        "candidate_revision": True,
        "factor_root": str(retained[0]),
    }
    cache = StudyCache(root, config)
    native_fit, fitted = rolling.fit_forecast, []

    def fit(*args, **kwargs):
        fitted.append(list(args[-1]))
        return native_fit(*args, **kwargs)

    monkeypatch.setattr(rolling, "fit_forecast", fit)
    first_llm = monthly.DecisionLLM(
        [news_fixtures.proposal(), provider_error("revision provider unavailable")]
    )
    with pytest.raises(
        RuntimeError, match="revision provider unavailable|revision operation failed"
    ):
        execute(retained, root, [], cache=cache, llm=first_llm)
    assert len(first_llm.contexts) == 2 and len(fitted) == 1
    assert len(json.loads((root / "2026-07/candidate_attempts.json").read_bytes())) == 1
    events = [json.loads(line) for line in (root / "research_calls.jsonl").read_text().splitlines()]
    assert [row["status"] for row in events] == ["attempted", "completed", "attempted", "failed"]
    assert [row["operation"] for row in events if row["status"] == "attempted"] == [
        "2026-07_decision",
        "2026-07_revision_decision",
    ]
    first_checkpoint = (root / "checkpoints/2026-07_decision.json").read_bytes()
    fixed_checkpoint = (root / "checkpoints/2026-07_fixed_fit.json").read_bytes()
    assert not (root / "checkpoints/2026-07_revision_decision.json").exists()
    # StudyCache's constructor expects the runtime record normally written by the CLI.
    (root / "runtime.json").write_text(
        json.dumps(
            {"status": "failed", "counts": {}, "token_usage": {}, "total_elapsed_seconds": 0}
        )
    )
    resumed = StudyCache(root, config, resume=True)
    final_llm = monthly.DecisionLLM([revised_proposal()])
    folds, _, _ = execute(retained, root, [], cache=resumed, llm=final_llm)
    assert len(final_llm.contexts) == 1 and len(fitted) == 2
    assert folds[0]["candidate_status"] == "fitted"
    assert (root / "checkpoints/2026-07_decision.json").read_bytes() == first_checkpoint
    assert (root / "checkpoints/2026-07_fixed_fit.json").read_bytes() == fixed_checkpoint
    assert len(folds[0]["candidate_attempts"]) == 2
    events = [json.loads(line) for line in (root / "research_calls.jsonl").read_text().splitlines()]
    assert [row["status"] for row in events] == [
        "attempted",
        "completed",
        "attempted",
        "failed",
        "attempted",
        "completed",
    ]
    assert [row["operation"] for row in events if row["status"] == "attempted"] == [
        "2026-07_decision",
        "2026-07_revision_decision",
        "2026-07_revision_decision",
    ]


def test_default_mode_still_uses_one_decision_and_no_attempt_fields(retained, tmp_path):
    folds, _, llm = execute(
        retained, tmp_path / "default", [news_fixtures.proposal()], candidate_revision=False
    )
    assert len(llm.contexts) == 1
    assert "candidate_review" not in llm.contexts[0]
    assert "candidate_attempts" not in folds[0]
    assert not (tmp_path / "default/2026-07/candidate_attempts.json").exists()
    assert not (tmp_path / "default/research_calls.jsonl").exists()


def test_revision_requires_llm_before_any_fit(retained, tmp_path, monkeypatch):
    _, data, info, calendar = retained
    monkeypatch.setattr(
        rolling, "fit_forecast", lambda *a, **kw: pytest.fail("must validate before fitting")
    )
    with pytest.raises(ValueError, match="(?i)LLM"):
        rolling.run_months(
            data,
            info,
            {},
            monthly_plan(data, calendar, "2026-07", "2026-07"),
            monthly.MemoryCache(),
            tmp_path,
            model="ridge",
            candidate_revision=True,
        )
    parser = argparse.ArgumentParser()
    rolling.add_arguments(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        [
            "research-rolling",
            "SOXX",
            "--prices-dir",
            str(retained[0]),
            "--output",
            str(tmp_path / "cli"),
            "--candidate-revision",
            "--first-month",
            "2026-07",
            "--last-month",
            "2026-07",
        ]
    )
    assert args.candidate_revision is True


def test_report_keeps_both_review_stages_and_baseline_comparison(retained, tmp_path):
    from quant_rsi.research.rolling_report import render_rolling

    first = news_fixtures.proposal()
    rejected = deepcopy(first["candidates"][0])
    rejected["name"] = "<img src=x onerror=bad()>"
    first["candidates"].append(rejected)
    folds, _, _ = execute(retained, tmp_path / "report", [first, revised_proposal()])
    html = render_rolling({"symbol": "SOXX", "folds": folds, "status": "completed"})
    assert "Candidate proposal and one revision" in html
    assert "<td>proposal</td>" in html and "<td>revision</td>" in html
    diagnostics = html.split("Candidate training coverage and correlation", 1)[1].split(
        "Screening rejections before final training", 1
    )[0]
    assert "discovery.export_pressure" in diagnostics
    assert "discovery.revised_export_pressure" in diagnostics
    assert "Distinct values" in diagnostics and "Nonzero rows" in diagnostics
    assert "Variance" in diagnostics and "Max |r| with baseline" in diagnostics
    assert "The proposal preview is not trained" in html
    assert "Candidate pool screening" in html
    assert "Fixed baseline" in html and "Adaptive research" in html
    assert "Unchanged price" in html and "Training-label mean" in html
    assert "<svg" in html and "Monthly price return (%)" in html
    assert "&lt;img src=x onerror=bad()&gt;" in html
    assert "<img src=x onerror=bad()>" not in html
