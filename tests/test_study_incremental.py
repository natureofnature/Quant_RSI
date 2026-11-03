"""Business-increment wiring and temporal boundaries; no market-quality claims."""

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from test_research_study import command

from quant_rsi.cli import main
from quant_rsi.research import study
from quant_rsi.research.demo import load_catalog
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.llm import LLMHypothesizer
from quant_rsi.research.study_incremental import simulated_increment
from quant_rsi.research.study_news import simulated_history
from quant_rsi.research.study_windows import configure_windows, retained_news


def run(tmp_path, monkeypatch, action=None, *extra):
    calls, matrices = [], []
    predict = LightGBMDemo.predict

    def capture(self, context, factors, hypothesis):
        result = predict(self, context, factors, hypothesis)
        matrices.append(self.comparison_inputs(len(self._fits)))
        return result

    def researcher(context, library, **kwargs):
        calls.append(deepcopy((context, library, kwargs)))
        result = simulated_increment(context, library, **kwargs)
        if action:
            result.update(action=action, factor_ids=[], proposed_business_factors=[])
            result["question_map"][0]["factor_ids"] = []
        return result

    monkeypatch.setattr(LightGBMDemo, "predict", capture)
    monkeypatch.setattr("quant_rsi.research.study_incremental.simulated_increment", researcher)
    root = tmp_path / "study"
    assert (
        main(
            command(
                root,
                "--research-mode",
                "incremental",
                "--financials",
                "--llm",
                "simulated",
                "--no-ui",
                *extra,
            )
        )
        == 0
    )
    return root, json.loads((root / "result.json").read_text()), calls, matrices


def test_baseline_and_business_revision_share_population_and_three_fit_budget(
    tmp_path, monkeypatch
):
    root, result, calls, matrices = run(tmp_path, monkeypatch)
    plan = result["study"]["incremental_plan"]
    attempts = result["iterations"]["attempts"]
    assert len(attempts) == len(matrices) == 3
    assert len(calls) == 2
    assert result["configuration"]["comparison_fit_budget"] == 0
    assert result["configuration"]["publication_fit_budget"] == 0
    assert attempts[0]["factor_ids"] == plan["baseline_ids"]
    assert "business.inventory_relative_growth" in attempts[1]["factor_ids"]
    assert "financial.operating_cashflow_yoy" in attempts[2]["factor_ids"]
    assert all(
        a["factor_ids"][: len(plan["baseline_ids"])] == plan["baseline_ids"] for a in attempts
    )
    assert len({a["prediction"]["training_rows"] for a in attempts}) == 1
    assert (
        len(
            {
                tuple(r["origin_session"] for r in a["prediction"]["development"]["rows"])
                for a in attempts
            }
        )
        == 1
    )
    assert all("terminal_evaluation" not in json.dumps(c) for c in calls)
    report = (root / "report.html").read_text()
    assert "Delta vs B" in report and "Frozen time and information boundaries" in report
    before = (root / "incremental_plan.json").read_bytes()
    monkeypatch.setattr(
        LightGBMDemo, "predict", lambda *a: pytest.fail("completed resume refitted")
    )
    assert main(["research-resume", str(root)]) == 1
    assert (root / "incremental_plan.json").read_bytes() == before


@pytest.mark.parametrize("action", ["retain", "withdraw", "stop"])
def test_no_fit_actions_finish_report_from_baseline(tmp_path, monkeypatch, action):
    root, result, calls, matrices = run(tmp_path, monkeypatch, action)
    assert len(matrices) == len(calls) == result["iterations"]["completed_fits"] == 1
    assert result["iterations"]["selected_round"] == 1
    assert result["iterations"]["stop_reason"] == "increment_" + action
    assert (root / "report.html").exists()
    assert result["terminal_evaluation"]["selected_round"] == 1
    from quant_rsi.research.study_ui import read_state

    assert (
        read_state(root)["iterations"]["completed_fits"] == result["iterations"]["completed_fits"]
    )


def test_no_business_data_reports_gap_and_stops_without_fake_revision(tmp_path):
    root = tmp_path / "missing"
    assert (
        main(command(root, "--research-mode", "incremental", "--llm", "simulated", "--no-ui")) == 0
    )
    result = json.loads((root / "result.json").read_text())
    assert result["iterations"]["completed_fits"] == 1
    assert not result["study"]["incremental_plan"]["candidate_input_ids"]
    assert "missing/disabled/unusable" in (root / "report.html").read_text()


def test_retained_news_checks_publication_and_original_receipt():
    model = LightGBMDemo(load_catalog(), evaluation_sessions=20, terminal_sessions=10)
    snapshot = simulated_history(model.data, "KO")[60]
    cutoff = model.data.timestamp(60, "close")
    late = deepcopy(snapshot)
    late["received_at"] = model.data.timestamp(61, "close")
    assert retained_news([late], cutoff, "KO", 10)["items"] == []
    for item in late["items"]:
        item["received_at"] = late["received_at"]
    assert retained_news([late], cutoff, "KO", 10)["items"] == []
    future = deepcopy(snapshot)
    for item in future["items"]:
        item["published_at"] = model.data.timestamp(61, "close")
    assert retained_news([future], cutoff, "KO", 10)["items"] == []
    assert retained_news([snapshot], cutoff, "KO", 10)["items"]


def test_explicit_windows_purge_immature_boundary_labels():
    model = LightGBMDemo(load_catalog(), evaluation_sessions=20, terminal_sessions=10)
    dates = model.data.dates
    args = SimpleNamespace(
        train_dates=[dates[30], dates[65]],
        development_dates=[dates[70], dates[95]],
        terminal_dates=[dates[100], dates[115]],
        terminal_sessions=10,
        information_mode="retained",
        research_cutoff=None,
    )
    plan = configure_windows(model, args)
    assert np.max(model.train + model.horizon) < np.min(model.origins)
    assert np.max(model.origins + model.horizon) < np.min(model.terminal_origins)
    assert plan["excluded_boundary_origins"]["train"] == [dates[65]]
    assert plan["excluded_boundary_origins"]["development"] == [dates[95]]
    assert model.origin == 94
    args.research_cutoff = model.data.timestamp(100, "close")
    with pytest.raises(ValueError, match="precede the terminal"):
        configure_windows(model, args)


def test_incremental_llm_schema_allows_no_price_recipe_and_empty_news(monkeypatch):
    captured = []
    response = {
        "hypothesis": "No eligible business evidence; stop.",
        "action": "stop",
        "factor_ids": [],
        "evidence_news_ids": [],
        "event_signal": 0.0,
        "proposed_factors": [],
        "proposed_business_factors": [],
        "question_map": [],
    }
    monkeypatch.setattr(
        LLMHypothesizer,
        "_hypothesis_text",
        lambda self, prompt, schema: captured.append((prompt, schema)) or json.dumps(response),
    )
    context = {
        "allow_factor_proposals": True,
        "incremental_plan": {"baseline_ids": ["B"]},
        "news": {"items": []},
    }
    library = {"definitions": [], "incremental_plan": context["incremental_plan"]}
    result = LLMHypothesizer("codex")(context, library)
    assert result["action"] == "stop"
    assert "Select at least one proposal in factor_ids" not in captured[0][0]
    assert captured[0][1]["properties"]["proposed_factors"]["maxItems"] == 0


def test_retained_mode_disables_search_and_current_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "issuer_news", lambda *a, **k: pytest.fail("current news fetched"))
    monkeypatch.setattr(
        study, "mature_feedback", lambda *a, **k: pytest.fail("current memory read")
    )
    root = tmp_path / "retained"
    assert (
        main(
            command(
                root,
                "--research-mode",
                "incremental",
                "--information-mode",
                "retained",
                "--news",
                "issuer-rss",
                "--history-dir",
                str(tmp_path / "history"),
                "--llm",
                "simulated",
                "--no-ui",
            )
        )
        == 0
    )
    result = json.loads((root / "result.json").read_text())
    steps = {e["stage"]: e["output"] for e in result["trace"]}
    assert steps["news"]["items"] == []
    assert steps["news"]["coverage_status"] == "missing_eligible_historical_material"
    assert steps["target"]["prior_outcomes"] is None
    assert result["study"]["research_windows"]["information_mode"] == "retained"


@pytest.mark.parametrize("mode", ["renamed_noop", "outside_domain"])
def test_noop_and_unavailable_proposals_do_not_train(tmp_path, monkeypatch, mode):
    def researcher(context, library, **kwargs):
        first_feedback = {"research_history": [{}]}
        proposal = simulated_increment(context, library, feedback=first_feedback)
        if mode == "outside_domain":
            proposal.update(factor_ids=["financial.not_supplied"], proposed_business_factors=[])
        elif len(kwargs["feedback"]["research_history"]) > 1:
            proposal["proposed_business_factors"][0]["name"] = "same_recipe_new_name"
            proposal["factor_ids"] = ["business.same_recipe_new_name"]
        return proposal

    monkeypatch.setattr("quant_rsi.research.study_incremental.simulated_increment", researcher)
    root = tmp_path / mode
    assert (
        main(
            command(
                root,
                "--research-mode",
                "incremental",
                "--financials",
                "--llm",
                "simulated",
                "--no-ui",
            )
        )
        == 0
    )
    result = json.loads((root / "result.json").read_text())
    assert result["iterations"]["completed_fits"] == (2 if mode == "renamed_noop" else 1)
    assert result["iterations"]["stop_reason"] == (
        "unchanged_increment" if mode == "renamed_noop" else "unavailable_increment"
    )


def test_resume_after_revision_failure_reuses_both_completed_fits(tmp_path, monkeypatch):
    import lightgbm

    calls, fits = [], []
    original_train = lightgbm.train

    def train(*args, **kwargs):
        fits.append(True)
        return original_train(*args, **kwargs)

    def researcher(context, library, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise ValueError("injected revision transport failure")
        return simulated_increment(context, library, **kwargs)

    monkeypatch.setattr(lightgbm, "train", train)
    monkeypatch.setattr("quant_rsi.research.study_incremental.simulated_increment", researcher)
    root = tmp_path / "resume"
    assert (
        main(
            command(
                root,
                "--research-mode",
                "incremental",
                "--financials",
                "--llm",
                "simulated",
                "--no-ui",
            )
        )
        == 1
    )
    assert len(fits) == 2
    before = (root / "research_windows.json").read_bytes()
    assert main(["research-resume", str(root)]) == 0
    assert len(fits) == 3  # The baseline and first business fit were restored, not trained again.
    assert len(calls) == 3  # One successful first proposal, one failed revision, one retry.
    assert (root / "research_windows.json").read_bytes() == before


def test_retained_context_excludes_latest_financial_row_and_memory():
    from quant_rsi.research.study_windows import restrict_context

    model = SimpleNamespace(
        origin=2, horizon=1, financial_features={"evidence": [0, 1, 2, "eligible", "future"]}
    )
    context = {
        "financial_features": {"latest_visible_facts": "future"},
        "prior_outcomes": {"secret": "future"},
        "factor_memory": {"entries": ["future"]},
        "factor_ledger": "future",
        "price_evidence": {"last": "future"},
    }
    filtered = restrict_context(context, model, {"information_mode": "retained"})
    assert "future" not in json.dumps(filtered)
    assert filtered["financial_features"]["latest_visible_facts"] == "eligible"


def test_fixed_baseline_is_not_counted_as_paid_llm_request(tmp_path, monkeypatch):
    response = {
        "hypothesis": "No useful incremental test proposed.",
        "action": "stop",
        "factor_ids": [],
        "evidence_news_ids": [],
        "event_signal": 0.0,
        "proposed_factors": [],
        "proposed_business_factors": [],
        "question_map": [],
    }
    monkeypatch.setattr(LLMHypothesizer, "_hypothesis_text", lambda *a: json.dumps(response))
    root = tmp_path / "counts"
    assert main(command(root, "--research-mode", "incremental", "--llm", "codex", "--no-ui")) == 0
    runtime = json.loads((root / "runtime.json").read_text())
    assert runtime["counts"]["llm_requests_started"] == 1
    assert runtime["counts"]["model_stages_started"] == 1


def test_withdraw_after_better_trial_reuses_baseline_and_reports_actual_selection(
    tmp_path, monkeypatch
):
    original = LightGBMDemo.predict
    scores = iter([5.0, 1.0])

    def predict(self, *args):
        result = original(self, *args)
        result["development"]["metrics"]["mae_price"] = next(scores)
        return result

    def researcher(context, library, **kwargs):
        result = simulated_increment(context, library, **kwargs)
        if len(kwargs["feedback"]["research_history"]) > 1:
            result.update(action="withdraw", factor_ids=[], proposed_business_factors=[])
        return result

    monkeypatch.setattr(LightGBMDemo, "predict", predict)
    monkeypatch.setattr("quant_rsi.research.study_incremental.simulated_increment", researcher)
    root = tmp_path / "withdraw"
    assert (
        main(
            command(
                root,
                "--research-mode",
                "incremental",
                "--financials",
                "--llm",
                "simulated",
                "--no-ui",
            )
        )
        == 0
    )
    result = json.loads((root / "result.json").read_text())
    assert result["iterations"]["completed_fits"] == 2
    assert result["iterations"]["attempts"][1]["accepted"]
    assert result["study"]["selection"]["selected_round"] == 1
    assert (
        result["study"]["selection"]["selected_factor_ids"]
        == result["iterations"]["attempts"][0]["factor_ids"]
    )
    assert result["terminal_evaluation"]["selected_round"] == 1


def test_retained_snapshot_already_in_archive_is_not_duplicated(tmp_path):
    from test_study_news import snapshot

    from quant_rsi.research.study_news import retain_snapshot

    # Fixture receipt is later than this synthetic historical research window.
    news = snapshot()
    news["items"][0]["text"] = "Retained test headline, not historical research evidence."
    archive = tmp_path / "archive"
    retain_snapshot(archive, news)
    source = tmp_path / "news.json"
    source.write_text(json.dumps(news))
    root = tmp_path / "retained-snapshot"
    assert (
        main(
            command(
                root,
                "--research-mode",
                "incremental",
                "--information-mode",
                "retained",
                "--news",
                "issuer-rss",
                "--news-archive",
                str(archive),
                "--news-snapshot",
                str(source),
                "--llm",
                "simulated",
                "--no-ui",
            )
        )
        == 0
    )
    assert len(json.loads((root / "news_history.json").read_text())) == 1
    result = json.loads((root / "result.json").read_text())
    assert next(e["output"] for e in result["trace"] if e["stage"] == "news")["items"] == []
