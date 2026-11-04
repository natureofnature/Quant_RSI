"""Workbench state and retained views; script checks do not establish browser usability."""

import json

import pytest
from test_study_onboarding import run_workbench_script

from quant_rsi.research.discovery_report import render_discovery
from quant_rsi.research.study_progress import StudyProgress
from quant_rsi.research.study_task_views import compare_retained_tasks, retained_task_view
from quant_rsi.research.study_ui import _discovery_output, read_state


def test_discovery_baseline_view_has_no_candidate_feature_lineage():
    view = _discovery_output(
        "discovery_fit_1",
        {"prediction": {"development": {"metrics": {"mae_price": 1.2}}}, "actual_features": None},
        {},
        "KO",
    )
    assert view["facts"]["price_mae"] == 1.2
    assert view["facts"]["input_factors"] == []


@pytest.mark.parametrize("llm,expected_calls", [("simulated", 0), ("codex", 5)])
def test_discovery_live_state_counts_exact_stages_and_matched_scores(tmp_path, llm, expected_calls):
    config = {"symbol": "KO", "research_mode": "discovery", "llm": llm, "selection_metric": "mae"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    progress = StudyProgress(tmp_path, llm, quiet=True)
    for stage in (
        "discovery_decision_1",
        "discovery_fit_1_factor_library",
        "discovery_fit_1",
        "discovery_decision_2",
    ):
        progress.start(stage)
        progress.finish(stage)
    assert progress.snapshot()["counts"]["model_stages_completed"] == 1
    assert progress.snapshot()["counts"]["llm_requests_started"] == (0 if llm == "simulated" else 2)
    progress.complete()
    (tmp_path / "discovery").mkdir()
    (tmp_path / "discovery" / "source_usage.json").write_text(
        json.dumps({"usage": {"counts": {"search": 1, "extract": 2, "analysis": 1}}})
    )
    result = {
        "discovery": {
            "selected": {"result": {"round": 1}},
            "groups": {
                "matched": {
                    "baseline": {"result": {"metric": 2.0}},
                    "incumbent": {"result": {"round": 1, "metric": 1.5}},
                }
            },
            "memory": [],
        }
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    (tmp_path / "trace.jsonl").write_text(
        json.dumps(
            {
                "stage": "discovery_decision_1",
                "output": {
                    "action": "acquire",
                    "reason": "Need demand history",
                    "candidates": [{"name": "demand"}],
                    "data_requests": [{"input_id": "acquired.demand"}],
                },
            }
        )
    )
    state = read_state(tmp_path)
    assert state["config"]["research_mode"] == "discovery"
    assert state["runtime"]["counts"]["llm_requests_started"] == expected_calls
    assert state["runtime"]["counts"]["model_stages_completed"] == 1
    assert state["runtime"]["counts"]["search_requests_started"] == 1
    assert state["discovery"]["completed_decisions"] == 2
    assert state["discovery"]["baseline_metric"] == 2.0
    assert state["discovery"]["selected_metric"] == 1.5
    assert state["events"][0]["view"]["facts"]["requested_inputs"] == ["acquired.demand"]
    assert state["events"][-1]["stage"] == "discovery_summary"
    result["discovery"]["selected"] = None
    (tmp_path / "result.json").write_text(json.dumps(result))
    state = read_state(tmp_path)
    assert state["discovery"]["selected_metric"] is None
    assert state["discovery"]["baseline_metric"] is None


def test_discovery_form_separates_actions_fits_and_keeps_date_windows():
    run_workbench_script(
        r"""
(async()=>{
  await new Promise(setImmediate);
  fields.research_mode.value='discovery';
  fields.compare_models.checked=true;fields.patience.value='2';
  fields.walk_forward_folds.value='3';
  syncResearchMode(true);
  assert.equal(Number(fields.rounds.value),6);
  assert.equal(Number(fields.max_fits.value),3);
  assert.equal(Number(fields.rounds.max),10);
  assert.equal(fields.latest_fit.value,'selected');
  assert.equal(nodes['rounds-label'].textContent,'Maximum decision actions');
  assert.equal(nodes['discovery-hint'].hidden,false);
  assert.equal(nodes['incremental-hint'].hidden,true);
  assert.equal(nodes['date-windows'].hidden,false);
  for(const key of ['compare_models','latest_fit','patience','walk_forward_folds'])
    assert.equal(fields[key].disabled,true);
  assert.equal(fields.compare_models.checked,false);
  fields.train_start.value='2020-01-01';fields.train_end.value='2020-04-01';
  const saved=options().options;
  assert.equal(saved.rounds,6);assert.equal(saved.max_fits,3);
  assert.equal(saved.latest_fit,'selected');assert.equal(saved.compare_models,false);
  assert.equal(saved.patience,null);assert.equal(saved.walk_forward_folds,0);
  assert.deepEqual(saved.train_dates,['2020-01-01','2020-04-01']);
  const summary=launchSummary({...saved,financials:true,external_drivers:true},1);
  assert.match(summary,/6 decision actions/);
  assert.match(summary,/3 research \+ 0 diagnostic/);
  loadStudyOptions(nodes['start-form'],{...saved,rounds:8,max_fits:4});
  assert.equal(Number(fields.rounds.value),8);
  assert.equal(Number(fields.max_fits.value),4);
  fields.research_mode.value='incremental';syncResearchMode(true);
  assert.equal(Number(fields.rounds.value),3);
  assert.equal(Number(fields.rounds.max),3);
  assert.equal(fields.latest_fit.disabled,false);
  fields.research_mode.value='legacy';syncResearchMode(true);
  assert.equal(Number(fields.rounds.value),2);
  assert.equal(nodes['discovery-hint'].hidden,true);
  assert.equal(nodes['date-windows'].hidden,true);
  assert.equal(fields.latest_fit.value,'refit');
  const texts=e=>[e.textContent||'',...e.children.flatMap(texts)].join(' ');
  renderTasks([{id:'a'.repeat(24),symbol:'KO',research_mode:'discovery',rounds:6,
    max_fits:3,status:'completed',llm:'simulated',model:'lightgbm',data:'synthetic'}]);
  assert.match(texts(nodes.tasks),/News-driven factor discovery/);
  assert.match(texts(nodes.tasks),/6 decision actions/);
  renderPlans([{id:'b'.repeat(24),name:'Discovery template',symbols:['KO'],
    enabled:false,interval_hours:24,options:{...saved,train_dates:null}}]);
  assert.match(texts(nodes.plans),/News-driven factor discovery/);
  assert.match(texts(nodes.plans),/6 decision actions/);
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    )


def task_result(root, selected=None):
    (root / "study").mkdir(parents=True)
    result = {
        "symbol": "KO",
        "configuration": {
            "prices": "synthetic",
            "research_mode": "discovery",
            "horizon_sessions": 5,
        },
        "iterations": {
            "selected_round": selected,
            "attempts": [{"prediction": {"horizon_sessions": 5}}],
        },
        "discovery": {
            "selection_note": "Incomparable experiments; no global selection.",
            "groups": {
                name: {"incumbent": {"result": {"round": index + 1}}, "candidates": []}
                for index, name in enumerate(("population_a", "population_b"))
            },
        },
    }
    (root / "study" / "result.json").write_text(json.dumps(result))
    (root / "task.json").write_text(json.dumps({"symbol": "KO", "created_at": "2026-11-04"}))
    return result


def test_unselected_discovery_results_are_viewable_without_global_comparison(tmp_path):
    left, right = tmp_path / "a", tmp_path / "b"
    task_result(left)
    task_result(right)
    view = retained_task_view(left)
    assert view["symbol"] == "KO"
    assert view["selected_round"] is None
    assert view["horizon_sessions"] == 5
    assert len(view["experiment_groups"]) == 2
    comparison = compare_retained_tasks(left, right)
    for segment in comparison["comparison"].values():
        assert not segment["comparable"]
        assert segment["differences"] == {}
        assert any("No single discovery experiment selected" in text for text in segment["reasons"])


def test_missing_selection_remains_an_error_for_legacy_corruption(tmp_path):
    task = tmp_path / "task"
    result = task_result(task)
    result.pop("discovery")
    (task / "study" / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="no retained selected"):
        retained_task_view(task)


def test_discovery_report_connects_acquired_sources_to_actual_fit():
    bundle = {
        "identity": "bundle-123",
        "status": "available",
        "distinct_observations": 1,
        "distinct_values": 1,
        "covered_rows": 40,
        "columns": {"acquired.demand": [2.0]},
        "observations": [
            {
                "input_id": "acquired.demand",
                "period_end": "2025-12-31",
                "value": 2.0,
                "unit": "percent",
                "source_url": "https://issuer.example/report",
                "effective_available_at": "2026-02-01T00:00:00Z",
                "availability_basis": "retained_receipt",
                "excerpt": "Demand was 2.0 percent. <script>unsafe</script>",
            }
        ],
    }
    definition = {
        "factor_id": "discovery.demand",
        "input_factor_ids": ["acquired.demand"],
        "evidence_ids": ["news-7"],
    }
    measured = {
        "round": 2,
        "fit_identity": "fit-456",
        "prediction": {"development": {"metrics": {"mae_price": 2.5}}},
        "actual_features": {
            "definition": definition,
            "distinct_training_values": 1,
            "training_values": [2.0],
            "input_bundle_refs": ["bundle-123"],
        },
    }
    experiment = {"ref": "record-789", "spec": {"kind": "candidate"}, "result": measured}
    result = {
        "symbol": "KO",
        "notice": "Simulated acceptance fixture",
        "configuration": {"question": "Demand?"},
        "study": {"conclusion": "No global winner"},
        "discovery": {
            "counts": {"fits": 2},
            "selected": None,
            "selection_note": "Separate populations",
            "groups": {
                name: {"baseline": experiment, "candidates": [], "incumbent": experiment}
                for name in ("population_a", "population_b")
            },
            "memory": [
                {
                    "kind": "acquire",
                    "request": {
                        "input_id": "acquired.demand",
                        "measurement": "Demand",
                        "unit": "percent",
                    },
                    "result": bundle,
                },
                {"kind": "experiment", **experiment},
            ],
        },
    }
    markdown, html = render_discovery(result)
    for text in (
        "Source observations",
        "https://issuer.example/report",
        "bundle-123",
        "fit-456",
        "news-7",
        "Candidate → acquired input → actual fit",
        "population_a",
        "population_b",
    ):
        assert text in markdown
    assert "<script>unsafe</script>" not in html
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in html
    assert "Separate populations" in html
