"""Preparation reports distinguish source acquisition, partial columns and readiness."""

from copy import deepcopy

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research.discovery_report import render_discovery


def candidate(name):
    return {
        "name": name,
        "description": "An engineering fixture for a proposed input.",
        "inputs": [{"name": "demand", "factor_id": "acquired.demand"}],
        "expression": {
            "schema_version": 2,
            "nodes": [{"name": "raw", "op": "leaf", "leaf": "demand"}],
        },
        "evidence_ids": ["measurement-source"],
    }


def preparation(name, computed):
    proposed = candidate(name)
    rows = [
        {
            "session": f"2026-09-{day:02d}",
            "prediction_time": f"2026-09-{day:02d}T20:00:00Z",
            "partition": "train" if day < 3 else "development",
            "value": float(day - 1) if day <= computed else None,
        }
        for day in range(1, 5)
    ]
    return {
        "kind": "preparation",
        "candidate": proposed,
        "result": {
            "status": "ready" if computed == 4 else "needs_data",
            "reason": None if computed == 4 else "incomplete_frozen_population",
            "preparation": {
                "candidate_ref": digest(canonical(proposed)),
                "recipe_ref": "recipe-reference",
                "definition_ref": "definition-reference",
                "column_ref": digest(canonical(rows)),
                "input_bindings": proposed["inputs"],
                "input_bundle_refs": ["acquisition-reference"],
                "source_ids": ["measurement-source"],
                "rows": rows,
                "coverage": {
                    "required_rows": 4,
                    "computed_rows": computed,
                    "missing_rows": 4 - computed,
                    "training_rows": 2,
                    "computed_training_rows": min(computed, 2),
                    "development_rows": 2,
                    "computed_development_rows": max(computed - 2, 0),
                },
            },
        },
    }


def report_result():
    return {
        "symbol": "KO",
        "configuration": {"question": "Prepare proposed business inputs", "fit_budget": 0},
        "notice": "Preparation fixture; no model fitted or prediction evaluated.",
        "study": {"conclusion": "Prepare available rows and retain remaining gaps."},
        "discovery": {
            "mode": "preparation_only",
            "counts": {"fits": 0, "acquisitions": 1},
            "groups": {},
            "selected": None,
            "memory": [
                {
                    "kind": "acquire",
                    "request": {
                        "symbol": "KO",
                        "input_id": "acquired.demand",
                        "measurement": "Demand measure",
                        "unit": "USD",
                    },
                    "result": {
                        "identity": "acquisition-reference",
                        "status": "available",
                        "observations": [],
                    },
                },
                preparation("complete", 4),
                preparation("partial", 2),
                {
                    "kind": "preparation",
                    "candidate": candidate("blocked"),
                    "result": {"status": "needs_data", "reason": "forecast_channel_unspecified"},
                },
            ],
        },
    }


def visible(value):
    markdown, html = render_discovery(value)
    return markdown, html.split("<details>", 1)[0]


def test_preparation_report_shows_distinct_readiness_states_without_fake_fit_results():
    value = report_result()
    before = deepcopy(value)
    markdown, html = visible(value)
    assert value == before
    for document in (markdown, html):
        assert "Preparation only — no model fitting" in document
        assert "Prior experiment feedback is not loaded" in document
        assert "Acquired: 1 source-bound bundle(s)" in document
        assert "4 / 4 rows computable" in document and "2 / 4 rows computable" in document
        assert "Full frozen population ready" in document and "Not ready" in document
        assert "forecast_channel_unspecified" in document and "Not computed" in document
        assert "Training rows" in document and "Development rows" in document
        assert "Missing rows" in document and "incomplete_frozen_population" in document
        for absent in (
            "No completed baseline fit",
            "Publication selection",
            "Development MAE",
            "Candidate → acquired input → actual fit",
            "Executed feature values",
            "Reserved terminal evaluation",
            "Experiments are incomparable",
        ):
            assert absent not in document
    assert (
        "complete | Acquired: 1 source-bound bundle(s) | 4 / 4 rows computable | "
        "2 / 2 | 2 / 2 | 0 | Full frozen population ready" in markdown
    )
    assert (
        "partial | Acquired: 1 source-bound bundle(s) | 2 / 4 rows computable | "
        "2 / 2 | 0 / 2 | 2 | Not ready" in markdown
    )
    gaps = markdown.split("## Gaps and failures", 1)[1]
    assert "incomplete_frozen_population" in gaps and "forecast_channel_unspecified" in gaps


def test_preparation_report_retains_source_and_candidate_references_without_inventing_columns():
    value = report_result()
    markdown, html = visible(value)
    for document in (markdown, html):
        for record in value["discovery"]["memory"][1:]:
            assert digest(canonical(record["candidate"])) in document
        assert "demand = acquired.demand" in document
        assert "acquisition-reference" in document and "measurement-source" in document
        assert "recipe-reference" in document and "definition-reference" in document
        for record in value["discovery"]["memory"][1:3]:
            assert record["result"]["preparation"]["column_ref"] in document
    provenance = markdown.split("## Preparation provenance", 1)[1].split("## ", 1)[0]
    assert "blocked" not in provenance
    # Detailed rows retain actual zero values and unavailable values, not invented targets.
    _, full_html = render_discovery(value)
    assert "&quot;value&quot;: 0.0" in full_html and "&quot;value&quot;: null" in full_html
    assert "&quot;target&quot;" not in full_html and "&quot;observed_price&quot;" not in full_html


def test_unresolved_column_and_missing_acquisition_receipt_do_not_claim_readiness():
    value = report_result()
    value["discovery"]["memory"] = [preparation("unresolved", 0)]
    markdown, html = visible(value)
    for document in (markdown, html):
        assert "Referenced acquisition receipt unavailable" in document
        assert "0 / 4 rows computable" in document
        assert "Full frozen population ready" not in document


def test_budget_refusal_remains_visible_after_researcher_stops():
    value = report_result()
    value["discovery"]["stop_reason"] = "stop"
    value["discovery"]["memory"].append(
        {
            "kind": "diagnostic",
            "status": "not_executed",
            "reason": "surveys_budget_exhausted",
            "ref": "budget-refusal",
        }
    )
    for document in visible(value):
        assert "not_executed: surveys_budget_exhausted" in document


def test_normal_report_without_preparation_mode_preserves_existing_comparison_sections():
    value = report_result()
    value["discovery"].pop("mode")
    value["configuration"]["fit_budget"] = 3
    value["discovery"]["memory"] = []
    for document in visible(value):
        assert "No completed baseline fit is available." in document
        assert "Publication selection" in document
        assert "Preparation only — no model fitting" not in document
