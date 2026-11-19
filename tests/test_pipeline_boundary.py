"""The active pipeline never imports pipeline A (#159 D1/D2, Milestone 1 Phase 3).

Pipeline A (the recursive campaign, package/controller flow and forecast workflow) is
frozen. The import-layering ratchet cannot see these leaks, because they run downward
(research -> market -> evaluation), so this test checks the import closure directly.
Moved shared symbols keep critical-tier unit tests here.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "instruments" / "shared" / "equities.json"

PIPELINE_A = (
    "quant_rsi.evaluation",
    "quant_rsi.forecast",
    "quant_rsi.worker",
    "quant_rsi.core.controller",
    "quant_rsi.core.model",
    "quant_rsi.core.accounting",
    "quant_rsi.execution.baseline",
    "quant_rsi.execution.observer",
    "quant_rsi.execution.observer_engine",
    "quant_rsi.execution.observer_journal",
    "quant_rsi.execution.strategy_replay",
    "quant_rsi.factors.discovery",
    "quant_rsi.factors.discovery_graph",
    "quant_rsi.factors.lifecycle",
    "quant_rsi.factors.news",
    "quant_rsi.factors.event_observer",
    "quant_rsi.market.asof",
    "quant_rsi.market.market_data",
    "quant_rsi.market.signal_sources",
    "quant_rsi.market.news_collection",
    "quant_rsi.research.researcher",
    "quant_rsi.research.research",
    "quant_rsi.research.research_journal",
    "quant_rsi.research.package_experiment",
    "quant_rsi.research.package_journal",
    "quant_rsi.research.task_definition",
    "quant_rsi.research.changes",
    "quant_rsi.research.trajectory",
    "quant_rsi.research.trajectory_state",
    "quant_rsi.research.trajectory_evidence",
    "quant_rsi.research.trajectory_delegation",
    "quant_rsi.research.workflow",
)

ACTIVE = (
    "quant_rsi.cli",
    "quant_rsi.research.study",
    "quant_rsi.research.rolling",
    "quant_rsi.research.study_workbench",
    "quant_rsi.research.study_queue",
    "quant_rsi.research.study_ui",
)

NEUTRAL = (
    "quant_rsi.core.model_parameters",
    "quant_rsi.market.event_csv",
    "quant_rsi.market.forecast_records",
    "quant_rsi.market.news_feed",
    "quant_rsi.factors.contracts",
)


def pipeline_a(names):
    return sorted(
        name for name in names if any(name == a or name.startswith(a + ".") for a in PIPELINE_A)
    )


def loaded_after(code):
    result = subprocess.run(
        [sys.executable, "-c", code + "\nimport sys, json; print(json.dumps(sorted(sys.modules)))"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("module", ACTIVE + NEUTRAL)
def test_importing_active_or_neutral_code_loads_no_pipeline_a_module(module):
    assert pipeline_a(loaded_after(f"import {module}")) == []


@pytest.mark.parametrize("mode", ["discovery", "incremental"])
def test_an_offline_study_run_loads_no_pipeline_a_module(tmp_path, mode):
    arguments = [
        "research-study",
        "KO",
        "--catalog",
        str(CATALOG),
        "--prices",
        "synthetic",
        "--related",
        "simulated",
        "--news",
        "simulated",
        "--llm",
        "simulated",
        "--research-mode",
        mode,
        "--no-ui",
        "--no-history",
        "--output",
        str(tmp_path / "study"),
    ]
    code = (
        "import contextlib, io\n"
        "from quant_rsi.cli import main\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        f"    assert main({arguments!r}) == 0\n"
    )
    assert pipeline_a(loaded_after(code)) == []


def test_pipeline_a_reexports_the_moved_symbols():
    # Frozen pipeline A keeps working unchanged through re-exports of the neutral homes.
    code = (
        "from quant_rsi.forecast import forecast_kernel, forecast_protocol, forecast_publication\n"
        "from quant_rsi.factors import lifecycle\n"
        "from quant_rsi.market import market_data, signal_sources\n"
        "from quant_rsi.core import model_parameters\n"
        "from quant_rsi.market import event_csv, forecast_records, news_feed\n"
        "assert forecast_kernel.forecast_errors is forecast_records.forecast_errors\n"
        "assert forecast_publication.confirmed_report is forecast_records.confirmed_report\n"
        "assert forecast_protocol.MODEL_PARAMETERS is model_parameters.MODEL_PARAMETERS\n"
        "assert forecast_protocol.BOOST_ROUNDS == model_parameters.BOOST_ROUNDS\n"
        "assert market_data.parse_csv is event_csv.parse_csv\n"
        "assert lifecycle.EventTemplate is news_feed.EventTemplate\n"
        "assert signal_sources._publication is news_feed._publication\n"
    )
    loaded_after(code)


def test_forecast_errors_are_percentage_points_and_never_clipped():
    from quant_rsi.market.forecast_records import forecast_errors

    result = forecast_errors([math.log(1.05), math.log(0.95)], [math.log(1.1), math.log(0.9)])
    assert result["mae_pp"] == pytest.approx(5)
    assert result["rmse_pp"] == pytest.approx(5)
    assert result["negative_mae_pp"] == pytest.approx(-5)
    assert result["observations"] == 2
    for invalid in (math.inf, math.nan, 1e6):
        with pytest.raises(ValueError):
            forecast_errors([invalid], [0.0])
    with pytest.raises(ValueError, match="complete fixed label grid"):
        forecast_errors([0.0], [0.0, 0.0])


def report():
    return {
        "status": "forecast",
        "issued_at": "2026-10-23T20:30:00+00:00",
        "origin_close_at": "2026-10-23T20:00:00+00:00",
        "forward_deadline": "2026-10-26T13:30:00+00:00",
    }


def test_confirmed_report_binds_the_committed_report_and_forward_window():
    from quant_rsi.core.artifacts import canonical, digest
    from quant_rsi.market.forecast_records import confirmed_report

    value = report()
    ref = digest(canonical(value))
    forward = confirmed_report(
        value, {"forecast_ref": ref, "confirmed_at": "2026-10-23T21:00:00+00:00"}
    )
    assert forward["forward"] and forward["mode"] == "forward" and forward["forecast_ref"] == ref
    late = confirmed_report(
        value, {"forecast_ref": ref, "confirmed_at": "2026-10-26T13:30:00+00:00"}
    )
    assert not late["forward"] and late["mode"] == "retrospective_or_abstained"
    with pytest.raises(ValueError, match="does not bind"):
        confirmed_report(value, {"forecast_ref": "0" * 64, "confirmed_at": value["issued_at"]})
    with pytest.raises(ValueError, match="does not bind"):
        confirmed_report(value, {"forecast_ref": ref, "confirmed_at": "2026-10-23T20:00:00+00:00"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Fri, 23 Oct 2026 20:15:00 GMT", ("2026-10-23T20:15:00+00:00", None, "second")),
        ("2026-10-23", (None, "2026-10-23", "day")),
        ("", (None, None, "unknown")),
        ("Fri, 23 Oct 2026 20:15:00", (None, None, "unknown")),  # no timezone: never guessed
        ("yesterday", (None, None, "unknown")),
    ],
)
def test_feed_publication_text_is_parsed_without_guessing(raw, expected):
    from quant_rsi.market.news_feed import _publication

    assert _publication(raw) == expected


def test_event_template_is_validated_in_its_neutral_home():
    from quant_rsi.market.news_feed import EventTemplate

    template = EventTemplate(template_id="news.issuer", description="Issuer headline", ttl_hours=24)
    assert template.decay == "none" and template.output_unit == "dimensionless"
    with pytest.raises(ValueError):
        EventTemplate(template_id="News", description="x", ttl_hours=0)
