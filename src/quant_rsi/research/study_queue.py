"""Workbench queue inputs and a small worker around the existing study CLI."""

import json
import math
import re
import sys
import time
from datetime import date
from decimal import Decimal
from pathlib import Path

from quant_rsi.core.artifacts import atomic_write, canonical
from quant_rsi.market.market_contracts import SideCosts, decimal_text
from quant_rsi.research.demo import research_question
from quant_rsi.research.study_history import collection_snapshot, history_view, observe_prices
from quant_rsi.research.study_iterations import validate_limits


def stock_symbol(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Z][A-Z0-9.-]{0,14}", value.strip().upper()
    ):
        raise ValueError("Enter a stock symbol, for example KO or AAPL")
    return value.strip().upper()


def study_options(options):
    defaults = {
        "llm": "codex",
        "model": "lightgbm",
        "compare_models": False,
        "latest_fit": "refit",
        "research_mode": "legacy",
        "train_dates": None,
        "development_dates": None,
        "terminal_dates": None,
        "data": "daily",
        "llm_model": "",
        "news_research": True,
        "financials": False,
        "external_drivers": False,
        "financials_source": "auto",
        "only_new_evidence": False,
        "rounds": 2,
        "max_fits": None,
        "patience": None,
        "selection_metric": "mae",
        "horizon_sessions": 5,
        "historical_sessions": None,
        "walk_forward_folds": 0,
        "question": "",
        "initial_cash": "10000",
        "commission_bps": "5",
        "slippage_bps": "5",
        "threshold": 0.0,
    }
    if not isinstance(options, dict) or set(options) - defaults.keys():
        raise ValueError("Unexpected study options")
    if options.get("research_mode") == "incremental":
        defaults.update(rounds=3, latest_fit="selected")
    result = defaults | options
    result["question"] = research_question(result["question"])
    for key, allowed in (
        ("llm", {"codex", "api", "simulated", "deepseek", "cursor"}),
        ("model", {"lightgbm", "ridge"}),
        ("research_mode", {"legacy", "incremental"}),
        ("latest_fit", {"refit", "selected"}),
        ("financials_source", {"auto", "sec", "issuer"}),
        ("data", {"daily", "synthetic"}),
        ("selection_metric", {"mae", "sharpe"}),
    ):
        if not isinstance(result[key], str) or result[key] not in allowed:
            raise ValueError("Unsupported provider, model, data mode or selection metric")
    if type(result["horizon_sessions"]) is not int or result["horizon_sessions"] not in (5, 20, 60):
        raise ValueError("Prediction horizon must be 5, 20 or 60 trading sessions")
    if type(result["walk_forward_folds"]) is not int or result["walk_forward_folds"] not in (
        0,
        2,
        3,
    ):
        raise ValueError("Walk-forward diagnostic needs 0, 2 or 3 folds")
    from quant_rsi.research.study_walk_forward import price_history_sessions

    price_history_sessions(
        result["horizon_sessions"], 20, result["walk_forward_folds"], result["historical_sessions"]
    )
    if result["historical_sessions"] is not None and result["data"] != "daily":
        raise ValueError("Historical sessions requires daily prices")
    model = result["llm_model"]
    if (
        not isinstance(model, str)
        or len(model) > 120
        or (result["llm"] == "api" and not model.strip())
    ):
        raise ValueError("API mode requires a model name; credentials come from the environment")
    if result["news_research"] and result["llm"] in {"deepseek", "cursor"}:
        raise ValueError("News research web search requires Codex or the OpenAI API")
    for key in (
        "news_research",
        "financials",
        "external_drivers",
        "only_new_evidence",
        "compare_models",
    ):
        if type(result[key]) is not bool:
            raise ValueError(key + " must be boolean")
    if result["max_fits"] is None:
        result["max_fits"] = result["rounds"]
    for key in ("rounds", "max_fits", "patience"):
        if result[key] is not None and type(result[key]) is not int:
            raise ValueError("Research limits must be integers")
    validate_limits(result["rounds"], result["max_fits"], result["patience"])
    incremental = result["research_mode"] == "incremental"
    if incremental and (result["rounds"] > 3 or result["max_fits"] > 3):
        raise ValueError("Incremental research permits at most three research fits")
    if incremental and result["walk_forward_folds"]:
        raise ValueError("Incremental research uses fixed windows; disable time-window diagnostics")
    previous_end = None
    for partition in ("train", "development", "terminal"):
        dates = result[partition + "_dates"]
        if dates is None:
            continue
        if not incremental:
            raise ValueError("Date windows require incremental research")
        if (
            not isinstance(dates, list)
            or len(dates) != 2
            or any(
                not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day)
                for day in dates
            )
        ):
            raise ValueError("Each date window needs a start and end in YYYY-MM-DD format")
        start, end = (date.fromisoformat(day) for day in dates)
        if start > end or (previous_end is not None and start <= previous_end):
            raise ValueError("Date windows must be ordered and must not overlap")
        previous_end = end
    for key in ("initial_cash", "commission_bps", "slippage_bps"):
        if not isinstance(result[key], str):
            raise ValueError("Capital and costs must be decimal strings")
    result["initial_cash"] = decimal_text(result["initial_cash"], positive=True)
    if Decimal(result["initial_cash"]) % Decimal("0.01"):
        raise ValueError("Initial USD capital must have cent precision")
    costs = SideCosts(
        commission_bps=result["commission_bps"], tax_bps="0", slippage_bps=result["slippage_bps"]
    )
    result["commission_bps"], result["slippage_bps"] = costs.commission_bps, costs.slippage_bps
    if type(result["threshold"]) not in (int, float) or not math.isfinite(result["threshold"]):
        raise ValueError("Signal threshold must be a finite number")
    return result


def watchlist(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 20:
        raise ValueError("A watchlist needs 1 to 20 unique stock symbols")
    symbols = [stock_symbol(item) for item in value]
    if len(set(symbols)) != len(symbols):
        raise ValueError("Watchlist symbols must be unique")
    return symbols


def study_arguments(task, workspace, symbol, options, *, batch_id=None):
    arguments = [
        "research-study",
        symbol,
        "--output",
        str(task / "study"),
        "--no-ui",
        "--factor-root",
        str(Path(__file__).resolve().parents[3] / "instruments"),
        "--history-dir",
        str(workspace / "history" / options["data"]),
    ]
    for key, flag in (
        ("llm", "llm"),
        ("model", "model"),
        ("initial_cash", "initial-cash"),
        ("commission_bps", "commission-bps"),
        ("slippage_bps", "slippage-bps"),
        ("threshold", "threshold"),
        ("research_mode", "research-mode"),
        ("latest_fit", "latest-fit"),
        ("data", "prices"),
        ("rounds", "rounds"),
        ("max_fits", "max-fits"),
        ("patience", "patience"),
        ("selection_metric", "selection-metric"),
        ("horizon_sessions", "horizon-sessions"),
        ("historical_sessions", "historical-sessions"),
        ("walk_forward_folds", "walk-forward-folds"),
    ):
        if options.get(key) is not None:
            arguments += ["--" + flag, str(options[key])]
    for partition in ("train", "development", "terminal"):
        if options.get(partition + "_dates"):
            arguments += ["--" + partition + "-dates", *options[partition + "_dates"]]
    if options["data"] == "synthetic":
        arguments += ["--news", "simulated", "--related", "simulated"]
    if options["llm"] == "cursor":
        arguments += ["--cursor-dir", str(task / "study" / "cursor-llm")]
    if options["llm_model"]:
        arguments += ["--llm-model=" + options["llm_model"]]
    if options.get("question"):
        arguments += ["--question=" + options["question"]]
    if options["news_research"]:
        arguments += ["--news-research"]
    if options.get("compare_models"):
        arguments += ["--compare-models"]
    if options["financials"]:
        arguments += ["--financials", "--financials-source", options["financials_source"]]
    if options.get("external_drivers"):
        arguments += ["--external-drivers"]
        if batch_id is not None and options["data"] == "daily":
            if not re.fullmatch(r"[0-9a-f]{24}", batch_id):
                raise ValueError("Invalid study batch identity")
            arguments += [
                "--external-drivers-cache-dir",
                str(workspace / "batches" / batch_id / "external_drivers"),
            ]
    return arguments


def observe_task(task, manifest):
    """Collect fresh evidence independently of news/LLM availability; never invent maturity."""
    history = task.parent.parent / "history" / manifest["data"]
    view = history_view(history, manifest["symbol"])
    actionable = any(e["status"] in {"pending", "insufficient_evidence"} for e in view["entries"])
    if manifest["data"] == "daily" and actionable:
        snapshot = collection_snapshot(manifest["symbol"], task / "prices")
        view = observe_prices(history, manifest["symbol"], snapshot)
        reason = "fresh_public_prices"
    else:
        reason = (
            "synthetic_not_observed" if manifest["data"] == "synthetic" else "no_pending_forecasts"
        )
    atomic_write(task / "observation.json", canonical(view | {"reason": reason}))


def run_task(task):
    """The launcher passes an inherited lock fd held for this entire process lifetime."""
    from quant_rsi.cli import main

    task = Path(task)
    manifest = json.loads((task / "task.json").read_bytes())
    state = {"status": "running", "started_at": time.time()}
    atomic_write(task / "worker.json", canonical(state))

    def preparation_stage(stage):
        state["stage"] = stage
        atomic_write(task / "worker.json", canonical(state))

    try:
        if manifest.get("action") == "observe":
            observe_task(task, manifest)
            code = 0
        else:
            arguments = manifest["arguments"]
            skip = False
            if (
                manifest.get("only_new_evidence")
                and manifest.get("action", "research") == "research"
                and arguments[0] == "research-study"
            ):
                from quant_rsi.research.study_trigger import refresh_evidence

                trigger, arguments = refresh_evidence(task, manifest, on_stage=preparation_stage)
                atomic_write(task / "trigger.json", canonical(trigger))
                skip = bool(trigger["previous_completed_task"]) and not manifest.get("force", False)
                preparation_stage(None)
            if skip:
                state.update(status="skipped", reason="unchanged_evidence", exit_code=0)
                atomic_write(task / "worker.json", canonical(state | {"finished_at": time.time()}))
                return 0
            code = main(arguments)
        state.update(status="completed" if code == 0 else "failed", exit_code=code)
    except Exception as exc:
        state.update(status="failed", error=str(exc), exit_code=1)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    atomic_write(task / "worker.json", canonical(state | {"finished_at": time.time()}))
    return state["exit_code"]


if __name__ == "__main__":
    raise SystemExit(run_task(sys.argv[1]))
