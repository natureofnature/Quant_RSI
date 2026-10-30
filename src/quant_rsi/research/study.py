"""One connected research run using the demo stages and existing numerical/account engines."""

import json
import math
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from pydantic import TypeAdapter

from quant_rsi.factors.contracts import InstrumentId
from quant_rsi.market.market_contracts import SideCosts, decimal_text
from quant_rsi.research.active_news import attach_research, research_news
from quant_rsi.research.demo import (
    load_catalog,
    run_demo,
    simulated_hypothesis,
    simulated_news,
    simulated_related,
)
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.demo_news import issuer_news
from quant_rsi.research.demo_prices import load_daily_prices
from quant_rsi.research.demo_proposals import FactorProposal
from quant_rsi.research.demo_related import profile_related
from quant_rsi.research.demo_strategy import evaluate_strategy
from quant_rsi.research.llm import LLMHypothesizer
from quant_rsi.research.study_estimators import model_configuration
from quant_rsi.research.study_news import (
    NewsHistory,
    load_archive,
    retain_snapshot,
    simulated_history,
)
from quant_rsi.research.study_peers import prepare_related_prices
from quant_rsi.research.study_progress import StudyProgress
from quant_rsi.research.study_selection import select_attempt


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_study(args):
    """Keep a completed stage log even when a later provider fails; never overwrite a run."""
    symbol = TypeAdapter(InstrumentId).validate_python(args.symbol.strip().upper())
    estimator_configuration = model_configuration(args.model, args.ridge_alpha)
    if args.prices == "synthetic" and args.prices_dir is not None:
        raise ValueError("--prices-dir requires --prices daily")
    if args.related_prices_dir and (args.prices != "daily" or not args.related_prices):
        raise ValueError("--related-prices-dir requires daily prices and enabled related prices")
    root = args.output.resolve()
    checkout = Path(__file__).resolve().parents[3]
    if root.is_relative_to(checkout):
        raise ValueError("Keep study artifacts outside the source checkout")
    if args.news_archive and args.news_archive.resolve().is_relative_to(checkout):
        raise ValueError("Keep news snapshots outside the source checkout")
    if args.related_prices_dir and args.related_prices_dir.resolve().is_relative_to(checkout):
        raise ValueError("Keep related price collections outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    configuration = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    configuration.update(
        symbol=symbol,
        model_configuration=estimator_configuration,
        research_rounds=2,
        generated_recipes="bounded_price_operators_and_retained_headline_counts",
    )
    _write(root / "config.json", configuration)
    last_stage = "initialization"
    progress = StudyProgress(root, args.llm, quiet=args.quiet)
    if getattr(args, "ui", False):
        from quant_rsi.research.study_ui import start_view

        start_view(root)

    def record(event):
        nonlocal last_stage
        progress.finish(event["stage"])
        with (root / "trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        last_stage = event["stage"]

    try:
        return _run(args, symbol, root, configuration, record, progress)
    except (Exception, KeyboardInterrupt) as exc:
        runtime = progress.fail()
        _write(
            root / "failure.json",
            {
                "status": "failed",
                "last_completed_stage": last_stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "runtime": runtime,
                "note": "Completed trace stages are retained. This directory is not resumable.",
            },
        )
        raise
    finally:
        if getattr(args, "ui", False):
            from quant_rsi.research.study_ui import save_view

            try:
                save_view(root)
            except OSError as exc:
                print(f"[quant-rsi] Could not save live.html: {exc}", file=sys.stderr)


def _run(args, symbol, root, configuration, record, progress):
    # Validate provider options before collection or fitting; no silent fallback.
    hypothesis = (
        simulated_hypothesis
        if args.llm == "simulated"
        else LLMHypothesizer(args.llm, args.llm_model, args.llm_timeout, on_usage=progress.usage)
    )
    SideCosts(commission_bps=args.commission_bps, slippage_bps=args.slippage_bps, tax_bps="0")
    decimal_text(args.initial_cash, positive=True)
    if not math.isfinite(args.threshold):
        raise ValueError("Strategy threshold must be finite")
    library = load_catalog(args.catalog)
    for op in ("qlib_roc_v1", "qlib_ma_v1", "log_return_volatility"):
        definition = FactorProposal(
            op=op,
            window=10,
            description="Sample standard deviation of 10 daily log returns (ddof=1), unannualized."
            if op == "log_return_volatility"
            else "Source-pinned local Qlib-style price adapter.",
        ).definition()
        if definition["factor_id"] not in {x["factor_id"] for x in library["definitions"]}:
            library["definitions"].append(definition)
    progress.start("load_prices")
    prices = (
        load_daily_prices(symbol, args.prices_dir or root / "prices")
        if args.prices == "daily"
        else None
    )
    model = LightGBMDemo(
        library,
        prices=prices,
        evaluation_sessions=args.evaluation_sessions,
        estimator=args.model,
        ridge_alpha=args.ridge_alpha,
    )
    if prices is None:
        model.data = replace(model.data, instruments=(symbol,))
    progress.finish(
        "load_prices",
        {"provider": model.price_provider, "symbol": symbol, "sessions": len(model.data.dates)},
    )
    progress.start("collect_news_context")
    related = (profile_related if args.related == "profile" else simulated_related)(symbol)
    news_context = {"symbol": symbol, "related": related}
    current_news = (
        issuer_news(news_context, limit=args.news_limit)
        if args.news == "issuer-rss"
        else simulated_news(news_context)
    )
    archive = args.news_archive or root / "news_archive"
    snapshots = load_archive(archive)
    if args.prices == "daily" and any(
        snapshot["provider"] != "issuer_rss" for snapshot in snapshots
    ):
        raise ValueError("Simulated news history cannot enter a daily-price study")
    if args.news == "issuer-rss":
        retain_snapshot(archive, current_news)
        snapshots = load_archive(archive)
    elif args.prices == "synthetic" and not snapshots:
        snapshots = simulated_history(model.data, symbol)
    history = NewsHistory(snapshots)
    cutoffs = (
        model.price_info["session_close_times"]
        if model.price_info
        else [model.data.timestamp(i, "close") for i in range(len(model.data.dates))]
    )
    model.configure_news(history, cutoffs, symbol)
    model.library["news_feature_support"] = model.context(symbol)["news_feature_support"]
    _write(root / "news_history.json", snapshots)
    _write(
        root / "inputs.json",
        {
            "price_provider": model.price_provider,
            "dates": list(model.data.dates),
            "target": symbol,
            "opens": model.data.opens[:, 0].tolist(),
            "closes": model.data.closes[:, 0].tolist(),
            "price_evidence": model.price_info,
            "news_history_file": "news_history.json",
            "news_snapshot_count": len(snapshots),
            "related_prices_file": "related_prices.json",
        },
    )
    progress.finish("collect_news_context", {"related": related, "news": current_news})
    prelude = []
    if args.news_research:

        def record_research(event):
            prelude.append(event)
            record(event)

        discovery = research_news(
            symbol,
            related,
            None if args.llm == "simulated" else hypothesis,
            on_stage=record_research,
            on_start=progress.start,
        )
        _write(root / "news_research.json", discovery)
        current_news, related = attach_research(current_news, related, discovery, symbol)

    progress.start("load_related_prices")
    peers = prepare_related_prices(
        model.data,
        cutoffs,
        list(model.train) + list(model.origins),
        related,
        mode=args.prices,
        directory=args.related_prices_dir or root / "related_prices",
        enabled=args.related_prices,
        simulated_suggestions=args.llm == "simulated",
    )
    model.configure_peers(peers)
    related["price_features"] = peers["support"]
    _write(
        root / "related_prices.json",
        {
            "support": peers["support"],
            "inputs": peers["inputs"],
            "definitions": peers["definitions"],
            "alignment": (
                "exact_session_and_peer_close_at_or_before_target_close; no_fill_no_row_drop"
            ),
        },
    )
    progress.finish("load_related_prices", {"related": related})

    def context(target):
        return model.context(target) | {
            "allow_factor_proposals": True,
            "selection_policy": {
                "metric": args.selection_metric,
                "rule": "strict_improvement_else_retain_initial; missing metrics retain initial",
                "scope": "development_only",
            },
        }

    def predict(context, factors, proposal):
        prediction = model.predict(context, factors, proposal)
        prediction["strategy"] = evaluate_strategy(
            model.data,
            symbol,
            prediction["development"]["rows"],
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
            initial_cash=args.initial_cash,
            threshold=args.threshold,
        )
        return prediction

    def observe(context, prediction):
        return model.observe(context, prediction) | {
            "strategy_diagnostics": {
                key: prediction["strategy"][key]["metrics"] for key in ("strategy", "buy_and_hold")
            },
        }

    def summarize(context, first, second, feedback):
        summary = model.summarize(context, first, second, feedback)
        selection = select_attempt(first, second, args.selection_metric)
        selected = second if selection["revision_accepted"] else first
        for key in ("latest_forecast", "latest_forecast_unavailable"):
            summary.pop(key, None)
            if key in selected:
                summary[key] = deepcopy(selected[key])
        summary["selection"] = selection
        summary["text"] = (
            f"{symbol}: two real fits and one exploratory factor revision. "
            f"{selection['reason']} The latest forecast, when available, follows the "
            f"{selection['selected_attempt']} attempt and has no observed outcome. "
            "Selection reuses development feedback; it does not establish independent gain."
        )
        return summary

    result = run_demo(
        symbol,
        catalog=lambda: model.library,
        price_context=context,
        related=lambda target: related,
        news=lambda context: current_news,
        hypothesize=hypothesis,
        predict=predict,
        observe=observe,
        summarize=summarize,
        on_stage=record,
        on_start=progress.start,
    )
    result["trace"] = prelude + result["trace"]
    steps = {event["stage"]: event["output"] for event in result["trace"]}
    first, second = (steps[key] for key in ("first_prediction", "second_prediction"))
    selection = result["summary"]["selection"]
    first_ids, second_ids = set(first["factor_ids"]), set(second["factor_ids"])
    metrics = second["development"]["metrics"]
    baseline = second["development"]["baselines"]["unchanged_price"]["metrics"]
    first_mae = first["development"]["metrics"]["mae_price"]
    change = metrics["mae_price"] - first_mae
    first_return = float(first["strategy"]["strategy"]["metrics"]["net_return"])
    revised_return = float(second["strategy"]["strategy"]["metrics"]["net_return"])
    buy_hold_return = float(second["strategy"]["buy_and_hold"]["metrics"]["net_return"])
    risk_accounts = [
        ("initial", first["strategy"]["strategy"]["metrics"]),
        ("revised", second["strategy"]["strategy"]["metrics"]),
        ("buy-and-hold", second["strategy"]["buy_and_hold"]["metrics"]),
    ]
    sharpe_summary = ", ".join(
        f"{label} "
        + (
            f"{account_metrics['sharpe_ratio']:.4f}"
            if account_metrics["sharpe_ratio"] is not None
            else f"N/A ({account_metrics['risk_statistics']['status']})"
        )
        for label, account_metrics in risk_accounts
    )
    risk_info = risk_accounts[0][1]["risk_statistics"]
    discovery_summary = ""
    if args.news_research:
        discovery_summary = (
            f"Active news research: {len(discovery['searches'])} search requests, "
            f"{len(discovery['evidence_news_ids'])}/{len(discovery['articles'])} articles read, "
            f"{len(discovery['related_candidates'])} suggested related companies "
            f"({discovery['provider']}). "
        )
    result["notice"] = (
        f"Connected research study with {model.price_provider} prices and "
        f"real {model.model_label} training. "
        f"Hypotheses: {steps['initial_hypothesis']['provider']}; "
        f"news: {steps['news']['provider']}; "
        f"relationships: {steps['related_stocks']['provider']}. "
        "Both attempts use the same retrospective development window; no independent gain "
        "is established. Inspect source coverage and the separate unobserved latest forecast."
    )
    result["configuration"] = configuration
    news_definitions = [
        item for item in steps["revised_factor_library"]["definitions"] if item["kind"] == "news"
    ]
    news_used = any(
        item["kind"] == "news"
        for key in ("initial_factors", "revised_factors")
        for item in steps[key]
    )
    result["study"] = {
        "engine": f"local_{args.model}_and_existing_finite_account",
        "selection": selection,
        "factor_changes": {
            "activated": sorted(second_ids - first_ids),
            "deactivated": sorted(first_ids - second_ids),
            "retained": sorted(first_ids & second_ids),
            "meaning": "Run-local selection changes; definitions and evidence remain in the trace.",
        },
        "news_use": (
            "Selected retained-headline counts entered the fitted feature matrix. "
            "Counts cover the selected RSS corpus, not all issuer news; no sentiment is inferred."
            if news_used
            else "News informed hypotheses and recipe proposals only; no news column was fitted."
        ),
        "news_factors": {
            "support": model.context(symbol)["news_feature_support"],
            "definitions": news_definitions,
            "snapshot_count": len(snapshots),
            "archive_directory": str(archive.resolve()) if args.news == "issuer-rss" else None,
            "executed_initial": [
                x["factor_id"] for x in steps["initial_factors"] if x["kind"] == "news"
            ],
            "executed_revised": [
                x["factor_id"] for x in steps["revised_factors"] if x["kind"] == "news"
            ],
            "timing_note": "Original receipt times are preserved source records, not independently "
            "authenticated clocks. Real price cutoffs use the retained exchange calendar.",
        },
        "external_engine": "Qlib-style local price adapters; no Qlib or QuantDinger execution.",
        "conclusion": (
            discovery_summary + f"{symbol}: completed two fits and one factor revision over "
            f"{metrics['sample_count']} development observations. Price MAE: initial "
            f"{first_mae:.4f}, revised {metrics['mae_price']:.4f}, "
            f"unchanged-price baseline {baseline['mae_price']:.4f}. "
            f"The revision {'increased' if change > 0 else 'decreased' if change < 0 else 'kept'} "
            f"MAE by {abs(change):.4f}. "
            f"{selection['reason']} "
            f"Net strategy return: initial {first_return:.2%}, revised {revised_return:.2%}, "
            f"buy-and-hold {buy_hold_return:.2%}. "
            f"Annualized Sharpe: {sharpe_summary} "
            f"({risk_info['daily_return_count']} daily returns; "
            "252 sessions/year; risk-free rate 0%). "
            "Short-sample Sharpe is descriptive, not a reliable long-term estimate. "
            "These are retrospective development results after factor selection and feedback, "
            "not independent accuracy or evidence of improvement. Strategy results are "
            "hypothetical and include explicit costs. The latest forecast, when present, "
            "has no observed outcome."
        ),
    }
    from quant_rsi.research.study_report import render_study

    # Freeze measured research time for the reports, but do not announce success yet.
    result["runtime"] = progress.snapshot("completed")
    markdown, html = render_study(result)
    (root / "report.md").write_text(markdown, encoding="utf-8")
    (root / "report.html").write_text(html, encoding="utf-8")
    _write(root / "result.json", result)
    progress.complete(result["runtime"])
    return result
