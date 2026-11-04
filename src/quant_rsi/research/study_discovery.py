"""Connect existing study preparation to replaceable discovery and experiment modules."""

import json
from copy import deepcopy
from pathlib import Path

from quant_rsi.core.artifacts import atomic_write, canonical, digest
from quant_rsi.research.demo_strategy import evaluate_strategy
from quant_rsi.research.discovery_contracts import research_policy
from quant_rsi.research.discovery_engine import DiscoveryExperimentRunner
from quant_rsi.research.discovery_factors import retrieve_candidates, survey_catalog
from quant_rsi.research.discovery_loop import run_discovery
from quant_rsi.research.discovery_sources import (
    DiscoverySources,
    discovery_decision,
    simulated_discovery_decision,
    simulated_discovery_sources,
)
from quant_rsi.research.llm import LLMHypothesizer


def _prior_memory(history, mode):
    if mode == "retained":
        return []
    entries = (history or {}).get("entries", [])[-5:]
    records = []
    for entry in entries:
        directory = entry.get("run_directory")
        if not directory:
            continue
        path = Path(directory) / "discovery" / "result.json"
        if not path.is_file() or path.stat().st_size > 20_000_000:
            continue
        saved = json.loads(path.read_text())
        # Prioritize measured facts before bounding retrieval, so intervening decisions
        # cannot hide an earlier experiment group. Preserve its comparison boundary.
        memory = saved.get("memory", [])
        facts = [record for record in memory if record.get("kind") in {"experiment", "acquire"}]
        for record in (facts or memory)[-6:]:
            if record.get("kind") in {"diagnostic", "interpretation", "experiment", "acquire"}:
                projected = deepcopy(record)
                projected["prior_study"] = {
                    key: entry.get(key)
                    for key in ("study_id", "symbol", "question", "horizon_sessions")
                }
                if record["kind"] == "experiment":
                    projected["result"] = {
                        key: deepcopy(record["result"].get(key))
                        for key in ("status", "metric", "fit_identity", "actual_features")
                    }
                    spec = projected.pop("spec", {})
                    projected["experiment_context"] = {
                        "comparison_key": spec.get("comparison_key"),
                        "metric": spec.get("contract", {}).get("metric"),
                        "metric_direction": "minimize; Sharpe is negated when selected",
                        "use": "Historical context only; not a matched current-task comparison",
                    }
                elif record["kind"] == "acquire":
                    projected["result"] = {
                        key: deepcopy(record["result"].get(key))
                        for key in ("status", "identity", "definitions", "limitations", "gaps")
                    }
                projected["projection_of"] = record["ref"]
                records.append(projected)
    return records[-12:]


def run_connected_discovery(
    args,
    *,
    model,
    root,
    configuration,
    cache,
    progress,
    record,
    news,
    related,
    windows,
    cutoffs,
    prelude,
    prior_outcomes=None,
):
    trace = deepcopy(prelude)

    def emit(stage, output):
        event = {"stage": stage, "output": output}
        if progress.active is None:
            progress.start(stage)
        if output is None:
            return
        record(event)
        trace.append(event)

    engine = DiscoveryExperimentRunner(model, args, cutoffs, emit)
    policy = research_policy()
    mode = configuration["information_mode"]
    evidence = []
    for item in news.get("items", []):
        evidence.append(
            deepcopy(item)
            | {
                "text": item.get("headline", item.get("title", "")),
                "received_at": item.get("received_at", news.get("received_at")),
                "version_available_at": item.get("received_at", news.get("received_at")),
                "version_at": item.get("received_at", news.get("received_at")),
            }
        )
    transport = (
        None
        if args.llm == "simulated"
        else LLMHypothesizer(
            args.llm,
            args.llm_model,
            args.llm_timeout,
            on_usage=progress.usage,
            handoff_dir=getattr(args, "cursor_dir", None) or root / "cursor-llm",
        )
    )
    sources = (
        simulated_discovery_sources(
            cutoffs, call=cache.call, usage_path=root / "discovery" / "source_usage.json"
        )
        if transport is None
        else DiscoverySources(
            transport,
            information_mode=mode,
            research_cutoff=windows.get("research_cutoff"),
            retained_articles=evidence,
            call=cache.call,
            usage_path=root / "discovery" / "source_usage.json",
            policy=policy,
            max_searches=2 if args.news_research else 0,
            max_reads=6 if args.news_research else 0,
            max_extractions=12,
        )
    )
    catalog = survey_catalog(available_inputs=engine.columns)
    atomic_write(root / "discovery_catalog.json", canonical(catalog))
    from quant_rsi.research.study_history import history_view

    prior_memory = cache.call(
        "discovery_prior_memory",
        {
            "symbol": configuration["symbol"],
            "question": configuration["question"],
            "horizon": model.horizon,
            "mode": mode,
        },
        lambda: _prior_memory(
            history_view(args.history_dir, configuration["symbol"]) if args.history_dir else None,
            mode,
        ),
    )
    context = {
        "symbol": configuration["symbol"],
        "question": configuration["question"],
        "research_question": configuration["question"],
        "information_mode": mode,
        "research_cutoff": windows.get("research_cutoff"),
        "evidence": evidence,
        "definitions": engine.definitions,
        "survey_candidates": retrieve_candidates(configuration["question"], catalog),
        "source_limits": sources.limits,
        "related": related,
        "prior_memory": prior_memory,
        "requested_data_period": {
            "start": model.data.dates[0],
            "end": model.data.dates[model.origins[-1]],
            "meaning": (
                "Feature observation dates, not a source publication filter. Acquire prior "
                "disclosures as needed to establish the value already available at the first "
                "date, within the declared maximum age. Acquire subsequent disclosures through "
                "the last date; never backfill an earlier row from a later publication. "
                "Temporal expressions also need causal warm-up before required training rows."
            ),
        },
        "experiment_contract": {
            "windows": windows,
            "baseline": engine.baseline,
            "model": model.model_configuration,
            "metric": args.selection_metric,
        },
    }
    atomic_write(root / "discovery_context.json", canonical(context))

    def decide(value):
        progress.start(f"discovery_decision_{value['iteration']}")
        result = (
            simulated_discovery_decision(value)
            if transport is None
            else discovery_decision(transport, value)
        )
        record({"stage": f"discovery_decision_{value['iteration']}", "output": result})
        return result

    operation_counts = {"acquire": 0, "survey": 0}

    def source_operation(kind, request):
        operation_counts[kind] += 1
        name = f"discovery_{kind}_{operation_counts[kind]}"
        emit(name, None)
        output = (
            sources.acquire(request, prediction_times=cutoffs, context={"policy": policy})
            if kind == "acquire"
            else sources.survey(
                request | {"symbol": configuration["symbol"]}, context={"policy": policy}
            )
        )
        emit(name, output)
        return output

    def restore_sources(value):
        engine.evidence_ids.update(row["id"] for row in value.get("evidence", []) if row.get("id"))
        for bundle in value.get("input_bundles", []):
            engine.evidence_ids.update(row["source_id"] for row in bundle.get("observations", []))
        sources.restore_state(
            [
                row["result"]
                for row in value.get("memory", [])
                if row.get("kind") in {"acquire", "survey"}
            ]
        )

    discovery = run_discovery(
        context,
        root=root,
        cache=cache,
        decide=decide,
        run_experiment=engine,
        validate_candidate=engine.validate,
        on_context=restore_sources,
        acquire=lambda request: source_operation("acquire", request),
        survey=lambda request: source_operation("survey", request),
        policy=policy,
        fit_budget=configuration["fit_budget"],
        decision_budget=args.rounds,
    )
    restore_sources(discovery)
    discovery["source_operations"] = sources.operations
    discovery["source_counts"] = sources.counts
    selected = discovery.get("selected")
    chosen = selected["result"]["round"] if selected else None
    for attempt in engine.attempts:
        attempt["accepted"] = any(
            attempt["round"] == g["incumbent"]["result"].get("round")
            for g in discovery["groups"].values()
        )
    conclusion = (
        f"{configuration['symbol']}: {discovery['counts']['proposed']} candidate proposals; "
        f"{discovery['counts']['fits']} fits in {len(discovery['groups'])} matched experiment(s). "
        f"Stopped: {discovery['stop_reason']}. "
        + (
            f"Selected fit {chosen} within its matched experiment. "
            if chosen
            else "Experiments are incomparable; no single forecast selected. "
        )
        + "Development results do not establish independent predictive gain."
    )
    result = {
        "symbol": configuration["symbol"],
        "target": configuration["symbol"],
        "configuration": configuration,
        "notice": "News-driven candidate discovery with "
        + (
            "simulated news/decision/acquisition providers"
            if transport is None
            else "real LLM/source providers"
        )
        + f" and actual {model.model_label} fits on {model.price_provider}. "
        + "Inspect source receipts and reconstructed public-availability assumptions.",
        "trace": trace,
        "discovery": discovery,
        "iterations": {
            "attempts": engine.attempts,
            "selected_round": chosen,
            "completed_fits": len(engine.attempts),
            "stop_reason": discovery["stop_reason"],
        },
        "summary": {
            "provider": "news_driven_discovery",
            "text": conclusion,
            "selection": {"selected_round": chosen, "reason": discovery["selection_note"]},
        },
        "study": {
            "conclusion": conclusion,
            "research_windows": windows,
            "selection": {"selected_round": chosen, "reason": discovery["selection_note"]},
        },
    }
    if chosen:
        prediction = engine.attempts[chosen - 1]["prediction"]
        result["summary"].update(
            {
                k: deepcopy(prediction[k])
                for k in ("latest_forecast", "latest_forecast_unavailable")
                if k in prediction
            }
        )
        if args.terminal_sessions:
            terminal = model.evaluate_terminal(chosen)
            if terminal["status"] == "completed":
                terminal["strategy"] = evaluate_strategy(
                    model.data,
                    configuration["symbol"],
                    terminal["evaluation"]["rows"],
                    commission_bps=args.commission_bps,
                    slippage_bps=args.slippage_bps,
                    initial_cash=args.initial_cash,
                    threshold=args.threshold,
                    horizon_sessions=model.horizon,
                )
            result["terminal_evaluation"] = terminal
            atomic_write(root / "terminal_evaluation.json", canonical(terminal))
    result["discovery"]["context_identity"] = digest(canonical(context))
    atomic_write(root / "discovery" / "result.json", canonical(discovery))
    return result
