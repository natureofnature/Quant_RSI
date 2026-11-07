"""Rolling forecasts through existing numerical, news and provider adapters."""

import json
import math
import shutil
import time
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import numpy as np

from quant_rsi.core.artifacts import atomic_write, canonical, digest
from quant_rsi.core.data import MarketData
from quant_rsi.forecast.forecast_protocol import BOOST_ROUNDS, MODEL_PARAMETERS
from quant_rsi.market.geopolitical_data import align_gpr_features, load_gpr_history
from quant_rsi.market.market_contracts import timestamp
from quant_rsi.market.risk_data import align_risk_features, load_risk_history
from quant_rsi.research.demo_features import price_metrics
from quant_rsi.research.demo_prices import read_daily_history
from quant_rsi.research.demo_related import profile_related
from quant_rsi.research.demo_strategy import evaluate_monthly_strategy, evaluate_session_strategy
from quant_rsi.research.discovery_factors import DiscoveryFactor
from quant_rsi.research.llm import LLMHypothesizer
from quant_rsi.research.rolling_acquired import AcquiredResolver, load_acquired_history
from quant_rsi.research.rolling_candidates import screen_candidates
from quant_rsi.research.rolling_config import load_window_profile
from quant_rsi.research.rolling_diagnostics import (
    compare_forecast_diagnostics,
    diagnose_candidates,
    diagnose_forecast,
    summarize_forecast,
)
from quant_rsi.research.rolling_news_sources import collect_history, load_history, select_evidence
from quant_rsi.research.rolling_ohlcv import load_ohlcv_history
from quant_rsi.research.rolling_research import (
    admit_decision,
    decision_context,
    feature_library,
    propose,
)
from quant_rsi.research.rolling_windows import monthly_plan, period_id, read_calendar, session_plan
from quant_rsi.research.study_estimators import StandardizedRidge
from quant_rsi.research.study_news import load_archive
from quant_rsi.research.study_peers import aligned_closes
from quant_rsi.research.study_resume import StudyCache, run_lock
from quant_rsi.research.study_windows import retained_news

NOTICE = (
    "Reconstructed monthly rolling evaluation, not contemporaneously captured forecasts. "
    "Each model uses only its six-month history, 30-row warm-up and matured daily labels. "
    "Daily training labels use the next month's trading-session horizon, not literal "
    "calendar-month-end labels for each historical day. Model/proposal predictions are "
    "frozen before settlement; only mature prior months enter feedback. Supplied news is "
    "cutoff-filtered; LLM pretrained knowledge is not. A few realized months do not establish "
    "predictive advantage. Pending months have no accuracy or realized strategy return."
)
SESSION_NOTICE = (
    "Reconstructed {horizon}-trading-session rolling evaluation, not contemporaneously "
    "captured forecasts. Each model uses its six-month history, 30-row warm-up and matured "
    "{horizon}-session daily labels. Forecast intervals are contiguous and non-overlapping; "
    "the next decision is made at the previous target close. Only mature prior outcomes "
    "enter feedback. Only complete intervals within the requested target-month range are "
    "planned, stopping after the first pending target. Supplied news is cutoff-filtered; "
    "LLM pretrained knowledge is not. Training labels can overlap and are not independent "
    "observations. Pending targets have no accuracy or realized strategy return. "
    "A shorter horizon does not itself establish predictive advantage."
)


def add_arguments(commands):
    parser = commands.add_parser(
        "research-rolling", help="rolling forecasts by calendar month or fixed trading sessions"
    )
    parser.add_argument("symbol")
    parser.add_argument("--prices-dir", type=Path, required=True)
    parser.add_argument(
        "--price-basis-review",
        type=Path,
        help="explicit reviewed split-only O/C evidence bundle for the retained target snapshot",
    )
    parser.add_argument("--related-prices-dir", type=Path)
    parser.add_argument(
        "--ohlcv-features",
        action="store_true",
        help="expose verified retained target high/low/volume to candidate expressions",
    )
    parser.add_argument("--first-month", required=True, help="first forecast month, YYYY-MM")
    parser.add_argument("--last-month", required=True, help="last forecast month, YYYY-MM")
    parser.add_argument("--profile", type=Path, help="instrument-owned rolling window JSON")
    training = parser.add_mutually_exclusive_group()
    training.add_argument("--train-months", type=int, help="calendar-month history; default 6")
    training.add_argument("--train-sessions", type=int, help="exact trading-session history window")
    parser.add_argument(
        "--horizon-sessions",
        type=int,
        help="fixed forecast and decision interval in trading sessions (1..20); e.g. 3",
    )
    parser.add_argument("--model", choices=("lightgbm", "ridge"), default="lightgbm")
    parser.add_argument("--llm", choices=("none", "codex", "api"), default="none")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-timeout", type=float, default=180)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=2,
        help="joint candidate-factor limit, 1..32; fixed factors are additional",
    )
    parser.add_argument(
        "--max-news-inputs",
        type=int,
        help="news input budget, 1..64; defaults to twice max-candidates",
    )
    parser.add_argument(
        "--candidate-revision",
        action="store_true",
        help="allow one same-cutoff revision after structural diagnostics; no additional fits",
    )
    parser.add_argument("--news-archive", type=Path)
    parser.add_argument(
        "--risk-history", type=Path, help="verified offline VIX/Treasury snapshot directory"
    )
    parser.add_argument(
        "--gpr-history", type=Path, help="retained publisher-vintage geopolitical risk directory"
    )
    parser.add_argument(
        "--acquired-history",
        type=Path,
        action="append",
        help="successful discovery acquisition checkpoint; repeat for additional sources",
    )
    parser.add_argument(
        "--max-extractions",
        type=int,
        default=6,
        help="cumulative measurement extraction attempts, including failures (default: 6)",
    )
    parser.add_argument(
        "--news-history", type=Path, help="replay a verified historical-news acquisition directory"
    )
    parser.add_argument(
        "--news-research",
        action="store_true",
        help="acquire historical headlines and execute news factor hypotheses",
    )
    parser.add_argument(
        "--news-query", help="fixed historical search universe; frozen before research"
    )
    parser.add_argument(
        "--news-information-mode", choices=("reconstructed", "retained"), default="reconstructed"
    )
    parser.add_argument("--commission-bps", default="5")
    parser.add_argument("--slippage-bps", default="5")
    parser.add_argument("--initial-cash", default="10000")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--price-note",
        default="Prices use the retained quote/share basis; "
        "future corporate actions require a consistent-basis comparison.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="reuse successful frozen operations")


def _write(path, value):
    atomic_write(path, canonical(value))


def _recorded_proposal(llm, context, root, operation, enabled):
    """Count physical provider attempts inside the cache operation, never on cache hits."""
    if not enabled:
        return propose(llm, context)
    record = {
        "operation": operation,
        "input_sha256": digest(canonical(context)),
        "started_at_unix": time.time(),
    }

    def event(status, **details):
        with (root / "research_calls.jsonl").open("ab") as stream:
            stream.write(canonical(record | {"status": status} | details) + b"\n")

    event("attempted")
    try:
        output = propose(llm, context)
    except Exception as exc:
        event("failed", error_type=type(exc).__name__, error=str(exc))
        raise
    event("completed", output_sha256=digest(canonical(output)))
    return output


def _view(data, dates):
    lookup = {day: i for i, day in enumerate(data.dates)}
    indices = [lookup[day] for day in dates]
    return MarketData(tuple(dates), data.instruments, data.opens[indices], data.closes[indices])


def fit_forecast(matrix, labels, train, name, factor_ids):
    x = matrix[train]
    if not np.isfinite(x).all() or not np.isfinite(matrix[-1]).all():
        raise ValueError("Features must cover all fixed training and prediction rows")
    if name == "ridge":
        estimator = StandardizedRidge(x, labels, 1.0)
        prediction = float(estimator.predict(matrix[-1:])[0])
        state = estimator.state()
        usage = {
            identity: {"standardized_coefficient": float(weight)}
            for identity, weight in zip(factor_ids, estimator.coefficients, strict=True)
        }
    else:
        import lightgbm as lgb

        estimator = lgb.train(
            MODEL_PARAMETERS | {"seed": 7},
            lgb.Dataset(x, label=labels),
            num_boost_round=BOOST_ROUNDS,
        )
        prediction = float(estimator.predict(matrix[-1:], num_threads=1)[0])
        state = {"model_string": estimator.model_to_string()}
        usage = {
            identity: {"split_count": int(count), "gain": float(gain)}
            for identity, count, gain in zip(
                factor_ids,
                estimator.feature_importance("split"),
                estimator.feature_importance("gain"),
                strict=True,
            )
        }
    if not math.isfinite(prediction):
        raise ValueError("Non-finite model forecast")
    return {
        "predicted_log_return": prediction,
        "factor_ids": factor_ids,
        "training_samples": len(train),
        "feature_usage": usage,
        "state": state,
    }


def _forecast(fold, reference, value):
    price = float(reference * np.exp(value))
    if not math.isfinite(price) or price <= 0:
        raise ValueError("Invalid predicted price")
    return {
        "origin_session": fold["origin_session"],
        "target_session": fold["target_session"],
        "reference_price": float(reference),
        "predicted_price": price,
        "predicted_log_return": float(value),
    }


def settle(fold, data):
    """Expose the target only after the forecast record has been frozen."""
    fold = deepcopy(fold)
    if fold["target_session"] in data.dates:
        index = data.dates.index(fold["target_session"])
        observed = float(data.closes[index, 0])
        for forecast in fold["forecasts"].values():
            forecast["observed_price"] = observed
        fold["outcome"] = "observed"
    else:
        fold["outcome"] = "not_observed"
    return fold


def _feedback(folds, cutoff):
    result = []
    for fold in folds:
        if timestamp(fold["target_close_at"]) > timestamp(cutoff) or fold["outcome"] != "observed":
            continue
        losses = {
            arm: abs(row["predicted_price"] - row["observed_price"]) / row["reference_price"] * 100
            for arm, row in fold["forecasts"].items()
        }
        result.append(
            {
                ("period_id" if "period_id" in fold else "month"): period_id(fold),
                "prediction_sha256": fold["prediction_sha256"],
                "target_session": fold["target_session"],
                "decision": fold["decision"],
                "candidate_status": fold["candidate_status"],
                "candidate_pool": fold.get("candidate_pool"),
                "candidate_rejections": fold.get("candidate_rejections", []),
                "forecast_diagnostics": {
                    arm: summarize_forecast(value)
                    for arm, value in fold.get("forecast_diagnostics", {}).items()
                },
                "paired_forecast_decomposition": deepcopy(
                    fold.get("paired_forecast_decomposition")
                ),
                "actual_adaptive_factors": fold["fits"].get("adaptive", {}).get("factor_ids", []),
                "forecasts": deepcopy(fold["forecasts"]),
                "errors": {
                    arm: row["predicted_price"] - row["observed_price"]
                    for arm, row in fold["forecasts"].items()
                },
                "absolute_return_errors_pp": losses,
                "adaptive_loss_deltas_pp": {
                    comparator: losses["adaptive"] - losses[comparator]
                    for comparator in ("fixed", "unchanged_price", "training_label_mean")
                    if "adaptive" in losses and comparator in losses
                },
                "comparison_note": (
                    "Negative delta means lower absolute return error on this same target. "
                    "Compare adaptive with fixed for candidate-pool increment, and with "
                    "unchanged price/training mean for absolute forecast performance. "
                    "One monthly comparison does not establish an individual factor's effect."
                ),
            }
        )
        if "period_id" in fold:
            result[-1]["comparison_note"] = result[-1]["comparison_note"].replace(
                "One monthly comparison", "One interval comparison"
            )
    return result


def _merge_acquired(columns, definitions, view, fold, train):
    if set(columns).intersection(view["columns"]):
        raise ValueError("Acquired inputs cannot replace native factors")
    fold["acquired_inputs"] = deepcopy(
        {key: value for key, value in view.items() if key not in {"columns", "raw_bundles"}}
    )
    for item in fold["acquired_inputs"]["coverage"]:
        identity = item["input_id"]
        values = view["columns"][identity]
        selected = values[train]
        finite = selected[np.isfinite(selected)]
        evidence = item["input_evidence"]
        item.update(
            training_available=len(finite),
            training_samples=len(train),
            distinct_training_values=len(np.unique(finite)),
            training_disclosures=len(
                {evidence[i]["observation_id"] for i in train if evidence[i]["observation_id"]}
            ),
            prediction_available=bool(np.isfinite(values[-1])),
        )
    view["coverage"] = deepcopy(fold["acquired_inputs"]["coverage"])
    return columns | view["columns"], definitions + view["definitions"]


def run_months(
    data,
    info,
    peers,
    plan,
    cache,
    root,
    *,
    model="lightgbm",
    llm=None,
    snapshots=(),
    on_fold=None,
    news_history=None,
    news_mode="reconstructed",
    news_required=False,
    max_candidates=2,
    max_news_inputs=None,
    risk_history=None,
    gpr_history=None,
    candidate_revision=False,
    ohlcv_history=None,
    acquired_resolver=None,
):
    """Cut off numerical views before proposing/fitting; settlement cannot enter those views."""
    if type(max_candidates) is not int or not 1 <= max_candidates <= 32:
        raise ValueError("Candidate limit must be between 1 and 32")
    max_news_inputs = 2 * max_candidates if max_news_inputs is None else max_news_inputs
    if type(max_news_inputs) is not int or not 1 <= max_news_inputs <= 64:
        raise ValueError("News input limit must be between 1 and 64")
    if candidate_revision and llm is None:
        raise ValueError("Candidate revision requires an LLM")
    folds, incumbent, incumbent_news = [], [], []
    for planned in plan:
        fold = deepcopy(planned)
        dates = fold.pop("history_sessions")
        train = np.asarray(fold.pop("training_indices"))
        prefix = _view(data, dates)
        start, end = data.dates.index(dates[0]), data.dates.index(dates[-1]) + 1
        peer_views = {symbol: values[start:end].copy() for symbol, values in peers.items()}
        price_inputs = None
        if ohlcv_history is not None:
            price_inputs = {
                "columns": {
                    identity: values[start:end].copy()
                    for identity, values in ohlcv_history["columns"].items()
                },
                "definitions": deepcopy(ohlcv_history["definitions"]),
            }
            for definition in price_inputs["definitions"]:
                values = price_inputs["columns"][definition["factor_id"]]
                # Expression screening owns required-row completeness: a lag can
                # legitimately avoid a missing current input.
                definition["runnable"] = bool(np.isfinite(values).any())
                definition["availability_reason"] = (
                    "retained_daily_field"
                    if definition["runnable"]
                    else "missing_retained_daily_field"
                )
            fold["ohlcv_inputs"] = {
                "provenance": ohlcv_history["provenance"],
                "features": [
                    {
                        "factor_id": identity,
                        "training_available": int(np.isfinite(values[train]).sum()),
                        "training_samples": len(train),
                        "distinct_training_values": len(
                            np.unique(values[train][np.isfinite(values[train])])
                        ),
                        "prediction_available": bool(np.isfinite(values[-1])),
                    }
                    for identity, values in price_inputs["columns"].items()
                ],
            }
        columns, definitions, baseline = feature_library(
            prefix, peer_views, price_inputs=price_inputs
        )
        for history, align in (
            (risk_history, align_risk_features),
            (gpr_history, align_gpr_features),
        ):
            if history is None:
                continue
            risk_columns, risk_definitions, diagnostics = align(history, dates)
            columns.update(risk_columns)
            definitions.extend(risk_definitions)
            risk = fold.setdefault(
                "risk_inputs", {"notice": "", "sources": [], "features": [], "coverage": []}
            )
            risk["notice"] = " ".join(filter(None, (risk["notice"], history["notice"])))
            risk["sources"].extend(history["sources"])
            risk["features"].extend(diagnostics)
            risk["coverage"].extend(
                [
                    {
                        "factor_id": identity,
                        "training_available": int(np.isfinite(values[train]).sum()),
                        "training_samples": len(train),
                        "prediction_available": bool(np.isfinite(values[-1])),
                    }
                    for identity, values in risk_columns.items()
                ]
            )
        matrix = np.column_stack([columns[name] for name in baseline])
        labels = np.log(
            prefix.closes[train + fold["horizon_sessions"], 0] / prefix.closes[train, 0]
        )
        folder = root / period_id(fold)
        folder.mkdir(exist_ok=True)
        fold["forecasts"], fold["fits"] = {}, {}
        baseline_count = len(baseline)

        def fit(
            arm,
            values,
            ids,
            *,
            train=train,
            baseline_count=baseline_count,
            labels=labels,
            planned=planned,
            fold=fold,
            folder=folder,
            prefix=prefix,
        ):
            inputs = {
                "x": values[train].tolist(),
                "y": labels.tolist(),
                "prediction_x": values[-1].tolist(),
                "model": model,
                "factor_ids": ids,
                "window": planned,
            }
            fitted = cache.call(
                f"{period_id(fold)}_{arm}_fit",
                inputs,
                lambda: fit_forecast(values, labels, train, model, ids),
            )
            _write(folder / f"{arm}_model.json", fitted)
            fold["fits"][arm] = {k: v for k, v in fitted.items() if k != "state"}
            fold["fits"][arm]["training_information_cutoff"] = fold["training_targets"][-1]
            diagnostic = diagnose_forecast(values, train, fitted, baseline_count)
            diagnostic["comparison_identity"] = digest(
                canonical(
                    {
                        "baseline_factor_ids": ids[:baseline_count],
                        "training_origins": fold["training_origins"],
                        "baseline_x": values[train, :baseline_count].tolist(),
                        "prediction_baseline_x": values[-1, :baseline_count].tolist(),
                        "labels": labels.tolist(),
                    }
                )
            )
            fold.setdefault("forecast_diagnostics", {})[arm] = diagnostic
            return _forecast(fold, prefix.closes[-1, 0], fitted["predicted_log_return"])

        fold["forecasts"]["fixed"] = fit("fixed", matrix, baseline)
        fold["forecasts"]["unchanged_price"] = _forecast(fold, prefix.closes[-1, 0], 0)
        fold["forecasts"]["training_label_mean"] = _forecast(
            fold, prefix.closes[-1, 0], float(labels.mean())
        )
        news = (
            select_evidence(news_history, fold["information_cutoff"], mode=news_mode, limit=None)
            if news_history is not None
            else retained_news(snapshots, fold["information_cutoff"], data.instruments[0], 2000)
        )
        news["research_required"] = news_required
        fold["news_evidence"] = deepcopy(news)
        # The corpus is already cutoff-filtered; each numerical row is filtered again.
        row_times = [info["session_close_times"][data.dates.index(day)] for day in dates]
        native_columns, native_definitions = columns, definitions
        acquired_view = None
        if acquired_resolver is not None:
            acquired_view = acquired_resolver.view(row_times, fold["information_cutoff"], news_mode)
            columns, definitions = _merge_acquired(
                native_columns, native_definitions, acquired_view, fold, train
            )
        context = decision_context(
            data.instruments[0],
            fold,
            definitions,
            incumbent,
            news,
            _feedback(folds, fold["information_cutoff"]),
            incumbent_news,
            max_candidates,
            max_news_inputs,
            acquired_inputs=acquired_view,
        )
        context["baseline_diagnostics"] = summarize_forecast(fold["forecast_diagnostics"]["fixed"])
        _write(folder / "research_context.json", context)
        fold["news_coverage"] = news["coverage_status"]
        fold["feedback_periods" if "period_id" in fold else "feedback_months"] = [
            period_id(value) for value in context["mature_feedback"]
        ]
        fold["decision"] = {"action": "retain", "reason": "LLM disabled; fixed baseline only"}
        fold["candidate_status"] = "disabled"
        if llm is not None:
            operation = f"{period_id(fold)}_decision"
            raw = cache.call(
                operation,
                context,
                lambda context=context, operation=operation: _recorded_proposal(
                    llm, context, root, operation, candidate_revision
                ),
            )
            try:
                preview, preview_news = incumbent, incumbent_news
                for number in range(2 if candidate_revision else 1):
                    attempt = {"stage": "proposal" if number == 0 else "revision"}
                    blocked_measurements = set()
                    try:
                        decision, proposed = admit_decision(
                            raw,
                            preview,
                            news,
                            max_candidates,
                            max_news_inputs,
                            screen_individually=True,
                            acquired_evidence=(acquired_view or {}).get("evidence", ()),
                            max_data_requests=2 if acquired_resolver is not None else 0,
                        )
                        if acquired_resolver is not None and decision["data_requests"]:
                            referenced = set()
                            for value in proposed:
                                try:
                                    candidate = DiscoveryFactor.model_validate_json(
                                        canonical(value)
                                    )
                                except ValueError:
                                    continue
                                referenced.update(item.factor_id for item in candidate.inputs)
                            acquisition = acquired_resolver.acquire(
                                decision["data_requests"],
                                referenced,
                                row_times,
                                fold["information_cutoff"],
                                news_mode,
                            )
                            acquired_view = acquisition["view"]
                            blocked_measurements = set(acquisition.get("blocked_input_ids", ()))
                            columns, definitions = _merge_acquired(
                                native_columns, native_definitions, acquired_view, fold, train
                            )
                            attempt["data_acquisitions"] = deepcopy(acquisition["acquisitions"])
                            attempt["data_gaps"] = deepcopy(acquisition["gaps"])
                            fold.setdefault("data_acquisitions", []).extend(
                                deepcopy(acquisition["acquisitions"])
                            )
                            fold.setdefault("data_gaps", []).extend(deepcopy(acquisition["gaps"]))
                        proposed_news = (
                            decision["news_inputs"]
                            if decision["action"] == "replace"
                            else []
                            if decision["action"] == "withdraw"
                            else preview_news
                        )
                        if decision["action"] == "select":
                            used = {
                                item["factor_id"]
                                for candidate in proposed
                                for item in candidate["inputs"]
                            }
                            proposed_news = [
                                value for value in preview_news if value["input_id"] in used
                            ]
                        parents = {
                            DiscoveryFactor.model_validate_json(canonical(value)).recipe_ref
                            for value in preview
                        }
                        screened = screen_candidates(
                            proposed,
                            proposed_news,
                            columns,
                            definitions,
                            baseline,
                            news,
                            row_times,
                            train,
                            news_required=news_required,
                            mode=news_mode if news_history is not None else "retained",
                            coverage=news_history["searches"] if news_history is not None else None,
                            allowed_parent_refs=parents,
                            accepted_incumbent_records=preview,
                            require_evidence=decision["basis"] == "news",
                            acquired_inputs=(
                                acquired_view
                                | {
                                    "qualified_input_ids": sorted(
                                        set(acquired_view["qualified_input_ids"])
                                        - blocked_measurements
                                    )
                                }
                                if acquired_view is not None
                                else None
                            ),
                        )
                        pool = {
                            "proposed": len(proposed),
                            "accepted": len(screened["candidates"]),
                            "limit": max_candidates,
                        }
                        attempt.update(
                            decision=decision,
                            status="screened",
                            candidate_pool=pool,
                            candidates=screened["candidates"],
                            news_inputs=screened["news_inputs"],
                            candidate_rejections=screened["rejections"],
                        )
                        if candidate_revision:
                            attempt["diagnostics"] = diagnose_candidates(
                                screened["matrix"], train, screened["ids"], len(baseline)
                            )
                        if not candidate_revision or number == 1:
                            fold["candidate_rejections"] = screened["rejections"]
                            fold["candidate_pool"] = pool
                            fold["news_features"] = screened["diagnostics"]
                        if proposed and not screened["candidates"]:
                            raise ValueError(
                                "No executable candidates: "
                                + "; ".join(item["reason"] for item in screened["rejections"])
                            )
                        proposed, proposed_news = screened["candidates"], screened["news_inputs"]
                        preview, preview_news = proposed, proposed_news
                    except ValueError as exc:
                        attempt.update(status="rejected", error=str(exc))
                        if not candidate_revision or number == 1:
                            raise
                    finally:
                        if candidate_revision:
                            fold.setdefault("candidate_attempts", []).append(deepcopy(attempt))
                            _write(folder / "candidate_attempts.json", fold["candidate_attempts"])
                    if candidate_revision and number == 0:
                        revision_context = decision_context(
                            data.instruments[0],
                            fold,
                            definitions,
                            preview,
                            news,
                            context["mature_feedback"],
                            preview_news,
                            max_candidates,
                            max_news_inputs,
                            acquired_inputs=acquired_view,
                        )
                        revision_context["candidate_review"] = deepcopy(attempt)
                        revision_context["baseline_diagnostics"] = deepcopy(
                            context["baseline_diagnostics"]
                        )
                        _write(folder / "research_revision_context.json", revision_context)
                        operation = f"{period_id(fold)}_revision_decision"
                        try:
                            raw = cache.call(
                                operation,
                                revision_context,
                                lambda revision_context=revision_context, operation=operation: (
                                    _recorded_proposal(
                                        llm,
                                        revision_context,
                                        root,
                                        operation,
                                        True,
                                    )
                                ),
                            )
                        except ValueError as exc:
                            # Provider/cache failures are not rejected candidate proposals.
                            raise RuntimeError("Candidate revision operation failed") from exc
                fold["decision"] = decision
                fold["candidate_rejections"] = screened["rejections"]
                fold["candidate_pool"] = pool
                candidate, ids = screened["matrix"], screened["ids"]
                news_columns = screened["news_columns"]
                used_news = set(news_columns)
                fold["news_features"] = screened["diagnostics"]
                if proposed:
                    fold["forecasts"]["adaptive"] = fit("adaptive", candidate, ids)
                    fold["candidate_status"] = (
                        "fitted_with_rejections" if screened["rejections"] else "fitted"
                    )
                    fold["news_training"] = {
                        identity: {
                            "training_values": news_columns[identity][train].tolist(),
                            "prediction_value": float(news_columns[identity][-1]),
                            "distinct_training_values": len(
                                np.unique(news_columns[identity][train])
                            ),
                        }
                        for identity in sorted(used_news)
                    }
                else:
                    fold["forecasts"]["adaptive"] = deepcopy(fold["forecasts"]["fixed"])
                    fold["candidate_status"] = "baseline_reused"
                incumbent, incumbent_news = proposed, proposed_news
            except ValueError as exc:
                fold["candidate_status"] = "rejected"
                fold["candidate_error"] = str(exc)
                fold["decision"] = {
                    "action": "rejected",
                    "reason": "LLM proposal rejected; fixed forecast used: " + str(exc),
                    "response_checkpoint": (
                        f"checkpoints/{period_id(fold)}_revision_decision.json"
                        if candidate_revision
                        else f"checkpoints/{period_id(fold)}_decision.json"
                    ),
                }
                fold["forecasts"]["adaptive"] = deepcopy(fold["forecasts"]["fixed"])
            if "adaptive" not in fold["fits"]:
                fold["fits"]["adaptive"] = deepcopy(fold["fits"]["fixed"]) | {"reused_arm": "fixed"}
                fold["forecast_diagnostics"]["adaptive"] = deepcopy(
                    fold["forecast_diagnostics"]["fixed"]
                )
            fold["paired_forecast_decomposition"] = compare_forecast_diagnostics(
                fold["forecast_diagnostics"]["fixed"], fold["forecast_diagnostics"]["adaptive"]
            )
        fold["candidate_recipes"] = deepcopy(incumbent)
        fold["candidate_news_inputs"] = deepcopy(incumbent_news)
        if acquired_view is not None:
            fitted_ids = set(fold["fits"].get("adaptive", fold["fits"]["fixed"])["factor_ids"])
            fold["fitted_measurement_inputs"] = sorted(
                {
                    item["factor_id"]
                    for candidate in incumbent
                    if "discovery." + candidate["name"] in fitted_ids
                    for item in candidate["inputs"]
                    if item["factor_id"] in acquired_view["qualified_input_ids"]
                }
            )
        fold["candidate_recipe_role"] = (
            "Research incumbent for next decision; actual factors in fits"
        )
        frozen = canonical(fold)
        destination = folder / "prediction.json"
        if destination.exists() and destination.read_bytes() != frozen:
            cadence = "session" if "period_id" in fold else "monthly"
            raise ValueError(f"Frozen {cadence} prediction changed; start a new study")
        atomic_write(destination, frozen)
        fold["prediction_sha256"] = digest(frozen)
        fold = settle(fold, data)
        _write(folder / "settlement.json", fold)
        folds.append(fold)
        if on_fold:
            on_fold(fold)
    return folds


def aggregate(data, folds, policy):
    completed = [fold for fold in folds if fold["outcome"] == "observed"]
    kinds = {fold.get("period_kind", "calendar_month") for fold in folds}
    if len(kinds) > 1:
        raise ValueError("Cannot aggregate mixed forecast interval kinds")
    fixed_sessions = kinds == {"trading_sessions"}
    horizons = {fold["horizon_sessions"] for fold in folds} if fixed_sessions else set()
    if fixed_sessions and len(horizons) != 1:
        raise ValueError("Session forecasts must share one fixed horizon")
    metrics, accounts = {}, {}
    for arm in ("fixed", "adaptive", "unchanged_price", "training_label_mean"):
        rows = [f["forecasts"][arm] for f in completed if arm in f["forecasts"]]
        if not rows:
            continue
        metrics[arm] = price_metrics(rows) | {
            "mae_return_pp": float(
                np.mean(
                    [
                        abs(r["predicted_price"] - r["observed_price"]) / r["reference_price"] * 100
                        for r in rows
                    ]
                )
            )
        }
        if arm in {"fixed", "adaptive"}:
            accounts[arm] = (
                evaluate_session_strategy(
                    data, data.instruments[0], rows, horizon_sessions=next(iter(horizons)), **policy
                )
                if fixed_sessions
                else evaluate_monthly_strategy(data, data.instruments[0], rows, **policy)
            )
    return metrics, accounts


def run_rolling(args):
    symbol = args.symbol.strip().upper()
    root = args.output.resolve()
    train_months = getattr(args, "train_months", None)
    train_sessions = getattr(args, "train_sessions", None)
    horizon = getattr(args, "horizon_sessions", None)
    profile_path, profile_raw = getattr(args, "profile", None), None
    if train_months is not None and train_sessions is not None:
        raise ValueError("Choose --train-months or --train-sessions, not both")
    if profile_path is not None:
        profile, profile_raw = load_window_profile(
            root / "window_profile.json" if args.resume else profile_path, symbol
        )
        if train_months is None and train_sessions is None:
            train_sessions = profile.training_window_sessions
        if horizon is None:
            horizon = profile.horizon_sessions
    if train_sessions is not None and horizon is None:
        raise ValueError("--train-sessions requires --horizon-sessions or a window profile")
    if train_months is None and train_sessions is None:
        train_months = 6
    review_path = getattr(args, "price_basis_review", None)
    if review_path and getattr(args, "ohlcv_features", False):
        raise ValueError("Reviewed O/C price basis does not qualify --ohlcv-features")
    review_options = (
        {"price_basis_review": root / "price_basis_review" if args.resume else review_path}
        if review_path
        else {}
    )
    data, info = read_daily_history(symbol, args.prices_dir, **review_options)
    ohlcv_history = (
        load_ohlcv_history(symbol, args.prices_dir, data, info)
        if getattr(args, "ohlcv_features", False)
        else None
    )
    calendar = read_calendar(args.prices_dir)
    plan = (
        monthly_plan(data, calendar, args.first_month, args.last_month, train_months)
        if horizon is None
        else session_plan(
            data,
            calendar,
            args.first_month,
            args.last_month,
            train_months if train_months is not None else 6,
            horizon,
            train_sessions=train_sessions,
        )
    )
    if review_path:
        cutoff = info["price_basis_review"]["review"]["qualification_cutoff"]
        if cutoff > plan[0]["origin_session"]:
            raise ValueError("Price basis qualification cutoff exceeds first forecast origin")
    for key in ("commission_bps", "slippage_bps", "initial_cash"):
        value = Decimal(getattr(args, key))
        if not value.is_finite() or value < 0 or (key == "initial_cash" and value == 0):
            raise ValueError("Finite nonnegative costs and positive initial cash required")
    if not math.isfinite(args.threshold):
        raise ValueError("Finite threshold required")
    peers, peer_info = {}, {}
    if args.related_prices_dir:
        for peer in profile_related(symbol)["stocks"]:
            peer_data, evidence = read_daily_history(peer, args.related_prices_dir / peer)
            peers[peer] = aligned_closes(
                data.dates, info["session_close_times"], peer_data, evidence
            )
            peer_info[peer] = evidence
    max_candidates = getattr(args, "max_candidates", 2)
    candidate_revision = getattr(args, "candidate_revision", False)
    if candidate_revision and args.llm == "none":
        raise ValueError("--candidate-revision requires --llm codex or api")
    if not 1 <= max_candidates <= 32:
        raise ValueError("--max-candidates must be between 1 and 32")
    max_news_inputs = getattr(args, "max_news_inputs", None)
    max_news_inputs = 2 * max_candidates if max_news_inputs is None else max_news_inputs
    if not 1 <= max_news_inputs <= 64:
        raise ValueError("--max-news-inputs must be between 1 and 64")
    news_required = getattr(args, "news_research", False)
    if news_required and args.llm == "none":
        raise ValueError("--news-research requires --llm codex or api")
    if getattr(args, "news_query", None) and not news_required:
        raise ValueError("--news-query requires --news-research")
    history_path = getattr(args, "news_history", None)
    frozen_history = None
    if history_path:
        if not news_required or args.news_archive:
            raise ValueError("--news-history requires --news-research and excludes --news-archive")
        frozen_history = load_history(
            history_path,
            symbol,
            plan[0]["training_window"]["start"],
            plan[-1]["origin_session"],
            query=getattr(args, "news_query", None),
        )
    snapshots = load_archive(args.news_archive) if args.news_archive else []
    risk_path = getattr(args, "risk_history", None)
    risk_history = (
        load_risk_history(root / "risk_history" if args.resume else risk_path)
        if risk_path
        else None
    )
    gpr_path = getattr(args, "gpr_history", None)
    gpr_history = (
        load_gpr_history(root / "gpr_history" if args.resume else gpr_path) if gpr_path else None
    )
    acquired_paths = getattr(args, "acquired_history", None) or []
    max_extractions = getattr(args, "max_extractions", 6)
    if type(max_extractions) is not int or not 0 <= max_extractions <= 6:
        raise ValueError("--max-extractions must be an integer in 0..6")
    retained_acquired = [
        root / "acquired_history" / f"{i:03d}.json" for i in range(len(acquired_paths))
    ]
    acquired_history = (
        load_acquired_history(retained_acquired if args.resume else acquired_paths)
        if acquired_paths
        else None
    )
    config = {
        k: str(v)
        if isinstance(v, Path)
        else [str(path) for path in v]
        if k == "acquired_history"
        else v
        for k, v in vars(args).items()
        if k not in {"resume", "command"}
        and not (k == "price_basis_review" and v is None)
        and not (k == "horizon_sessions" and v is None)
        and not (k in {"profile", "train_sessions"} and v is None)
        and not (k == "gpr_history" and v is None)
        and not (k == "ohlcv_features" and not v)
        and not (k in {"acquired_history", "max_extractions"} and not acquired_paths)
    }
    config.update(
        symbol=symbol,
        effective_max_news_inputs=max_news_inputs,
        output=str(root),
        factor_root=str(Path(__file__).resolve().parents[3] / "instruments"),
    )
    if train_sessions is not None:
        config.pop("train_months", None)
        config["train_sessions"] = train_sessions
    else:
        config["train_months"] = train_months
    if horizon is not None:
        config["horizon_sessions"] = horizon
    config["input_identity"] = {
        "target": info,
        "peers": peer_info,
        "news_sha256": digest(canonical(snapshots)),
        "history_sha256": digest(canonical(frozen_history)) if frozen_history is not None else None,
        "risk_sha256": digest(canonical(risk_history)) if risk_history is not None else None,
    }
    if profile_raw is not None:
        config["profile"] = str(profile_path.resolve())
        config["input_identity"]["window_profile_sha256"] = digest(profile_raw)
    if gpr_history is not None:
        config["input_identity"]["gpr_sha256"] = digest(canonical(gpr_history))
    if ohlcv_history is not None:
        config["input_identity"]["ohlcv"] = ohlcv_history["provenance"]
    if acquired_history is not None:
        config["input_identity"]["acquired_sha256"] = digest(canonical(acquired_history))
    config["estimator_parameters"] = (
        MODEL_PARAMETERS | {"seed": 7, "num_boost_round": BOOST_ROUNDS}
        if args.model == "lightgbm"
        else {"alpha": 1.0}
    )
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {
        "kind": "MonthlyRollingStudy" if horizon is None else "SessionRollingStudy",
        "symbol": symbol,
        "configuration": config,
        "notice": (NOTICE if horizon is None else SESSION_NOTICE.format(horizon=horizon)).replace(
            "six-month",
            f"{train_sessions}-trading-session"
            if train_sessions is not None
            else f"{train_months}-month",
        ),
        "price_basis_notice": args.price_note
        + (
            " "
            + info["price_basis_interpretation"]
            + " Basis: split-adjusted, excluding distributions. "
            + " ".join(info["price_basis_review"]["limitations"])
            if review_path
            else ""
        ),
        "folds": [],
        "metrics": {},
        "accounts": {},
    }
    runtime = {
        "status": "running",
        "counts": {},
        "token_usage": {},
        "previous_attempts": [],
        "total_elapsed_seconds": 0,
    }
    from quant_rsi.research.rolling_report import render_rolling

    with run_lock(root):
        cache = StudyCache(root, config, resume=args.resume)
        if profile_raw is not None and not args.resume:
            atomic_write(root / "window_profile.json", profile_raw)
        if review_path and not args.resume:
            destination = root / "price_basis_review"
            for name in ["review.json"] + [
                source["path"] for source in info["price_basis_review"]["review"]["sources"]
            ]:
                retained = destination / name
                retained.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(review_path / name, retained)
            _, retained_info = read_daily_history(
                symbol, args.prices_dir, price_basis_review=destination
            )
            if retained_info != info:
                raise ValueError("Price basis evidence changed during retention; start a new study")
        if acquired_history is not None and not args.resume:
            (root / "acquired_history").mkdir()
            for source, destination in zip(acquired_paths, retained_acquired, strict=True):
                shutil.copyfile(source, destination)
            if canonical(load_acquired_history(retained_acquired)) != canonical(acquired_history):
                raise ValueError("Acquired sources changed during retention; start a new study")
        for name, source_path, history, loader in (
            ("risk_history", risk_path, risk_history, load_risk_history),
            ("gpr_history", gpr_path, gpr_history, load_gpr_history),
        ):
            if history is None or args.resume:
                continue
            destination = root / name
            (destination / "raw").mkdir(parents=True)
            shutil.copyfile(source_path / "receipts.json", destination / "receipts.json")
            for source in history["sources"]:
                relative = Path("raw") / source["file"]
                shutil.copyfile(source_path / relative, destination / relative)
            retained_risk = loader(destination)
            if canonical(retained_risk) != canonical(history):
                raise ValueError("Risk snapshot changed during retention; start a new study")
        runtime["previous_attempts"] = cache.previous
        _write(root / "config.json", config)
        _write(root / ("monthly_plan.json" if horizon is None else "session_plan.json"), plan)
        if ohlcv_history is not None:
            _write(
                root / "ohlcv_inputs.json",
                {key: ohlcv_history[key] for key in ("definitions", "provenance", "coverage")},
            )
        _write(root / "runtime.json", runtime)

        def usage(value):
            with (root / "provider_usage.jsonl").open("ab") as stream:
                stream.write(canonical(value) + b"\n")

        def completed(fold):
            result["folds"].append(fold)
            print(
                f"[quant-rsi] {period_id(fold)}: {fold['outcome']}; "
                f"candidate {fold['candidate_status']}",
                flush=True,
            )

        try:
            llm = (
                None
                if args.llm == "none"
                else LLMHypothesizer(args.llm, args.llm_model, args.llm_timeout, on_usage=usage)
            )
            acquired_resolver = (
                AcquiredResolver(
                    acquired_history, llm, cache, root, max_extractions=max_extractions
                )
                if acquired_history is not None
                else None
            )
            news_history = None
            if news_required and not snapshots:
                start = plan[0]["training_window"]["start"]
                end = plan[-1]["origin_session"]
                request = {
                    "symbol": symbol,
                    "start": start,
                    "end": end,
                    "query": getattr(args, "news_query", None),
                }

                def acquire_history():
                    if frozen_history is None:
                        return collect_history(
                            symbol, start, end, root / "news_history", query=request["query"]
                        )
                    destination = root / "news_history"
                    destination.mkdir(exist_ok=True)
                    for search in frozen_history["searches"]:
                        for relative in (
                            Path(search["raw_path"]),
                            Path(f"{search['start']}_{search['end']}") / "success.json",
                        ):
                            target = destination / relative
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(history_path / relative, target)
                    _write(destination / "news_history.json", frozen_history)
                    return load_history(destination, symbol, start, end, query=request["query"])

                print("[quant-rsi] loading historical news", flush=True)
                news_history = cache.call(
                    "news_history",
                    request,
                    acquire_history,
                )
            run_months(
                data,
                info,
                peers,
                plan,
                cache,
                root,
                model=args.model,
                llm=llm,
                snapshots=snapshots,
                on_fold=completed,
                news_history=news_history,
                news_mode=getattr(args, "news_information_mode", "reconstructed"),
                news_required=news_required,
                max_candidates=max_candidates,
                max_news_inputs=max_news_inputs,
                risk_history=risk_history,
                gpr_history=gpr_history,
                candidate_revision=candidate_revision,
                ohlcv_history=ohlcv_history,
                acquired_resolver=acquired_resolver,
            )
            policy = {
                k: getattr(args, k)
                for k in ("commission_bps", "slippage_bps", "initial_cash", "threshold")
            }
            result["metrics"], result["accounts"] = aggregate(data, result["folds"], policy)
            runtime["status"] = result["status"] = "completed"
        except Exception as exc:
            runtime["status"] = result["status"] = "failed"
            result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            _write(root / "failure.json", result["failure"])
            raise
        finally:
            runtime["total_elapsed_seconds"] = time.monotonic() - started
            runtime["counts"] = {
                "completed_folds": len(result["folds"]),
                "retained_fit_operations": len(list(cache.root.glob("*_fit.json"))),
                "retained_decision_operations": len(list(cache.root.glob("*_decision.json"))),
            }
            if candidate_revision:
                calls = root / "research_calls.jsonl"
                events = (
                    [json.loads(line) for line in calls.read_bytes().splitlines()]
                    if calls.exists()
                    else []
                )
                runtime["counts"]["research_provider_attempts_total"] = sum(
                    event["status"] == "attempted" for event in events
                )
                runtime["counts"]["research_provider_failures_total"] = sum(
                    event["status"] == "failed" for event in events
                )
            source_usage = root / "acquired_source_usage.json"
            if acquired_history is not None and source_usage.exists():
                receipt = json.loads(source_usage.read_bytes())
                if receipt["sha256"] != digest(canonical(receipt["usage"])):
                    raise ValueError("Measurement source usage receipt changed")
                usage_record = receipt["usage"]
                runtime["counts"]["measurement_extraction_attempts_total"] = usage_record["counts"][
                    "extract"
                ]
                runtime["counts"]["measurement_extraction_failures_total"] = sum(
                    event["kind"] == "extract" and event["status"] == "failed"
                    for event in usage_record["operations"]
                )
                result["measurement_usage"] = usage_record
            _write(root / "runtime.json", runtime)
            _write(root / "result.json", result)
            atomic_write(root / "report.html", render_rolling(result).encode())
    return result
