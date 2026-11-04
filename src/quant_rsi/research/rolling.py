"""Fixed-width monthly rolling forecasts through existing numerical/provider adapters."""

import math
import time
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import numpy as np

from quant_rsi.core.artifacts import atomic_write, canonical, digest
from quant_rsi.core.data import MarketData
from quant_rsi.forecast.forecast_protocol import BOOST_ROUNDS, MODEL_PARAMETERS
from quant_rsi.market.market_contracts import timestamp
from quant_rsi.research.demo_features import price_metrics
from quant_rsi.research.demo_prices import read_daily_history
from quant_rsi.research.demo_related import profile_related
from quant_rsi.research.demo_strategy import evaluate_monthly_strategy
from quant_rsi.research.llm import LLMHypothesizer
from quant_rsi.research.rolling_research import (
    admit_decision,
    candidate_matrix,
    decision_context,
    feature_library,
    propose,
)
from quant_rsi.research.rolling_windows import monthly_plan, read_calendar
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


def add_arguments(commands):
    parser = commands.add_parser(
        "research-rolling", help="six-month training / next-month forecasts"
    )
    parser.add_argument("symbol")
    parser.add_argument("--prices-dir", type=Path, required=True)
    parser.add_argument("--related-prices-dir", type=Path)
    parser.add_argument("--first-month", required=True, help="first forecast month, YYYY-MM")
    parser.add_argument("--last-month", required=True, help="last forecast month, YYYY-MM")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--model", choices=("lightgbm", "ridge"), default="lightgbm")
    parser.add_argument("--llm", choices=("none", "codex", "api"), default="none")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-timeout", type=float, default=180)
    parser.add_argument("--news-archive", type=Path)
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
    else:
        import lightgbm as lgb

        estimator = lgb.train(
            MODEL_PARAMETERS | {"seed": 7},
            lgb.Dataset(x, label=labels),
            num_boost_round=BOOST_ROUNDS,
        )
        prediction = float(estimator.predict(matrix[-1:], num_threads=1)[0])
        state = {"model_string": estimator.model_to_string()}
    if not math.isfinite(prediction):
        raise ValueError("Non-finite model forecast")
    return {
        "predicted_log_return": prediction,
        "factor_ids": factor_ids,
        "training_samples": len(train),
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
        result.append(
            {
                "month": fold["month"],
                "prediction_sha256": fold["prediction_sha256"],
                "target_session": fold["target_session"],
                "decision": fold["decision"],
                "forecasts": deepcopy(fold["forecasts"]),
                "errors": {
                    arm: row["predicted_price"] - row["observed_price"]
                    for arm, row in fold["forecasts"].items()
                },
            }
        )
    return result


def run_months(
    data, info, peers, plan, cache, root, *, model="lightgbm", llm=None, snapshots=(), on_fold=None
):
    """Cut off numerical views before proposing/fitting; settlement cannot enter those views."""
    folds, incumbent = [], []
    for planned in plan:
        fold = deepcopy(planned)
        dates = fold.pop("history_sessions")
        train = np.asarray(fold.pop("training_indices"))
        prefix = _view(data, dates)
        start, end = data.dates.index(dates[0]), data.dates.index(dates[-1]) + 1
        peer_views = {symbol: values[start:end].copy() for symbol, values in peers.items()}
        columns, definitions, baseline = feature_library(prefix, peer_views)
        matrix = np.column_stack([columns[name] for name in baseline])
        labels = np.log(
            prefix.closes[train + fold["horizon_sessions"], 0] / prefix.closes[train, 0]
        )
        folder = root / fold["month"]
        folder.mkdir(exist_ok=True)
        fold["forecasts"], fold["fits"] = {}, {}

        def fit(
            arm,
            values,
            ids,
            *,
            train=train,
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
                f"{fold['month']}_{arm}_fit",
                inputs,
                lambda: fit_forecast(values, labels, train, model, ids),
            )
            _write(folder / f"{arm}_model.json", fitted)
            fold["fits"][arm] = {k: v for k, v in fitted.items() if k != "state"}
            fold["fits"][arm]["training_information_cutoff"] = fold["training_targets"][-1]
            return _forecast(fold, prefix.closes[-1, 0], fitted["predicted_log_return"])

        fold["forecasts"]["fixed"] = fit("fixed", matrix, baseline)
        fold["forecasts"]["unchanged_price"] = _forecast(fold, prefix.closes[-1, 0], 0)
        fold["forecasts"]["training_label_mean"] = _forecast(
            fold, prefix.closes[-1, 0], float(labels.mean())
        )
        news = retained_news(snapshots, fold["information_cutoff"], data.instruments[0], 20)
        context = decision_context(
            data.instruments[0],
            fold,
            definitions,
            incumbent,
            news,
            _feedback(folds, fold["information_cutoff"]),
        )
        _write(folder / "research_context.json", context)
        fold["news_coverage"] = news["coverage_status"]
        fold["feedback_months"] = [value["month"] for value in context["mature_feedback"]]
        fold["decision"] = {"action": "retain", "reason": "LLM disabled; fixed baseline only"}
        fold["candidate_status"] = "disabled"
        if llm is not None:
            raw = cache.call(
                f"{fold['month']}_decision", context, lambda context=context: propose(llm, context)
            )
            try:
                decision, proposed = admit_decision(raw, incumbent, news)
                fold["decision"] = decision
                candidate, ids = candidate_matrix(
                    proposed, columns, definitions, baseline, np.append(train, len(dates) - 1)
                )
                if proposed:
                    fold["forecasts"]["adaptive"] = fit("adaptive", candidate, ids)
                    fold["candidate_status"] = "fitted"
                else:
                    fold["forecasts"]["adaptive"] = deepcopy(fold["forecasts"]["fixed"])
                    fold["candidate_status"] = "baseline_reused"
                incumbent = proposed
            except ValueError as exc:
                fold["candidate_status"] = "rejected"
                fold["candidate_error"] = str(exc)
                fold["decision"] = {
                    "action": "rejected",
                    "reason": "LLM proposal rejected; fixed forecast used: " + str(exc),
                    "response_checkpoint": f"checkpoints/{fold['month']}_decision.json",
                }
                fold["forecasts"]["adaptive"] = deepcopy(fold["forecasts"]["fixed"])
            if "adaptive" not in fold["fits"]:
                fold["fits"]["adaptive"] = deepcopy(fold["fits"]["fixed"]) | {"reused_arm": "fixed"}
        fold["candidate_recipes"] = deepcopy(incumbent)
        fold["candidate_recipe_role"] = (
            "Research incumbent for next decision; actual factors in fits"
        )
        frozen = canonical(fold)
        destination = folder / "prediction.json"
        if destination.exists() and destination.read_bytes() != frozen:
            raise ValueError("Frozen monthly prediction changed; start a new study")
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
            accounts[arm] = evaluate_monthly_strategy(data, data.instruments[0], rows, **policy)
    return metrics, accounts


def run_rolling(args):
    symbol = args.symbol.strip().upper()
    data, info = read_daily_history(symbol, args.prices_dir)
    calendar = read_calendar(args.prices_dir)
    plan = monthly_plan(data, calendar, args.first_month, args.last_month, args.train_months)
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
    snapshots = load_archive(args.news_archive) if args.news_archive else []
    root = args.output.resolve()
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in {"resume", "command"}
    }
    config.update(
        symbol=symbol,
        output=str(root),
        factor_root=str(Path(__file__).resolve().parents[3] / "instruments"),
    )
    config["input_identity"] = {
        "target": info,
        "peers": peer_info,
        "news_sha256": digest(canonical(snapshots)),
    }
    config["estimator_parameters"] = (
        MODEL_PARAMETERS | {"seed": 7, "num_boost_round": BOOST_ROUNDS}
        if args.model == "lightgbm"
        else {"alpha": 1.0}
    )
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {
        "kind": "MonthlyRollingStudy",
        "symbol": symbol,
        "configuration": config,
        "notice": NOTICE.replace("six-month", f"{args.train_months}-month"),
        "price_basis_notice": args.price_note,
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
        runtime["previous_attempts"] = cache.previous
        _write(root / "config.json", config)
        _write(root / "monthly_plan.json", plan)
        _write(root / "runtime.json", runtime)

        def usage(value):
            with (root / "provider_usage.jsonl").open("ab") as stream:
                stream.write(canonical(value) + b"\n")

        def completed(fold):
            result["folds"].append(fold)
            print(
                f"[quant-rsi] {fold['month']}: {fold['outcome']}; "
                f"candidate {fold['candidate_status']}",
                flush=True,
            )

        try:
            llm = (
                None
                if args.llm == "none"
                else LLMHypothesizer(args.llm, args.llm_model, args.llm_timeout, on_usage=usage)
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
            _write(root / "runtime.json", runtime)
            _write(root / "result.json", result)
            atomic_write(root / "report.html", render_rolling(result).encode())
    return result
