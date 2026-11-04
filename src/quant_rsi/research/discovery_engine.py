"""Adapt discovered columns to the existing study model and strategy evaluator."""

from copy import deepcopy

import numpy as np

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research.demo_features import price_feature
from quant_rsi.research.demo_strategy import evaluate_strategy
from quant_rsi.research.discovery_contracts import guard_numerical
from quant_rsi.research.discovery_factors import (
    DiscoveryFactor,
    adapter_catalog,
    compute_factor,
    execute_survey,
)
from quant_rsi.research.study_incremental import freeze_plan


def serial_column(values):
    return [float(value) if np.isfinite(value) else None for value in values]


class DiscoveryExperimentRunner:
    """One callback means one fit; candidate preparation never mutates sample indices."""

    def __init__(self, model, args, cutoffs, emit):
        self.model, self.args, self.cutoffs, self.emit = model, args, cutoffs, emit
        self.plan = freeze_plan(model)
        self.baseline = [
            deepcopy(d)
            for d in model.library["definitions"]
            if d["factor_id"] in self.plan["baseline_ids"]
        ]
        self.definitions, self.columns = [], {}
        for field in ("open", "close"):
            identity = "price." + field
            self.columns[identity] = getattr(model.data, field + "s")[:, 0]
            self.definitions.append(
                {
                    "factor_id": identity,
                    "description": "Retained target " + field,
                    "kind": "numeric",
                    "unit": "price",
                    "runnable": True,
                    "recipe": {"op": "retained_price", "field": field},
                }
            )
        for bundle in (
            model.financial_features,
            model.external_features,
            model.peer_features,
            model.profile_features,
        ):
            if bundle:
                for definition in bundle.get("definitions", []):
                    identity = definition["factor_id"]
                    if identity in bundle.get("columns", {}):
                        self.definitions.append(deepcopy(definition))
                        self.columns[identity] = np.asarray(
                            bundle["columns"][identity], dtype=float
                        )
        for definition in adapter_catalog(available_inputs=self.columns):
            self.definitions.append(definition)
            recipe = definition["recipe"]
            self.columns[definition["factor_id"]] = execute_survey(
                recipe["survey_id"],
                self.columns,
                window=recipe["window"] or 10,
            )
        for definition in model.library["definitions"]:
            if definition["factor_id"] in self.columns:
                continue
            if definition.get("recipe", {}).get("op") in {
                "momentum",
                "mean_return",
                "log_return_volatility",
                "qlib_ma_v1",
                "qlib_roc_v1",
            }:
                self.definitions.append(deepcopy(definition))
                self.columns[definition["factor_id"]] = price_feature(
                    model.data, definition["recipe"]
                )
        self.attempts = []
        self.evidence_ids = set()

    def inputs(self, bundles):
        definitions, columns = deepcopy(self.definitions), dict(self.columns)
        from quant_rsi.research.discovery_sources import align_input_bundle

        for bundle in bundles:
            expected = align_input_bundle(bundle["request"], bundle["observations"], self.cutoffs)
            for key in ("identity", "columns", "input_evidence", "definitions"):
                if bundle.get(key) != expected[key]:
                    raise ValueError("Acquired input differs from its source-bound alignment")
            for identity, rows in bundle["input_evidence"].items():
                finite = [i for i, v in enumerate(bundle["columns"][identity]) if v is not None]
                report = guard_numerical(
                    prediction_times=[self.cutoffs[i] for i in finite],
                    available_times=[rows[i]["effective_available_at"] for i in finite],
                )
                if report["status"] != "pass":
                    raise ValueError("Acquired values violate per-row availability")
            for definition in bundle.get("definitions", []):
                key = definition["factor_id"]
                if key in columns:
                    raise ValueError("Acquired input cannot replace an admitted definition")
                values = np.asarray(bundle["columns"][key], dtype=float)
                if values.shape != (len(self.model.data.dates),):
                    raise ValueError("Acquired input has a different calendar")
                definitions.append(deepcopy(definition))
                columns[key] = values
        return definitions, columns

    def prepare(self, candidate, bundles):
        proposal = DiscoveryFactor.model_validate_json(canonical(candidate))
        definitions, columns = self.inputs(bundles)
        definition = proposal.definition(definitions)
        if not definition["runnable"]:
            return definition, None, definitions
        values = compute_factor(definition, columns, definitions)
        return definition, values, definitions

    def validate(self, candidate, bundles):
        try:
            evidence_ids = candidate.get("evidence_ids", [])
            if not evidence_ids or not set(evidence_ids) <= self.evidence_ids:
                return {"status": "rejected", "reason": "candidate_requires_admitted_evidence"}
            definition, values, _ = self.prepare(candidate, bundles)
            required = np.concatenate((self.model.train, self.model.origins))
            if values is None:
                return {"status": "needs_data", "definition": definition}
            missing = [self.model.data.dates[i] for i in required if not np.isfinite(values[i])]
            if missing:
                return {
                    "status": "needs_data",
                    "reason": "incomplete_frozen_population",
                    "missing_sessions": missing,
                    "definition": definition,
                }
            return {
                "status": "ready",
                "recipe_ref": DiscoveryFactor.model_validate_json(canonical(candidate)).recipe_ref,
                "definition": definition,
                "distinct_training_values": len(np.unique(values[self.model.train])),
            }
        except (ValueError, TypeError, KeyError) as exc:
            return {"status": "rejected", "reason": str(exc)}

    def __call__(self, spec):
        number = len(self.model._fits) + 1
        self._fit_started = False
        try:
            return self._execute(spec)
        except Exception as exc:
            cache = self.model.fit_cache
            receipt = cache.root / f"fit_{number}.json" if cache is not None else None
            # A successful fit receipt survives reporting/handoff failure. Its logical
            # fit is charged when replay reconstructs that result, not a second time here.
            completed = receipt is not None and receipt.is_file()
            exc.fit_count = int(self._fit_started and not completed)
            exc.fit_accounting = (
                "completed_fit_receipt"
                if completed
                else "attempted_fit_without_receipt"
                if self._fit_started
                else "before_fit"
            )
            raise

    def _execute(self, spec):
        model, args = self.model, self.args
        factors = deepcopy(self.baseline)
        actual = None
        if spec["kind"] == "candidate":
            diagnostic = self.validate(spec["candidate"], spec["input_bundles"])
            if diagnostic["status"] != "ready":
                return diagnostic | {"fit_count": 0}
            definition, values, definitions = self.prepare(spec["candidate"], spec["input_bundles"])
            dependencies = [
                d for d in definitions if d["factor_id"] in definition["input_factor_ids"]
            ]
            model.configure_discovery(
                {
                    "definitions": [definition],
                    "columns": {definition["factor_id"]: serial_column(values)},
                    "inputs": dependencies,
                }
            )
            factors.append(definition)
            actual = {
                "definition": definition,
                "input_definitions": dependencies,
                "training_values": serial_column(values[model.train]),
                "development_values": serial_column(values[model.origins]),
                "distinct_training_values": len(np.unique(values[model.train])),
                "input_bundle_refs": [b.get("identity") for b in spec["input_bundles"]],
            }
        # Existing price/financial adapters enforce per-row source eligibility.
        # Acquired columns are reconstructed and checked against source evidence in inputs().
        guard = guard_numerical(
            prediction_times=[],
            available_times=[],
            label_times=[self.cutoffs[i + model.horizon] for i in model.train],
            fit_time=self.cutoffs[int(model.origins[0])],
        )
        if guard["status"] != "pass":
            return {"status": "rejected", "fit_count": 0, "guard": guard}
        number = len(model._fits) + 1
        hypothesis = {
            "provider": "discovery",
            "hypothesis": spec["candidate"]["description"]
            if spec["candidate"]
            else "Fixed matched baseline",
            "factor_ids": [d["factor_id"] for d in factors],
            "evidence_news_ids": (spec["candidate"] or {}).get("evidence_ids", []),
            "event_signal": 0.0,
        }
        self.emit(f"discovery_fit_{number}_factor_library", {"definitions": factors})
        self.emit(f"discovery_fit_{number}", None)
        self._fit_started = True
        prediction = model.predict({}, factors, hypothesis)
        prediction["strategy"] = evaluate_strategy(
            model.data,
            args.symbol.upper(),
            prediction["development"]["rows"],
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
            initial_cash=args.initial_cash,
            threshold=args.threshold,
            horizon_sessions=model.horizon,
        )
        metric = prediction["development"]["metrics"]["mae_price"]
        if args.selection_metric == "sharpe":
            sharpe = prediction["strategy"]["strategy"]["metrics"]["sharpe_ratio"]
            if sharpe is None:
                metric = None
            else:
                metric = -sharpe
        attempt = {
            "round": number,
            "hypothesis": hypothesis,
            "factor_ids": hypothesis["factor_ids"],
            "prediction": prediction,
            "development_metrics": prediction["development"]["metrics"],
            "accepted": False,
            "decision": None,
            "comparison_key": spec["comparison_key"],
        }
        self.attempts.append(attempt)
        self.emit(f"discovery_fit_{number}", {"prediction": prediction, "actual_features": actual})
        return {
            "status": "completed",
            "fit_count": 1,
            "round": number,
            "metric": metric,
            "prediction": {
                key: value
                for key, value in prediction.items()
                if key not in {"latest_forecast", "latest_forecast_unavailable"}
            },
            "actual_features": actual,
            "guard": guard,
            "fit_identity": digest(
                canonical(
                    {
                        "configuration": model.model_configuration,
                        "factors": factors,
                        "train": model.train.tolist(),
                        "development": model.origins.tolist(),
                        "inputs": [b.get("identity") for b in spec["input_bundles"]],
                        "matrix": [serial_column(row) for row in model._fits[-1][1]],
                        "source": model.price_info,
                    }
                )
            ),
        }
