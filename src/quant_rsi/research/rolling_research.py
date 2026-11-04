"""Small causal feature/proposal adapter for the monthly rolling task."""

from typing import Literal

import numpy as np
from pydantic import Field

from quant_rsi.core.artifacts import canonical
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.research.demo_features import price_feature
from quant_rsi.research.discovery_factors import (
    DiscoveryFactor,
    adapter_catalog,
    compute_factor,
    execute_survey,
    operator_descriptions,
    retrieve_candidates,
    survey_catalog,
)
from quant_rsi.research.discovery_sources import strict_schema
from quant_rsi.research.study_peers import peer_feature


class MonthlyDecision(StrictModel):
    action: Literal["retain", "replace", "withdraw"]
    basis: Literal["price_only", "news"]
    reason: str = Field(min_length=1, max_length=3000)
    candidates: tuple[DiscoveryFactor, ...] = Field(max_length=2)


def feature_library(data, peers):
    columns = {"price.open": data.opens[:, 0], "price.close": data.closes[:, 0]}
    definitions = [
        {"factor_id": key, "unit": "price", "kind": "numeric", "runnable": True} for key in columns
    ]
    baseline = []
    for op in ("momentum", "log_return_volatility"):
        identity = f"research.{op}_10"
        columns[identity] = price_feature(data, {"op": op, "window": 10})
        definitions.append(
            {
                "factor_id": identity,
                "unit": "dimensionless",
                "kind": "numeric",
                "runnable": True,
                "recipe": {"op": op, "window": 10},
            }
        )
        baseline.append(identity)
    for symbol, values in sorted(peers.items()):
        for op in ("peer_return", "relative_return"):
            identity = f"peer.{symbol}.{op}_10"
            columns[identity] = peer_feature(data.closes[:, 0], values, op, 10)
            definitions.append(
                {
                    "factor_id": identity,
                    "unit": "dimensionless",
                    "kind": "numeric",
                    "runnable": True,
                    "recipe": {"op": op, "symbol": symbol, "window": 10},
                }
            )
            if op == "relative_return":
                baseline.append(identity)
    for definition in adapter_catalog(available_inputs=columns):
        recipe = definition["recipe"]
        columns[definition["factor_id"]] = execute_survey(
            recipe["survey_id"], columns, window=recipe["window"] or 10
        )
        definitions.append(definition)
    return columns, definitions, baseline


def decision_context(symbol, fold, definitions, incumbent, news, feedback):
    return {
        "symbol": symbol,
        "information_cutoff": fold["information_cutoff"],
        "training_window": fold["training_window"],
        "forecast_target": fold["target_session"],
        "horizon_sessions": fold["horizon_sessions"],
        "definitions": definitions,
        "operators": operator_descriptions(),
        "experimental_survey": retrieve_candidates(
            "price momentum volatility relative sector market",
            survey_catalog(available_inputs=[d["factor_id"] for d in definitions]),
        ),
        "incumbent": incumbent,
        "incumbent_recipe_refs": {
            value["name"]: DiscoveryFactor.model_validate_json(canonical(value)).recipe_ref
            for value in incumbent
        },
        "news": news,
        "mature_feedback": feedback,
    }


def propose(llm, context):
    prompt = (
        "Make one monthly rolling research decision, before the target month's outcome. "
        "The fixed baseline always runs separately. The adaptive arm may retain previous "
        "candidate expressions, withdraw them, or replace them with at most two expressions "
        "appended to baseline features. This is a hypothesis, not a claim of improvement. "
        "Only supplied numerical inputs can execute; do not invent financial/news measurements. "
        "Inspect experimental survey definitions and prior mature monthly feedback. "
        "Price-only hypotheses are allowed and must use basis price_only, without pretending "
        "they came from news. News hypotheses need actual eligible article IDs on each "
        "candidate. No current search, outside knowledge of later events, or future outcomes. "
        "Trailing warm-up starts at the six-month window boundary; recipes requiring too much "
        "history will be unavailable. No dropping samples. Candidates for retain/withdraw must "
        "be empty. Parent recipe refs may name only the supplied incumbent recipes. "
        "Return only schema JSON.\nINPUT_JSON:\n" + canonical(context).decode()
    )
    return {"text": llm.structured(prompt, strict_schema(MonthlyDecision))}


def admit_decision(raw, incumbent, news):
    decision = MonthlyDecision.model_validate_json(raw["text"])
    if decision.action != "replace" and decision.candidates:
        raise ValueError("Only replacement decisions may contain candidates")
    if decision.action == "replace" and not decision.candidates:
        raise ValueError("Replacement requires a candidate")
    evidence = {item["id"] for item in news["items"]}
    previous = {DiscoveryFactor.model_validate_json(canonical(c)).recipe_ref for c in incumbent}
    for candidate in decision.candidates:
        if not set(candidate.evidence_ids) <= evidence:
            raise ValueError("Candidate cites unknown or cutoff-ineligible evidence")
        if decision.basis == "news" and not candidate.evidence_ids:
            raise ValueError("News hypothesis requires cutoff-eligible evidence")
        if not set(candidate.parent_recipe_refs) <= previous:
            raise ValueError("Candidate cites unknown parent recipe")
    candidates = (
        [c.model_dump(mode="json") for c in decision.candidates]
        if decision.action == "replace"
        else []
        if decision.action == "withdraw"
        else incumbent
    )
    return decision.model_dump(mode="json"), candidates


def candidate_matrix(candidates, columns, definitions, baseline, required):
    matrix = [columns[name] for name in baseline]
    identities = list(baseline)
    refs = set()
    for value in candidates:
        candidate = DiscoveryFactor.model_validate_json(canonical(value))
        if candidate.recipe_ref in refs:
            raise ValueError("Duplicate candidate recipe")
        refs.add(candidate.recipe_ref)
        definition = candidate.definition(definitions)
        values = compute_factor(definition, columns, definitions)
        if not np.isfinite(values[required]).all():
            raise ValueError("Candidate unavailable on the fixed training/prediction sample")
        if any(np.array_equal(values[required], col[required]) for col in matrix):
            raise ValueError("Candidate duplicates an existing model column")
        if definition["factor_id"] in identities:
            raise ValueError("Duplicate candidate name")
        matrix.append(values)
        identities.append(definition["factor_id"])
    return np.column_stack(matrix), identities
