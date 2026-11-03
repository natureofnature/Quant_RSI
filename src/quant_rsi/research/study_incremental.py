"""Fixed controls and bounded business increments for the existing study loop."""

from copy import deepcopy

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research.demo_proposals import FactorProposal, extend_library


def freeze_plan(model):
    """Admit only inputs already usable on every frozen training/development row."""
    baseline = []
    for op in ("momentum", "log_return_volatility"):
        definition = FactorProposal(
            op=op, window=10, description="Fixed price control."
        ).definition()
        if not any(d["factor_id"] == definition["factor_id"] for d in model.library["definitions"]):
            model.library["definitions"].append(definition)
        baseline.append(definition["factor_id"])
    peers = sorted(
        d["factor_id"]
        for d in model.library["definitions"]
        if d.get("runnable")
        and d.get("recipe", {}).get("op") == "relative_return"
        and d["recipe"]["window"] == 10
    )
    baseline += peers[:1]
    domain, gaps = [], []
    for d in model.library["definitions"]:
        if d.get("recipe", {}).get("op") not in {"financial", "external"}:
            continue
        if d.get("runnable"):
            domain.append(d["factor_id"])
        else:
            gaps.append(
                {
                    "factor_id": d["factor_id"],
                    "reason": d.get("availability_reason", "unavailable on frozen population"),
                }
            )
    plan = {
        "version": 1,
        "baseline_ids": baseline,
        "baseline_definitions": [
            deepcopy(next(d for d in model.library["definitions"] if d["factor_id"] == key))
            for key in baseline
        ],
        "candidate_input_ids": sorted(domain),
        "gaps": gaps,
        "max_incremental_features": 2,
        "common_population": (
            "all frozen training/development rows; no candidate-specific dropping or filling"
        ),
        "baseline_note": (
            "Price trend/volatility and at most one available configured "
            "peer-relative return; a peer is not a sector benchmark."
        ),
    }
    model.library["incremental_plan"] = plan
    return plan


def baseline_hypothesis(plan):
    return {
        "provider": "fixed_baseline",
        "hypothesis": "Frozen controls before LLM increments.",
        "factor_ids": list(plan["baseline_ids"]),
        "evidence_news_ids": [],
        "event_signal": 0.0,
        "proposed_factors": [],
        "proposed_business_factors": [],
        "question_map": [],
        "action": "baseline",
    }


def normalize_hypothesis(hypothesis, library):
    """The caller supplies increments only; baseline ownership stays with the engine."""
    plan = library["incremental_plan"]
    result = deepcopy(hypothesis)
    if result.get("proposed_factors") or result.get("proposed_news_factors"):
        raise ValueError(
            "Incremental mode permits existing business inputs "
            "and bounded business expressions only"
        )
    action = result.get("action", "test")
    if action not in {"test", "retain", "withdraw", "stop"}:
        raise ValueError("Unknown incremental action")
    increments = result["factor_ids"]
    if len(increments) > 2 or len(set(increments)) != len(increments):
        raise ValueError("Select at most two distinct incremental features")
    extended = extend_library(library, result)
    definitions = {d["factor_id"]: d for d in extended["definitions"]}
    domain = set(plan["candidate_input_ids"])
    for key in increments:
        definition = definitions.get(key)
        if not definition or not definition.get("runnable", True):
            raise ValueError(f"Unavailable incremental feature: {key}")
        if key not in domain:
            recipe = definition["recipe"]
            if recipe["op"] not in {"financial_expression", "business_expression"} or not all(
                binding["factor_id"] in domain for binding in recipe["inputs"]
            ):
                raise ValueError(f"Feature is outside the frozen candidate domain: {key}")
    if action != "test" and (increments or result.get("proposed_business_factors")):
        raise ValueError("retain/withdraw/stop must not introduce features")
    if action == "test" and not increments:
        raise ValueError("Testing requires an increment; use retain/withdraw/stop for no fit")
    result["incremental_factor_ids"] = increments
    result["factor_ids"] = plan["baseline_ids"] + increments
    result["action"] = action
    return result


def simulated_increment(context, library, *, previous=None, feedback=None):
    """Offline wiring fixture: real business columns, no invented investment conclusion."""
    ids = library["incremental_plan"]["candidate_input_ids"]
    inventory = next((key for key in ids if "inventory_yoy" in key), None)
    revenue = next((key for key in ids if "revenue_yoy" in key), None)
    prior = (feedback or {}).get("research_history", [])
    first_trial = len(prior) <= 1
    proposal = []
    selected = []
    if first_trial and inventory and revenue:
        proposal = [
            {
                "name": "inventory_relative_growth",
                "description": "Inventory growth minus revenue growth; not turnover or causality.",
                "inputs": [
                    {"name": "inventory", "factor_id": inventory},
                    {"name": "revenue", "factor_id": revenue},
                ],
                "expression": {
                    "schema_version": 1,
                    "nodes": [
                        {"name": "i", "op": "leaf", "inputs": [], "leaf": "inventory"},
                        {"name": "r", "op": "leaf", "inputs": [], "leaf": "revenue"},
                        {
                            "name": "difference",
                            "op": "subtract",
                            "inputs": ["i", "r"],
                            "leaf": None,
                        },
                    ],
                },
            }
        ]
        selected = ["business.inventory_relative_growth"]
    elif ids:
        selected = [next((key for key in ids if "operating_cashflow_yoy" in key), ids[0])]
    return {
        "provider": "simulated_increment",
        "action": "test" if selected else "stop",
        "hypothesis": (
            "SIMULATED business increment; test relative inventory growth, then a "
            "cash-flow variable. Missing data is a gap."
        ),
        "factor_ids": selected,
        "evidence_news_ids": [],
        "event_signal": 0.0,
        "proposed_factors": [],
        "proposed_business_factors": proposal,
        "question_map": [
            {
                "claim": context.get("research_question") or "Business increment",
                "factor_ids": selected,
                "rationale": "Offline execution fixture.",
                "limitations": ["Not an investment finding or causal test."],
            }
        ],
    }


def recipe_identity(recipe):
    """Ignore expression/binding names; preserve operations, ordering and source identities."""
    if recipe["op"] not in {"financial_expression", "business_expression"}:
        return recipe
    bindings = {binding["name"]: binding["factor_id"] for binding in recipe["inputs"]}
    nodes = {}
    for node in recipe["expression"]["nodes"]:
        nodes[node["name"]] = (
            {"factor_id": bindings[node["leaf"]]}
            if node["op"] == "leaf"
            else {"op": node["op"], "inputs": [nodes[name] for name in node["inputs"]]}
        )
    return nodes[recipe["expression"]["nodes"][-1]["name"]]


def recipe_set(factors):
    return sorted(digest(canonical(recipe_identity(factor["recipe"]))) for factor in factors)
