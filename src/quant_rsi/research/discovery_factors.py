"""Survey retrieval and executable, evidence-linked expressions for connected research.

The survey is knowledge, not automatic execution permission. The numerical path
uses the shared expression grammar and existing price adapters, with explicit
input bindings and no data collection or population changes inside an executor.
"""

import json
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
from pydantic import Field, model_validator

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.factors.expressions import Name, SeriesFactorExpression, evaluate_series
from quant_rsi.factors.price_adapters import CandidatePriceRecipe, evaluate_price_adapter
from quant_rsi.market.market_contracts import Hash, timestamp

SURVEY_BINDINGS = {
    "qlib.kmid": {"open": "price.open", "close": "price.close"},
    "qlib.klen": {"high": "price.high", "low": "price.low", "open": "price.open"},
    "qlib.roc": {"close": "price.close"},
    "qlib.ma": {"close": "price.close"},
}


def operator_descriptions():
    """Versioned, finite signatures supplied to the proposer, not arbitrary code."""
    return {
        "schema_version": 2,
        "max_nodes": 31,
        "max_depth": 8,
        "operators": {
            "leaf": "One named bound input; no node inputs or window.",
            "add": "Two preceding nodes; compatible known units; no window.",
            "subtract": "Two preceding nodes; compatible known units; no window.",
            "multiply": "Two preceding nodes; no window.",
            "divide": "Two preceding nodes; zero divisor produces missing; no window.",
            "lag": "One preceding node; window 1..252 earlier observations, never future.",
            "rolling_mean": "One node; full trailing window 1..252, including current row.",
            "rolling_std": "One node; full trailing window 2..252; sample std, ddof=1.",
            "rolling_zscore": "One node; (current-mean)/sample std over full window 2..252; "
            "constant windows missing, no learned full-sample normalization.",
        },
        "missingness": "Missing/nonfinite inputs propagate; no filling or row dropping.",
        "availability": "Caller must admit source values at each row's prediction time.",
    }


def survey_catalog(root=None, available_inputs=()):
    """Read all detailed survey records without modifying or importing upstream code."""
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    directory = root / "experimental" / "candidate_factors"
    sources = {
        item["id"]: item for item in json.loads((directory / "sources.json").read_text())["sources"]
    }
    available = set(available_inputs)
    result = []
    for path in sorted((directory / "definitions").glob("*.json")):
        for record in json.loads(path.read_text())["candidates"]:
            source = sources[record["source"]["source_id"]]
            bindings = SURVEY_BINDINGS.get(record["id"], {})
            missing = sorted(set(bindings.values()) - available)
            status = (
                "noncausal_reference_only"
                if record["review_status"] == "reference_only_noncausal"
                else "needs_operator_adapter"
                if not bindings
                else "needs_input"
                if missing
                else "executable"
            )
            result.append(
                deepcopy(record)
                | {
                    "survey_id": record["id"],
                    "factor_id": f"survey.{record['id']}",
                    "execution_status": status,
                    "input_bindings": bindings,
                    "missing_inputs": missing,
                    "provenance": {
                        "revision": source["commit"],
                        "license_spdx": source["license_spdx"],
                        "license_artifacts": deepcopy(source["license_artifacts"]),
                        "source_files": deepcopy(source["files"]),
                        "definition_file": str(path.relative_to(root)),
                        "definition_sha256": digest(path.read_bytes()),
                    },
                }
            )
    return result


def retrieve_candidates(query, catalog, limit=8):
    """Bounded lexical retrieval; aliases aid lookup without asserting equivalence."""
    if type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError("Candidate retrieval limit must be between one and 32")
    tokens = set(re.findall(r"[a-z0-9_]+", query.lower()))
    aliases = {
        "volatility": {"volatility", "range", "std"},
        "momentum": {"roc", "return", "trend"},
        "valuation": {"yield", "earnings", "book", "market_cap"},
        "volume": {"volume", "liquidity", "vwap", "obv"},
    }
    terms = tokens | set().union(*(aliases.get(token, set()) for token in tokens))

    def score(item):
        content = " ".join(
            str(item.get(key, ""))
            for key in ("id", "formula_summary", "inputs", "semantic_edges", "output_units")
        ).lower()
        words = set(re.findall(r"[a-z0-9_]+", content))
        return len(words & terms)

    ranked = sorted(catalog, key=lambda item: (-score(item), item["survey_id"]))
    return [deepcopy(item) for item in ranked if score(item) > 0][:limit]


def execute_survey(survey_id, columns, *, window=10):
    """Execute only the admitted source-pinned subset on aligned numerical inputs."""
    if survey_id not in SURVEY_BINDINGS:
        raise ValueError(f"Survey candidate {survey_id} needs an operator/adapter")
    required = SURVEY_BINDINGS[survey_id]
    if any(identity not in columns for identity in required.values()):
        raise ValueError(f"Survey candidate {survey_id} needs input")
    bound = {name: np.asarray(columns[key], dtype=float) for name, key in required.items()}
    if (
        any(value.ndim != 1 for value in bound.values())
        or len({len(value) for value in bound.values()}) != 1
    ):
        raise ValueError("Survey inputs require aligned one-dimensional series")
    if survey_id == "qlib.klen":
        highs, lows, opens = bound["high"], bound["low"], bound["open"]
        valid = (
            np.isfinite(highs)
            & np.isfinite(lows)
            & np.isfinite(opens)
            & (lows > 0)
            & (opens > 0)
            & (highs > 0)
            & (lows <= opens)
            & (opens <= highs)
        )
        with np.errstate(all="ignore"):
            result = np.divide(highs - lows, opens, out=np.full_like(opens, np.nan), where=valid)
        result[~np.isfinite(result)] = np.nan
        return result
    closes = bound["close"]
    if survey_id == "qlib.kmid":
        opens = bound["open"]
        with np.errstate(all="ignore"):
            result = np.divide(
                closes - opens,
                opens,
                out=np.full_like(opens, np.nan),
                where=(opens > 0) & (closes > 0),
            )
        result[~np.isfinite(result)] = np.nan
        return result
    recipe = CandidatePriceRecipe(op=f"qlib_{survey_id.split('.')[1]}_v1", window=window)
    result = np.full(len(closes), np.nan)
    for index in range(recipe.observations - 1, len(closes)):
        value = evaluate_price_adapter(
            recipe, closes[index + 1 - recipe.observations : index + 1].tolist()
        )
        if value is not None:
            result[index] = value
    return result


def adapter_catalog(root=None, available_inputs=(), *, window=10):
    """Executable leaf definitions retain exact source versions and requirements."""
    if type(window) is not int or not 1 <= window <= 30:
        raise ValueError("Survey price window must be between one and 30")
    return [
        {
            "factor_id": item["factor_id"],
            "description": item["formula_summary"],
            "kind": "numeric",
            "unit": item["output_units"],
            "recipe": {
                "op": "survey_adapter",
                "survey_id": item["survey_id"],
                "window": None if item["survey_id"] in {"qlib.kmid", "qlib.klen"} else window,
                "upstream_revision": item["provenance"]["revision"],
            },
            "input_factor_ids": list(item["input_bindings"].values()),
            "runnable": item["execution_status"] == "executable",
            "availability_reason": item["execution_status"],
            "provenance": item["provenance"],
        }
        for item in survey_catalog(root, available_inputs)
        if item["survey_id"] in SURVEY_BINDINGS
    ]


class DiscoveryInput(StrictModel):
    name: Name
    factor_id: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9_.-]+$")


class DiscoveryFactor(StrictModel):
    name: Name
    description: str = Field(min_length=1, max_length=2000)
    inputs: tuple[DiscoveryInput, ...] = Field(min_length=1, max_length=12)
    expression: SeriesFactorExpression
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=20)
    # Research revisions/replacements, not numerical expression dependencies.
    parent_recipe_refs: tuple[Hash, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def bindings(self):
        names = [item.name for item in self.inputs]
        leaves = {node.leaf for node in self.expression.nodes if node.op == "leaf"}
        if len(names) != len(set(names)) or set(names) != leaves:
            raise ValueError("Every unique input binding must be used by the expression")
        if len(self.parent_recipe_refs) != len(set(self.parent_recipe_refs)):
            raise ValueError("Parent recipe references must be unique")
        if self.recipe_ref in self.parent_recipe_refs:
            raise ValueError("A candidate cannot be its own parent")
        return self

    @property
    def recipe_ref(self):
        return digest(
            canonical(
                self.expression.normalized({item.name: item.factor_id for item in self.inputs})
            )
        )

    def definition(self, definitions):
        sources = {item["factor_id"]: item for item in definitions}
        missing = [
            item.factor_id
            for item in self.inputs
            if item.factor_id not in sources or sources[item.factor_id].get("runnable") is False
        ]
        if any(
            sources.get(item.factor_id, {}).get("execution_status") == "noncausal_reference_only"
            for item in self.inputs
        ):
            raise ValueError("Noncausal reference-only inputs cannot enter forecasting")
        unit = _expression_unit(self, sources)
        definition = {
            "factor_id": f"discovery.{self.name}",
            "description": self.description,
            "kind": "numeric",
            "unit": unit,
            "recipe_ref": self.recipe_ref,
            "recipe": {
                "op": "discovery_expression",
                "operator_version": 2,
                "inputs": [item.model_dump(mode="json") for item in self.inputs],
                "expression": self.expression.model_dump(mode="json"),
            },
            "input_factor_ids": [item.factor_id for item in self.inputs],
            "evidence_ids": list(self.evidence_ids),
            "runnable": not missing,
            "missing_inputs": sorted(set(missing)),
            "availability_reason": "needs_input" if missing else "bound_expression",
        }
        if self.parent_recipe_refs:
            definition["parent_recipe_refs"] = list(self.parent_recipe_refs)
        groups = {}
        for item in self.inputs:
            forecast = sources.get(item.factor_id, {}).get("forecast") or {}
            if forecast.get("pairing_group"):
                groups.setdefault(forecast["pairing_group"], set()).add(item.factor_id)
        paired = [sorted(group) for _, group in sorted(groups.items()) if len(group) > 1]
        if paired:
            definition["forecast_pairing"] = {"version": 1, "groups": paired}
        return definition


def _expression_unit(proposal, sources):
    units = {item.name: sources.get(item.factor_id, {}).get("unit") for item in proposal.inputs}
    values = {}
    for node in proposal.expression.nodes:
        if node.op == "leaf":
            unit = units[node.leaf]
        else:
            args = [values[name] for name in node.inputs]
            if node.op in {"add", "subtract"}:
                if all(args) and args[0] != args[1]:
                    raise ValueError("Addition/subtraction requires compatible known units")
                unit = args[0] if args[0] == args[1] else None
            elif node.op == "rolling_zscore":
                unit = "dimensionless"
            elif node.op == "divide":
                unit = "dimensionless" if args[0] and args[0] == args[1] else None
            elif node.op == "multiply":
                unit = (
                    args[1]
                    if args[0] == "dimensionless"
                    else (args[0] if args[1] == "dimensionless" else None)
                )
            else:
                unit = args[0]
        values[node.name] = unit
    return values[proposal.expression.nodes[-1].name]


def compute_factor(definition, columns, definitions=None, *, input_evidence=None):
    """Resolve an exact expression to a numerical column; source admission is external."""
    recipe = definition["recipe"]
    if recipe.get("op") != "discovery_expression" or recipe.get("operator_version") != 2:
        raise ValueError("Expected a version-two discovery expression")
    proposal = DiscoveryFactor.model_validate_json(
        canonical(
            {
                "name": definition["factor_id"].removeprefix("discovery."),
                "description": definition["description"],
                "inputs": recipe["inputs"],
                "expression": recipe["expression"],
                "evidence_ids": definition.get("evidence_ids", []),
            }
        )
    )
    if proposal.recipe_ref != definition["recipe_ref"]:
        raise ValueError("Discovery recipe identity does not match its executable expression")
    if definitions is not None:
        expected = proposal.definition(definitions)
        if not expected["runnable"]:
            raise ValueError("Discovery expression needs admitted input definitions")
        if definition.get("forecast_pairing") != expected.get("forecast_pairing"):
            raise ValueError("Forecast pairing differs from admitted input definitions")
    missing = [item.factor_id for item in proposal.inputs if item.factor_id not in columns]
    if missing:
        raise ValueError(f"Discovery expression needs input: {', '.join(missing)}")
    bound = {item.name: columns[item.factor_id] for item in proposal.inputs}
    pairing = definition.get("forecast_pairing", {"version": 1, "groups": []})
    if pairing["version"] != 1:
        raise ValueError("Unknown forecast pairing version")
    for group_ids in pairing["groups"]:
        forecasts = [item for item in proposal.inputs if item.factor_id in group_ids]
        if len({item.factor_id for item in forecasts}) != len(group_ids) or len(group_ids) < 2:
            raise ValueError("Invalid forecast pairing group")
        _check_paired_temporal_expression(proposal, {item.name for item in forecasts})
        evidence = input_evidence or {}
        size = len(bound[forecasts[0].name])
        rows = [evidence.get(item.factor_id, []) for item in forecasts]
        if any(len(row) != size for row in rows):
            raise ValueError("Combined forecasts need aligned vintage evidence")
        for series in rows:
            for row in series:
                if "forecast_vintage" not in row:
                    raise ValueError("Combined forecasts need complete vintage evidence")
                vintage = row["forecast_vintage"]
                if vintage is None:
                    continue
                if (
                    not isinstance(vintage, dict)
                    or set(vintage) != {"source_id", "published_at", "target_quarter"}
                    or not all(
                        isinstance(value, str) and value.strip() for value in vintage.values()
                    )
                    or not re.fullmatch(r"[1-9][0-9]{3}-Q[1-4]", vintage["target_quarter"])
                ):
                    raise ValueError("Invalid forecast vintage evidence")
                timestamp(vintage["published_at"])
        matched = np.array(
            [
                bool(group[0].get("forecast_vintage"))
                and all(
                    row.get("forecast_vintage") == group[0]["forecast_vintage"] for row in group
                )
                for group in zip(*rows, strict=True)
            ],
            dtype=bool,
        )
        # Mask inputs before temporal operators, so lags cannot consume mismatched vintages.
        for item in forecasts:
            bound[item.name] = np.where(matched, bound[item.name], np.nan)
    return evaluate_series(proposal.expression, bound)


def _check_paired_temporal_expression(proposal, paired_names):
    """Support paired arithmetic followed by a unary temporal chain, without cross-vintage math."""
    temporal = {"lag", "rolling_mean", "rolling_std", "rolling_zscore"}
    dependencies, shifted = {}, {}
    for node in proposal.expression.nodes:
        names = (
            {node.leaf}
            if node.op == "leaf"
            else set().union(*(dependencies[name] for name in node.inputs))
        )
        uses_pair = bool(names & paired_names)
        prior_shift = any(shifted[name] for name in node.inputs)
        if node.op in temporal and uses_pair and not paired_names <= names:
            raise ValueError("Combine paired forecast inputs before temporal operators")
        if prior_shift and node.op not in temporal:
            raise ValueError("Paired forecast temporal operators must be an outermost unary chain")
        dependencies[node.name] = names
        shifted[node.name] = prior_shift or (node.op in temporal and uses_pair)
