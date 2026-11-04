"""Shared finite factor-expression grammar and scalar arithmetic."""

import math
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.core.contracts_base import StrictModel

Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]


class ExpressionNode(StrictModel):
    name: Name
    op: Literal["leaf", "add", "subtract", "multiply"]
    inputs: tuple[Name, ...] = ()
    leaf: Name | None = None

    @model_validator(mode="after")
    def arity(self) -> Self:
        if self.op == "leaf":
            if self.leaf is None or self.inputs:
                raise ValueError("leaf node requires one named source")
        elif self.leaf is not None or len(self.inputs) != 2:
            raise ValueError("binary node requires two preceding nodes")
        return self


class FactorExpression(StrictModel):
    schema_version: Literal[1] = 1
    nodes: tuple[ExpressionNode, ...] = Field(min_length=1, max_length=7)

    @model_validator(mode="after")
    def dag(self) -> Self:
        depths = {}
        for node in self.nodes:
            if node.name in depths or any(name not in depths for name in node.inputs):
                raise ValueError("expression requires unique topologically ordered nodes")
            depths[node.name] = 1 + max((depths[name] for name in node.inputs), default=0)
            if depths[node.name] > 3:
                raise ValueError("expression depth exceeds registered bound")
        needed = {self.nodes[-1].name}
        for node in reversed(self.nodes):
            if node.name in needed:
                needed.update(node.inputs)
        if set(depths) != needed:
            raise ValueError("expression contains unused nodes")
        return self

    @property
    def ref(self):
        # Names and commutative argument order do not create novel definitions.
        values = {}
        for node in self.nodes:
            args = [values[name] for name in node.inputs]
            if node.op in {"add", "multiply"}:
                args.sort(key=canonical)
            values[node.name] = [node.op, node.leaf] if node.op == "leaf" else [node.op, args]
        return digest(canonical(values[self.nodes[-1].name]))


def evaluate_expression(expression, sources):
    reasons = []
    values = {}
    for node in expression.nodes:
        if node.op == "leaf":
            if node.leaf not in sources:
                raise ValueError("expression refers to an unregistered leaf")
            value = sources[node.leaf]
        else:
            a, b = (values[name] for name in node.inputs)
            value = (
                None
                if a is None or b is None
                else (a + b if node.op == "add" else a - b if node.op == "subtract" else a * b)
            )
        if value is not None and not math.isfinite(value):
            reasons.append("expression_nonfinite")
            value = None
        values[node.name] = value
    return dict(
        value=values[expression.nodes[-1].name],
        reasons=sorted(set(reasons)),
    )


class SeriesExpressionNode(ExpressionNode):
    """Version-two nodes extend the same DAG grammar with causal series operators."""

    op: Literal[
        "leaf",
        "add",
        "subtract",
        "multiply",
        "divide",
        "lag",
        "rolling_mean",
        "rolling_std",
        "rolling_zscore",
    ]
    window: int | None = Field(default=None, ge=1, le=252)

    @model_validator(mode="after")
    def arity(self) -> Self:
        if self.op == "leaf":
            if self.leaf is None or self.inputs or self.window is not None:
                raise ValueError("leaf node requires one source and no window")
        elif self.op in {"add", "subtract", "multiply", "divide"}:
            if self.leaf is not None or len(self.inputs) != 2 or self.window is not None:
                raise ValueError("binary node requires two preceding nodes and no window")
        elif self.leaf is not None or len(self.inputs) != 1 or self.window is None:
            raise ValueError("temporal node requires one preceding node and a window")
        elif self.op in {"rolling_std", "rolling_zscore"} and self.window < 2:
            raise ValueError("sample standard deviation requires a window of at least two")
        return self


class SeriesFactorExpression(FactorExpression):
    """A versioned extension; v1 dumps, bounds and recipe identities stay unchanged."""

    schema_version: Literal[2] = 2
    nodes: tuple[SeriesExpressionNode, ...] = Field(min_length=1, max_length=31)

    @model_validator(mode="after")
    def dag(self) -> Self:
        depths = {}
        for node in self.nodes:
            if node.name in depths or any(name not in depths for name in node.inputs):
                raise ValueError("expression requires unique topologically ordered nodes")
            depths[node.name] = 1 + max((depths[name] for name in node.inputs), default=0)
            if depths[node.name] > 8:
                raise ValueError("expression depth exceeds registered bound")
        needed = {self.nodes[-1].name}
        for node in reversed(self.nodes):
            if node.name in needed:
                needed.update(node.inputs)
        if set(depths) != needed:
            raise ValueError("expression contains unused nodes")
        return self

    def normalized(self, bindings=None):
        """Remove incidental names; optionally resolve leaf aliases to actual input IDs."""
        values = {}
        for node in self.nodes:
            args = [values[name] for name in node.inputs]
            if node.op in {"add", "multiply"}:
                args.sort(key=canonical)
            values[node.name] = (
                ["leaf", (bindings or {}).get(node.leaf, node.leaf)]
                if node.op == "leaf"
                else [node.op, args, node.window]
            )
        return ["series_expression_v2", values[self.nodes[-1].name]]

    @property
    def ref(self):
        return digest(canonical(self.normalized()))


def evaluate_series(expression, sources):
    """Execute the shared DAG on aligned series, without filling or dropping rows.

    Inputs must already be checked for per-row availability. Temporal operators
    never inspect a later row; rolling windows require all observations and use
    sample standard deviation (ddof=1). Zero divisors/constant z-scores are missing.
    """
    import numpy as np

    inputs = {}
    for name, column in sources.items():
        array = np.asarray(column, dtype=float)
        if array.ndim != 1:
            raise ValueError("expression inputs must be one-dimensional aligned series")
        inputs[name] = array
    if not inputs or len({len(column) for column in inputs.values()}) != 1:
        raise ValueError("expression inputs must share one population")
    values = {}
    for node in expression.nodes:
        if node.op == "leaf":
            if node.leaf not in inputs:
                raise ValueError("expression refers to an unregistered leaf")
            value = inputs[node.leaf].copy()
        elif node.op in {"add", "subtract", "multiply", "divide"}:
            a, b = (values[name] for name in node.inputs)
            with np.errstate(all="ignore"):
                if node.op == "add":
                    value = a + b
                elif node.op == "subtract":
                    value = a - b
                elif node.op == "multiply":
                    value = a * b
                else:
                    value = np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)
        else:
            source = values[node.inputs[0]]
            value = np.full_like(source, np.nan)
            window = node.window
            if node.op == "lag":
                value[window:] = source[:-window]
            else:
                for index in range(window - 1, len(source)):
                    sample = source[index + 1 - window : index + 1]
                    if not np.isfinite(sample).all():
                        continue
                    with np.errstate(all="ignore"):
                        mean = np.mean(sample)
                        if node.op == "rolling_mean":
                            value[index] = mean
                        elif node.op == "rolling_std":
                            value[index] = np.std(sample, ddof=1)
                        else:
                            std = np.std(sample, ddof=1)
                            if std > 0:
                                value[index] = (sample[-1] - mean) / std
        value[~np.isfinite(value)] = np.nan
        values[node.name] = value
    return values[expression.nodes[-1].name]
