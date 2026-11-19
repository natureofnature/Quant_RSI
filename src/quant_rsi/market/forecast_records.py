"""Forecast scoring and publication-confirmation records (neutral market layer).

Shared by the research study's history ledger and the frozen forecast workflow
(#159 D2). Imports only core artifacts and market contracts.
"""

import math

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.market.market_contracts import timestamp


def forecast_errors(predicted, actual):
    """Positive errors and their explicit maximization score; never clip invalid output."""
    if not predicted or len(predicted) != len(actual):
        raise ValueError("forecast scoring requires the complete fixed label grid")
    errors = []
    for pred, label in zip(predicted, actual, strict=True):
        if not math.isfinite(pred) or not math.isfinite(label):
            raise ValueError("nonfinite forecast or label")
        try:
            predicted_ratio, actual_ratio = math.exp(pred), math.exp(label)
        except OverflowError as error:
            raise ValueError("forecast has a nonfinite implied price") from error
        if not (math.isfinite(predicted_ratio) and predicted_ratio > 0):
            raise ValueError("forecast has an invalid implied positive price")
        if not (math.isfinite(actual_ratio) and actual_ratio > 0):
            raise ValueError("label has an invalid implied positive price")
        error = 100 * (math.expm1(pred) - math.expm1(label))
        if not math.isfinite(error):
            raise ValueError("nonfinite percentage-point forecast error")
        errors.append(error)
    mae = math.fsum(abs(value) for value in errors) / len(errors)
    try:
        rmse = math.sqrt(math.fsum(value * value for value in errors) / len(errors))
    except OverflowError as error:
        raise ValueError("nonfinite forecast squared error") from error
    if not math.isfinite(mae) or not math.isfinite(rmse):
        raise ValueError("nonfinite forecast aggregate error")
    return {"mae_pp": mae, "rmse_pp": rmse, "negative_mae_pp": -mae, "observations": len(errors)}


def confirmed_report(report, confirmation):
    ref = digest(canonical(report))
    if confirmation["forecast_ref"] != ref or timestamp(confirmation["confirmed_at"]) < timestamp(
        report["issued_at"]
    ):
        raise ValueError("publication confirmation does not bind the committed report")
    forward = (
        report["status"] == "forecast"
        and timestamp(report["origin_close_at"])
        <= timestamp(report["issued_at"])
        < timestamp(report["forward_deadline"])
        and timestamp(confirmation["confirmed_at"]) < timestamp(report["forward_deadline"])
    )
    return report | dict(
        forecast_ref=ref,
        post_commit_confirmed_at=confirmation["confirmed_at"],
        forward=forward,
        mode="forward" if forward else "retrospective_or_abstained",
    )
