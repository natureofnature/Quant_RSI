"""Read-only candidate-matrix structure; no fitting, scoring or factor selection."""

from itertools import combinations

import numpy as np

CORRELATION_THRESHOLD = 0.8
MAX_CORRELATION_PAIRS = 20
NOTICE = (
    "Structural training-matrix diagnostics only. These summaries do not measure "
    "predictive usefulness or causality and do not select factors. "
    "Training observations may have overlapping return-label horizons."
)
FORECAST_NOTICE = (
    "Current-cutoff input ranges and saved-model accounting only; no observed outcome, "
    "feature selection or causal importance. Additive contributions use log_return units, "
    "not simple-return percentage points. Training ranges do not guarantee future validity."
)


def diagnose_candidates(matrix, train, ids, baseline_count):
    """Summarize admitted training rows without reading prediction/warm-up values.

    Correlations exclude the column itself. Baseline and candidate maxima refer to
    their respective column groups; constant columns have no defined correlation.
    Rank uses standardized, centered columns and NumPy's default numerical tolerance.
    Raw variance is None only when the representable numerical range is exceeded.
    """
    matrix, train = np.asarray(matrix, dtype=float), np.asarray(train)
    if matrix.ndim != 2 or not matrix.shape[1] or len(ids) != matrix.shape[1]:
        raise ValueError("Diagnostics require a nonempty matrix with matching factor IDs")
    if any(not isinstance(identity, str) for identity in ids) or len(set(ids)) != len(ids):
        raise ValueError("Diagnostic factor IDs must be distinct strings")
    if type(baseline_count) is not int or not 0 <= baseline_count <= matrix.shape[1]:
        raise ValueError("Invalid diagnostic baseline column count")
    if (
        train.ndim != 1
        or not train.size
        or train.dtype.kind not in "iu"
        or np.any(train < 0)
        or np.any(train >= len(matrix))
        or len(np.unique(train)) != len(train)
    ):
        raise ValueError("Diagnostics require distinct valid integer training-row indices")
    x = matrix[train]
    if not np.isfinite(x).all():
        raise ValueError("Diagnostics require finite values on every admitted training row")

    # Scale before centering to avoid overflow in correlation/rank calculations.
    magnitude = np.max(np.abs(x), axis=0)
    normalized = x / np.where(magnitude > 0, magnitude, 1)
    centered = normalized - normalized.mean(axis=0)
    scale = centered.std(axis=0)
    varying = scale > 0
    standardized = centered / np.where(varying, scale, 1)
    correlation = np.clip(standardized.T @ standardized / len(x), -1, 1)

    def maximum(index, group):
        peers = [other for other in group if other != index and varying[other]]
        return (
            float(np.max(np.abs(correlation[index, peers]))) if varying[index] and peers else None
        )

    columns = []
    for index, identity in enumerate(ids):
        values = x[:, index]
        with np.errstate(over="ignore", invalid="ignore"):
            variance = float(np.square(scale[index] * magnitude[index]))
        nonzero = int(np.count_nonzero(values))
        columns.append(
            {
                "factor_id": identity,
                "role": "baseline" if index < baseline_count else "candidate",
                "distinct_values": len(np.unique(values)),
                "nonzero_rows": nonzero,
                "nonzero_fraction": nonzero / len(x),
                "variance": variance if np.isfinite(variance) else None,
                "constant": not bool(varying[index]),
                "max_abs_baseline_correlation": maximum(index, range(baseline_count)),
                "max_abs_candidate_correlation": maximum(index, range(baseline_count, len(ids))),
            }
        )
    pairs = []
    for left, right in combinations(range(len(ids)), 2):
        value = float(correlation[left, right])
        if varying[left] and varying[right] and abs(value) >= CORRELATION_THRESHOLD:
            pairs.append(
                {
                    "left": ids[left],
                    "right": ids[right],
                    "correlation": value,
                    "kind": "baseline_pair"
                    if right < baseline_count
                    else "candidate_pair"
                    if left >= baseline_count
                    else "candidate_vs_baseline",
                }
            )
    pairs.sort(key=lambda row: -abs(row["correlation"]))
    return {
        "training_samples": len(x),
        "model_columns": len(ids),
        "baseline_columns": baseline_count,
        "candidate_columns": len(ids) - baseline_count,
        "standardized_rank": int(np.linalg.matrix_rank(standardized)),
        "columns": columns,
        "high_correlation_pairs": pairs[:MAX_CORRELATION_PAIRS],
        "high_correlation_pair_count": len(pairs),
        "omitted_correlation_pairs": max(0, len(pairs) - MAX_CORRELATION_PAIRS),
        "absolute_correlation_threshold": CORRELATION_THRESHOLD,
        "notice": NOTICE,
    }


def diagnose_forecast(matrix, train, fit, baseline_count):
    """Explain the last matrix row using admitted training rows and a saved fit only.

    Prediction z uses the population training standard deviation, and is undefined
    for a constant column. Ridge accounting uses the saved model's own scaler.
    Missing/nonlinear states still permit input-range diagnostics.
    """
    from quant_rsi.research.study_estimators import StandardizedRidge

    ids = fit["factor_ids"]
    structure = diagnose_candidates(matrix, train, ids, baseline_count)
    matrix, train = np.asarray(matrix, dtype=float), np.asarray(train)
    x, point = matrix[train], matrix[-1]
    predicted = float(fit["predicted_log_return"])
    if not np.isfinite(point).all() or not np.isfinite(predicted):
        raise ValueError("Forecast diagnostics require finite prediction values")
    columns = []
    for index, column in enumerate(structure["columns"]):
        values, value = x[:, index], float(point[index])
        # Normalization keeps range summaries serializable even for large inputs.
        magnitude = max(float(np.max(np.abs(values))), abs(value), 1.0)
        scaled = values / magnitude
        scale = float(scaled.std())
        constant = bool(np.all(values == values[0]))
        z = (value / magnitude - float(scaled.mean())) / scale if scale and not constant else None
        columns.append(
            {
                "factor_id": column["factor_id"],
                "role": column["role"],
                "training_min": float(values.min()),
                "training_max": float(values.max()),
                "distinct_values": column["distinct_values"],
                "constant": constant,
                "prediction_value": value,
                "prediction_z": float(z) if z is not None and np.isfinite(z) else None,
                "outside_training_range": bool(value < values.min() or value > values.max()),
            }
        )
    decomposition = {
        "available": False,
        "units": "log_return",
        "predicted_log_return": predicted,
        "reason": "Linear saved state unavailable; no nonlinear importance decomposition inferred",
    }
    state = fit.get("state") or {}
    if "coefficients" in state:
        model = StandardizedRidge.from_state(state)
        if len(model.mean) != len(ids):
            raise ValueError("Saved ridge state does not match forecast factor columns")
        contributions = ((point - model.mean) / model.scale) * model.coefficients
        baseline, candidate = (
            float(contributions[:baseline_count].sum()),
            float(contributions[baseline_count:].sum()),
        )
        reconstructed = model.intercept + baseline + candidate
        if not np.isfinite(contributions).all() or not np.isclose(
            reconstructed, predicted, rtol=1e-10, atol=1e-12
        ):
            raise ValueError("Saved ridge accounting does not reconstruct the forecast")
        decomposition = {
            "available": True,
            "units": "log_return",
            "intercept": model.intercept,
            "baseline_group": baseline,
            "candidate_group": candidate,
            "reconstructed_prediction": reconstructed,
            "predicted_log_return": predicted,
            "reconstruction_error": reconstructed - predicted,
        }
    return {
        **{
            key: structure[key]
            for key in (
                "training_samples",
                "model_columns",
                "baseline_columns",
                "candidate_columns",
            )
        },
        "columns": columns,
        "outside_training_range_count": sum(c["outside_training_range"] for c in columns),
        "decomposition": decomposition,
        "notice": FORECAST_NOTICE,
    }


def summarize_forecast(diagnostic):
    """Bound the LLM view without ranking factors or introducing outcome fields."""
    outside = [c for c in diagnostic["columns"] if c["outside_training_range"]]
    return {
        "training_samples": diagnostic["training_samples"],
        "model_columns": diagnostic["model_columns"],
        "decomposition": dict(diagnostic["decomposition"]),
        "outside_training_range_count": len(outside),
        "outside_training_range_columns": [dict(column) for column in outside[:8]],
        "omitted_outside_training_range_columns": max(0, len(outside) - 8),
        "notice": FORECAST_NOTICE,
    }


def compare_forecast_diagnostics(fixed, adaptive):
    """Account for refitted baseline weights separately from added columns."""

    def base_ids(item):
        return [c["factor_id"] for c in item["columns"] if c["role"] == "baseline"]

    result = {"available": False, "units": "log_return", "notice": FORECAST_NOTICE}
    if (
        not fixed.get("comparison_identity")
        or fixed["comparison_identity"] != adaptive.get("comparison_identity")
        or base_ids(fixed) != base_ids(adaptive)
        or fixed["candidate_columns"]
    ):
        return result | {
            "reason": "Matching training rows, labels and baseline input identity are required"
        }
    left, right = fixed["decomposition"], adaptive["decomposition"]
    if not (left["available"] and right["available"]):
        return result | {"reason": "Both saved linear decompositions are required"}
    return result | {
        "available": True,
        "intercept_change": right["intercept"] - left["intercept"],
        "baseline_refit_change": right["baseline_group"] - left["baseline_group"],
        "candidate_group": right["candidate_group"],
        "predicted_log_return_change": right["predicted_log_return"] - left["predicted_log_return"],
    }
