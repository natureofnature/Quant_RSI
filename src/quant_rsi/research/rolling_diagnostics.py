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
