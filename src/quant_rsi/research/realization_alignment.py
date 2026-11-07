"""Candidate-local actual/guidance joins over admitted disclosure history.

This is a management-reference measurement, not an analyst/market surprise.
Original-release anchors are reviewed source constraints, not first-ever-publication proof.
"""

from datetime import timedelta

import numpy as np

from quant_rsi.core.artifacts import canonical
from quant_rsi.market.market_contracts import timestamp
from quant_rsi.research.discovery_sources import DataRequest


def _unique(rows):
    if not rows or len({row["value"] for row in rows}) != 1:
        return None
    return min(rows, key=lambda row: row["observation_id"])


def prepare_realization_inputs(proposal, bundles, times):
    """Resolve once, before screening and numerical dependency probes; never fill gaps."""
    relation = proposal.realization_alignment
    names = {item.name: item.factor_id for item in proposal.inputs}
    ids = {role: names[getattr(relation, role)] for role in ("actual", "lower", "upper")}
    by_id = {bundle["request"]["input_id"]: bundle for bundle in bundles}
    if len(set(ids.values())) != 3 or not set(ids.values()) <= by_id.keys():
        raise ValueError("Realization alignment needs three distinct admitted raw histories")
    requests = {
        role: DataRequest.model_validate_json(canonical(by_id[identity]["request"]))
        for role, identity in ids.items()
    }
    actual, lower, upper = (requests[role] for role in ("actual", "lower", "upper"))
    if (
        actual.forecast is not None
        or actual.realized_measure is None
        or not actual.original_actual_releases
        or lower.forecast is None
        or upper.forecast is None
        or lower.forecast.statistic != "lower"
        or upper.forecast.statistic != "upper"
        or not lower.forecast.pairing_group
        or lower.forecast.pairing_group != upper.forecast.pairing_group
        or lower.forecast.product != upper.forecast.product
        or lower.forecast.basis != upper.forecast.basis
        or actual.realized_measure.product != lower.forecast.product
        or actual.realized_measure.basis != lower.forecast.basis
        or len({request.symbol for request in requests.values()}) != 1
        or len({request.unit for request in requests.values()}) != 1
    ):
        raise ValueError("Realization alignment needs anchored actuals and compatible range bounds")
    anchors = {
        item.period_end: item.model_dump(mode="json") for item in actual.original_actual_releases
    }
    columns = {identity: [] for identity in ids.values()}
    evidence = {identity: [] for identity in ids.values()}
    sources = {identity: set() for identity in ids.values()}
    for instant in times:
        cutoff = timestamp(instant)
        available = {
            role: [
                row
                for row in by_id[identity]["observations"]
                if timestamp(row["effective_available_at"]) <= cutoff
                and timestamp(row["published_at"]) <= cutoff
                and (cutoff - timestamp(row["published_at"])).days <= requests[role].max_age_days
            ]
            for role, identity in ids.items()
        }
        actuals = [
            row
            for row in available["actual"]
            if row.get("original_actual_release") is not None
            and row["original_actual_release"] == anchors.get(row["period_end"])
        ]
        latest_period = max(
            (
                end
                for end, anchor in anchors.items()
                if timestamp(anchor["publication_date"] + "T12:00:00Z") + timedelta(days=1)
                <= cutoff
            ),
            default=None,
        )
        realized = _unique([row for row in actuals if row["period_end"] == latest_period])
        selected, proof, reason = {}, None, "missing_or_conflicting_original_actual"
        if realized:
            quarter = (
                f"{realized['period_end'][:4]}-Q{(int(realized['period_end'][5:7]) - 1) // 3 + 1}"
            )
            forecasts = {
                role: [
                    row
                    for row in available[role]
                    if row.get("target_quarter") == quarter
                    and row.get("period_start") == realized["period_start"]
                    and row["period_end"] == realized["period_end"]
                    and timestamp(row["published_at"]) < timestamp(realized["published_at"])
                ]
                for role in ("lower", "upper")
            }
            latest = max(
                (timestamp(row["published_at"]) for rows in forecasts.values() for row in rows),
                default=None,
            )
            pairs = {
                role: [row for row in rows if timestamp(row["published_at"]) == latest]
                for role, rows in forecasts.items()
            }
            source_ids = {row["source_id"] for rows in pairs.values() for row in rows}
            lo, hi = (_unique(pairs[role]) for role in ("lower", "upper"))
            reason = "missing_or_conflicting_latest_pre_actual_range"
            if lo and hi and len(source_ids) == 1:
                reason = "nonpositive_guidance_width"
                if (
                    np.isfinite([realized["value"], lo["value"], hi["value"]]).all()
                    and hi["value"] > lo["value"]
                ):
                    selected = {"actual": realized, "lower": lo, "upper": hi}
                    proof = {
                        "version": relation.version,
                        "policy": relation.policy,
                        "target_quarter": quarter,
                        "period_start": realized["period_start"],
                        "period_end": realized["period_end"],
                        "actual_observation_id": realized["observation_id"],
                        "actual_published_at": realized["published_at"],
                        "guidance_source_id": lo["source_id"],
                        "guidance_published_at": lo["published_at"],
                    }
                    reason = None
        for role, identity in ids.items():
            row = selected.get(role)
            columns[identity].append(row["value"] if row else np.nan)
            item = {
                "prediction_time": instant,
                "observation_id": row["observation_id"] if row else None,
                "effective_available_at": row["effective_available_at"] if row else None,
                "source_id": row["source_id"] if row else None,
                "realization_alignment": proof,
                "missing_reason": reason,
            }
            if role != "actual":
                item["forecast_vintage"] = (
                    {key: row[key] for key in ("source_id", "published_at", "target_quarter")}
                    if row
                    else None
                )
            evidence[identity].append(item)
            if row:
                sources[identity].add(row["source_id"])
    return {
        "columns": {key: np.asarray(values, dtype=float) for key, values in columns.items()},
        "input_evidence": evidence,
        "input_evidence_ids": {key: sorted(value) for key, value in sources.items()},
    }


def mask_realization_inputs(proposal, bound, evidence):
    """Require prepared evidence, but evaluate the supplied columns (including probes)."""
    from quant_rsi.research.discovery_factors import _check_paired_temporal_expression

    relation = proposal.realization_alignment
    names = [getattr(relation, role) for role in ("actual", "lower", "upper")]
    ids = {item.name: item.factor_id for item in proposal.inputs}
    _check_paired_temporal_expression(proposal, set(names))
    rows = [(evidence or {}).get(ids[name], []) for name in names]
    size = len(bound[names[0]])
    if any(len(series) != size for series in rows):
        raise ValueError("Realization alignment requires prepared source evidence")
    matched = []
    for group in zip(*rows, strict=True):
        if any("realization_alignment" not in row for row in group):
            raise ValueError("Realization alignment requires prepared source evidence")
        proof = group[0]["realization_alignment"]
        valid = bool(proof) and all(row["realization_alignment"] == proof for row in group)
        if valid:
            valid = (
                proof["version"] == relation.version
                and proof["policy"] == relation.policy
                and proof["actual_observation_id"] == group[0]["observation_id"]
                and timestamp(proof["guidance_published_at"])
                < timestamp(proof["actual_published_at"])
                and all(
                    row.get("forecast_vintage")
                    == {
                        "source_id": proof["guidance_source_id"],
                        "published_at": proof["guidance_published_at"],
                        "target_quarter": proof["target_quarter"],
                    }
                    for row in group[1:]
                )
                and all(
                    timestamp(row["effective_available_at"]) <= timestamp(row["prediction_time"])
                    for row in group
                )
            )
        matched.append(valid)
    valid = np.asarray(matched) & (
        np.asarray(bound[relation.upper]) > np.asarray(bound[relation.lower])
    )
    return bound | {name: np.where(valid, bound[name], np.nan) for name in names}
