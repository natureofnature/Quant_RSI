"""Research policy identities and deterministic visibility gates for connected discovery.

The protected PolicyPatch in changes.py has a different, fixed neighbor-order API;
these records use the same content identities without extending its authority.
"""

from copy import deepcopy
from datetime import UTC, datetime

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.market.market_contracts import timestamp

POLICY_FIELDS = {"investigation", "proposal", "screening", "allocation"}
DEFAULT_POLICY = {
    "investigation": "Prefer primary evidence and seek missing measurements.",
    "proposal": "Synthesize measurable expressions; never substitute unrelated price proxies.",
    "screening": "Prefer evidence-linked executable candidates; retain negative results.",
    "allocation": "Reuse comparable baselines and reserve a full matched pair for new samples.",
}


def research_policy(settings=None, *, parent=None):
    settings = deepcopy(DEFAULT_POLICY if settings is None else settings)
    if set(settings) != POLICY_FIELDS or any(
        not isinstance(value, str) or not value.strip() or len(value) > 4000
        for value in settings.values()
    ):
        raise ValueError(
            "Policy must contain bounded investigation/proposal/screening/allocation text"
        )
    value = {"schema_version": 1, "settings": settings, "parent": parent}
    return value | {"identity": digest(canonical(value))}


def patch_policy(policy, patch):
    """Explicit patches cannot change evaluator, data access or total budget fields."""
    if set(patch) != {"base_identity", "changes", "reason", "evidence_refs"}:
        raise ValueError("Invalid research policy patch fields")
    if patch["base_identity"] != policy["identity"]:
        raise ValueError("Policy patch base identity mismatch")
    if not patch["changes"] or not set(patch["changes"]) <= POLICY_FIELDS:
        raise ValueError("Policy patch changes protected or unknown fields")
    if not patch["reason"] or not isinstance(patch["evidence_refs"], list):
        raise ValueError("Policy patch requires reason and evidence references")
    result = research_policy(policy["settings"] | patch["changes"], parent=policy["identity"])
    return result | {"patch": deepcopy(patch)}


def guard_records(records, *, mode="retrospective", cutoff=None, gate="G1"):
    """Return eligible records, never rejected contents; timestamps use shared UTC validation.

    Call on original source/version records before summarization or extraction. Source
    readers must provide actual publication/receipt/version times, not guessed dates.
    """
    if mode not in {"retrospective", "retained"} or gate not in {"G1", "G3"}:
        raise ValueError("Unknown discovery visibility mode/gate")
    boundary = timestamp(cutoff) if mode == "retained" and cutoff else None
    admitted, excluded, reports = [], [], []
    for record in records:
        ref = record.get("id") or record.get("ref") or digest(canonical(record))
        reason = None
        if _contains_terminal(record):
            reason = "terminal_not_research_feedback"
        elif mode == "retained":
            keys = (
                ("created_at",) if gate == "G3" else ("published_at", "received_at", "version_at")
            )
            values = [
                record.get(key, record.get("version_available_at") if key == "version_at" else None)
                for key in keys
            ]
            if boundary is None or not all(values):
                reason = "missing_time_proof"
            else:
                try:
                    if any(timestamp(value) > boundary for value in values):
                        reason = "after_research_cutoff"
                except (TypeError, ValueError):
                    reason = "invalid_time_proof"
        if reason:
            excluded.append(ref)
        else:
            admitted.append(deepcopy(record))
        reports.append({"ref": ref, "rule": reason or "eligible"})
    return admitted, {
        "schema_version": 1,
        "gate": gate,
        "status": "needs_data" if excluded else "pass",
        "cutoff": cutoff,
        "admitted": [r["ref"] for r in reports if r["rule"] == "eligible"],
        "excluded": excluded,
        "rules": reports,
    }


def guard_numerical(
    *,
    prediction_times,
    available_times,
    label_times=(),
    fit_time=None,
    preprocessing_partition="train",
):
    """G2 checks availability, label maturity and learned transform ownership, not scores."""
    reasons = []
    if len(prediction_times) != len(available_times):
        reasons.append("availability_shape")
    else:
        for index, (origin, available) in enumerate(
            zip(prediction_times, available_times, strict=True)
        ):
            try:
                if available is None or timestamp(available) > timestamp(origin):
                    reasons.append(f"unavailable_row:{index}")
            except (TypeError, ValueError):
                reasons.append(f"invalid_time_row:{index}")
    if label_times:
        if fit_time is None or any(
            timestamp(label) >= timestamp(fit_time) for label in label_times
        ):
            reasons.append("immature_training_label")
    if preprocessing_partition != "train":
        reasons.append("preprocessing_outside_training")
    return {
        "schema_version": 1,
        "gate": "G2",
        "status": "reject" if reasons else "pass",
        "rules": reasons,
    }


def now():
    return datetime.now(UTC).isoformat()


def _contains_terminal(value):
    if isinstance(value, dict):
        if value.get("partition") == "terminal" or any(
            key in value for key in ("terminal_evaluation", "terminal_results")
        ):
            return True
        return any(_contains_terminal(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_terminal(item) for item in value)
    return False
