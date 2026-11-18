"""Retrieved knowledge references and cumulative attempt counts for a study (#159 D8)."""

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research.sector_map import post_period


def knowledge_references(*, sector, related, prior_memory, first_decision_time):
    """Every cross-run or reference item a study's decisions can see, with trust labels.

    Reference knowledge reviewed after the first decision carries post_period_knowledge.
    Prior-study records are results or interpretations from earlier runs, read through the
    run's history watermark.
    """
    rows = []
    if sector:
        rows.append(
            {
                "kind": "reference",
                "source": "sector_map",
                "ref": sector["ref"],
                "scope": sector["sector_id"],
                "reviewed_on": sector["reviewed_on"],
                "post_period_knowledge": post_period(sector["reviewed_on"], first_decision_time),
            }
        )
    if related and related.get("reviewed_on"):
        rows.append(
            {
                "kind": "reference",
                "source": "relationship_profile",
                "ref": digest(canonical(related.get("relationships", []))),
                "scope": ", ".join(related.get("stocks", [])),
                "reviewed_on": related["reviewed_on"],
                "post_period_knowledge": post_period(related["reviewed_on"], first_decision_time),
            }
        )
    for record in prior_memory or []:
        rows.append(
            {
                "kind": "interpretation" if record.get("kind") == "interpretation" else "result",
                "source": "prior_study",
                "ref": record.get("ref") or record.get("projection_of"),
                "scope": record.get("prior_study") or "prior study",
                "reviewed_on": (record.get("created_at") or "")[:10] or None,
                "post_period_knowledge": False,
            }
        )
    return rows


def evaluated_attempts(history_dir, symbol, horizon_sessions, through):
    """Evaluated fits of earlier studies for this symbol and horizon, through the watermark."""
    if not history_dir:
        return {"prior_studies": 0, "prior_evaluated_attempts": 0, "through": through}
    from quant_rsi.research.study_history import ledger_studies

    studies = ledger_studies(
        history_dir, symbol, horizon_sessions=horizon_sessions, through=through
    )
    return {
        "prior_studies": len(studies),
        "prior_evaluated_attempts": sum(len(s.get("attempts") or []) for s in studies),
        "through": through,
        "rule": "history_ledger_attempts_through_watermark_v1",
    }
