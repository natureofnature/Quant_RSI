"""Publication-dated financial facts as session-constant intraday leaves (design section 5.4).

A session's `f_` values are the research study's quarterly business metrics computed as of
that session's open, so only facts usable before trading began are read. Both sources record
dates only, so facts become usable at 05:00 UTC on the day after the SEC filing or the
issuer's publication; a release-day session never sees its own release. The values then hold
for every decision bar of the session.
"""

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.intraday.contracts import IntradayDevelopmentPlan
from quant_rsi.intraday.features import BarSource
from quant_rsi.market.market_contracts import MarketSession
from quant_rsi.research.study_financials import METRICS, prepare_financials

DAYS_SINCE_REPORT = "f_days_since_report"


def financial_binding_digest(snapshot: dict) -> str:
    """Bind the verified payload and all metadata that changes its interpretation."""
    provider = snapshot["provider"]
    if provider not in {"issuer_earnings", "sec_companyfacts"}:
        raise ValueError(f"unsupported financial provider {provider}")
    payload = snapshot["records"] if provider == "issuer_earnings" else snapshot["companyfacts"]
    if digest(canonical(payload)) != snapshot["sha256"]:
        raise ValueError("financial snapshot does not match its digest")
    profile = None if provider == "issuer_earnings" else snapshot["profile"]
    if profile is not None and (
        profile["symbol"] != snapshot["symbol"] or int(profile["cik"]) != int(payload["cik"])
    ):
        raise ValueError("financial snapshot issuer/profile identity mismatch")
    return digest(
        canonical(
            dict(
                provider=provider,
                symbol=snapshot["symbol"],
                payload_sha256=snapshot["sha256"],
                profile=profile,
            )
        )
    )


def load_financial_snapshot(path: Path) -> dict:
    """A retained snapshot whose payload still matches its recorded digest."""
    snapshot = json.loads(Path(path).read_bytes())
    financial_binding_digest(snapshot)
    return snapshot


def session_values(snapshot: dict, sessions: Sequence[MarketSession]) -> dict[str, dict]:
    """Session ID -> leaf -> value (None when missing), as of each session's open."""
    cutoffs = [session.open_at.isoformat() for session in sessions]
    bundle = prepare_financials(snapshot, cutoffs, required=[])
    values = {}
    for row, (session, evidence) in enumerate(zip(sessions, bundle["evidence"], strict=True)):
        leaves = {"f_" + m: bundle["columns"]["financial." + m][row] for m in METRICS}
        revenue = (evidence.get("current") or {}).get("facts", {}).get("revenue")
        leaves[DAYS_SINCE_REPORT] = (
            None
            if revenue is None
            else (session.open_at - datetime.fromisoformat(revenue["available_at"])).total_seconds()
            / 86400
        )
        values[session.session_id] = leaves
    return values


class FundamentalSource:
    """A bar source plus the per-session fundamental values its `f_` leaves read."""

    def __init__(self, source: BarSource, fundamentals: dict[str, dict[str, dict]]):
        self.source, self.fundamentals = source, fundamentals

    @property
    def contract(self):
        return self.source.contract

    def block(self, instrument, grid, session_id):
        return self.source.block(instrument, grid, session_id)

    def __getattr__(self, name):
        return getattr(self.source, name)


def with_fundamentals(
    source: BarSource, plan: IntradayDevelopmentPlan, snapshots: Sequence[dict]
) -> BarSource:
    """Bind the plan's financial snapshots to `source`; refuse missing or substituted ones."""
    if not plan.fundamentals:
        if snapshots:
            raise ValueError("financial snapshots were given but the plan uses no f_ leaves")
        return source
    by_digest = {financial_binding_digest(snapshot): snapshot for snapshot in snapshots}
    targets = set(source.contract.targets)
    bound = {item.instrument for item in plan.fundamentals}
    if bound != targets:
        raise ValueError(f"fundamentals must cover exactly the targets {sorted(targets)}")
    sessions = source.contract.calendar.sessions
    values = {}
    for item in plan.fundamentals:
        if item.binding_sha256 is None:
            raise ValueError(
                "fundamental plan lacks binding_sha256; create a new plan binding the retained "
                "payload and its interpretation; preserve completed artifacts unchanged"
            )
        snapshot = by_digest.get(item.binding_sha256)
        if snapshot is None:
            raise ValueError(
                f"no financial snapshot binding {item.binding_sha256} for {item.instrument}"
            )
        if snapshot["sha256"] != item.snapshot_sha256:
            raise ValueError("financial payload digest differs from the plan")
        if snapshot["symbol"] != item.instrument:
            raise ValueError(f"financial snapshot {snapshot['symbol']} is not {item.instrument}")
        values[item.instrument] = session_values(snapshot, sessions)
    return FundamentalSource(source, values)
