"""Curated, dated sector research maps (#157, #159 Phase 2 and D8 reference knowledge).

A sector map is reference knowledge, changed only by human review. It names the sector
and market comparators a sector-aware baseline uses, and the demand-chain tiers a
discovery run should investigate. Tier entities are reviewed leads with identity sources,
not established relationships: the decision agent still needs evidence for any link.
"""

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, model_validator

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.factors.contracts import InstrumentId
from quant_rsi.market.availability import date_only_available_at, usable_at

TIERS = (
    "direct_customers",
    "customers_of_customers",
    "end_demand_and_funding",
    "cycle_and_capital",
)
COVERAGE_RULE = "sector_tier_coverage_v1"
BASELINE_KIND = "sector_aware_trailing_controls_v1"
COMPARATOR_WINDOW = 10


class Comparator(StrictModel):
    symbol: InstrumentId
    reason: str = Field(min_length=1)
    source_urls: list[HttpUrl] = Field(min_length=1, max_length=3)


class TierEntity(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    symbol: InstrumentId | None = None
    lead: str = Field(min_length=1, max_length=400)
    source_urls: list[HttpUrl] = Field(min_length=1, max_length=3)


class Tier(StrictModel):
    tier: Literal[
        "direct_customers",
        "customers_of_customers",
        "end_demand_and_funding",
        "cycle_and_capital",
    ]
    question: str = Field(min_length=1, max_length=400)
    entities: list[TierEntity] = Field(min_length=1, max_length=12)


class UnavailableInput(StrictModel):
    name: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class SectorMap(StrictModel):
    sector_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    reviewed_on: date
    members: list[InstrumentId] = Field(min_length=1, max_length=40)
    sector_comparator: Comparator
    market_comparator: Comparator
    tiers: list[Tier] = Field(min_length=1, max_length=len(TIERS))
    unavailable_inputs: list[UnavailableInput] = Field(default_factory=list, max_length=12)
    notice: str = Field(min_length=1)

    @model_validator(mode="after")
    def consistent(self):
        if len(set(self.members)) != len(self.members):
            raise ValueError("Sector members must be distinct")
        if self.sector_comparator.symbol == self.market_comparator.symbol:
            raise ValueError("Sector and market comparators must differ")
        names = [t.tier for t in self.tiers]
        if len(set(names)) != len(names):
            raise ValueError("Each tier appears once")
        return self


def load_sector_map(symbol, factor_root):
    """The one reviewed sector map whose members include symbol, or None.

    Maps live in <factor_root>/shared/sectors/*.json. A symbol in two maps is ambiguous
    and refused rather than resolved by file order.
    """
    directory = Path(factor_root) / "shared" / "sectors"
    if not directory.is_dir():
        return None
    selected = []
    for path in sorted(directory.glob("*.json")):
        sector = SectorMap.model_validate_json(path.read_bytes())
        if symbol in sector.members:
            selected.append((path, sector))
    if not selected:
        return None
    if len(selected) > 1:
        raise ValueError(f"{symbol} belongs to more than one sector map")
    path, sector = selected[0]
    body = sector.model_dump(mode="json")
    return body | {"ref": digest(canonical(body)), "path": f"shared/sectors/{path.name}"}


def comparator_symbols(sector, target):
    """Comparators a target's baseline uses; a comparator is never the target itself."""
    return [
        sector[role]["symbol"]
        for role in ("sector_comparator", "market_comparator")
        if sector[role]["symbol"] != target
    ]


def comparator_baseline_ids(sector, target):
    """Trailing comparator returns ending at the origin close (never over the label)."""
    return [
        f"peer.{symbol}.peer_return_{COMPARATOR_WINDOW}"
        for symbol in comparator_symbols(sector, target)
    ]


def post_period(available, first_decision_time):
    """Whether knowledge was unavailable at the study's first decision (D8, under D3).

    ``available`` is a review date or a creation timestamp. It is post-period unless usable
    at the first decision instant: strictly before it (D3.1), and a date-only value from the
    start of the next day (D3.2). Missing or unreadable times are unknown (None), never
    in period.
    """
    if not available or not first_decision_time:
        return None
    try:
        instant = datetime.fromisoformat(str(first_decision_time))
        text = str(available)
        if len(text) == 10:
            available_at = date_only_available_at(text)
        else:
            available_at = datetime.fromisoformat(text)
        if instant.tzinfo is None or available_at.tzinfo is None:
            return None
        return not usable_at(available_at, instant)
    except ValueError:
        return None


def _mentions(entity, text):
    names = [entity["name"]] + ([entity["symbol"]] if entity.get("symbol") else [])
    return any(re.search(r"(?<![\w])" + re.escape(n) + r"(?![\w])", text, re.I) for n in names)


def tier_coverage(sector, target, memory, bundles, evidence):
    """Per-tier coverage of a decision's visible knowledge (sector_tier_coverage_v1).

    measured: an available acquired input for the entity's symbol;
    evidenced: a retained source or survey finding names the entity;
    gap: neither. The issuer tier is the prediction target itself.
    """
    measured = {
        (b.get("request") or {}).get("symbol")
        for b in bundles or []
        if b.get("status") == "available"
    }
    texts = [
        " ".join(str(a.get(k) or "") for k in ("title", "headline", "text")) for a in evidence or []
    ]
    for entry in memory or []:
        result = entry.get("result")
        if entry.get("kind") == "survey" and isinstance(result, dict):
            texts.append(json.dumps(result.get("findings", []), ensure_ascii=False))
            texts.extend(
                " ".join(str(a.get(k) or "") for k in ("title", "headline", "text"))
                for a in result.get("articles", [])
            )
        if entry.get("kind") == "acquire" and isinstance(result, dict):
            if result.get("status") == "available":
                measured.add((result.get("request") or {}).get("symbol"))

    def status(entity):
        if entity.get("symbol") and entity["symbol"] in measured:
            return "measured"
        if any(_mentions(entity, text) for text in texts):
            return "evidenced"
        return "gap"

    issuer = {"name": target, "symbol": target}
    rows = [{"tier": "issuer", "entities": [{"name": target, "status": status(issuer)}]}]
    for tier in sector["tiers"]:
        leads = tier.get("entities") or tier.get("leads") or []
        entities = [{"name": e["name"], "status": status(e)} for e in leads]
        rows.append({"tier": tier["tier"], "question": tier["question"], "entities": entities})
    for row in rows:
        statuses = [e["status"] for e in row["entities"]]
        row["counts"] = {s: statuses.count(s) for s in ("measured", "evidenced", "gap")}
    return {"rule": COVERAGE_RULE, "sector_ref": sector["ref"], "tiers": rows}


def decision_sector_context(sector, target):
    """The compact sector map a decision sees; identity sources stay in the reference."""
    return {
        "sector_id": sector["sector_id"],
        "ref": sector["ref"],
        "reviewed_on": sector["reviewed_on"],
        "comparators": comparator_symbols(sector, target),
        "tiers": [
            {
                "tier": t["tier"],
                "question": t["question"],
                "leads": [
                    {k: e[k] for k in ("name", "symbol", "lead") if e.get(k)} for e in t["entities"]
                ],
            }
            for t in sector["tiers"]
        ],
        "unavailable_inputs": sector["unavailable_inputs"],
        "notice": "Reviewed leads, not established relationships; evidence is still required.",
    }
