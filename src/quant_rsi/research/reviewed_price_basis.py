"""Bind a human-reviewed, identity-only research quote basis to retained evidence.

This validates hashes, declared scope and numerical anchors. It does not interpret
source documents, certify feed independence, authenticate receipts or establish PIT.
"""

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import AwareDatetime, Field, field_validator

from quant_rsi.core.artifacts import decode, digest
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.market.market_contracts import DataContract, Hash, decimal_text
from quant_rsi.market.market_data import parse_csv

FILES = ("collection_receipt.json", "yahoo_chart_raw.json", "data_contract.json", "events.csv")
ROLES = {
    "issuer_action",
    "provider_semantics",
    "independent_quote_semantics",
    "independent_quotes",
    "human_review",
}
LIMITATIONS = (
    "Human-reviewed attestation; document meaning and source independence "
    "are not automatically verified.",
    "Snapshot research only; not PIT, as-traded, total-return, or execution-account certification.",
    "Identity transformation only; original quotes and corporate-action notices remain unchanged.",
)


class ReviewSource(StrictModel):
    role: Literal[
        "issuer_action",
        "provider_semantics",
        "independent_quote_semantics",
        "independent_quotes",
        "human_review",
    ]
    path: str = Field(min_length=1, max_length=240)
    sha256: Hash
    url: str | None = Field(max_length=2000)
    description: str = Field(min_length=1, max_length=2000)


class ReviewedSplit(StrictModel):
    event_id: str
    action_sha256: Hash
    effective_at: AwareDatetime
    numerator: str
    denominator: str
    issuer_source_path: str

    @field_validator("numerator", "denominator")
    @classmethod
    def positive_ratio(cls, value):
        return decimal_text(value, positive=True)


class ReviewedAnchor(StrictModel):
    session_id: str
    open: str
    close: str
    source_path: str

    @field_validator("open", "close")
    @classmethod
    def nearest_cent(cls, value):
        decimal_text(value, positive=True)
        if Decimal(value).as_tuple().exponent != -2:
            raise ValueError("Manual anchor must explicitly record nearest-cent precision")
        return value


class ReviewedResearchPriceBasis(StrictModel):
    kind: Literal["ReviewedResearchPriceBasis"]
    schema_version: Literal[1]
    scope: Literal["snapshot_research_only"]
    symbol: str
    currency: str
    price_basis: Literal["split_adjusted_excluding_distributions"]
    transform: Literal["identity"]
    qualification_cutoff: date
    reviewer: str = Field(min_length=1, max_length=200)
    reviewed_at: AwareDatetime
    collection_hashes: dict[str, Hash]
    sources: tuple[ReviewSource, ...] = Field(min_length=5, max_length=16)
    resolved_splits: tuple[ReviewedSplit, ...] = Field(min_length=1, max_length=16)
    anchors: tuple[ReviewedAnchor, ...] = Field(min_length=2, max_length=64)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(root, relative, maximum=8 * 1024 * 1024):
    name = Path(relative)
    root = Path(root).resolve()
    path = (root / name).resolve()
    _require(
        not name.is_absolute() and ".." not in name.parts and path.is_relative_to(root),
        "Review artifact must be a safe relative file",
    )
    with path.open("rb") as stream:
        body = stream.read(maximum + 1)
    _require(len(body) <= maximum, "Reviewed price artifact exceeds size bound")
    return body


def review_price_basis(symbol, collection_root, review_dir, contract, events, receipt):
    """Return resolved IDs plus auditable metadata; never alter or filter events/quotes."""
    raw_review = _read(review_dir, "review.json", 128 * 1024)
    review = ReviewedResearchPriceBasis.model_validate_json(raw_review)
    _require(
        review.symbol == symbol and review.currency == contract.currency,
        "Reviewed symbol/currency differs from collection",
    )
    _require(
        review.reviewed_at.date() >= review.qualification_cutoff,
        "Review precedes qualification cutoff",
    )
    _require(set(review.collection_hashes) == set(FILES), "Incomplete collection hash bindings")
    originals = {name: _read(collection_root, name) for name in FILES}
    _require(
        review.collection_hashes == {k: digest(v) for k, v in originals.items()},
        "Reviewed collection hash binding differs",
    )
    _require(
        decode(originals["collection_receipt.json"]) == receipt,
        "Passed receipt differs from bound collection",
    )
    _require(
        DataContract.model_validate_json(originals["data_contract.json"]) == contract,
        "Passed contract differs from bound collection",
    )
    _require(
        tuple(parse_csv(contract, originals["events.csv"], receipt["completed_at"]))
        == tuple(events),
        "Passed events differ from bound collection",
    )
    _require(contract.instruments == (symbol,), "Review requires one matching instrument")
    _require(
        receipt["normalized_inputs"]
        == {
            "contract_file_sha256": digest(originals["data_contract.json"]),
            "events_file_sha256": digest(originals["events.csv"]),
        },
        "Collection normalized hashes differ",
    )
    raw_hash = digest(originals["yahoo_chart_raw.json"])
    _require(
        any(d.get("sha256") == raw_hash for d in receipt["downloads"]),
        "Raw chart is not bound to a collection download",
    )
    chart = decode(originals["yahoo_chart_raw.json"])["chart"]["result"][0]
    _require(
        chart["meta"]["symbol"] == symbol and chart["meta"]["currency"] == review.currency,
        "Raw provider symbol/currency differs",
    )
    sources = {s.path: s for s in review.sources}
    _require(
        len(sources) == len(review.sources) and {s.role for s in review.sources} == ROLES,
        "Missing or duplicate reviewed source artifacts",
    )
    for source in review.sources:
        _require(source.path != "review.json", "Review cannot cite itself as source evidence")
        _require(
            source.role == "human_review"
            or (source.url is not None and source.url.startswith("https://")),
            "Source evidence needs its original HTTPS URL",
        )
        _require(
            digest(_read(review_dir, source.path)) == source.sha256,
            "Reviewed source artifact hash differs",
        )
    sessions = {
        s.session_id: s for s in contract.calendar.sessions[: receipt["historical_sessions"]]
    }
    zone = ZoneInfo(contract.calendar.timezone)
    quotes = {(e.kind, e.session_id): e for e in events if e.kind in {"OPEN_QUOTE", "CLOSE_QUOTE"}}
    _require(
        len(quotes) == sum(e.kind in {"OPEN_QUOTE", "CLOSE_QUOTE"} for e in events),
        "Ambiguous normalized quotes",
    )
    _require(
        len({a.session_id for a in review.anchors}) == len(review.anchors),
        "Duplicate reviewed anchor sessions",
    )
    anchor_days = []
    for anchor in review.anchors:
        _require(anchor.session_id in sessions, "Anchor is outside historical collection")
        day = sessions[anchor.session_id].open_at.astimezone(zone).date()
        _require(day <= review.qualification_cutoff, "Anchor exceeds qualification cutoff")
        _require(
            anchor.source_path in sources
            and sources[anchor.source_path].role == "independent_quotes",
            "Anchor requires independent quote artifact",
        )
        for kind, expected in (("OPEN_QUOTE", anchor.open), ("CLOSE_QUOTE", anchor.close)):
            event = quotes.get((kind, anchor.session_id))
            _require(
                event is not None
                and event.quality == "valid"
                and event.revision == 0
                and event.instrument == symbol
                and event.price is not None,
                "Anchor has no unique valid normalized quote",
            )
            _require(
                abs(Decimal(event.price) - Decimal(expected)) <= Decimal("0.005"),
                "Independent anchor differs beyond nearest-cent tolerance",
            )
        anchor_days.append(day)
    actions = {e.event_id: e for e in events if e.kind == "CORPORATE_ACTION_NOTICE"}
    resolved, original_events = [], []
    for split in review.resolved_splits:
        _require(
            split.event_id not in resolved and split.event_id in actions,
            "Unknown or duplicate resolved action ID",
        )
        event = actions[split.event_id]
        payload = decode(event.action.encode())
        _require(
            payload.get("event_kind") == "splits"
            and event.instrument == symbol
            and event.quality == "valid"
            and event.revision == 0,
            "Only valid unrevised split notices can be resolved",
        )
        _require(
            digest(event.action.encode()) == split.action_sha256
            and event.event_at == split.effective_at,
            "Resolved action payload/time differs",
        )
        ratio = (Decimal(split.numerator), Decimal(split.denominator))
        _require(
            ratio[0] != ratio[1]
            and ratio
            == (Decimal(str(payload.get("numerator"))), Decimal(str(payload.get("denominator")))),
            "Resolved split ratio differs",
        )
        _require(
            type(payload.get("date")) is int and event.event_at.timestamp() == payload["date"],
            "Normalized split time differs from provider action",
        )
        ratio_text = str(payload.get("splitRatio", "")).split(":")
        _require(
            len(ratio_text) == 2
            and ratio == tuple(Decimal(decimal_text(v, positive=True)) for v in ratio_text),
            "Provider split ratio fields disagree",
        )
        _require(
            any(
                action == {k: v for k, v in payload.items() if k not in {"provider", "event_kind"}}
                for action in chart.get("events", {}).get("splits", {}).values()
            ),
            "Resolved split does not match retained provider action",
        )
        _require(
            split.issuer_source_path in sources
            and sources[split.issuer_source_path].role == "issuer_action",
            "Resolved split lacks issuer action evidence",
        )
        day = event.event_at.astimezone(zone).date()
        _require(
            day <= review.qualification_cutoff
            and any(d < day for d in anchor_days)
            and any(d >= day for d in anchor_days),
            "Every resolved split needs pre/post anchors within cutoff",
        )
        resolved.append(split.event_id)
        original_events.append(event.model_dump(mode="json"))
    return {
        "resolved_event_ids": resolved,
        "metadata": {
            "review_identity": digest(raw_review),
            "review": review.model_dump(mode="json"),
            "original_resolved_events": original_events,
            "limitations": list(LIMITATIONS),
        },
    }
