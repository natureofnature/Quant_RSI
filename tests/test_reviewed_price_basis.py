"""Synthetic reviewed-basis fixtures validate bindings, not real market qualification."""

import json
from copy import deepcopy
from decimal import Decimal

import pytest
from test_daily_data import chart, install

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.market.daily_data import collect_daily
from quant_rsi.market.market_contracts import DataContract
from quant_rsi.market.market_data import parse_csv
from quant_rsi.research.reviewed_price_basis import FILES, review_price_basis


def reviewed_fixture(tmp_path, monkeypatch):
    """Create a genuine offline collection and explicitly synthetic manual review."""
    prices, evidence = tmp_path / "prices", tmp_path / "review"
    payload = chart()
    install(monkeypatch, payload)
    collect_daily("SOXL", prices)
    receipt = json.loads((prices / "collection_receipt.json").read_bytes())
    contract = DataContract.model_validate_json((prices / "data_contract.json").read_bytes())
    events = parse_csv(contract, (prices / "events.csv").read_bytes(), receipt["completed_at"])
    split = next(
        e
        for e in events
        if e.kind == "CORPORATE_ACTION_NOTICE" and json.loads(e.action)["event_kind"] == "splits"
    )
    evidence.mkdir()
    sources = []
    for role in (
        "issuer_action",
        "provider_semantics",
        "independent_quote_semantics",
        "independent_quotes",
        "human_review",
    ):
        name = role + ".txt"
        body = ("Synthetic offline attestation, not actual market evidence: " + role).encode()
        (evidence / name).write_bytes(body)
        sources.append(
            {
                "role": role,
                "path": name,
                "sha256": digest(body),
                "url": None if role == "human_review" else "https://example.org/" + name,
                "description": "Synthetic fixture reviewed only for engineering tests",
            }
        )
    quotes = {(e.kind, e.session_id): e for e in events if e.kind in {"OPEN_QUOTE", "CLOSE_QUOTE"}}
    anchors = []
    for session in contract.calendar.sessions:
        if session.open_at.date().isoformat() in {"2026-05-29", "2026-06-01", "2026-06-30"}:
            anchors.append(
                {
                    "session_id": session.session_id,
                    "open": format(Decimal(quotes["OPEN_QUOTE", session.session_id].price), ".2f"),
                    "close": format(
                        Decimal(quotes["CLOSE_QUOTE", session.session_id].price), ".2f"
                    ),
                    "source_path": "independent_quotes.txt",
                }
            )
    review = {
        "kind": "ReviewedResearchPriceBasis",
        "schema_version": 1,
        "scope": "snapshot_research_only",
        "symbol": "SOXL",
        "currency": "USD",
        "price_basis": "split_adjusted_excluding_distributions",
        "transform": "identity",
        "qualification_cutoff": "2026-06-30",
        "reviewer": "Synthetic offline fixture",
        "reviewed_at": "2026-10-26T12:00:00Z",
        "collection_hashes": {name: digest((prices / name).read_bytes()) for name in FILES},
        "sources": sources,
        "resolved_splits": [
            {
                "event_id": split.event_id,
                "action_sha256": digest(split.action.encode()),
                "effective_at": split.event_at.isoformat(),
                "numerator": "2",
                "denominator": "1",
                "issuer_source_path": "issuer_action.txt",
            }
        ],
        "anchors": anchors,
    }
    (evidence / "review.json").write_bytes(canonical(review))
    return prices, evidence, contract, events, receipt, review


def verify(fixture):
    prices, evidence, contract, events, receipt, review = fixture
    (evidence / "review.json").write_bytes(canonical(review))
    return review_price_basis("SOXL", prices, evidence, contract, events, receipt)


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    return reviewed_fixture(tmp_path, monkeypatch)


def test_review_retains_original_actions_and_does_not_mutate_collection(reviewed):
    prices, _, _, events, _, review = reviewed
    before = {p.name: p.read_bytes() for p in prices.iterdir()}
    original_events = deepcopy(events)
    result = verify(reviewed)
    assert result["resolved_event_ids"] == [review["resolved_splits"][0]["event_id"]]
    assert events == original_events
    assert {p.name: p.read_bytes() for p in prices.iterdir()} == before
    metadata = result["metadata"]
    assert metadata["review"]["transform"] == "identity"
    assert metadata["original_resolved_events"][0]["action"] == next(
        e.action for e in events if e.event_id in result["resolved_event_ids"]
    )
    assert "Human-reviewed attestation" in metadata["limitations"][0]
    assert "not PIT" in metadata["limitations"][1]
    assert "Synthetic offline attestation" not in json.dumps(metadata)


@pytest.mark.parametrize("name", FILES)
def test_every_bound_collection_file_is_checked(reviewed, name):
    prices = reviewed[0]
    path = prices / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="collection hash binding"):
        verify(reviewed)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_pre",
        "missing_post",
        "cutoff",
        "outside",
        "duplicate",
        "mismatch",
        "too_precise",
        "negative",
        "wrong_source",
    ],
)
def test_anchor_admission_is_complete_bounded_and_same_type(reviewed, problem):
    review = reviewed[-1]
    if problem == "missing_pre":
        review["anchors"] = review["anchors"][1:]
    elif problem == "missing_post":
        review["anchors"] = review["anchors"][:1]
    elif problem == "cutoff":
        review["qualification_cutoff"] = "2026-05-29"
    elif problem == "outside":
        review["anchors"][0]["session_id"] = "outside.collection"
    elif problem == "duplicate":
        review["anchors"][1] = deepcopy(review["anchors"][0])
    elif problem == "mismatch":
        review["anchors"][0]["open"] = "999.00"
    elif problem == "too_precise":
        review["anchors"][0]["open"] += "0"
    elif problem == "negative":
        review["anchors"][0]["open"] = "-1.00"
    else:
        review["anchors"][0]["source_path"] = "provider_semantics.txt"
    with pytest.raises(ValueError):
        verify(reviewed)


@pytest.mark.parametrize(
    "problem", ["unknown", "dividend", "ratio", "time", "payload", "duplicate"]
)
def test_only_exact_declared_splits_are_resolved(reviewed, problem):
    review = reviewed[-1]
    split = review["resolved_splits"][0]
    if problem == "unknown":
        split["event_id"] = "invented"
    elif problem == "dividend":
        split["event_id"] = next(
            e.event_id
            for e in reviewed[3]
            if e.kind == "CORPORATE_ACTION_NOTICE"
            and json.loads(e.action)["event_kind"] == "dividends"
        )
    elif problem == "ratio":
        split["numerator"] = "3"
    elif problem == "time":
        split["effective_at"] = "2026-06-02T13:30:00Z"
    elif problem == "payload":
        split["action_sha256"] = "f" * 64
    else:
        review["resolved_splits"].append(deepcopy(split))
    with pytest.raises(ValueError):
        verify(reviewed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("price_basis", "raw"),
        ("price_basis", "dividend_adjusted"),
        ("transform", "divide_by_3"),
        ("scope", "live_execution"),
        ("currency", "EUR"),
        ("symbol", "SOXX"),
    ],
)
def test_other_units_or_scope_cannot_be_smuggled_into_review(reviewed, field, value):
    reviewed[-1][field] = value
    with pytest.raises(ValueError):
        verify(reviewed)


@pytest.mark.parametrize("problem", ["tamper", "missing_role", "traversal", "symlink", "self"])
def test_source_evidence_hashes_and_safe_paths(reviewed, problem, tmp_path):
    evidence, review = reviewed[1], reviewed[-1]
    source = review["sources"][0]
    if problem == "tamper":
        (evidence / source["path"]).write_bytes(b"changed")
    elif problem == "missing_role":
        review["sources"][0]["role"] = "provider_semantics"
    elif problem == "traversal":
        source["path"] = "../outside.txt"
    elif problem == "symlink":
        outside = tmp_path / "outside.txt"
        outside.write_bytes(b"outside")
        (evidence / "link.txt").symlink_to(outside)
        source.update(path="link.txt", sha256=digest(b"outside"))
    else:
        source["path"] = "review.json"
    with pytest.raises(ValueError):
        verify(reviewed)
