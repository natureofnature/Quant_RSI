"""Reuse a retained public daily collection without creating a research campaign."""

import json
from pathlib import Path

import numpy as np

from quant_rsi.core.artifacts import digest
from quant_rsi.core.data import MarketData
from quant_rsi.market.daily_data import HISTORY_SESSIONS, collect_daily
from quant_rsi.market.market_contracts import DataContract
from quant_rsi.market.market_data import parse_csv


def load_daily_prices(symbol, directory, *, horizon_sessions=5, historical_sessions=None):
    if type(horizon_sessions) is not int or horizon_sessions not in (3, 5, 20, 60):
        raise ValueError("Prediction horizon must be 3, 5, 20 or 60 trading sessions")
    required = HISTORY_SESSIONS + 3 * max(0, horizon_sessions - 5)
    if historical_sessions is not None:
        if type(historical_sessions) is not int or not required <= historical_sessions <= 445:
            raise ValueError("Historical sessions must cover the horizon and not exceed 445")
        required = historical_sessions
    root = Path(directory).resolve()
    if not root.exists() or not any(root.iterdir()):
        if horizon_sessions == 5 and required == HISTORY_SESSIONS:
            collect_daily(symbol, root)
        else:
            collect_daily(
                symbol,
                root,
                historical_sessions=required,
                future_sessions=horizon_sessions,
            )
    try:
        return _read_collection(
            symbol, root, horizon_sessions=horizon_sessions, historical_sessions=required
        )
    except (KeyError, TypeError, IndexError):
        raise ValueError(
            "Incomplete daily collection; use a fresh --prices-dir to collect"
        ) from None


def read_daily_history(symbol, directory, *, price_basis_review=None):
    """Validate retained quotes without attaching a fixed prediction horizon or fetching."""
    return _read_collection(
        symbol,
        Path(directory).resolve(),
        horizon_sessions=None,
        price_basis_review=price_basis_review,
    )


def _read_collection(
    symbol,
    root,
    *,
    horizon_sessions=5,
    historical_sessions=HISTORY_SESSIONS,
    price_basis_review=None,
):
    receipt = json.loads((root / "collection_receipt.json").read_bytes())
    if (receipt.get("kind"), receipt.get("status"), receipt.get("ticker")) != (
        "PublicDailyCollection",
        "collected",
        symbol,
    ):
        raise ValueError("Daily collection must be successful and match the requested symbol")
    contract_raw = (root / "data_contract.json").read_bytes()
    events_raw = (root / "events.csv").read_bytes()
    if receipt["normalized_inputs"] != {
        "contract_file_sha256": digest(contract_raw),
        "events_file_sha256": digest(events_raw),
    }:
        raise ValueError("Daily collection input hashes differ from the retained receipt")
    contract = DataContract.model_validate_json(contract_raw)
    count = receipt["historical_sessions"]
    if contract.instruments != (symbol,) or type(count) is not int or count < HISTORY_SESSIONS:
        raise ValueError("Daily collection universe or historical window differs")
    if count < historical_sessions:
        raise ValueError("Insufficient retained history for this horizon; use a fresh --prices-dir")
    sessions = contract.calendar.sessions[:count]
    if horizon_sessions is not None and len(contract.calendar.sessions) < count + horizon_sessions:
        raise ValueError(
            "Daily collection needs a published future calendar for this horizon; "
            "use a fresh --prices-dir"
        )
    events = parse_csv(contract, events_raw, receipt["completed_at"])
    reviewed = None
    if price_basis_review is not None:
        from quant_rsi.research.reviewed_price_basis import review_price_basis

        reviewed = review_price_basis(symbol, root, price_basis_review, contract, events, receipt)
    resolved = set(reviewed["resolved_event_ids"]) if reviewed else set()
    quotes = {}
    for event in events:
        if event.kind in {"OPEN_QUOTE", "CLOSE_QUOTE"}:
            key = (event.kind, event.session_id)
            if event.revision or event.quality != "valid" or key in quotes:
                raise ValueError("Daily demo needs one valid unrevised quote per session and phase")
            quotes[key] = float(event.price)
        elif event.kind == "CORPORATE_ACTION_NOTICE" and event.event_at >= sessions[0].open_at:
            if (
                json.loads(event.action).get("event_kind") != "dividends"
                and event.event_id not in resolved
            ):
                raise ValueError("Unresolved corporate action in the daily price window")
    expected = {
        (kind, session.session_id) for session in sessions for kind in ("OPEN_QUOTE", "CLOSE_QUOTE")
    }
    if quotes.keys() != expected or receipt["origin_session_id"] != sessions[-1].session_id:
        raise ValueError("Daily collection has incomplete or unexpected price sessions")
    data = MarketData(
        tuple(session.session_id for session in sessions),
        (symbol,),
        np.array([[quotes["OPEN_QUOTE", session.session_id]] for session in sessions]),
        np.array([[quotes["CLOSE_QUOTE", session.session_id]] for session in sessions]),
    )
    info = {
        "provider": "public_daily_prices",
        "collection_directory": str(root),
        "received_at": receipt["completed_at"],
        "price_cutoff": receipt.get("price_cutoff"),
        "source_id": contract.source_id,
        "input_hashes": receipt["normalized_inputs"],
        "price_basis": "raw",
        "currency": contract.currency,
        "qualifications": receipt["qualifications"],
        "session_close_times": [session.close_at.isoformat() for session in sessions],
    }
    if reviewed:
        info.update(
            price_basis="split_adjusted_excluding_distributions",
            price_basis_review=reviewed["metadata"],
            price_basis_interpretation=(
                "Reviewed research interpretation supersedes the collector's explicit raw "
                "currency-per-share assumption for this snapshot. Original files and prices "
                "are unchanged; this does not certify as-traded or execution prices."
            ),
        )
    if horizon_sessions is not None:
        info.update(
            horizon_sessions=horizon_sessions,
            latest_target_session=contract.calendar.sessions[
                count - 1 + horizon_sessions
            ].session_id,
        )
    return data, info
