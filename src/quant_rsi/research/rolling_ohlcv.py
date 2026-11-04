"""Read optional daily leaves from hash-bound raw evidence without changing MarketData."""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

import numpy as np

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.market.daily_data import CHART_URL, EASTERN, MAX_RESPONSE_BYTES, _chart, _price
from quant_rsi.market.market_contracts import timestamp
from quant_rsi.research.demo_prices import read_daily_history
from quant_rsi.research.rolling_windows import read_calendar

NOTICE = (
    "High/low prices and volume are retained Yahoo daily fields on the provider's quote/share "
    "basis, not certified original as-traded units or PIT data. Actual receipt timestamps are "
    "preserved; historical close-time availability is the existing research assumption. "
    "Missing optional values stay unavailable; zero volume is an observed zero."
)


def _source_record(receipt, symbol):
    records = [
        row
        for row in receipt.get("downloads", [])
        if Path(row.get("path", "")).name == "yahoo_chart_raw.json"
    ]
    if len(records) != 1:
        raise ValueError("OHLCV requires one retained chart download receipt")
    record = records[0]
    url = record.get("url", "")
    parsed = urlsplit(url)
    expected = urlsplit(CHART_URL + quote(symbol, safe=""))
    query = parse_qs(parsed.query, keep_blank_values=True)
    window = {key: value for key, value in query.items() if key not in {"interval", "events"}}
    valid_window = window in ({"range": ["1y"]}, {"range": ["2y"]})
    if set(window) == {"period1", "period2"}:
        valid_window = all(len(v) == 1 and v[0].isdigit() for v in window.values())
        if valid_window:
            valid_window = int(window["period1"][0]) < int(window["period2"][0])
    if (
        (parsed.scheme, parsed.netloc, parsed.path)
        != (expected.scheme, expected.netloc, expected.path)
        or parsed.fragment
        or record.get("response_url") != url
        or query.get("interval") != ["1d"]
        or query.get("events") != ["div,splits"]
        or not valid_window
        or record.get("http_status") != 200
        or record.get("truncated") is not False
    ):
        raise ValueError("OHLCV receipt differs from the registered successful daily provider")
    if (
        not timestamp(record["requested_at"])
        <= timestamp(record["received_at"])
        <= timestamp(receipt["completed_at"])
    ):
        raise ValueError("OHLCV receipt timestamps are inconsistent")
    return record


def load_ohlcv_history(symbol, directory, data, info):
    """Return independent aligned columns plus JSON definitions, provenance and coverage.

    The normalized reader retains corporate-action and calendar/quote admission
    rules. Original files, caller arrays and completed studies are never rewritten.
    """
    root = Path(directory).resolve()
    validated, validated_info = read_daily_history(symbol, root)
    if (
        data.dates != validated.dates
        or data.instruments != validated.instruments
        or not np.array_equal(data.opens, validated.opens)
        or not np.array_equal(data.closes, validated.closes)
        or any(
            info.get(k) != validated_info[k]
            for k in ("source_id", "input_hashes", "price_cutoff", "session_close_times")
        )
    ):
        raise ValueError("OHLCV target differs from the validated daily collection")
    calendar = read_calendar(root)
    receipt = json.loads((root / "collection_receipt.json").read_bytes())
    record = _source_record(receipt, symbol)
    path = root / "yahoo_chart_raw.json"
    if not path.resolve().is_relative_to(root) or path.stat().st_size > MAX_RESPONSE_BYTES:
        raise ValueError("OHLCV raw source escapes collection or exceeds byte limit")
    body = path.read_bytes()
    raw_hash = digest(body)
    if record.get("bytes") != len(body) or record.get("sha256") != raw_hash:
        raise ValueError("OHLCV raw bytes/hash differ from receipt")
    identity = {"response_ref": raw_hash, "calendar_ref": calendar["sha256"]}
    identities = {"raw_calendar": identity}
    if receipt.get("price_cutoff") is not None:
        identities["raw_calendar_cutoff"] = identity | {"price_cutoff": receipt["price_cutoff"]}
    prefix = f"yahoo_{symbol}_{data.dates[0]}_{data.dates[-1]}_"
    matched_schema = next(
        (
            schema
            for schema, evidence in identities.items()
            if validated_info["source_id"] == prefix + digest(canonical(evidence))
        ),
        None,
    )
    if matched_schema is None:
        raise ValueError("OHLCV raw/calendar evidence differs from the normalized source identity")
    # Older retained collections recorded a cutoff without hashing it into source_id.
    # Accept that exact identity schema, but enforce its explicit observation boundary.
    latest_close = max(timestamp(value) for value in validated_info["session_close_times"])
    boundaries = [timestamp(receipt["completed_at"])]
    if receipt.get("price_cutoff") is not None:
        boundaries.append(timestamp(receipt["price_cutoff"]))
    if any(latest_close > boundary for boundary in boundaries):
        raise ValueError("OHLCV selected close exceeds retained cutoff or completion time")
    _, times, quotes = _chart(body, symbol)
    contract = json.loads((root / "data_contract.json").read_bytes())
    sessions = contract["calendar"]["sessions"][: len(data.dates)]
    by_open = {value: index for index, value in enumerate(times)}
    counts = Counter(datetime.fromtimestamp(value, EASTERN).date().isoformat() for value in times)
    if any(counts[day] != 1 for day in data.dates):
        raise ValueError("OHLCV source needs exactly one row for each selected session date")
    columns = {
        "price." + field: np.full(len(data.dates), np.nan) for field in ("high", "low", "volume")
    }
    for index, session in enumerate(sessions):
        opening = datetime.fromisoformat(session["open_at"]).timestamp()
        if session["session_id"] != data.dates[index] or opening not in by_open:
            raise ValueError("OHLCV source lacks an exact selected session opening")
        source_index = by_open[opening]
        for field, expected in (("open", data.opens[index, 0]), ("close", data.closes[index, 0])):
            if float(_price(quotes[field][source_index])) != expected:
                raise ValueError("OHLCV raw open/close differ from normalized quotes")
        o, c = quotes["open"][source_index], quotes["close"][source_index]
        for field in ("high", "low", "volume"):
            value = quotes[field][source_index] if field in quotes else None
            if value is None:
                continue
            if type(value) not in {int, float} or not np.isfinite(value):
                raise ValueError(f"Invalid nonmissing OHLCV {field}")
            if field == "volume":
                if value < 0 or value != int(value):
                    raise ValueError("OHLCV volume must be a nonnegative integral share count")
            else:
                if (
                    value <= 0
                    or (field == "high" and value < max(o, c))
                    or (field == "low" and value > min(o, c))
                ):
                    raise ValueError("OHLCV high/low conflict with the retained bar")
                value = float(_price(value))
            columns["price." + field][index] = value
    coverage, definitions = [], []
    for identity, values in columns.items():
        field = identity.split(".", 1)[1]
        available = int(np.isfinite(values).sum())
        coverage.append(
            {
                "factor_id": identity,
                "field_present": field in quotes,
                "sessions": len(values),
                "available": available,
                "missing": len(values) - available,
                "observed_zero": int((values == 0).sum()),
            }
        )
        definitions.append(
            {
                "factor_id": identity,
                "kind": "numeric",
                "unit": "shares" if field == "volume" else "price",
                "runnable": available > 0,
                "availability_reason": "retained_daily_field"
                if available
                else "missing_retained_daily_field",
            }
        )
    return {
        "dates": list(data.dates),
        "columns": columns,
        "definitions": definitions,
        "coverage": coverage,
        "notice": NOTICE,
        "provenance": {
            "symbol": symbol,
            "source_id": validated_info["source_id"],
            "source_id_schema": matched_schema,
            "normalized_inputs": validated_info["input_hashes"],
            "calendar_sha256": calendar["sha256"],
            "raw_sha256": raw_hash,
            "requested_url": record["url"],
            "response_url": record["response_url"],
            "requested_at": record["requested_at"],
            "received_at": record["received_at"],
            "price_cutoff": validated_info["price_cutoff"],
            "session_close_times": validated_info["session_close_times"],
            "price_unit_claim": "provider_quote_basis_USD_per_share_assumption",
            "volume_unit_claim": "provider_share_count_assumption",
        },
    }
