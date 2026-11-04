"""Read-only official market-risk data and explicit prior-date reconstruction.

Cboe VIX closes are index points; Treasury 10-year par yields are percent, not
fractions. Historical feeds retrieved now are latest vintages. A strict prior-date
alignment is a stated lag assumption, not proof of historical publication time.
"""

import csv
import io
import json
import math
import re
import xml.etree.ElementTree as ET
from bisect import bisect_left
from datetime import date, datetime
from pathlib import Path

import numpy as np

from quant_rsi.core.artifacts import digest
from quant_rsi.market.market_contracts import timestamp

VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
    "?data=daily_treasury_yield_curve&field_tdr_date_value="
)
ATOM = "{http://www.w3.org/2005/Atom}"
META = "{http://schemas.microsoft.com/ado/2007/08/dataservices/metadata}"
DATA = "{http://schemas.microsoft.com/ado/2007/08/dataservices}"
MAX_AGE_DAYS = 7
NOTICE = (
    "Reconstructed latest-vintage official data, not retained historical releases. "
    "Use only observations dated strictly before the equity session, at most seven "
    "calendar days old. This date lag is an explicit availability assumption. "
    "No same-day values, interpolation, zero fill, or carry across explicit missing values."
)


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid numerical risk observation") from exc
    if not math.isfinite(result):
        raise ValueError("Non-finite risk observation")
    return result


def _insert(values, day, value):
    if day in values and values[day] != value:
        raise ValueError(f"Conflicting duplicate risk observation: {day}")
    values[day] = value


def parse_vix(body):
    """Parse the official CSV close column, retaining the published index-point unit."""
    reader = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
    fields = reader.fieldnames or []
    if not {"DATE", "CLOSE"} <= set(fields) or len(set(fields)) != len(fields):
        raise ValueError("VIX CSV requires unambiguous DATE and CLOSE columns")
    values = {}
    for row in reader:
        if None in row:
            raise ValueError("Malformed VIX CSV row")
        day = datetime.strptime(row["DATE"], "%m/%d/%Y").date().isoformat()
        value = _number(row["CLOSE"])
        if value < 0:
            raise ValueError("VIX close cannot be negative")
        _insert(values, day, value)
    if not values:
        raise ValueError("VIX CSV contains no observations")
    return dict(sorted(values.items()))


def parse_treasury(body):
    """Parse annual nominal 10-year par yields; explicit missing entries remain None."""
    text = body.decode("utf-8-sig")
    if re.search(r"<!DOCTYPE|<!ENTITY", text, re.IGNORECASE):
        raise ValueError("Treasury XML declarations/entities are unsupported")
    root = ET.fromstring(text)
    if root.tag != ATOM + "feed":
        raise ValueError("Expected official Treasury Atom feed")
    values = {}
    for entry in root.findall(ATOM + "entry"):
        properties = entry.find(ATOM + "content/" + META + "properties")
        if properties is None:
            raise ValueError("Treasury entry lacks data properties")
        dates, yields = (
            properties.findall(DATA + "NEW_DATE"),
            properties.findall(DATA + "BC_10YEAR"),
        )
        if len(dates) != 1 or not dates[0].text or len(yields) > 1:
            raise ValueError("Ambiguous Treasury date or 10-year field")
        day = datetime.fromisoformat(dates[0].text.strip()).date().isoformat()
        field = yields[0] if yields else None
        content = field.text.strip() if field is not None and field.text else ""
        if field is not None and field.get(META + "null") == "true" and content:
            raise ValueError("Treasury missing marker conflicts with a numerical value")
        _insert(values, day, _number(content) if content else None)
    if not values:
        raise ValueError("Treasury feed contains no observations")
    return dict(sorted(values.items()))


def load_risk_history(directory):
    """Verify retained official-source receipts and raw bytes without fetching/writing."""
    root = Path(directory).resolve()
    manifest = root / "receipts.json"
    if manifest.stat().st_size > 128 * 1024:
        raise ValueError("Risk receipt manifest exceeds byte limit")
    receipts = json.loads(manifest.read_bytes())
    if not isinstance(receipts, list) or not 2 <= len(receipts) <= 32:
        raise ValueError("Risk bundle needs VIX and bounded yearly Treasury receipts")
    series, seen = {"vix": {}, "treasury_10y": {}}, set()
    for receipt in receipts:
        name = receipt["file"]
        year = re.fullmatch(r"treasury-(\d{4})\.xml", name)
        if name in seen or (name != "vix.csv" and year is None):
            raise ValueError("Invalid or duplicate risk source filename")
        seen.add(name)
        url = VIX_URL if name == "vix.csv" else TREASURY_URL + year[1]
        response_urls = {url}
        if name == "vix.csv":
            response_urls.add(url.replace("cdn.cboe.com", "cdn-api.cboe.com"))
        if (
            receipt["status"] != 200
            or receipt["requested_url"] != url
            or receipt["response_url"] not in response_urls
            or timestamp(receipt["requested_at"]) > timestamp(receipt["received_at"])
        ):
            raise ValueError("Risk receipt does not identify a successful official source")
        path = (root / "raw" / name).resolve()
        if not path.is_relative_to(root) or path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Risk source escapes bundle or exceeds byte limit")
        body = path.read_bytes()
        if len(body) != receipt["bytes"] or digest(body) != receipt["sha256"]:
            raise ValueError("Risk source bytes/hash differ from receipt")
        key = "vix" if name == "vix.csv" else "treasury_10y"
        parsed = parse_vix(body) if key == "vix" else parse_treasury(body)
        for day, value in parsed.items():
            if year is not None and day[:4] != year[1]:
                raise ValueError("Treasury observation year differs from source URL/filename")
            previous = series[key].get(day)
            if previous is not None and previous["value"] != value:
                raise ValueError(f"Conflicting duplicate risk observation: {day}")
            record = series[key].setdefault(day, {"date": day, "value": value, "source_refs": []})
            if receipt["sha256"] not in record["source_refs"]:
                record["source_refs"].append(receipt["sha256"])
    if not all(series.values()):
        raise ValueError("Risk bundle must contain both VIX and Treasury observations")
    return {
        "kind": "RetainedMarketRiskHistory",
        "notice": NOTICE,
        "sources": receipts,
        "series": {key: [rows[day] for day in sorted(rows)] for key, rows in series.items()},
    }


def _align(records, sessions):
    days = [date.fromisoformat(record["date"]) for record in records]
    if any(a >= b for a, b in zip(days, days[1:], strict=False)):
        raise ValueError("Risk observation dates must be strictly increasing")
    values, provenance = [], []
    for session in sessions:
        index = bisect_left(days, session) - 1
        record = records[index] if index >= 0 else None
        age = (session - days[index]).days if record is not None else None
        status = (
            "no_prior_observation"
            if record is None
            else "explicit_missing"
            if record["value"] is None
            else "stale"
            if age > MAX_AGE_DAYS
            else "available"
        )
        value = _number(record["value"]) if status == "available" else None
        values.append(value)
        provenance.append(
            {
                "equity_session": session.isoformat(),
                "observation_date": record["date"] if record else None,
                "source_refs": record["source_refs"] if record else [],
                "age_days": age,
                "status": status,
                "value": value,
            }
        )
    return values, provenance


def align_risk_features(bundle, session_dates):
    """Return four columns, definitions and per-feature endpoint provenance.

    Changes are absolute differences over five/twenty *equity rows*. Both aligned
    endpoint values must be available; warm-up or missing endpoints stay NaN.
    """
    sessions = [date.fromisoformat(day) for day in session_dates]
    if not sessions or any(a >= b for a, b in zip(sessions, sessions[1:], strict=False)):
        raise ValueError("Equity session dates must be nonempty and strictly increasing")
    columns, definitions, diagnostics = {}, [], []
    for key, level_id, change_id, lag, level_unit, change_unit in (
        ("vix", "risk.vix_close", "risk.vix_change_5", 5, "vix_index_points", "vix_index_points"),
        (
            "treasury_10y",
            "risk.treasury_10y_percent",
            "risk.treasury_10y_change_20",
            20,
            "percent",
            "percentage_points",
        ),
    ):
        levels, provenance = _align(bundle["series"][key], sessions)
        for factor_id, offset, unit in ((level_id, 0, level_unit), (change_id, lag, change_unit)):
            values, rows = [], []
            for index, session in enumerate(sessions):
                endpoints = [provenance[index]]
                value = None
                status = "warmup" if index < offset else "unavailable_endpoint"
                if index >= offset:
                    if offset:
                        endpoints.append(provenance[index - offset])
                    if all(endpoint["status"] == "available" for endpoint in endpoints):
                        value = levels[index] - levels[index - offset] if offset else levels[index]
                        status = "available"
                values.append(value)
                rows.append(
                    {
                        "session_date": session.isoformat(),
                        "value": value,
                        "status": status,
                        "endpoints": endpoints,
                    }
                )
            columns[factor_id] = np.asarray(values, dtype=float)
            definitions.append(
                {
                    "factor_id": factor_id,
                    "kind": "numeric",
                    "unit": unit,
                    "runnable": True,
                    "recipe": {
                        "op": "prior_date_risk" if not offset else "prior_date_risk_difference",
                        "series": key,
                        "equity_row_lag": offset,
                        "max_age_days": MAX_AGE_DAYS,
                    },
                }
            )
            diagnostics.append(
                {"factor_id": factor_id, "unit": unit, "notice": NOTICE, "rows": rows}
            )
    return columns, definitions, diagnostics
