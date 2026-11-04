"""Retain NASS Quick Stats monthly flock series. The key stays in the environment."""

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, build_opener
from zoneinfo import ZoneInfo

from quant_rsi.core.artifacts import atomic_write, canonical, digest
from quant_rsi.market.news_feed import NoRedirect

API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
MAX_BYTES = 8 * 1024 * 1024
EASTERN = ZoneInfo("America/New_York")
MONTHS = {
    name: index
    for index, name in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"),
        start=1,
    )
}
SERIES = (
    {
        "field": "inventory",
        "entity": "US table-egg layers",
        "short_desc": "CHICKENS, LAYERS, TABLE - INVENTORY, AVG, MEASURED IN HEAD",
        "domain_desc": "INVENTORY OF LAYERS, TABLE",
    },
    {
        "field": "placements",
        "entity": "US egg-type pullets",
        "short_desc": (
            "CHICKENS, CHICKS, EGG TYPE, PULLET - PLACEMENTS, INTENDED, MEASURED IN HEAD"
        ),
        "domain_desc": "TOTAL",
    },
    {
        "field": "hatched",
        "entity": "US egg-type chicks hatched",
        "short_desc": "CHICKENS, CHICKS, EGG TYPE - HATCHED, MEASURED IN HEAD",
        "domain_desc": "TOTAL",
    },
)
NOTICE = (
    "NASS Quick Stats national monthly series: average table-egg layer inventory, "
    "egg-type pullet placements, and egg-type chicks hatched. A value is hidden until "
    "its load_time, which is the database load time interpreted as US Eastern time, "
    "not the reference month. Quick Stats returns the current row, so a later reload "
    "is not a point-in-time revision history. Missing rows stay empty and are then "
    "carried forward only after they have loaded."
)


def _key():
    key = os.environ.get("NASS_QUICKSTATS_API_KEY", "")
    if not re.fullmatch(r"[A-Za-z0-9-]{8,80}", key):
        raise ValueError("NASS_QUICKSTATS_API_KEY is not set")
    return key


def parse_nass_rows(payload, series):
    """Keep one national row per month. A later load replaces that month only."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Unexpected NASS response")
    if payload.get("error"):
        raise ValueError("NASS query failed")
    selected = {}
    for row in payload["data"]:
        if (
            row.get("short_desc") != series["short_desc"]
            or row.get("domain_desc") != series["domain_desc"]
            or row.get("agg_level_desc") != series.get("agg_level_desc", "NATIONAL")
            or row.get("freq_desc") != series.get("freq_desc", "MONTHLY")
        ):
            continue
        if series.get("state_alpha") and row.get("state_alpha") != series["state_alpha"]:
            continue
        period = _period(row)
        value = _number(row.get("Value"), allow_zero=bool(series.get("allow_zero")))
        loaded = _loaded(row.get("load_time"))
        if period is None or value is None or loaded is None:
            continue
        current = selected.get(period)
        if current is None or loaded > current["load_time"]:
            selected[period] = {
                "field": series["field"],
                "entity": series["entity"],
                "year": period[0],
                "month": period[1],
                "value": value,
                "load_time": loaded,
                "unit": series.get("unit", "head"),
            }
    if not selected:
        raise ValueError(f"NASS returned no {series['field']} rows")
    return [selected[key] for key in sorted(selected)]


def align_nass(records, cutoffs, field):
    """Use the newest reference month whose load_time is already past."""
    values = []
    parsed = [
        (
            datetime.fromisoformat(record["load_time"]),
            (record["year"], record["month"]),
            record["value"],
        )
        for record in records
        if record["field"] == field
    ]
    for cutoff in cutoffs:
        stamp = datetime.fromisoformat(cutoff)
        if stamp.tzinfo is None:
            raise ValueError("NASS cutoffs require timezones")
        visible = [item for item in parsed if item[0] <= stamp]
        if not visible:
            values.append(None)
            continue
        values.append(max(visible, key=lambda item: item[1])[2])
    return values


def _period(row):
    desc = row.get("reference_period_desc")
    if not isinstance(desc, str):
        return None
    month = MONTHS.get(desc.removeprefix("FIRST OF "))
    year = row.get("year")
    if month is None or not str(year).isdigit():
        return None
    return int(year), month


def _number(value, allow_zero=False):
    if not isinstance(value, str) or not re.fullmatch(r"[\d,]+(\.\d+)?", value):
        return None
    number = float(value.replace(",", ""))
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        return None
    return number


def _loaded(value):
    if not isinstance(value, str) or len(value) < 19:
        return None
    try:
        stamp = datetime.strptime(value[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return stamp.replace(tzinfo=EASTERN).isoformat()


def _download(url):
    request = Request(
        url,
        headers={"User-Agent": "QuantRSI research (github.com/natureofnature/Quant_RSI)"},
    )
    with build_opener(NoRedirect).open(request, timeout=90) as response:
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("NASS response exceeds the collection bound")
    return raw


def collect_nass(directory, first_year):
    """Read the key from the environment. Do not store it or the request URL."""
    directory = Path(directory)
    if type(first_year) is not int or not 1990 <= first_year <= 2100:
        raise ValueError("Invalid NASS start year")
    retained = directory / "snapshot.json"
    if retained.exists():
        snapshot = json.loads(retained.read_bytes())
        if (
            snapshot.get("provider") != "nass_chickens_eggs"
            or snapshot.get("first_year") != first_year
            or digest(canonical(snapshot["records"])) != snapshot["sha256"]
        ):
            raise ValueError("Retained NASS snapshot identity changed")
        return snapshot
    directory.mkdir(parents=True, exist_ok=True)
    key = _key()
    records, resources = [], []
    for series in SERIES:
        query = {
            "short_desc": series["short_desc"],
            "domain_desc": series["domain_desc"],
            "agg_level_desc": "NATIONAL",
            "freq_desc": "MONTHLY",
            "year__GE": str(first_year),
            "format": "JSON",
        }
        raw = _download(API_URL + "?" + urlencode(query | {"key": key}))
        name = digest(raw) + ".body"
        atomic_write(directory / name, raw)
        records.extend(parse_nass_rows(json.loads(raw), series))
        resources.append(
            {"file": name, "sha256": digest(raw), "field": series["field"], "query": query}
        )
    snapshot = {
        "provider": "nass_chickens_eggs",
        "first_year": first_year,
        "records": records,
        "sha256": digest(canonical(records)),
        "resources": resources,
        "notice": NOTICE,
    }
    atomic_write(retained, canonical(snapshot))
    return snapshot
