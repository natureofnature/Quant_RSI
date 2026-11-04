"""Retain the public daily national shell-egg index for one instrument binding.

The MARS report API answers 403 without a key. History here is the ESMIS PDF
archive plus the current AMS file. A report date is not visible on that date.
"""

import io
import json
import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, build_opener

from pypdf import PdfReader

from quant_rsi.core.artifacts import atomic_write, canonical, digest
from quant_rsi.market.news_feed import NoRedirect

INDEX_URL = (
    "https://esmis.nal.usda.gov/publication/"
    "daily-national-shell-egg-index-report-5-day-rolling-average"
)
CURRENT_URL = "https://www.ams.usda.gov/mnreports/ams_2843.pdf"
MAX_BYTES = 2 * 1024 * 1024
MAX_PAGES = 24
MAX_FILES = 240
CARRY_DAYS = 6
NOTICE = (
    "USDA AMS Daily National Shell Egg Index, national FOB caged graded-loose "
    "white large weighted average, cents per dozen. The published figure is already "
    "a five-day rolling average. A report date becomes visible on the next target "
    "session, not on the report date. Files are the ESMIS PDF archive and the current "
    "AMS PDF, read now; this is not a MARS JSON history and not a point-in-time "
    "correction log. A gap longer than six calendar days is left empty rather than "
    "carried forward."
)


def parse_shell_egg_report(text):
    """Read one report date and the caged white-large weighted average."""
    compact = " ".join(text.split())
    dated = re.search(r"Report for:\s*(\d{2}/\d{2}/\d{4})", compact)
    if not dated:
        raise ValueError("Shell egg report has no report date")
    report_date = datetime.strptime(dated[1], "%m/%d/%Y").date()
    caged = compact.split("Cage-Free", 1)[0]
    match = re.search(r"White Large ([\d,]+) ([\d.]+) - ([\d.]+) (-?[\d.]+)", caged)
    if not match:
        raise ValueError("Shell egg report has no caged white large weighted average")
    value = float(match[4])
    volume = float(match[1].replace(",", ""))
    if not math.isfinite(value) or value <= 0 or not math.isfinite(volume) or volume <= 0:
        raise ValueError("Invalid shell egg weighted average")
    return {
        "report_date": report_date.isoformat(),
        "value": value,
        "volume": volume,
        "volume_unit": "cases_30_dozen",
        "unit": "cents_per_dozen",
        "row": "national_fob_caged_graded_loose_white_large",
    }


def pdf_text(raw):
    reader = PdfReader(io.BytesIO(raw))
    if len(reader.pages) > 8:
        raise ValueError("Shell egg PDF has too many pages")
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def release_links(raw, page_url):
    """Keep same-host ESMIS PDF links. Ignore every other href."""
    if page_url.split("?", 1)[0] != INDEX_URL:
        raise ValueError("Shell egg index URL is not the ESMIS publication")
    found = []
    for href in re.findall(r'href=["\']([^"\']+)', raw.decode("utf-8", "replace")):
        url = urljoin(page_url, href)
        if not url.startswith("https://esmis.nal.usda.gov/sites/default/release-files/"):
            continue
        if not url.lower().endswith("/ams_2843.pdf"):
            continue
        if url not in found:
            found.append(url)
    return found


def align_shell_eggs(records, session_dates):
    """Show a report only after its date, and do not bridge a missing week."""
    ordered = sorted(records, key=lambda record: record["report_date"])
    dates = [date.fromisoformat(record["report_date"]) for record in ordered]
    if len(dates) != len(set(dates)):
        raise ValueError("Shell egg reports repeat a report date")
    values = []
    for session in session_dates:
        session = date.fromisoformat(session)
        selected = None
        for index, record in enumerate(ordered):
            published = dates[index]
            if session <= published:
                continue
            successor = dates[index + 1] if index + 1 < len(dates) else None
            # A later report supersedes this one even when that later report
            # is not visible yet. Do not keep this value across the gap.
            if successor is not None and session > successor:
                continue
            if successor is None or (successor - published).days > CARRY_DAYS:
                if session > published + timedelta(days=CARRY_DAYS):
                    continue
            selected = record["value"]
        values.append(selected)
    return values


def _download(url):
    request = Request(
        url,
        headers={"User-Agent": "QuantRSI research (github.com/natureofnature/Quant_RSI)"},
    )
    with build_opener(NoRedirect).open(request, timeout=40) as response:
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Shell egg file exceeds the collection bound")
    return raw


def _retain(directory, url, raw):
    name = digest(raw) + ".body"
    path = directory / name
    if not path.exists():
        atomic_write(path, raw)
    elif path.read_bytes() != raw:
        raise ValueError("Retained shell egg file changed")
    return {"url": url, "file": name, "sha256": digest(raw)}


def collect_shell_eggs(directory, since, through):
    """Reuse a valid snapshot. A new collection does not replace one."""
    directory = Path(directory)
    since, through = date.fromisoformat(since), date.fromisoformat(through)
    if since > through:
        raise ValueError("Invalid shell egg collection interval")
    bounds = {"since": since.isoformat(), "through": through.isoformat()}
    retained = directory / "snapshot.json"
    if retained.exists():
        snapshot = json.loads(retained.read_bytes())
        if (
            snapshot.get("provider") != "usda_ams_shell_egg_index"
            or snapshot.get("bounds") != bounds
            or digest(canonical(snapshot["records"])) != snapshot["sha256"]
        ):
            raise ValueError("Retained shell egg snapshot identity changed")
        for resource in snapshot["resources"]:
            body = directory / resource["file"]
            if resource["file"] != digest(body.read_bytes()) + ".body":
                raise ValueError("Retained shell egg file changed")
        return snapshot
    directory.mkdir(parents=True, exist_ok=True)
    urls, failures = [], []
    for page in range(MAX_PAGES):
        page_url = INDEX_URL if page == 0 else f"{INDEX_URL}?page={page}"
        try:
            raw = _download(page_url)
        except (OSError, ValueError) as exc:
            failures.append({"url": page_url, "reason": str(exc)})
            break
        found = release_links(raw, page_url)
        fresh = [url for url in found if url not in urls]
        if not fresh:
            break
        urls.extend(fresh)
        if len(urls) > MAX_FILES:
            raise ValueError("Shell egg archive exceeds the file bound")
    urls.append(CURRENT_URL)
    records, resources = [], []
    earliest = since - timedelta(days=14)
    for url in urls:
        try:
            raw = _download(url)
            parsed = parse_shell_egg_report(pdf_text(raw))
            published = date.fromisoformat(parsed["report_date"])
            if earliest <= published <= through:
                records.append(parsed | {"source_url": url, "resource_sha256": digest(raw)})
                resources.append(_retain(directory, url, raw))
        except (OSError, ValueError) as exc:
            failures.append({"url": url, "reason": str(exc)})
    by_date = {}
    for record in records:
        previous = by_date.get(record["report_date"])
        if previous is not None and previous["value"] != record["value"]:
            raise ValueError("Conflicting shell egg values for one report date")
        by_date[record["report_date"]] = record
    records = [by_date[key] for key in sorted(by_date)]
    if not records:
        raise ValueError("No shell egg reports were read")
    snapshot = {
        "provider": "usda_ams_shell_egg_index",
        "bounds": bounds,
        "records": records,
        "sha256": digest(canonical(records)),
        "resources": resources,
        "failures": failures,
        "notice": NOTICE,
    }
    atomic_write(retained, canonical(snapshot))
    return snapshot
