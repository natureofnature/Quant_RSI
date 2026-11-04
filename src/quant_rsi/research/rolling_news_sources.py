"""Bounded historical headline search with honest reconstruction/receipt boundaries.

Google News search is a selected, currently retrieved corpus, not a point-in-time
archive or an exhaustive event feed. Its dated headlines can support explicitly
reconstructed research; the real receipt/version timestamps are never backdated.
"""

import json
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from datetime import UTC, date, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode, urlsplit

from quant_rsi.core.artifacts import atomic_write, canonical, digest
from quant_rsi.market import news_feed
from quant_rsi.market.market_contracts import timestamp

NOTICE = (
    "Selected historical Google News headlines retrieved now, not a complete news feed. "
    "Publication dates do not attest historical text versions or contemporary receipt. "
    "Search ranking and coverage may change; empty results do not prove no events occurred."
)


def _months(start, end):
    current = start
    while current <= end:
        following = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield current, min(end, following - timedelta(days=1))
        current = following


def _query(symbol, query):
    if query is None:
        query = (
            '(SOXX OR semiconductor OR chipmakers OR Nvidia OR TSMC OR "chip exports" '
            'OR "stock market" OR "Federal Reserve" OR Taiwan OR "Middle East")'
            if symbol == "SOXX"
            else f'("{symbol}" stock OR "{symbol}" earnings)'
        )
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise ValueError("News query must contain 1..500 characters")
    return query.strip()


def _parse(body, symbol, start, end, received):
    text = body.decode("utf-8-sig")
    if re.search(r"<!DOCTYPE|<!ENTITY", text, re.IGNORECASE):
        raise ValueError("RSS declarations/entities are unsupported")
    root = ET.fromstring(text)
    if root.find("channel") is None:
        raise ValueError("Historical news response is not an RSS channel")
    raw_items = root.findall("./channel/item")
    if len(raw_items) > 200:
        raise ValueError("Historical news response exceeded 200 records")
    items, excluded = {}, {}
    for entry in raw_items:
        title = " ".join((entry.findtext("title") or "").split())
        link = (entry.findtext("link") or "").strip()
        reason = None
        try:
            published = parsedate_to_datetime(entry.findtext("pubDate") or "")
            if published.tzinfo is None:
                raise ValueError("Missing publication timezone")
            published = published.astimezone(UTC)
        except (ValueError, TypeError, OverflowError):
            reason = "invalid_publication_time"
        parsed = urlsplit(link)
        if (
            not title
            or len(title) > 2000
            or (
                parsed.scheme != "https"
                or parsed.hostname != "news.google.com"
                or parsed.username
                or parsed.password
            )
        ):
            reason = "invalid_title_or_url"
        if not reason and not start <= published.date() <= end:
            reason = "outside_requested_dates"
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        identity = entry.findtext("guid") or link
        item_id = "google-news-" + digest(canonical(identity))[:32]
        value = {
            "id": item_id,
            "symbol": symbol,
            "title": title,
            "text": title,
            "url": link,
            "source_name": (entry.findtext("source") or "")[:300],
            "published_at": published.isoformat(),
            "received_at": received,
            "version_available_at": received,
            "content_extent": "retrieved_headline_only",
        }
        if item_id in items and items[item_id] != value:
            raise ValueError("Conflicting duplicate historical headline")
        items[item_id] = value
    return sorted(items.values(), key=lambda item: (item["published_at"], item["id"])), excluded


def collect_history(symbol, start, end, directory, *, query=None, fetcher=None):
    """Fetch one date-bounded RSS search per calendar month; reuse verified successes.

    start/end are inclusive ISO dates. HTTP/parse failures stop acquisition while
    retaining receipts and successful months, rather than silently producing a
    price-only study. A retry never overwrites a failed request's raw response.
    """
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", symbol):
        raise ValueError("Invalid news symbol")
    start, end = date.fromisoformat(start), date.fromisoformat(end)
    if start > end or (end - start).days > 1100:
        raise ValueError("Historical news range must be ordered and at most 1100 days")
    query = _query(symbol, query)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fetcher = fetcher or news_feed.fetch
    searches, articles = [], {}
    for first, last in _months(start, end):
        bounded = f"{query} after:{first - timedelta(days=1)} before:{last + timedelta(days=1)}"
        url = "https://news.google.com/rss/search?" + urlencode(
            {"q": bounded, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        )
        request = {"symbol": symbol, "start": str(first), "end": str(last), "query": bounded}
        folder = directory / f"{first}_{last}"
        folder.mkdir(exist_ok=True)
        saved = folder / "success.json"
        if saved.exists():
            record = json.loads(saved.read_bytes())
            if record["request"] != request:
                raise ValueError("Historical news cache request differs; use a fresh directory")
            raw_path = folder / record["raw_file"]
            if raw_path.parent != folder or raw_path.name != record["raw_file"]:
                raise ValueError("Invalid historical news raw path")
            body = raw_path.read_bytes()
            if digest(body) != record["raw_sha256"]:
                raise ValueError("Historical news raw response hash mismatch")
            receipt = record["receipt"]
        else:
            attempt = len(list(folder.glob("attempt-*.json"))) + 1
            body, receipt = fetcher(SimpleNamespace(url=url), 1024 * 1024)
            timestamp(receipt["received_at"])
            raw_name = f"attempt-{attempt}.xml"
            atomic_write(folder / raw_name, body)
            record = {
                "request": request,
                "url": url,
                "receipt": receipt,
                "raw_file": raw_name,
                "raw_sha256": digest(body),
            }
            atomic_write(folder / f"attempt-{attempt}.json", canonical(record))
            if receipt.get("error") or receipt.get("http_status") != 200:
                raise ValueError(f"Historical news search failed for {first}: {receipt}")
        items, excluded = _parse(body, symbol, first, last, receipt["received_at"])
        if not saved.exists():
            atomic_write(saved, canonical(record))
        searches.append(
            request
            | {
                "url": url,
                "status": "partial_results" if items else "empty_search_results",
                "received_at": receipt["received_at"],
                "raw_sha256": record["raw_sha256"],
                "raw_path": str((folder / record["raw_file"]).relative_to(directory)),
                "item_count": len(items),
                "excluded": excluded,
            }
        )
        for item in items:
            articles.setdefault(item["id"], item)
    snapshot = {
        "provider": "historical_google_news_search",
        "symbol": symbol,
        "start": str(start),
        "end": str(end),
        "query": query,
        "received_at": max(search["received_at"] for search in searches),
        "coverage_status": "partial_search_results",
        "notice": NOTICE,
        "searches": searches,
        "items": sorted(articles.values(), key=lambda item: (item["published_at"], item["id"])),
    }
    atomic_write(directory / "news_history.json", canonical(snapshot))
    return snapshot


def select_evidence(snapshot, cutoff, *, mode="reconstructed", limit=None):
    """Filter the full numerical corpus before optional LLM-context truncation."""
    if mode not in {"retained", "reconstructed"}:
        raise ValueError("Unknown historical news information mode")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("News evidence limit must be positive or None")
    boundary = timestamp(cutoff)
    admitted = []
    for raw in snapshot["items"]:
        fields = (
            ("published_at", "received_at", "version_available_at")
            if mode == "retained"
            else ("published_at",)
        )
        try:
            if any(timestamp(raw[field]) > boundary for field in fields):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        admitted.append(deepcopy(raw))
    admitted.sort(key=lambda item: (item["published_at"], item["id"]))
    selected = admitted if limit is None else admitted[-limit:]
    return {
        "provider": snapshot["provider"],
        "symbol": snapshot["symbol"],
        "information_mode": mode,
        "research_cutoff": cutoff,
        "items": selected,
        "eligible_count": len(admitted),
        "context_omitted_count": len(admitted) - len(selected),
        "coverage_status": "partial_search_results" if admitted else "missing_eligible_material",
        "notice": NOTICE
        + (
            " Reconstructed: current headline versions aligned to publication times."
            if mode == "reconstructed"
            else " Retained: publication, receipt and version timestamps must precede cutoff."
        ),
    }
