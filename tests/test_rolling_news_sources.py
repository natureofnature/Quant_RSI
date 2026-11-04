"""Offline source contract checks; fixture headlines do not establish research quality."""

import json
from copy import deepcopy
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from quant_rsi.research.rolling_news_sources import collect_history, load_history, select_evidence


def rss(rows):
    return (
        "<rss><channel>"
        + "".join(
            f"<item><guid>{identity}</guid><title>{escape(title)}</title>"
            f"<link>https://news.google.com/rss/articles/{identity}</link>"
            f"<pubDate>{day}</pubDate><source>Example publisher</source></item>"
            for identity, title, day in rows
        )
        + "</channel></rss>"
    ).encode()


def receipt(url):
    return {
        "http_status": 200,
        "response_url": url,
        "received_at": "2026-11-04T18:00:00+00:00",
    }


def test_collection_filters_source_dates_preserves_receipts_and_reuses_raw(tmp_path):
    calls = []

    def fetch(source, allowance):
        calls.append(source.url)
        assert allowance == 1024 * 1024
        return rss(
            [
                ("old", "Chip demand falls", "Mon, 01 Jun 2026 10:00:00 GMT"),
                ("late", "Tariff announced", "Wed, 01 Jul 2026 10:00:00 GMT"),
                ("undated", "Untimed headline", "unknown"),
            ]
        ), receipt(source.url)

    value = collect_history("SOXX", "2026-06-01", "2026-06-30", tmp_path, fetcher=fetch)
    assert len(calls) == 1
    assert "after%3A2026-05-31" in calls[0] and "before%3A2026-07-01" in calls[0]
    assert [item["title"] for item in value["items"]] == ["Chip demand falls"]
    assert value["items"][0]["received_at"] == "2026-11-04T18:00:00+00:00"
    assert value["items"][0]["version_available_at"] == value["items"][0]["received_at"]
    assert value["coverage_status"] == "partial_search_results"
    assert value["searches"][0]["excluded"] == {
        "outside_requested_dates": 1,
        "invalid_publication_time": 1,
    }
    repeated = collect_history("SOXX", "2026-06-01", "2026-06-30", tmp_path, fetcher=fetch)
    assert repeated == value and len(calls) == 1
    assert json.loads((tmp_path / "news_history.json").read_bytes()) == value
    raw = tmp_path / value["searches"][0]["raw_path"]
    raw.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        collect_history("SOXX", "2026-06-01", "2026-06-30", tmp_path, fetcher=fetch)


def test_failed_month_does_not_repeat_successful_request_or_erase_failure(tmp_path):
    calls = []

    def fetch(source, allowance):
        calls.append(source.url)
        if "after%3A2026-06-30" in source.url and len(calls) == 2:
            return b"service unavailable", receipt(source.url) | {"http_status": 503}
        return rss([]), receipt(source.url)

    with pytest.raises(ValueError, match="search failed"):
        collect_history("SOXX", "2026-06-01", "2026-07-31", tmp_path, fetcher=fetch)
    result = collect_history("SOXX", "2026-06-01", "2026-07-31", tmp_path, fetcher=fetch)
    assert len(calls) == 3
    assert calls[1] == calls[2]
    assert [s["status"] for s in result["searches"]] == ["empty_search_results"] * 2
    folder = tmp_path / "2026-07-01_2026-07-31"
    assert (folder / "attempt-1.xml").read_bytes() == b"service unavailable"
    assert (folder / "attempt-2.xml").exists()
    assert not result["items"]


def test_request_change_requires_new_cache_and_rejects_entity_xml(tmp_path):
    def fetch(source, allowance):
        return rss([]), receipt(source.url)

    collect_history("SOXX", "2026-06-01", "2026-06-30", tmp_path, fetcher=fetch)
    with pytest.raises(ValueError, match="request differs"):
        collect_history("SOXX", "2026-06-01", "2026-06-30", tmp_path, query="war", fetcher=fetch)
    with pytest.raises(ValueError, match="declarations"):
        collect_history(
            "SOXX",
            "2026-07-01",
            "2026-07-31",
            tmp_path,
            fetcher=lambda s, _: (b"<!DOCTYPE rss><rss><channel/></rss>", receipt(s.url)),
        )
    assert not (tmp_path / "2026-07-01_2026-07-31" / "success.json").exists()


def test_cutoff_is_applied_before_context_limit_and_retained_requires_real_receipt(tmp_path):
    rows = [
        ("first", "Chip demand", "Mon, 01 Jun 2026 10:00:00 GMT"),
        ("second", "Export restrictions", "Tue, 02 Jun 2026 10:00:00 GMT"),
        ("future", "War escalation", "Wed, 03 Jun 2026 10:00:00 GMT"),
    ]
    snapshot = collect_history(
        "SOXX",
        "2026-06-01",
        "2026-06-30",
        tmp_path,
        fetcher=lambda s, _: (rss(rows), receipt(s.url)),
    )
    cutoff = "2026-06-02T20:00:00+00:00"
    value = select_evidence(snapshot, cutoff)
    assert [v["title"] for v in value["items"]] == ["Chip demand", "Export restrictions"]
    assert value["eligible_count"] == 2
    assert select_evidence(snapshot, cutoff, limit=1)["context_omitted_count"] == 1
    assert select_evidence(snapshot, cutoff, mode="retained")["items"] == []
    changed = deepcopy(snapshot)
    changed["items"][2]["title"] = "Future data changed"
    assert select_evidence(changed, cutoff) == value
    changed["items"][0]["received_at"] = "2026-06-01T11:00:00+00:00"
    assert select_evidence(changed, cutoff, mode="retained")["items"] == []
    changed["items"][0]["version_available_at"] = "2026-06-01T11:00:00+00:00"
    assert len(select_evidence(changed, cutoff, mode="retained")["items"]) == 1
    assert "Reconstructed" in value["notice"]
    assert len(list(Path(tmp_path).rglob("*.xml"))) == 1


@pytest.fixture
def saved_history(tmp_path):
    root = tmp_path / "corpus"
    value = collect_history(
        "SOXX",
        "2026-06-01",
        "2026-07-31",
        root,
        fetcher=lambda s, _: (
            rss([("old", "Chip demand falls", "Mon, 01 Jun 2026 10:00:00 GMT")]),
            receipt(s.url),
        ),
    )
    return root, value


def test_load_history_reconstructs_same_corpus_without_network_or_writes(
    saved_history, monkeypatch
):
    root, value = saved_history
    before = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }

    def prohibited(*args, **kwargs):
        raise AssertionError("Read-only corpus loading must not fetch or write")

    monkeypatch.setattr("quant_rsi.market.news_feed.fetch", prohibited)
    monkeypatch.setattr("quant_rsi.research.rolling_news_sources.atomic_write", prohibited)
    loaded = load_history(root, "SOXX", "2026-06-01", "2026-07-31")
    assert loaded == value
    assert loaded["items"][0]["received_at"] == "2026-11-04T18:00:00+00:00"
    assert {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    } == before


@pytest.mark.parametrize(
    "damage", ["raw", "items", "receipt", "missing_interval", "missing_raw", "escape"]
)
def test_load_history_rejects_tamper_and_missing_evidence(saved_history, tmp_path, damage):
    root, value = saved_history
    snapshot = deepcopy(value)
    raw = root / snapshot["searches"][0]["raw_path"]
    if damage == "raw":
        raw.write_bytes(b"tampered")
    elif damage == "items":
        snapshot["items"][0]["title"] = "Changed title"
    elif damage == "receipt":
        snapshot["searches"][0]["received_at"] = "2026-06-02T00:00:00+00:00"
    elif damage == "missing_interval":
        snapshot["searches"].pop()
    elif damage == "missing_raw":
        raw.unlink()
    else:
        escaped = tmp_path / "outside.xml"
        escaped.write_bytes(raw.read_bytes())
        snapshot["searches"][0]["raw_path"] = "../outside.xml"
    (root / "news_history.json").write_text(json.dumps(snapshot))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_history(root, "SOXX", "2026-06-01", "2026-07-31")


@pytest.mark.parametrize(
    "symbol,start,end,query",
    [
        ("KO", "2026-06-01", "2026-07-31", None),
        ("SOXX", "2026-06-01", "2026-06-30", None),
        ("SOXX", "2026-06-02", "2026-07-31", None),
        ("SOXX", "2026-06-01", "2026-07-31", "different universe"),
    ],
)
def test_load_history_rejects_different_requested_universe(
    saved_history, symbol, start, end, query
):
    root, _ = saved_history
    with pytest.raises(ValueError, match="requested metadata"):
        load_history(root, symbol, start, end, query=query)
