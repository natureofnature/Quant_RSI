"""Offline fixtures for the shell-egg index and NASS monthly flock series."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from quant_rsi.core.data import synthetic_data
from quant_rsi.core.schemas import DataSpec
from quant_rsi.factors.catalog import resolve_instrument_factors
from quant_rsi.research import ams_shell_eggs as ams
from quant_rsi.research import nass_chickens as nass
from quant_rsi.research.study_factor_profiles import prepare_bound_features

ROOT = Path(__file__).resolve().parents[1] / "instruments"
REPORT = """
Daily National Shell Egg Index Report
Report for: 09/30/2025
NATIONAL SHELL EGGS - Caged
White Large 58,848 88.50 - 148.00 120.73 -2.64 123.37
NATIONAL SHELL EGGS - Cage-Free
White Large 1,230 100.00 - 125.00 107.93 0.00 107.93
"""


def test_shell_egg_parser_uses_the_caged_white_large_average_only():
    parsed = ams.parse_shell_egg_report(REPORT)
    assert parsed["report_date"] == "2025-09-30"
    assert parsed["value"] == pytest.approx(120.73)
    assert parsed["volume"] == pytest.approx(58848)
    with pytest.raises(ValueError, match="no report date"):
        ams.parse_shell_egg_report("White Large 1 1.00 - 2.00 1.50")


def test_report_date_is_hidden_that_session_and_a_gap_is_not_carried():
    records = [
        {"report_date": "2026-10-30", "value": 100.0},
        {"report_date": "2026-11-02", "value": 110.0},
    ]
    assert ams.align_shell_eggs(records, ["2026-10-30", "2026-11-02", "2026-11-03"]) == [
        None,
        100.0,
        110.0,
    ]
    stale = [
        {"report_date": "2026-09-30", "value": 80.0},
        {"report_date": "2026-11-02", "value": 90.0},
    ]
    assert ams.align_shell_eggs(
        stale, ["2026-10-01", "2026-10-06", "2026-10-07", "2026-11-02", "2026-11-03"]
    ) == [80.0, 80.0, None, None, 90.0]
    bridged = [
        {"report_date": "2026-09-29", "value": 123.37},
        {"report_date": "2026-09-30", "value": 120.73},
        {"report_date": "2026-11-02", "value": 35.42},
    ]
    assert ams.align_shell_eggs(
        bridged,
        ["2026-09-30", "2026-10-01", "2026-10-06", "2026-10-07", "2026-11-02", "2026-11-03"],
    ) == [123.37, 120.73, 120.73, None, None, 35.42]


def test_index_links_stay_on_the_esmis_report(tmp_path):
    page = (
        '<a href="/sites/default/release-files/aa/bb/cc/AMS_2843.PDF">a</a>'
        '<a href="https://example.com/AMS_2843.PDF">b</a>'
        '<a href="/sites/default/release-files/aa/bb/cc/other.pdf">c</a>'
    ).encode()
    links = ams.release_links(page, ams.INDEX_URL)
    assert links == ["https://esmis.nal.usda.gov/sites/default/release-files/aa/bb/cc/AMS_2843.PDF"]
    with pytest.raises(ValueError, match="not the ESMIS"):
        ams.release_links(page, "https://example.com/publication")


def test_collection_uses_the_injected_response_and_keeps_the_file(tmp_path, monkeypatch):
    pages = {
        ams.INDEX_URL: (
            '<a href="/sites/default/release-files/aa/bb/cc/AMS_2843.PDF">a</a>'
        ).encode(),
        ams.INDEX_URL + "?page=1": b"",
    }
    pdf = b"%PDF-fixture"

    def download(url):
        if url in pages:
            return pages[url]
        if url.endswith("/AMS_2843.PDF") or url == ams.CURRENT_URL:
            return pdf
        raise AssertionError(url)

    monkeypatch.setattr(ams, "_download", download)
    monkeypatch.setattr(ams, "pdf_text", lambda raw: REPORT)
    snapshot = ams.collect_shell_eggs(tmp_path, "2025-09-01", "2025-10-01")
    assert snapshot["records"][0]["value"] == pytest.approx(120.73)
    assert (tmp_path / snapshot["resources"][0]["file"]).read_bytes() == pdf
    monkeypatch.setattr(ams, "_download", lambda url: (_ for _ in ()).throw(AssertionError(url)))
    assert (
        ams.collect_shell_eggs(tmp_path, "2025-09-01", "2025-10-01")["sha256"] == snapshot["sha256"]
    )


def test_nass_load_time_hides_the_reference_month_until_it_is_loaded():
    records = [
        {
            "field": "inventory",
            "year": 2026,
            "month": 1,
            "value": 10.0,
            "load_time": "2026-02-17T15:00:00-05:00",
        },
        {
            "field": "inventory",
            "year": 2026,
            "month": 2,
            "value": 12.0,
            "load_time": "2026-03-23T15:00:00-04:00",
        },
        {
            "field": "inventory",
            "year": 2026,
            "month": 1,
            "value": 11.0,
            "load_time": "2026-04-01T15:00:00-04:00",
        },
    ]
    # The April reload of January must not replace February.
    aligned = nass.align_nass(
        records,
        [
            "2026-02-17T14:00:00-05:00",
            "2026-02-17T16:00:00-05:00",
            "2026-03-23T14:00:00-04:00",
            "2026-03-23T16:00:00-04:00",
            "2026-04-02T16:00:00-04:00",
        ],
        "inventory",
    )
    assert aligned == [None, 10.0, 10.0, 12.0, 12.0]


def test_nass_parser_keeps_the_named_domain_and_drops_suppressed_values():
    series = nass.SERIES[0]
    payload = {
        "data": [
            {
                "short_desc": series["short_desc"],
                "domain_desc": series["domain_desc"],
                "agg_level_desc": "NATIONAL",
                "freq_desc": "MONTHLY",
                "year": "2026",
                "reference_period_desc": "JAN",
                "Value": "281,158,000",
                "load_time": "2026-02-17 15:00:00.000",
            },
            {
                "short_desc": series["short_desc"],
                "domain_desc": "TOTAL",
                "agg_level_desc": "NATIONAL",
                "freq_desc": "MONTHLY",
                "year": "2026",
                "reference_period_desc": "JAN",
                "Value": "999",
                "load_time": "2026-02-17 15:00:00.000",
            },
            {
                "short_desc": series["short_desc"],
                "domain_desc": series["domain_desc"],
                "agg_level_desc": "NATIONAL",
                "freq_desc": "MONTHLY",
                "year": "2026",
                "reference_period_desc": "FEB",
                "Value": "(D)",
                "load_time": "2026-03-23 15:00:00.000",
            },
        ]
    }
    rows = nass.parse_nass_rows(payload, series)
    assert [row["value"] for row in rows] == [281158000.0]
    assert rows[0]["load_time"].startswith("2026-02-17T15:00:00")


def test_nass_collection_reads_the_env_key_and_does_not_store_it(tmp_path, monkeypatch):
    monkeypatch.delenv("NASS_QUICKSTATS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NASS_QUICKSTATS_API_KEY"):
        nass.collect_nass(tmp_path, 2024)
    monkeypatch.setenv("NASS_QUICKSTATS_API_KEY", "TESTKEY12")
    seen = []

    def download(url):
        seen.append(url)
        series = nass.SERIES[len(seen) - 1]
        body = {
            "data": [
                {
                    "short_desc": series["short_desc"],
                    "domain_desc": series["domain_desc"],
                    "agg_level_desc": "NATIONAL",
                    "freq_desc": "MONTHLY",
                    "year": "2026",
                    "reference_period_desc": "JAN",
                    "Value": "1000",
                    "load_time": "2026-02-17 15:00:00.000",
                }
            ]
        }
        return json.dumps(body).encode()

    monkeypatch.setattr(nass, "_download", download)
    snapshot = nass.collect_nass(tmp_path, 2024)
    assert len(snapshot["records"]) == 3
    stored = (tmp_path / "snapshot.json").read_text()
    assert "TESTKEY12" not in stored
    assert all("TESTKEY12" in url and "key=" in url for url in seen)
    assert "TESTKEY12" not in json.dumps(snapshot["resources"])


def test_bound_profile_applies_the_egg_series_instead_of_the_share_price():
    data = replace(
        synthetic_data(
            DataSpec(source="synthetic", sessions=40, instruments=1, train_cutoff=30, seed=7)
        ),
        instruments=("CALM",),
    )
    profile = resolve_instrument_factors(ROOT, "US", "CALM")
    prices = np.full(len(data.dates), 100.0)
    prices[20] = 150.0
    series = {
        ("usda_ams_egg_reports", "US shell eggs", "wholesale_price"): prices,
        ("nass_chickens_eggs", "US table-egg layers", "inventory"): prices,
    }
    required = np.array([20])
    bundle = prepare_bound_features(profile, data, {}, required, series)
    columns = bundle["columns"]
    assert columns["bound.egg_price"][20] == pytest.approx(0.5)
    assert columns["bound.table_egg_layers"][20] == pytest.approx(0.5)
    assert not np.allclose(columns["bound.egg_price"], columns["bound.price_momentum"])
    bare = prepare_bound_features(profile, data, {}, required, None)
    assert np.isnan(bare["columns"]["bound.egg_price"]).all()
    assert bare["definitions"][2]["availability_reason"] == "retained_series_not_supplied"
