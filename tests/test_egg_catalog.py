"""Offline fixtures for the extra egg-catalog series."""

import json
from datetime import date

import pytest

from quant_rsi.research import egg_catalog as catalog
from quant_rsi.research.nass_chickens import parse_nass_rows

NY = """
Report for: 09/29/2025
Extra Large 162.00 - 166.00 164.00 -4.00 168.00
Large 160.00 - 164.00 162.00 -4.00 166.00
Explanatory Notes: carton prices
"""
BREAKING = """
Report for: 11/03/2026
48 lb 4.00 - 11.00 8.00 0.00 8.00
"""


def test_ny_large_ignores_extra_large_and_breaking_uses_the_48lb_average():
    parsed = catalog.parse_ny_shell_egg(NY)
    assert parsed["report_date"] == "2025-09-29"
    assert parsed["value"] == pytest.approx(162.0)
    assert catalog.parse_breaking_stock(BREAKING)["value"] == pytest.approx(8.0)
    with pytest.raises(ValueError, match="no large average"):
        catalog.parse_ny_shell_egg("Report for: 09/29/2025 Extra Large 1.00 - 2.00 1.50")


def test_esmis_links_keep_only_the_named_report():
    page = (
        '<a href="/sites/default/release-files/aa/bb/cc/AMS_2734.PDF">a</a>'
        '<a href="https://example.com/AMS_2734.PDF">b</a>'
        '<a href="/sites/default/release-files/aa/bb/cc/AMS_3888.PDF">c</a>'
    ).encode()
    links = catalog.esmis_links(
        page, catalog.NY_PUBLICATION, catalog.NY_PUBLICATION, "/ams_2734.pdf"
    )
    assert links == ["https://esmis.nal.usda.gov/sites/default/release-files/aa/bb/cc/AMS_2734.PDF"]
    with pytest.raises(ValueError, match="does not match"):
        catalog.esmis_links(
            page,
            "https://example.com/publication",
            catalog.NY_PUBLICATION,
            "/ams_2734.pdf",
        )


def test_nass_first_of_month_keeps_a_zero_molt_share():
    series = catalog.EXTRA_SERIES[2]
    payload = {
        "data": [
            {
                "short_desc": series["short_desc"],
                "domain_desc": series["domain_desc"],
                "agg_level_desc": "NATIONAL",
                "freq_desc": "POINT IN TIME",
                "year": "2026",
                "reference_period_desc": "FIRST OF JAN",
                "Value": "0",
                "load_time": "2026-02-17 15:00:00.000",
            },
            {
                "short_desc": series["short_desc"],
                "domain_desc": "OTHER",
                "agg_level_desc": "NATIONAL",
                "freq_desc": "POINT IN TIME",
                "year": "2026",
                "reference_period_desc": "FIRST OF JAN",
                "Value": "9",
                "load_time": "2026-02-17 15:00:00.000",
            },
        ]
    }
    rows = parse_nass_rows(payload, series)
    assert rows[0]["month"] == 1
    assert rows[0]["value"] == 0
    assert rows[0]["unit"] == "percent_of_inventory"


def test_public_parsers_keep_one_named_series():
    chart = {
        "chart": {
            "result": [
                {
                    "meta": {"currency": "USX"},
                    "timestamp": [1704067200, 1704153600],
                    "indicators": {"quote": [{"close": [100.0, None]}]},
                }
            ],
            "error": None,
        }
    }
    # 2024-01-01 00:00 UTC is still 2023-12-31 in New York.
    assert catalog.parse_yahoo_chart(chart, "ZC=F")[0]["date"] == "2023-12-31"
    bls = catalog.parse_bls(
        {
            "Results": {
                "series": [
                    {
                        "seriesID": "APU0000708111",
                        "data": [{"year": "2026", "period": "M09", "value": "2.272"}],
                    }
                ]
            }
        },
        "APU0000708111",
    )
    assert bls[0]["visible_on"] == "2026-10-15"
    effr = catalog.parse_effr(
        "Effective Date,Rate Type,Rate (%)\n11/02/2026,EFFR,3.88\n11/02/2026,OBFR,3.80\n"
    )
    assert [row["value"] for row in effr] == [pytest.approx(3.88)]
    diesel = catalog.parse_eia_diesel(
        {
            "response": {
                "data": [
                    {"series": catalog.DIESEL_SERIES, "period": "2026-11-02", "value": "6.292"},
                    {"series": "OTHER", "period": "2026-11-02", "value": "1"},
                ]
            }
        }
    )
    assert diesel[0]["value"] == pytest.approx(6.292)
    drought = catalog.parse_drought(
        "MapDate,AreaOfInterest,None,D0,D1,D2,D3,D4\n"
        "20250128,IA,10,20,5,1,0,0\n"
        "20250128,CONUS,10,20,5,1,0,0\n"
    )
    assert drought[0]["value"] == pytest.approx(6.0)
    national = catalog.parse_drought(
        "MapDate,AreaOfInterest,None,D0,D1,D2,D3,D4\n20250128,CONUS,10,20,1,0,0,0\n"
    )
    assert national[0]["region"] == "CONUS"
    assert national[0]["value"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no Iowa or national"):
        catalog.parse_drought("MapDate,AreaOfInterest,D1,D2,D3,D4\n20250128,XX,1,0,0,0\n")


def test_calendar_and_feed_index_and_concentration():
    rows = catalog.calendar_rows("2025-04-19", "2025-04-20")
    assert [row["easter"] for row in rows] == [1, 0]
    assert catalog._thanksgiving(2025) == date(2025, 11, 27)
    feed = catalog._feed_index(
        [
            {"date": "2024-01-02", "value": 100.0},
            {"date": "2024-01-03", "value": 110.0},
        ],
        [
            {"date": "2024-01-02", "value": 200.0},
            {"date": "2024-01-03", "value": 200.0},
        ],
    )
    assert feed[0]["value"] == pytest.approx(100.0)
    assert feed[1]["value"] == pytest.approx(105.0)
    found = {
        "_layer_values": {"2024-01": 100.0},
        "_state_layers": {
            f"layers_{state.lower()}": [{"date": "2024-01", "value": 10.0}]
            for state in catalog.STATE_LAYERS
        },
    }
    found["_state_layers"]["layers_ia"] = [{"date": "2024-01", "value": 40.0}]
    catalog._derive_concentration(found)
    assert found["concentration"]["_rows"][0]["value"] == pytest.approx(0.8)


def test_carried_value_stops_at_a_gap_and_can_stay_hidden_on_its_own_date():
    records = [
        {"date": "2026-01-02", "value": 10},
        {"date": "2026-01-05", "value": 12},
        {"date": "2026-01-20", "value": 14},
    ]
    sessions = ["2026-01-02", "2026-01-05", "2026-01-08", "2026-01-12", "2026-01-21"]
    assert catalog.align_carried(records, sessions, carry_days=6) == [10, 12, 12, None, 14]
    assert catalog.align_carried(records, sessions, carry_days=6, hide_on_date=True) == [
        None,
        10,
        12,
        None,
        14,
    ]


def test_coverage_lists_every_catalog_row():
    empty = catalog.coverage_rows({})
    assert len(empty) == 70
    assert len({row["label"] for row in empty}) == 70
    assert all(row["status"] == "missing" for row in empty)
    found = {
        "layers": {
            "count": 2,
            "start": "2024-01",
            "end": "2024-02",
            "unit": "head",
            "note": "kept",
        }
    }
    selected = [row for row in catalog.coverage_rows(found) if row["status"] == "collected"]
    assert [row["label"] for row in selected] == ["餐桌蛋產蛋雞群規模（table-egg layers）"]
    text = catalog.render_coverage(empty)
    assert "有數 0 列，缺 70 列。" in text


def test_extra_nass_collection_does_not_store_the_key(tmp_path, monkeypatch):
    monkeypatch.setenv("NASS_QUICKSTATS_API_KEY", "TESTKEY12")
    seen = []

    def download(url):
        seen.append(url)
        series = catalog._state_series()[len(seen) - 1]
        body = {
            "data": [
                {
                    "short_desc": series["short_desc"],
                    "domain_desc": series["domain_desc"],
                    "agg_level_desc": series.get("agg_level_desc", "NATIONAL"),
                    "freq_desc": series.get("freq_desc", "MONTHLY"),
                    "state_alpha": series.get("state_alpha", "US"),
                    "year": "2026",
                    "reference_period_desc": (
                        "FIRST OF JAN" if series.get("freq_desc") == "POINT IN TIME" else "JAN"
                    ),
                    "Value": "0" if series.get("allow_zero") else "10",
                    "load_time": "2026-02-17 15:00:00.000",
                }
            ]
        }
        return json.dumps(body).encode()

    monkeypatch.setattr(catalog, "nass_download", download)
    found = {}
    snapshot = catalog._collect_extra_nass(tmp_path, found, 2024)
    stored = (tmp_path / "snapshot.json").read_text()
    assert "TESTKEY12" not in stored
    assert all("TESTKEY12" in url for url in seen)
    assert len(snapshot["records"]) == len(catalog._state_series())
    assert found["molt"]["count"] == 1
    assert found["molt"]["_rows"][0]["value"] == 0
