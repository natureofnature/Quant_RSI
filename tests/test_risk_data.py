"""Offline measured-risk fixtures; synthetic levels do not establish research value."""

import json
from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pytest

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.market.risk_data import (
    ATOM,
    DATA,
    META,
    TREASURY_URL,
    VIX_URL,
    align_risk_features,
    load_risk_history,
    parse_treasury,
    parse_vix,
)


def treasury_xml(rows):
    entries = []
    for day, value in rows:
        field = (
            '<d:BC_10YEAR m:null="true"/>'
            if value is None
            else f"<d:BC_10YEAR>{value}</d:BC_10YEAR>"
        )
        entries.append(
            f'<entry><content type="application/xml"><m:properties>'
            f"<d:NEW_DATE>{day}T00:00:00</d:NEW_DATE>{field}</m:properties></content></entry>"
        )
    return (
        f'<feed xmlns="{ATOM[1:-1]}" xmlns:m="{META[1:-1]}" xmlns:d="{DATA[1:-1]}">'
        + "".join(entries)
        + "</feed>"
    ).encode()


def write_bundle(root, vix_rows, treasury_rows):
    """Build bounded official-format fixtures and matching receipts for integration tests."""
    root.mkdir()
    (root / "raw").mkdir()
    bodies = {
        "vix.csv": (
            "DATE,CLOSE\n"
            + "".join(f"{date.fromisoformat(day):%m/%d/%Y},{value}\n" for day, value in vix_rows)
        ).encode()
    }
    years = sorted({day[:4] for day, _ in treasury_rows})
    bodies.update(
        {
            f"treasury-{year}.xml": treasury_xml(
                [(day, value) for day, value in treasury_rows if day[:4] == year]
            )
            for year in years
        }
    )
    receipts = []
    for name, body in bodies.items():
        url = VIX_URL if name == "vix.csv" else TREASURY_URL + name[9:13]
        (root / "raw" / name).write_bytes(body)
        receipts.append(
            {
                "file": name,
                "requested_url": url,
                "response_url": url,
                "status": 200,
                "requested_at": "2026-11-04T19:30:00+00:00",
                "received_at": "2026-11-04T19:31:00+00:00",
                "sha256": digest(body),
                "bytes": len(body),
            }
        )
    (root / "receipts.json").write_bytes(canonical(receipts))
    return root


def test_parse_units_missing_and_identical_duplicates():
    assert parse_vix(b"DATE,CLOSE\n01/02/2025,18.25\n01/02/2025,18.250\n") == {"2025-01-02": 18.25}
    assert parse_treasury(
        treasury_xml(
            [
                ("2025-01-02", 4.57),
                ("2025-01-03", None),
                ("2025-01-02", 4.570),
            ]
        )
    ) == {"2025-01-02": 4.57, "2025-01-03": None}
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        parse_vix(b"DATE,CLOSE\n01/02/2025,18.25\n01/02/2025,18.26\n")
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        parse_treasury(treasury_xml([("2025-01-02", 4.57), ("2025-01-02", None)]))


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "invalid"])
def test_nonfinite_and_invalid_values_are_rejected(value):
    with pytest.raises(ValueError):
        parse_vix(f"DATE,CLOSE\n01/02/2025,{value}\n".encode())
    with pytest.raises(ValueError):
        parse_treasury(treasury_xml([("2025-01-02", value)]))


def test_invalid_dates_and_xml_entities_are_rejected():
    with pytest.raises(ValueError):
        parse_vix(b"DATE,CLOSE\n02/30/2025,18\n")
    with pytest.raises(ValueError):
        parse_treasury(treasury_xml([("2025-02-30", 4.5)]))
    with pytest.raises(ValueError, match="declarations"):
        parse_treasury(b'<!DOCTYPE feed [<!ENTITY x "4.5">]><feed/>')


def test_alignment_prior_date_holiday_missing_and_stale(tmp_path):
    root = write_bundle(
        tmp_path / "history",
        [("2025-01-03", 15), ("2025-01-06", 20)],
        [("2025-01-03", 4.2), ("2025-01-06", None), ("2025-01-07", 4.4)],
    )
    bundle = load_risk_history(root)
    columns, definitions, diagnostics = align_risk_features(
        bundle, ["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-10", "2025-01-14", "2025-01-15"]
    )
    np.testing.assert_allclose(
        columns["risk.vix_close"], [np.nan, 15, 20, 20, np.nan, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        columns["risk.treasury_10y_percent"],
        [np.nan, 4.2, np.nan, 4.4, 4.4, np.nan],
        equal_nan=True,
    )
    by_id = {row["factor_id"]: row for row in diagnostics}
    treasury = by_id["risk.treasury_10y_percent"]["rows"]
    assert treasury[2]["endpoints"][0]["status"] == "explicit_missing"
    assert treasury[3]["endpoints"][0]["observation_date"] == "2025-01-07"
    assert treasury[4]["endpoints"][0]["age_days"] == 7
    assert treasury[5]["endpoints"][0]["status"] == "stale"
    assert {d["factor_id"]: d["unit"] for d in definitions}[
        "risk.treasury_10y_percent"
    ] == "percent"
    assert all(np.isnan(columns["risk.treasury_10y_change_20"]))
    canonical(diagnostics)  # Unavailable provenance is None, never non-JSON NaN.


def test_equity_row_changes_and_future_invariance(tmp_path):
    days = []
    current = date(2025, 1, 2)
    while len(days) < 27:
        if current.weekday() < 5 and current != date(2025, 1, 9):
            days.append(current.isoformat())
        current += timedelta(days=1)
    root = write_bundle(
        tmp_path / "history",
        [(day, 10 + i) for i, day in enumerate(days)],
        [(day, 4 + i / 100) for i, day in enumerate(days)],
    )
    bundle = load_risk_history(root)
    original, _, diagnostic = align_risk_features(bundle, days)
    assert np.isnan(original["risk.vix_change_5"][5])  # First endpoint has no prior record.
    assert original["risk.vix_change_5"][6] == 5
    assert original["risk.treasury_10y_change_20"][21] == pytest.approx(0.20)
    assert original["risk.treasury_10y_percent"][21] == pytest.approx(4.20)
    changed = deepcopy(bundle)
    for series in changed["series"].values():
        for row in series:
            if row["date"] >= days[21]:
                row["value"] = 99.0
    revised, _, revised_diagnostic = align_risk_features(changed, days)
    for key in original:
        np.testing.assert_array_equal(original[key][:22], revised[key][:22])
    assert diagnostic[3]["rows"][21] == revised_diagnostic[3]["rows"][21]
    endpoints = diagnostic[3]["rows"][21]["endpoints"]
    assert [value["observation_date"] for value in endpoints] == [days[20], days[0]]
    assert all(endpoint["source_refs"] for endpoint in endpoints)
    assert revised["risk.vix_close"][22] == 99


def test_bundle_is_read_only_and_canonical(tmp_path):
    root = write_bundle(tmp_path / "history", [("2025-01-02", 18)], [("2025-01-02", 4.5)])
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    bundle = load_risk_history(root)
    assert bundle["series"]["treasury_10y"][0]["value"] == 4.5
    assert bundle["series"]["vix"][0]["source_refs"] == [bundle["sources"][0]["sha256"]]
    canonical(bundle)
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("damage", ["raw", "status", "source_url", "filename", "year", "missing"])
def test_bundle_rejects_tamper_and_wrong_provenance(tmp_path, damage):
    root = write_bundle(tmp_path / "history", [("2025-01-02", 18)], [("2025-01-02", 4.5)])
    path = root / "receipts.json"
    receipts = json.loads(path.read_bytes())
    if damage == "raw":
        (root / "raw/vix.csv").write_bytes(b"changed")
    elif damage == "status":
        receipts[0]["status"] = 500
    elif damage == "source_url":
        receipts[0]["response_url"] = "https://example.com/vix.csv"
    elif damage == "filename":
        receipts[0]["file"] = "../vix.csv"
    elif damage == "missing":
        (root / "raw/treasury-2025.xml").unlink()
    else:
        body = treasury_xml([("2024-12-31", 4.5)])
        (root / "raw/treasury-2025.xml").write_bytes(body)
        receipts[1].update(sha256=digest(body), bytes=len(body))
    path.write_bytes(canonical(receipts))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_risk_history(root)
