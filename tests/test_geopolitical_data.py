"""Synthetic publisher-shaped XLS fixtures test timing mechanics, not economic value.

fixtures/gpr_vintages.zip was generated once with xlwt 1.3.0; its README describes
the deterministic rows. xlwt is not a test/runtime dependency. All reads use xlrd.
"""

import json
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from quant_rsi.core.artifacts import digest
from quant_rsi.market.geopolitical_data import align_gpr_features, load_gpr_history, parse_gpr

FIXTURES = Path(__file__).parent / "fixtures" / "gpr_vintages.zip"


def write_gpr_bundle(root, releases=None):
    """Write verified-format synthetic archives; default is weekly January–August 2026."""
    root = Path(root)
    (root / "raw").mkdir(parents=True)
    receipts = []
    with ZipFile(FIXTURES) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for key in (
            releases if releases is not None else sorted(k for k in manifest if k[:4] == "2026")
        ):
            release = manifest[key]
            name = f"data_gpr_daily_recent_{release.replace('-', '')}.xls"
            body = archive.read(key + ".xls")
            commit = digest(key.encode())[:40]
            url = (
                "https://raw.githubusercontent.com/iacoviel/iacoviel.github.io/"
                f"{commit}/gpr_archive_files/{name}"
            )
            (root / "raw" / name).write_bytes(body)
            receipts.append(
                {
                    "file": name,
                    "requested_url": url,
                    "response_url": url,
                    "status": 200,
                    "requested_at": "2026-11-04T20:00:00+00:00",
                    "received_at": "2026-11-04T20:01:00+00:00",
                    "bytes": len(body),
                    "sha256": digest(body),
                    "release_date": release,
                    "commit_sha": commit,
                    "committed_at": release + "T18:00:00+00:00",
                }
            )
    (root / "receipts.json").write_text(json.dumps(receipts))
    return root


def edit_receipts(root, edit):
    path = root / "receipts.json"
    receipts = json.loads(path.read_bytes())
    edit(receipts)
    path.write_text(json.dumps(receipts))


def aligned(bundle, dates):
    columns, definitions, diagnostics = align_gpr_features(bundle, dates)
    assert definitions[0]["factor_id"] == "risk.gpr_level"
    assert definitions[0]["unit"] == "index_1985_2019_100"
    return columns["risk.gpr_level"], diagnostics[0]["rows"]


def test_binary_vintage_revision_cannot_rewrite_earlier_equity_rows(tmp_path):
    early = load_gpr_history(write_gpr_bundle(tmp_path / "early", ["first"]))
    all_vintages = load_gpr_history(write_gpr_bundle(tmp_path / "all", ["first", "revised"]))
    dates = ["2024-06-17", "2024-06-18", "2024-06-24", "2024-06-25"]
    values, rows = aligned(all_vintages, dates)
    np.testing.assert_equal(values, [np.nan, 10, 10, 30])
    previous, previous_rows = aligned(early, dates[:3])
    np.testing.assert_equal(values[:3], previous)
    assert rows[:3] == previous_rows
    assert rows[-1]["observation_date"] == "2024-06-16"
    assert rows[-1]["release_date"] == "2024-06-24"
    assert rows[-1]["source_ref"] == all_vintages["sources"][1]["sha256"]


def test_commit_utc_date_and_equity_calendar_delay_availability(tmp_path):
    root = write_gpr_bundle(tmp_path / "late", ["first"])
    edit_receipts(root, lambda rows: rows[0].update(committed_at="2024-06-21T23:30:00-04:00"))
    bundle = load_gpr_history(root)
    assert bundle["vintages"][0]["available_after_date"] == "2024-06-22"
    values, _ = aligned(bundle, ["2024-06-17", "2024-06-21", "2024-06-24"])
    np.testing.assert_equal(values, [np.nan, np.nan, 10])
    edit_receipts(root, lambda rows: rows[0].update(available_after_date="1900-01-01"))
    overridden = load_gpr_history(root)
    assert overridden["vintages"][0]["available_after_date"] == "2024-06-22"


def test_latest_vintage_missing_value_blocks_older_vintage(tmp_path):
    bundle = load_gpr_history(write_gpr_bundle(tmp_path / "missing", ["first", "missing"]))
    values, rows = aligned(bundle, ["2024-06-24", "2024-06-25"])
    np.testing.assert_equal(values, [10, np.nan])
    assert rows[-1]["status"] == "explicit_missing"
    assert rows[-1]["observation_date"] == "2024-06-24"


def test_absent_and_stale_archives_or_observations_stay_unavailable(tmp_path):
    recent_only = load_gpr_history(write_gpr_bundle(tmp_path / "only", ["revised"]))
    values, rows = aligned(recent_only, ["2024-06-18", "2024-06-25", "2024-07-01", "2024-07-09"])
    np.testing.assert_equal(values, [np.nan, 30, np.nan, np.nan])
    assert [row["status"] for row in rows] == [
        "no_available_vintage",
        "available",
        "stale_observation",
        "stale_release",
    ]
    old = load_gpr_history(write_gpr_bundle(tmp_path / "old", ["old_observation"]))
    values, rows = aligned(old, ["2024-06-25"])
    assert np.isnan(values[0]) and rows[0]["status"] == "stale_observation"


@pytest.mark.parametrize(
    "key, reason",
    [
        ("negative", "nonnegative"),
        ("nonfinite", "finite"),
        ("duplicate", "Duplicate"),
        ("future", "after its declared release"),
        ("bad_header", "unambiguous"),
        ("bad_date", "disagree"),
    ],
)
def test_invalid_binary_workbooks_are_rejected(tmp_path, key, reason):
    root = write_gpr_bundle(tmp_path / key, [key])
    with pytest.raises(ValueError, match=reason):
        load_gpr_history(root)


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"requested_url": "https://example.com/gpr.xls"}, "commit-pinned"),
        (
            {
                "response_url": "https://raw.githubusercontent.com/iacoviel/iacoviel.github.io/main/x.xls"
            },
            "commit-pinned",
        ),
        ({"committed_at": "2024-06-16T23:59:00Z"}, "timestamps"),
        ({"committed_at": "2026-11-05T00:00:00Z"}, "timestamps"),
        ({"requested_at": "2026-11-05T00:00:00Z"}, "timestamps"),
        ({"release_date": "2024-06-18"}, "release date"),
        ({"commit_sha": "main"}, "commit SHA"),
        ({"status": 403}, "successful HTTP"),
        ({"bytes": 8 * 1024 * 1024 + 1}, "byte count"),
        ({"sha256": "f" * 64}, "bytes/hash"),
    ],
)
def test_receipts_fail_closed(tmp_path, changes, reason):
    root = write_gpr_bundle(tmp_path / "receipt", ["first"])
    edit_receipts(root, lambda rows: rows[0].update(changes))
    with pytest.raises(ValueError, match=reason):
        load_gpr_history(root)


def test_duplicate_receipt_missing_fields_and_bounded_manifest(tmp_path):
    root = write_gpr_bundle(tmp_path / "manifest", ["first"])
    original = json.loads((root / "receipts.json").read_bytes())
    for rows, reason in (
        (original * 2, "duplicated"),
        (original * 129, "128 archive"),
        ([{}], "Incomplete"),
    ):
        (root / "receipts.json").write_text(json.dumps(rows))
        with pytest.raises(ValueError, match=reason):
            load_gpr_history(root)
    (root / "receipts.json").write_text(" " * (512 * 1024 + 1))
    with pytest.raises(ValueError, match="manifest exceeds"):
        load_gpr_history(root)


def test_weekly_2026_fixture_covers_equity_rows_without_future_backfill(tmp_path):
    bundle = load_gpr_history(write_gpr_bundle(tmp_path / "weekly"))
    values, rows = aligned(bundle, ["2026-01-02", "2026-02-02", "2026-06-30", "2026-08-28"])
    assert np.isnan(values[0]) and np.isfinite(values[1:]).all()
    assert all(row["release_date"] < row["session_date"] for row in rows[1:])
    assert len(bundle["sources"]) == 35
    assert len(bundle["vintages"]) == 35


def test_invalid_session_order_and_non_xls_are_rejected(tmp_path):
    bundle = load_gpr_history(write_gpr_bundle(tmp_path / "early", ["first"]))
    original = deepcopy(bundle)
    for dates in ([], ["2024-06-25", "2024-06-24"], ["2024-06-24"] * 2):
        with pytest.raises(ValueError, match="strictly increasing"):
            align_gpr_features(bundle, dates)
    with pytest.raises(ValueError, match="Invalid publisher"):
        parse_gpr(b"not an XLS")
    assert bundle == original
