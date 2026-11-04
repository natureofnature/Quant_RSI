"""Publisher-vintage GPR levels with conservative, date-based availability gates."""

import json
import math
import re
from bisect import bisect_left
from datetime import date, datetime
from pathlib import Path

import numpy as np

from quant_rsi.core.artifacts import digest
from quant_rsi.market.market_contracts import timestamp

MAX_AGE_DAYS = 14
NOTICE = (
    "Archived publisher GPR vintages, not a certified point-in-time archive. "
    "Retained GitHub commit metadata is not a cryptographic publication-time attestation. "
    "Use a vintage only on equity dates strictly after both its filename release date "
    "and its commit UTC date; observations must precede the equity date. "
    "Release and observation ages may not exceed fourteen calendar days. "
    "No newest-vintage backfill, interpolation, zero fill or missing-value carry. "
    "GPR is a newspaper-based geopolitical-risk index, not a direct event probability."
)
_FILE = re.compile(r"data_gpr_daily_recent_(\d{8})\.xls")
_REQUIRED = {
    "file",
    "requested_url",
    "response_url",
    "status",
    "requested_at",
    "received_at",
    "bytes",
    "sha256",
    "release_date",
    "commit_sha",
    "committed_at",
}


def parse_gpr(body):
    """Parse DAY/GPRD/date from the publisher's binary XLS; preserve explicit missing."""
    try:
        import xlrd
    except ImportError as exc:
        raise ValueError("GPR XLS support requires: pip install 'quant-rsi[gpr]'") from exc
    if not body or len(body) > 8 * 1024 * 1024:
        raise ValueError("GPR workbook must be nonempty and at most 8 MiB")
    try:
        book = xlrd.open_workbook(file_contents=body, on_demand=True)
    except xlrd.XLRDError as exc:
        raise ValueError("Invalid publisher GPR XLS workbook") from exc
    try:
        if book.nsheets != 1:
            raise ValueError("GPR workbook requires one unambiguous worksheet")
        sheet = book.sheet_by_index(0)
        if not 2 <= sheet.nrows <= 100_000 or not 3 <= sheet.ncols <= 32:
            raise ValueError("GPR worksheet exceeds bounds or has no observations")
        headers = sheet.row_values(0)
        if any(not isinstance(value, str) or not value for value in headers):
            raise ValueError("GPR worksheet needs named columns")
        if len(set(headers)) != len(headers) or not {"DAY", "GPRD", "date"} <= set(headers):
            raise ValueError("GPR worksheet requires unambiguous DAY, GPRD and date columns")
        day_col, level_col, serial_col = [headers.index(name) for name in ("DAY", "GPRD", "date")]
        values = {}
        for index in range(1, sheet.nrows):
            day_cell, level, serial = [
                sheet.cell(index, column) for column in (day_col, level_col, serial_col)
            ]
            if day_cell.ctype != xlrd.XL_CELL_TEXT or not re.fullmatch(r"\d{8}", day_cell.value):
                raise ValueError("GPR DAY must be an eight-digit publisher date")
            day = datetime.strptime(day_cell.value, "%Y%m%d").date().isoformat()
            if serial.ctype not in {xlrd.XL_CELL_DATE, xlrd.XL_CELL_NUMBER}:
                raise ValueError("GPR date column must contain Excel dates")
            excel_date = xlrd.xldate_as_datetime(serial.value, book.datemode)
            if excel_date.date().isoformat() != day or excel_date.time().isoformat() != "00:00:00":
                raise ValueError("GPR DAY and Excel date disagree")
            if level.ctype in {xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK} or (
                level.ctype == xlrd.XL_CELL_TEXT and level.value == ""
            ):
                value = None
            elif level.ctype == xlrd.XL_CELL_NUMBER:
                value = float(level.value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("GPR level must be finite and nonnegative")
            else:
                raise ValueError("GPR level must be numeric or explicitly empty")
            if day in values:
                raise ValueError("Duplicate GPR observation date")
            values[day] = value
        return dict(sorted(values.items()))
    finally:
        book.release_resources()


def load_gpr_history(directory):
    """Verify pinned publisher archives and derive availability from retained receipts."""
    root = Path(directory).resolve()
    manifest = root / "receipts.json"
    if manifest.stat().st_size > 512 * 1024:
        raise ValueError("GPR receipt manifest exceeds byte limit")
    receipts = json.loads(manifest.read_bytes())
    if not isinstance(receipts, list) or not 1 <= len(receipts) <= 128:
        raise ValueError("GPR bundle requires one to 128 archive receipts")
    vintages, seen = [], set()
    for receipt in receipts:
        if not isinstance(receipt, dict) or not _REQUIRED <= receipt.keys():
            raise ValueError("Incomplete GPR source receipt")
        strings = _REQUIRED - {"status", "bytes"}
        if any(not isinstance(receipt[key], str) for key in strings):
            raise ValueError("GPR receipt text fields must be strings")
        match = _FILE.fullmatch(receipt["file"])
        if not match or not re.fullmatch(r"[0-9a-f]{40}", receipt["commit_sha"]):
            raise ValueError("GPR source needs a dated filename and full commit SHA")
        release = datetime.strptime(match[1], "%Y%m%d").date()
        if receipt["release_date"] != release.isoformat() or release in seen:
            raise ValueError("GPR release date differs from filename or is duplicated")
        seen.add(release)
        expected_url = (
            "https://raw.githubusercontent.com/iacoviel/iacoviel.github.io/"
            f"{receipt['commit_sha']}/gpr_archive_files/{receipt['file']}"
        )
        if receipt["requested_url"] != expected_url or receipt["response_url"] != expected_url:
            raise ValueError("GPR source must use its commit-pinned publisher URL")
        if type(receipt["status"]) is not int or receipt["status"] != 200:
            raise ValueError("GPR source requires a successful HTTP receipt")
        if type(receipt["bytes"]) is not int or not 0 < receipt["bytes"] <= 8 * 1024 * 1024:
            raise ValueError("Invalid GPR source byte count")
        if not re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"]):
            raise ValueError("Invalid GPR source SHA-256")
        requested, received, committed = [
            timestamp(receipt[key]) for key in ("requested_at", "received_at", "committed_at")
        ]
        if requested > received or committed > received or committed.date() < release:
            raise ValueError("Inconsistent GPR release, commit or retrieval timestamps")
        path = (root / "raw" / receipt["file"]).resolve()
        if not path.is_relative_to(root) or path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("GPR source escapes bundle or exceeds byte limit")
        body = path.read_bytes()
        if len(body) != receipt["bytes"] or digest(body) != receipt["sha256"]:
            raise ValueError("GPR source bytes/hash differ from receipt")
        observations = parse_gpr(body)
        if max(observations) > release.isoformat():
            raise ValueError("GPR observation occurs after its declared release date")
        vintages.append(
            {
                "file": receipt["file"],
                "release_date": release.isoformat(),
                "commit_sha": receipt["commit_sha"],
                "committed_at": committed.isoformat(),
                "available_after_date": max(release, committed.date()).isoformat(),
                "source_ref": receipt["sha256"],
                "observations": [
                    {"date": day, "value": value} for day, value in observations.items()
                ],
            }
        )
    return {
        "kind": "ArchivedGPRVintages",
        "notice": NOTICE,
        "sources": receipts,
        "vintages": sorted(vintages, key=lambda item: item["release_date"]),
    }


def align_gpr_features(bundle, session_dates):
    """Select the latest available vintage first; never backfill from another vintage."""
    sessions = [date.fromisoformat(day) for day in session_dates]
    if not sessions or any(a >= b for a, b in zip(sessions, sessions[1:], strict=False)):
        raise ValueError("Equity session dates must be nonempty and strictly increasing")
    values, rows = [], []
    for session in sessions:
        eligible = [
            item
            for item in bundle["vintages"]
            if item["available_after_date"] < session.isoformat()
        ]
        row = {"session_date": session.isoformat(), "value": None, "status": "no_available_vintage"}
        if eligible:
            vintage = max(eligible, key=lambda item: item["release_date"])
            release_age = (session - date.fromisoformat(vintage["release_date"])).days
            row.update(
                {
                    key: vintage[key]
                    for key in (
                        "file",
                        "release_date",
                        "commit_sha",
                        "committed_at",
                        "available_after_date",
                        "source_ref",
                    )
                }
            )
            row["release_age_days"] = release_age
            observations = vintage["observations"]
            index = bisect_left([item["date"] for item in observations], session.isoformat()) - 1
            row["status"] = "no_prior_observation"
            if index >= 0:
                observation = observations[index]
                age = (session - date.fromisoformat(observation["date"])).days
                row.update(observation_date=observation["date"], observation_age_days=age)
                row["status"] = (
                    "stale_release"
                    if release_age > MAX_AGE_DAYS
                    else "stale_observation"
                    if age > MAX_AGE_DAYS
                    else "explicit_missing"
                    if observation["value"] is None
                    else "available"
                )
                if row["status"] == "available":
                    row["value"] = observation["value"]
        values.append(row["value"])
        rows.append(row)
    identity = "risk.gpr_level"
    definition = {
        "factor_id": identity,
        "kind": "numeric",
        "unit": "index_1985_2019_100",
        "runnable": True,
        "description": "Latest available archived daily newspaper geopolitical-risk index level",
        "recipe": {"op": "gpr_vintage_level", "max_age_days": MAX_AGE_DAYS},
    }
    return (
        {identity: np.asarray(values, dtype=float)},
        [definition],
        [{"factor_id": identity, "unit": definition["unit"], "notice": NOTICE, "rows": rows}],
    )
