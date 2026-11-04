"""Collect the CALM supply sources and align them to the price sessions."""

import os

import numpy as np

from quant_rsi.research.ams_shell_eggs import align_shell_eggs, collect_shell_eggs
from quant_rsi.research.egg_catalog import load_egg_catalog
from quant_rsi.research.nass_chickens import SERIES, align_nass, collect_nass


def _column(values):
    return np.array([np.nan if value is None else value for value in values], dtype=float)


def load_bound_supply(profile, dates, cutoffs, directory):
    """Leave a source empty when its collector fails. Do not impute it."""
    sources = {item["source"]["source_id"] for item in profile["factors"]}
    series, report = {}, {"failures": []}
    if "usda_ams_egg_reports" in sources:
        try:
            snapshot = collect_shell_eggs(directory / "shell_eggs", dates[0], dates[-1])
            series[("usda_ams_egg_reports", "US shell eggs", "wholesale_price")] = _column(
                align_shell_eggs(snapshot["records"], dates)
            )
            report["shell_eggs"] = {
                "provider": snapshot["provider"],
                "notice": snapshot["notice"],
                "sha256": snapshot["sha256"],
                "bounds": snapshot["bounds"],
                "observations": len(snapshot["records"]),
                "failures": snapshot["failures"],
            }
        except (OSError, ValueError) as exc:
            report["failures"].append({"source": "usda_ams_egg_reports", "reason": str(exc)})
    if "nass_chickens_eggs" in sources:
        try:
            snapshot = collect_nass(directory / "nass", int(dates[0][:4]) - 1)
            for item in SERIES:
                series[("nass_chickens_eggs", item["entity"], item["field"])] = _column(
                    align_nass(snapshot["records"], cutoffs, item["field"])
                )
            report["nass"] = {
                "provider": snapshot["provider"],
                "notice": snapshot["notice"],
                "sha256": snapshot["sha256"],
                "first_year": snapshot["first_year"],
                "observations": len(snapshot["records"]),
            }
        except (OSError, ValueError) as exc:
            report["failures"].append({"source": "nass_chickens_eggs", "reason": str(exc)})
    if "egg_catalog" in sources:
        catalog_dir = os.environ.get("EGG_CATALOG_DIR", "")
        if not catalog_dir:
            report["failures"].append(
                {"source": "egg_catalog", "reason": "EGG_CATALOG_DIR is not set"}
            )
        else:
            try:
                loaded = load_egg_catalog(catalog_dir, dates, cutoffs)
                for field, values in loaded.items():
                    series[("egg_catalog", "US", field)] = _column(values)
                report["egg_catalog"] = {
                    "directory": catalog_dir,
                    "fields": sorted(loaded),
                }
            except (OSError, ValueError) as exc:
                report["failures"].append({"source": "egg_catalog", "reason": str(exc)})
    return series, report
