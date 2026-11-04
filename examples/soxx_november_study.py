"""Freeze Jan-Oct 2026 SOXX/peer inputs, then run the existing discovery CLI."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from html import escape
from pathlib import Path

from quant_rsi.market.daily_data import collect_daily
from quant_rsi.research.demo_prices import load_daily_prices
from quant_rsi.research.demo_related import profile_related

CUTOFF = "2026-10-31T23:59:59-04:00"
HISTORY = 209
HORIZON = 20
SPLIT_EFFECTIVE = "2026-11-04T16:00:00-05:00"
SPLIT_SOURCE = "https://www.sec.gov/Archives/edgar/data/1100663/000119312526361162/d133118d497.htm"


def prepare_inputs(directory):
    """Reuse only verified immutable inputs; never change an existing collection."""
    directory = Path(directory).resolve()
    symbols = ["SOXX", *profile_related("SOXX")["stocks"]]
    dates = None
    for symbol in symbols:
        root = directory / symbol
        if not root.exists():
            collect_daily(
                symbol,
                root,
                historical_sessions=HISTORY,
                future_sessions=HORIZON,
                as_of=datetime.fromisoformat(CUTOFF),
            )
        data, info = load_daily_prices(
            symbol, root, horizon_sessions=HORIZON, historical_sessions=HISTORY
        )
        if datetime.fromisoformat(info["received_at"]) >= datetime.fromisoformat(SPLIT_EFFECTIVE):
            raise ValueError(
                "Use preserved pre-split collections: newly fetched OHLC can be "
                "retrospectively split-adjusted; this example does not reverse adjustments"
            )
        if (
            len(data.dates) != HISTORY
            or data.dates[0] != "2026-01-02"
            or data.dates[-1] != "2026-10-30"
            or info["latest_target_session"] != "2026-11-30"
            or info["price_cutoff"] != CUTOFF
            or (dates is not None and data.dates != dates)
        ):
            raise ValueError(f"{symbol} collection does not match the frozen Jan-Oct task")
        dates = data.dates
    return directory


def study_command(inputs, output, *, llm="codex", news_archive=None):
    command = [
        sys.executable,
        "-m",
        "quant_rsi",
        "research-study",
        "SOXX",
        "--prices",
        "daily",
        "--prices-dir",
        str(inputs / "SOXX"),
        "--related-prices-dir",
        str(inputs),
        "--historical-sessions",
        str(HISTORY),
        "--horizon-sessions",
        str(HORIZON),
        "--research-mode",
        "discovery",
        "--information-mode",
        "retained",
        "--research-cutoff",
        CUTOFF,
        "--train-dates",
        "2026-01-02",
        "2026-08-31",
        "--development-dates",
        "2026-09-01",
        "2026-10-02",
        "--terminal-sessions",
        "0",
        "--latest-fit",
        "selected",
        "--rounds",
        "6",
        "--max-fits",
        "4",
        "--llm",
        llm,
        "--no-history",
        "--no-ui",
        "--output",
        str(output),
        "--question",
        "Forecast SOXX from the 2026-10-30 close to 2026-11-30 using only the "
        "supplied Jan-Oct prices and cutoff-eligible evidence. Inspect experimental "
        "factor definitions. Investigate semiconductor sector and broad-market "
        "relationships; propose executable candidates only when evidence supports them. "
        "If historical news or inputs are missing, report the gap and retain the "
        "baseline rather than invent evidence. November outcomes are unavailable. "
        "The model predicts on the October price/share basis, not a split-adjusted "
        "November nominal quote. Keep development results separate from the forecast.",
    ]
    if news_archive is not None:
        command.extend(["--news-archive", str(news_archive)])
    return command


def forecast_interpretation(result):
    """Presentation only: the split conversion does not enter research or selection."""
    forecast = result["summary"].get("latest_forecast")
    interpretation = {
        "symbol": "SOXX",
        "status": "available" if forecast else "unavailable",
        "price_basis": "2026-10-30_share_equivalent_USD",
        "split_ratio": 3,
        "split_source": SPLIT_SOURCE,
        "split_source_publication_date": "2026-08-21",
        "source_use": "presentation_only; not a pre-cutoff retained research article",
        "notice": "Selected research fit; no publication refit. November outcome unobserved. "
        "The converted quote is arithmetic, not a second prediction or an accuracy score. "
        "No eligible historical news archive was supplied unless explicitly configured. "
        "Historical prices are reconstructed; LLM pretrained knowledge is not time-filtered.",
    }
    if forecast:
        if (forecast["origin_session"], forecast["target_session"]) != ("2026-10-30", "2026-11-30"):
            raise ValueError("Forecast does not match the frozen October-to-November task")
        interpretation.update(
            forecast=forecast,
            predicted_price_return=forecast["predicted_price"] / forecast["reference_price"] - 1,
            indicative_post_split_price=forecast["predicted_price"] / 3,
        )
    else:
        interpretation["reason"] = result["summary"].get(
            "latest_forecast_unavailable", "No comparable selected fit"
        )
    return interpretation


def write_forecast_summary(output):
    result = json.loads((output / "result.json").read_bytes())
    value = forecast_interpretation(result)
    text = json.dumps(value, indent=2, allow_nan=False)
    (output / "november_forecast.json").write_text(text + "\n")
    rows = []
    if value["status"] == "available":
        forecast = value["forecast"]
        rows = [
            ("Reference: October 30", f"${forecast['reference_price']:.2f}"),
            ("November 30 forecast — October share basis", f"${forecast['predicted_price']:.2f}"),
            ("Predicted price return", f"{value['predicted_price_return']:.2%}"),
            (
                "Indicative November quote — after 3:1 split",
                f"${value['indicative_post_split_price']:.2f}",
            ),
            ("Training label cutoff", forecast["training_information_cutoff"]),
        ]
    html = (
        '<!doctype html><meta charset="utf-8"><title>SOXX November forecast</title>'
        "<style>body{background:#0c192b;color:#e1e9f5;font:18px system-ui;max-width:960px;"
        "margin:3rem auto;padding:1rem}td{padding:1rem;border-bottom:1px solid #345}"
        "a{color:#62d8ff}pre{white-space:pre-wrap}</style><h1>SOXX · November 2026</h1>"
        "<p>209 price sessions: January 2–October 30. Target: November 30.</p><table>"
        + "".join(f"<tr><td>{escape(k)}</td><td>{escape(v)}</td></tr>" for k, v in rows)
        + "</table><p>"
        + escape(value["notice"])
        + "</p>"
        + f'<p><a href="{SPLIT_SOURCE}">SEC split announcement</a> · '
        '<a href="report.html">Full research, baseline comparisons and charts</a></p>'
        "<details><summary>Interpretation record</summary><pre>" + escape(text) + "</pre></details>"
    )
    (output / "november_forecast.html").write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llm", choices=("codex", "api"), default="codex")
    parser.add_argument("--news-archive", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory; completed studies are preserved")
    inputs = prepare_inputs(args.inputs)
    command = study_command(
        inputs, args.output.resolve(), llm=args.llm, news_archive=args.news_archive
    )
    print(
        json.dumps(
            {
                "command": command,
                "input_dates": ["2026-01-02", "2026-10-30"],
                "target_session": "2026-11-30",
            }
        ),
        flush=True,
    )
    if not args.prepare_only:
        code = subprocess.run(command, check=False).returncode
        if code == 0:
            write_forecast_summary(args.output)
        return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
