"""Render retained discovery comparisons without recomputing experiments."""

import math
from html import escape


def _finite(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _number(value, *, percent=False):
    number = _finite(value)
    return None if number is None else f"{number * (100 if percent else 1):.4f}"


def _rows(rows, date_key, value_key):
    return [(str(row[date_key]), _finite(row.get(value_key))) for row in rows if row.get(date_key)]


def _polyline(points, color):
    if len(points) < 2:
        return ""
    return f"<polyline points='{' '.join(points)}' fill='none' stroke='{color}' stroke-width='2'/>"


def _prediction_row(label, metrics, retained=None):
    return [
        label,
        retained,
        metrics.get("sample_count"),
        _number(metrics.get("mae_price")),
        _number(metrics.get("rmse_price")),
        _number(metrics.get("mape_percent")),
        _number(metrics.get("direction_hit_rate"), percent=True),
    ]


def _line_chart(title, series):
    dates = sorted({date for _, rows in series for date, _ in rows})
    values = [value for _, rows in series for _, value in rows if value is not None]
    if not dates or not values:
        return f"<p>{escape(title)}: unavailable — no recorded observations to plot.</p>"
    low, high = min(values), max(values)
    padding = (high - low) * 0.08 or max(abs(low) * 0.02, 1)
    low, high = low - padding, high + padding
    palette = ("#8ae8d2", "#76b6ff", "#ffcf70", "#dca5ff", "#ff99a8", "#a5cf7e")
    html = [
        f"<figure class='comparison-chart'><figcaption>{escape(title)}</figcaption>",
        f"<svg viewBox='0 0 1000 350' role='img' aria-label='{escape(title, quote=True)}'>",
    ]
    for step in range(5):
        value = low + (high - low) * step / 4
        y = 285 - 240 * step / 4
        html.append(
            f"<line x1='90' x2='960' y1='{y}' y2='{y}' stroke='#30425b'/>"
            f"<text x='8' y='{y + 4}'>{value:.2f}</text>"
        )
    for index, (label, rows) in enumerate(series):
        color = palette[index % len(palette)]
        by_date = dict(rows)
        html.append(f"<g data-series='{escape(label, quote=True)}'>")
        segment = []

        for position, date in enumerate(dates):
            value = by_date.get(date)
            if value is None:
                html.append(_polyline(segment, color))
                segment.clear()
                continue
            x = 90 + position * 870 / max(1, len(dates) - 1)
            y = 285 - (value - low) * 240 / (high - low)
            segment.append(f"{x:.2f},{y:.2f}")
            html.append(
                f"<circle data-session='{escape(date, quote=True)}' cx='{x:.2f}' cy='{y:.2f}' "
                f"r='3' fill='{color}'><title>{escape(label)} · {escape(date)}: "
                f"{value:.4f}</title></circle>"
            )
        html.append(_polyline(segment, color))
        html.append("</g>")
    for x, date, anchor in ((90, dates[0], "start"), (960, dates[-1], "end")):
        html.append(f"<text x='{x}' y='320' text-anchor='{anchor}'>{escape(date)}</text>")
    html.append("</svg><ul class='comparison-legend'>")
    for index, (label, rows) in enumerate(series):
        unavailable = " (unavailable)" if not any(v is not None for _, v in rows) else ""
        html.append(
            f"<li style='color:{palette[index % len(palette)]}'>{escape(label + unavailable)}</li>"
        )
    html.append("</ul></figure>")
    return "".join(html)


def _date_range(rows, key):
    dates = sorted(str(row[key]) for row in rows if row.get(key))
    return f"{dates[0]} → {dates[-1]}" if dates else None


def append_comparisons(discovery, table, parts):
    """Keep every matched population separate, including baseline-only populations."""
    groups = discovery.get("groups", {})
    if not groups:
        table("Experiment comparisons", ["Status"], [["No completed baseline fit is available."]])
    for key, group in groups.items():
        suffix = " · " + key[:12]
        parts.append(
            f"<section class='comparison-group' data-comparison-group='{escape(key, quote=True)}'>"
            f"<h2>Comparable experiment {escape(key[:12])}</h2>"
        )
        records = [group["baseline"], *group.get("candidates", [])]
        base = records[0].get("result", {}).get("prediction") or {}
        development = base.get("development") or {}
        observed = development.get("rows") or []
        table(
            "Comparison context" + suffix,
            ["Group", "Samples", "Horizon (sessions)", "Currency", "Origins", "Targets"],
            [
                [
                    key,
                    development.get("metrics", {}).get("sample_count"),
                    base.get("horizon_sessions"),
                    development.get("currency"),
                    _date_range(observed, "origin_session"),
                    _date_range(observed, "target_session"),
                ]
            ],
        )
        if not group.get("candidates"):
            table(
                "Candidate status" + suffix,
                ["Trained candidates", "Status"],
                [
                    [
                        0,
                        "No candidate completed training. The fixed-factor baseline and its "
                        "simple prediction/account benchmarks remain available below.",
                    ]
                ],
            )
        rows, predictions, accounts = [], [], []
        benchmarks = []

        predictions.append(("Observed price", _rows(observed, "target_session", "observed_price")))
        for index, record in enumerate(records):
            measured = record.get("result") or {}
            prediction = measured.get("prediction") or {}
            dev = prediction.get("development") or {}
            role = "Fixed-factor baseline" if index == 0 else "Candidate"
            label = f"{role} · Fit {measured.get('round', 'unavailable')}"
            rows.append(
                _prediction_row(
                    label,
                    dev.get("metrics") or {},
                    record.get("ref") == group.get("incumbent", {}).get("ref"),
                )
            )
            predictions.append(
                (label, _rows(dev.get("rows") or [], "target_session", "predicted_price"))
            )
            strategy = prediction.get("strategy") or {}
            accounts.append((label + " strategy", strategy.get("strategy") or {}))
            benchmark = strategy.get("buy_and_hold")
            if benchmark and not any(benchmark == other for _, other in benchmarks):
                benchmarks.append(
                    (f"Buy-and-hold · Fit {measured.get('round', 'unavailable')}", benchmark)
                )
        for name, label in (
            ("unchanged_price", "Unchanged price"),
            ("training_label_mean", "Training-label mean"),
        ):
            benchmark = (development.get("baselines") or {}).get(name) or {}
            rows.append(_prediction_row(label, benchmark.get("metrics") or {}, "Benchmark"))
            predictions.append(
                (label, _rows(benchmark.get("rows") or [], "target_session", "predicted_price"))
            )
        table(
            "Prediction comparison" + suffix,
            ["Arm", "Retained", "N", "Price MAE", "Price RMSE", "MAPE (%)", "Direction hits (%)"],
            rows,
        )
        parts.append(
            "<p>Fixed-factor baseline is a trained model. Unchanged price and training-label mean "
            "are simple prediction benchmarks. Curves use recorded target dates and observed "
            "outcomes; missing values remain gaps. These are development results, "
            "not independent validation.</p>"
        )
        parts.append(_line_chart("Development predictions" + suffix, predictions))
        if len(benchmarks) == 1:
            benchmarks[0] = ("Buy-and-hold", benchmarks[0][1])
        accounts.extend(benchmarks)
        if not benchmarks:
            accounts.append(("Buy-and-hold", {}))
        rows = []
        for label, account in accounts:
            metrics = account.get("metrics") or {}
            fees = metrics.get("fees") or {}
            risk = metrics.get("risk_statistics") or {}
            rows.append(
                [
                    label,
                    _number(metrics.get("net_return"), percent=True),
                    _number(metrics.get("sharpe_ratio")),
                    _number(metrics.get("annualized_volatility"), percent=True),
                    _number(metrics.get("max_drawdown"), percent=True),
                    _number(fees.get("commission")),
                    _number(fees.get("slippage")),
                    _number(fees.get("tax")),
                    risk.get("status"),
                ]
            )
        table(
            "Strategy comparison" + suffix,
            [
                "Account",
                "Net return (%)",
                "Sharpe (ann.)",
                "Volatility (ann.)",
                "Max drawdown (%)",
                "Commission",
                "Slippage",
                "Tax",
                "Risk status",
            ],
            rows,
        )
        parts.append(
            "<p>Hypothetical costed accounts, including the recorded buy-and-hold benchmark. "
            "NAV uses a shared session axis within this group only. Sharpe is an annualized "
            "development statistic on a short sample; unavailable statistics are not zero.</p>"
        )
        parts.append(
            _line_chart(
                "Hypothetical account NAV" + suffix,
                [
                    (label, _rows(account.get("valuations") or [], "session_id", "nav"))
                    for label, account in accounts
                ],
            )
        )
        parts.append("</section>")
