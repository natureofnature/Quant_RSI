"""Display frozen monthly forecasts separately from observed evaluation outcomes."""

import json
from html import escape

from quant_rsi.research.discovery_report_comparisons import _finite, _line_chart, _number, _rows

ARMS = {
    "fixed": "Fixed baseline",
    "adaptive": "Adaptive research",
    "unchanged_price": "Unchanged price",
    "training_label_mean": "Training-label mean",
}


def _return(row, field):
    reference, value = _finite(row.get("reference_price")), _finite(row.get(field))
    return (value / reference - 1) * 100 if reference and reference > 0 and value else None


def _text(value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return escape("—" if value is None else str(value))


def render_rolling(result):
    parts = [
        f"<h1>{_text(result['symbol'])} · Monthly rolling research</h1>",
        f"<p>{_text(result.get('notice'))}</p>",
    ]
    if result.get("status") == "failed":
        parts.append(
            "<section class='notice'><h2>Incomplete run — failed</h2>"
            f"<p>{_text(result.get('failure'))}</p></section>"
        )
    if result.get("price_basis_notice"):
        parts.append(
            "<section class='notice'><h2>Price basis</h2>"
            f"<p>{_text(result['price_basis_notice'])}</p></section>"
        )

    def table(title, headers, rows):
        parts.append(
            f"<section><h2>{escape(title)}</h2><div class='scroll'><table><thead><tr>"
            + "".join(f"<th>{escape(h)}</th>" for h in headers)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{_text(v)}</td>" for v in row) + "</tr>" for row in rows
            )
            + "</tbody></table></div></section>"
        )

    folds = result.get("folds") or []
    table(
        "Training and forecast windows",
        [
            "Month",
            "Requested training window",
            "Actual training origins",
            "Last training label",
            "Samples",
            "Forecast origin",
            "Target",
            "Trading sessions",
        ],
        [
            [
                f.get("month"),
                f"{f.get('training_window', {}).get('start')} → "
                f"{f.get('training_window', {}).get('end')}",
                f"{f['training_origins'][0]} → {f['training_origins'][-1]}"
                if f.get("training_origins")
                else None,
                (f.get("training_targets") or [None])[-1],
                len(f.get("training_origins") or []),
                f.get("origin_session"),
                f.get("target_session"),
                f.get("horizon_sessions"),
            ]
            for f in folds
        ],
    )
    parts.append(
        "<p>Each model is retrained on its fixed calendar window using only labels "
        "mature at that forecast origin. Feature warm-up and unmatured labels reduce "
        "the actual training sample. Returns below are close-to-close price returns.</p>"
    )
    for outcome, title in (
        ("observed", "Observed monthly forecasts"),
        ("not_observed", "Pending forecasts — excluded from accuracy"),
    ):
        rows = []
        for fold in folds:
            if fold.get("outcome") != outcome:
                continue
            for arm, row in (fold.get("forecasts") or {}).items():
                actual = _return(row, "observed_price") if outcome == "observed" else None
                prediction = _return(row, "predicted_price")
                rows.append(
                    [
                        fold.get("month"),
                        ARMS.get(arm, arm),
                        _number(row.get("reference_price")),
                        _number(row.get("predicted_price")),
                        _number(prediction),
                        _number(actual),
                        _number(prediction - actual)
                        if prediction is not None and actual is not None
                        else None,
                    ]
                )
        table(
            title,
            [
                "Month",
                "Model",
                "Reference price",
                "Predicted price",
                "Predicted return (%)",
                "Observed return (%)",
                "Error (pp)",
            ],
            rows,
        )
    observed = any(f.get("outcome") == "observed" for f in folds)
    table(
        "Observed forecast metrics",
        ["Model", "Months", "Price MAE", "Price RMSE", "Return MAE (pp)", "Direction hit (%)"],
        [
            [
                ARMS.get(arm, arm),
                metrics.get("sample_count"),
                _number(metrics.get("mae_price")),
                _number(metrics.get("rmse_price")),
                _number(metrics.get("mae_return_pp")),
                _number(metrics.get("direction_hit_rate"), percent=True),
            ]
            for arm, metrics in (result.get("metrics", {}) if observed else {}).items()
        ],
    )
    parts.append(
        "<p>Metrics use observed months only; pending forecasts have no accuracy score. "
        "A few months are limited evidence, not proof of predictive advantage.</p>"
    )
    arms = [arm for arm in ARMS if any(arm in (f.get("forecasts") or {}) for f in folds)]
    series = [
        (
            ARMS[arm],
            [
                (
                    f["target_session"],
                    _return((f.get("forecasts") or {}).get(arm, {}), "predicted_price"),
                )
                for f in folds
            ],
        )
        for arm in arms
    ]
    series.append(
        (
            "Observed return",
            [
                (
                    f["target_session"],
                    _return(next(iter((f.get("forecasts") or {}).values()), {}), "observed_price")
                    if f.get("outcome") == "observed"
                    else None,
                )
                for f in folds
            ],
        )
    )
    parts.append(_line_chart("Monthly price return (%) — pending outcomes remain gaps", series))
    accounts, benchmarks = [], []
    for arm, account in (result.get("accounts") or {}).items():
        if not account:
            continue
        accounts.append((ARMS.get(arm, arm), account.get("strategy") or {}))
        benchmark = account.get("buy_and_hold") or {}
        if benchmark and not any(benchmark == prior for _, prior in benchmarks):
            benchmarks.append((f"Buy-and-hold ({ARMS.get(arm, arm)})", benchmark))
    if len(benchmarks) == 1:
        benchmarks[0] = ("Buy-and-hold", benchmarks[0][1])
    accounts.extend(benchmarks)
    rows = []
    for label, account in accounts:
        m = account.get("metrics") or {}
        fees, risk = m.get("fees") or {}, m.get("risk_statistics") or {}
        rows.append(
            [
                label,
                _number(m.get("net_return"), percent=True),
                _number(m.get("sharpe_ratio")),
                _number(m.get("max_drawdown"), percent=True),
                _number(fees.get("commission")),
                _number(fees.get("slippage")),
                _number(fees.get("tax")),
                risk.get("daily_return_count"),
                risk.get("status"),
            ]
        )
    table(
        "Continuous observed strategy accounts",
        [
            "Account",
            "Net return (%)",
            "Sharpe (ann.)",
            "Max drawdown (%)",
            "Commission",
            "Slippage",
            "Tax",
            "Daily returns",
            "Risk status",
        ],
        rows,
    )
    parts.append(
        "<p>One continuous account per model across observed months, with explicit costs "
        "and next-open fills. Sharpe uses consecutive daily net NAV returns, never an "
        "average of monthly Sharpes. Pending months are excluded from account replay.</p>"
    )
    parts.append(
        _line_chart(
            "Continuous net account NAV",
            [
                (label, _rows(a.get("valuations") or [], "session_id", "nav"))
                for label, a in accounts
            ],
        )
    )
    table(
        "Research decisions and evidence gaps",
        ["Month", "Candidate status", "Historical news", "Decision", "Fits / actual factors"],
        [
            [
                f.get("month"),
                f.get("candidate_status"),
                f.get("news_coverage"),
                f.get("decision"),
                f.get("fits"),
            ]
            for f in folds
        ],
    )
    if result.get("failure") or result.get("failures") or result.get("status"):
        parts.append(
            f"<section><h2>Run status</h2><p>{_text(result.get('status'))}</p>"
            f"<pre>{_text(result.get('failure') or result.get('failures'))}</pre></section>"
        )
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Monthly rolling research</title>
<style>body{margin:0;background:#0c1322;color:#e8eef8;font:16px system-ui}
main{max-width:1250px;margin:auto;padding:28px}h1,th{color:#8ae8d2}
section{margin:22px 0;padding:18px;background:#152137;border:1px solid #30425b;border-radius:12px}
.notice{border-color:#e9b453}table{width:100%;border-collapse:collapse}.scroll{overflow:auto}
th,td{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #30425b}
td,pre{overflow-wrap:anywhere}p{line-height:1.6}pre{white-space:pre-wrap}
.comparison-chart{margin:22px 0;padding:12px;background:#0c192b;border-radius:12px}
.comparison-chart svg{width:100%;height:auto}.comparison-chart text{fill:#b7c9dd;font-size:13px}
.comparison-legend{display:flex;flex-wrap:wrap;gap:10px 24px;padding-left:20px}
@media(max-width:680px){main{padding:12px}section{padding:10px}}
</style><main>"""
        + "".join(parts)
        + "</main></html>"
    )
