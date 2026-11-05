"""Display frozen rolling forecasts separately from observed evaluation outcomes."""

import json
from html import escape

from quant_rsi.research.discovery_report_comparisons import _finite, _line_chart, _number, _rows
from quant_rsi.research.rolling_windows import period_id

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
    folds = result.get("folds") or []
    session_mode = result.get("kind") == "SessionRollingStudy" or any(
        fold.get("period_kind") == "trading_sessions" for fold in folds
    )
    horizon = result.get("configuration", {}).get("horizon_sessions")
    if horizon is None and folds:
        horizon = folds[0].get("horizon_sessions")
    cadence = f"{horizon}-session" if horizon is not None else "Trading-session"
    heading = f"{cadence} rolling research" if session_mode else "Monthly rolling research"
    parts = [
        f"<h1>{_text(result['symbol'])} · {_text(heading)}</h1>",
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
        if session_mode:
            headers = [{"Month": "Period", "Months": "Periods"}.get(h, h) for h in headers]
        table_class = " class='period-table'" if session_mode and headers[0] == "Period" else ""
        parts.append(
            f"<section><h2>{escape(title)}</h2><div class='scroll'><table{table_class}><thead><tr>"
            + "".join(f"<th>{escape(h)}</th>" for h in headers)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{_text(v)}</td>" for v in row) + "</tr>" for row in rows
            )
            + "</tbody></table></div></section>"
        )

    diagnosed = [
        (period_id(fold), arm, diagnostic)
        for fold in folds
        for arm, diagnostic in fold.get("forecast_diagnostics", {}).items()
    ]
    if diagnosed:
        parts.append(
            "<details><summary>Forecast input ranges and model decomposition</summary>"
            "<p>Ranges use admitted training rows only. Outside-range inputs indicate "
            "extrapolation, not a wrong forecast. Constant columns have no defined z-score. "
            "Model terms below are log-return percentage points (100 × log return), "
            "not simple-return percentage points or causal factor importance. "
            "No additional models are fitted for these diagnostics.</p>"
        )
        table(
            "Saved model accounting",
            [
                "Month",
                "Model",
                "Training rows",
                "Columns",
                "Outside range",
                "Intercept",
                "Baseline terms",
                "Candidate terms",
                "Total log return",
                "Accounting status",
            ],
            [
                [
                    month,
                    ARMS.get(arm, arm),
                    diagnostic["training_samples"],
                    diagnostic["model_columns"],
                    diagnostic["outside_training_range_count"],
                    *[
                        _number(100 * decomposition[key]) if key in decomposition else None
                        for key in (
                            "intercept",
                            "baseline_group",
                            "candidate_group",
                            "predicted_log_return",
                        )
                    ],
                    "Available" if decomposition["available"] else decomposition.get("reason"),
                ]
                for month, arm, diagnostic in diagnosed
                for decomposition in [diagnostic["decomposition"]]
            ],
        )
        table(
            "Prediction inputs versus training range",
            [
                "Month",
                "Model",
                "Factor",
                "Role",
                "Training min",
                "Training max",
                "Distinct values",
                "Prediction input",
                "Training z-score",
                "Outside range",
            ],
            [
                [
                    month,
                    ARMS.get(arm, arm),
                    column["factor_id"],
                    column["role"],
                    _number(column["training_min"]),
                    _number(column["training_max"]),
                    column["distinct_values"],
                    _number(column["prediction_value"]),
                    _number(column["prediction_z"]),
                    column["outside_training_range"],
                ]
                for month, arm, diagnostic in diagnosed
                for column in diagnostic["columns"]
            ],
        )
        table(
            "Adaptive minus fixed accounting",
            [
                "Month",
                "Intercept change",
                "Baseline coefficient refit",
                "Candidate terms",
                "Total log-return change",
                "Accounting status",
            ],
            [
                [
                    period_id(fold),
                    *[
                        _number(100 * pair[key]) if key in pair else None
                        for key in (
                            "intercept_change",
                            "baseline_refit_change",
                            "candidate_group",
                            "predicted_log_return_change",
                        )
                    ],
                    "Available" if pair["available"] else pair.get("reason"),
                ]
                for fold in folds
                if (pair := fold.get("paired_forecast_decomposition"))
            ],
        )
        parts.append("</details>")
    attempts = [
        (period_id(fold), attempt)
        for fold in folds
        for attempt in fold.get("candidate_attempts", [])
    ]
    if attempts:
        parts.append(
            "<p>Same-cutoff candidate review uses training-only structural diagnostics, "
            "not validation scores or factor importance. The proposal preview is not trained. "
            "Only the final accepted pool reaches the adaptive model; rejection falls back "
            "to the fixed forecast. Correlated or sparse inputs are not automatically useless. "
            "Overlapping training labels do not provide independent observations.</p>"
        )
        table(
            "Candidate proposal and one revision",
            [
                "Month",
                "Stage",
                "Action",
                "Status",
                "Accepted",
                "Training rows",
                "Columns",
                "Rank",
                "Error",
            ],
            [
                [
                    month,
                    attempt["stage"],
                    attempt.get("decision", {}).get("action"),
                    attempt["status"],
                    attempt.get("candidate_pool", {}).get("accepted"),
                    attempt.get("diagnostics", {}).get("training_samples"),
                    attempt.get("diagnostics", {}).get("model_columns"),
                    attempt.get("diagnostics", {}).get("standardized_rank"),
                    attempt.get("error"),
                ]
                for month, attempt in attempts
            ],
        )
        table(
            "Candidate training coverage and correlation",
            [
                "Month",
                "Stage",
                "Factor",
                "Distinct values",
                "Nonzero rows",
                "Variance",
                "Max |r| with baseline",
                "Max |r| with another candidate",
            ],
            [
                [
                    month,
                    attempt["stage"],
                    column["factor_id"],
                    column["distinct_values"],
                    column["nonzero_rows"],
                    _number(column["variance"]),
                    _number(column["max_abs_baseline_correlation"]),
                    _number(column["max_abs_candidate_correlation"]),
                ]
                for month, attempt in attempts
                for column in attempt.get("diagnostics", {}).get("columns", [])
                if column["role"] == "candidate"
            ],
        )
        table(
            "Screening rejections before final training",
            ["Month", "Stage", "Kind", "Name", "Reason"],
            [
                [month, attempt["stage"], row["kind"], row["name"], row["reason"]]
                for month, attempt in attempts
                for row in attempt.get("candidate_rejections", [])
            ],
        )
    if any(fold.get("ohlcv_inputs") for fold in folds):
        parts.append(
            "<p>Optional high/low/volume leaves come from the retained target quote source. "
            "They do not alter the fixed baseline. ETF volume measures trading activity, "
            "not semiconductor sales or demand. Provider quote/volume units are not certified "
            "historical share units; missing inputs are not filled or removed from the sample. "
            "These are reconstructed historical inputs, not contemporaneously captured data.</p>"
        )
        table(
            "Retained OHLCV inputs",
            [
                "Month",
                "Input",
                "Available training rows",
                "Training rows",
                "Distinct values",
                "Forecast available",
            ],
            [
                [
                    period_id(fold),
                    item["factor_id"],
                    item["training_available"],
                    item["training_samples"],
                    item["distinct_training_values"],
                    item["prediction_available"],
                ]
                for fold in folds
                for item in fold.get("ohlcv_inputs", {}).get("features", [])
            ],
        )

    if any(fold.get("acquired_inputs") for fold in folds):
        acquired = next(fold["acquired_inputs"] for fold in folds if fold.get("acquired_inputs"))
        parts.append(f"<p>{_text(acquired['notice'])}</p>")
        table(
            "Acquired business measurements",
            [
                "Month",
                "Input",
                "Issuer",
                "Measurement",
                "Unit",
                "Training coverage",
                "Distinct training values",
                "Source observations in training",
                "Forecast available",
                "Used by fitted model",
            ],
            [
                [
                    period_id(fold),
                    item["input_id"],
                    item.get("symbol"),
                    item["measurement"],
                    item["unit"],
                    f"{item['training_available']} / {item['training_samples']}",
                    item["distinct_training_values"],
                    item["training_disclosures"],
                    item["prediction_available"],
                    item["input_id"] in fold.get("fitted_measurement_inputs", []),
                ]
                for fold in folds
                for item in fold.get("acquired_inputs", {}).get("coverage", [])
            ],
        )
        table(
            "Measurement source excerpts",
            [
                "Month",
                "Source ID",
                "Issuer",
                "URL",
                "Available",
                "Literal excerpt",
                "Projection hash",
            ],
            [
                [
                    period_id(fold),
                    item["id"],
                    item["symbol"],
                    item["source_url"],
                    item["published_at"],
                    item["excerpt"],
                    item["projection_sha256"],
                ]
                for fold in folds
                for item in fold.get("acquired_inputs", {}).get("evidence", [])
            ],
        )
        table(
            "Measurement acquisition and gaps",
            ["Month", "Requested measurements", "Unavailable / rejected requests"],
            [
                [
                    period_id(fold),
                    [
                        {
                            "input": row["request"]["input_id"],
                            "status": row["status"],
                            "source_observations": row["distinct_observations"],
                            "source_counts": row.get("counts"),
                        }
                        for row in fold.get("data_acquisitions", [])
                    ],
                    fold.get("data_gaps", []),
                ]
                for fold in folds
            ],
        )
        if result.get("measurement_usage"):
            parts.append(
                "<p>Measurement source operations (cumulative, including failed attempts): "
                f"{_text(result['measurement_usage']['counts'])}. Extraction calls are additional "
                + (
                    "to per-period decision calls; "
                    if session_mode
                    else "to monthly decision calls; "
                )
                + "replayed checkpoints do not repeat successful work.</p>"
            )

    if any(fold.get("risk_inputs") for fold in folds):
        risk = next(fold["risk_inputs"] for fold in folds if fold.get("risk_inputs"))
        parts.append(f"<p>{_text(risk['notice'])}</p>")
        table(
            "Measured market-risk inputs",
            ["Month", "Factor", "Available training rows", "Prediction available"],
            [
                [
                    period_id(fold),
                    coverage["factor_id"],
                    f"{coverage['training_available']} / {coverage['training_samples']}",
                    coverage["prediction_available"],
                ]
                for fold in folds
                for coverage in fold.get("risk_inputs", {}).get("coverage", [])
            ],
        )
        table(
            "Risk source provenance",
            ["File", "Source", "Retrieved", "SHA-256"],
            [
                [source["file"], source["requested_url"], source["received_at"], source["sha256"]]
                for source in risk["sources"]
            ],
        )
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
                period_id(f),
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
    window_kind = (
        "trading-session"
        if result.get("configuration", {}).get("train_sessions") is not None
        else "calendar"
    )
    parts.append(
        f"<p>Each model is retrained on its fixed {window_kind} window using only labels "
        "mature at that forecast origin. Feature warm-up and unmatured labels reduce "
        "the actual training sample. Returns below are close-to-close price returns.</p>"
    )
    for outcome, title in (
        ("observed", "Observed period forecasts" if session_mode else "Observed monthly forecasts"),
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
                        period_id(fold),
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
        "<p>Metrics use observed periods only; pending forecasts have no accuracy score. "
        "A few periods are limited evidence, not proof of predictive advantage.</p>"
        if session_mode
        else "<p>Metrics use observed months only; pending forecasts have no accuracy score. "
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
    parts.append(
        _line_chart(
            (cadence if session_mode else "Monthly")
            + " price return (%) — pending outcomes remain gaps",
            series,
        )
    )
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
        "<p>One continuous account per model across observed periods, with explicit costs "
        "and next-open fills. Sharpe uses consecutive daily net NAV returns, never an "
        "average of per-period Sharpes. Pending periods are excluded from account replay.</p>"
        if session_mode
        else "<p>One continuous account per model across observed months, with explicit costs "
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
    if any(f.get("candidate_pool") for f in folds):
        table(
            "Candidate pool screening",
            [
                "Month",
                "Limit",
                "Proposed",
                "Accepted",
                "Actual model columns",
                "Rejected / unavailable",
            ],
            [
                [
                    period_id(f),
                    f.get("candidate_pool", {}).get("limit"),
                    f.get("candidate_pool", {}).get("proposed"),
                    f.get("candidate_pool", {}).get("accepted"),
                    len(f.get("fits", {}).get("adaptive", {}).get("factor_ids", [])),
                    f.get("candidate_rejections", []),
                ]
                for f in folds
            ],
        )
        table(
            "Actual model feature usage",
            ["Month", "Arm", "Feature", "Fit-specific usage"],
            [
                [period_id(f), arm, identity, usage]
                for f in folds
                for arm, fit in f.get("fits", {}).items()
                for identity, usage in fit.get("feature_usage", {}).items()
            ],
        )
        parts.append(
            "<p>Feature usage describes this fitted model; it does not establish a reliable factor "
            "or causal effect. Exact duplicate columns are screened on the admitted sample; "
            "correlated but distinct columns are retained.</p>"
        )
    table(
        "Research decisions and evidence gaps",
        ["Month", "Candidate status", "Historical news", "Decision", "Fits / actual factors"],
        [
            [
                period_id(f),
                f.get("candidate_status"),
                f.get("news_coverage"),
                f.get("decision"),
                f.get("fits"),
            ]
            for f in folds
        ],
    )
    if any(f.get("news_evidence", {}).get("research_required") for f in folds):
        fitted = sum(bool(f.get("news_training")) for f in folds)
        parts.append(
            "<section class='notice'><h2>News research execution</h2>"
            f"<p>{fitted} of {len(folds)} "
            + ("per-period" if session_mode else "monthly")
            + " models used executable news inputs. "
            "Headline counts and lexical balances measure the retrieved corpus; they are "
            "not observed revenue, verified sentiment, or a complete record of world events. "
            "Zero means no matching retrieved headline. Search coverage is partial. "
            "Historical reconstruction uses publication timestamps of headlines retrieved "
            "now; it does not prove those versions were available historically. "
            "Recipe selection itself uses the fold's research corpus, so this remains "
            "reconstructed research rather than contemporaneous blind validation.</p></section>"
        )
        input_rows = []
        for fold in folds:
            recipes = {r["input_id"]: r for r in fold.get("candidate_news_inputs", [])}
            for identity, measurement in fold.get("news_training", {}).items():
                recipe = recipes[identity]
                values = measurement["training_values"]
                input_rows.append(
                    [
                        period_id(fold),
                        identity,
                        recipe["hypothesis"],
                        recipe["operation"],
                        recipe["terms"],
                        recipe["negative_terms"],
                        recipe["window_days"],
                        measurement["distinct_training_values"],
                        f"{min(values)} → {max(values)}",
                        measurement["prediction_value"],
                        recipe["evidence_ids"],
                    ]
                )
        table(
            "News inputs actually used in training",
            [
                "Month",
                "Input",
                "Hypothesis",
                "Measure",
                "Terms",
                "Negative terms",
                "Window days",
                "Distinct training values",
                "Training range",
                "Forecast value",
                "Evidence IDs",
            ],
            input_rows,
        )
        table(
            "Historical news coverage",
            ["Month", "Mode", "Coverage", "Eligible headlines", "Notice"],
            [
                [
                    period_id(f),
                    f.get("news_evidence", {}).get("information_mode"),
                    f.get("news_coverage"),
                    len(f.get("news_evidence", {}).get("items", [])),
                    f.get("news_evidence", {}).get("notice"),
                ]
                for f in folds
            ],
        )
        table(
            "News evidence sample (full corpus in result.json)",
            ["Month", "ID", "Headline", "Published", "Actually retrieved", "Source URL"],
            [
                [
                    period_id(f),
                    a["id"],
                    a.get("title"),
                    a.get("published_at"),
                    a.get("received_at"),
                    a.get("url"),
                ]
                for f in folds
                for a in f.get("news_evidence", {}).get("items", [])[-10:]
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
<title>"""
        + escape(heading)
        + """</title>
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
"""
        + (".period-table td:first-child{white-space:nowrap}\n" if session_mode else "")
        + "</style><main>"
        + "".join(parts)
        + "</main></html>"
    )
