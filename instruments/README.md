# Instrument Profiles and Local Outputs

All instruments share the engine in `src/quant_rsi/`. Keep instrument-specific configuration here; do not copy source modules into instrument directories.

## Layered Factor Specifications

Shared stock definitions live in [`shared/equities.json`](shared/equities.json).
Each stock selects them and declares its own sources in `<MARKET>/<SYMBOL>/factors.json`:
see [CALM](US/CALM/factors.json), [KO](US/KO/factors.json) and [XOM](US/XOM/factors.json).
CALM's egg-price and disease-event definitions stay inside CALM; they are not
stock-wide defaults.

Run `quant-rsi factors validate US CALM` to resolve and inspect the configuration.
This is read-only specification validation, not data collection or forecasting.
The [factor-layer guide](../docs/factor_layers.md) explains numeric/text types,
ownership, examples and the remaining execution boundary.

## Tracked Profiles

### Structured disclosure issuers

[`US/TSM/disclosures.json`](US/TSM/disclosures.json) binds the US-listed TSM ADS to
Taiwan issuer 2330 for official MOPS announcement queries. The profile identifies
the source issuer and the official page supporting that identity; it does not
choose investigation months, announcements, measurements or factor formulas.
Active discovery loads these profiles only with `--news-research`. A missing
mapping produces a data gap rather than guessing a company ID. See
[structured disclosures](../docs/news_driven_discovery.md#official-structured-disclosures).

### Rolling research windows

[`US/SOXX/rolling.json`](US/SOXX/rolling.json) configures the SOXX rolling study:

```json
{
  "schema_version": 1,
  "kind": "RollingWindowProfile",
  "instrument": "SOXX",
  "training_window_sessions": 126,
  "horizon_sessions": 3
}
```

Load it explicitly with `research-rolling SOXX --profile instruments/US/SOXX/rolling.json`
alongside the required price, target-month and output arguments. Both lengths count
exchange trading sessions. The first is retained price history, including 30 warm-up
rows and the final horizon rows with immature labels: 126/3 yields 93 fitted rows.
The second is the forecast/test interval and decision stride, not a separate
independent validation set. The target-month range determines the total test span.

CLI `--train-sessions` and `--horizon-sessions` override the corresponding profile
values. `--train-months` instead selects calendar-month training; it cannot be
combined with `--train-sessions`. The profile requires 61..445 history sessions,
1..20 horizon sessions and at least `60 + horizon_sessions` history sessions.
Copy this profile into another stock's directory and change its instrument identity;
the requested symbol must match. It does not provision market or news data.

Each run retains `window_profile.json` and its hash. `--resume` with the original
command uses that retained copy, so editing the source profile affects new runs
only; changing effective CLI settings requires a new output directory. Without
`--profile`, existing defaults remain six calendar months and calendar-month
forecasts. See the [complete rolling example](../docs/monthly_rolling_research.md#three-trading-session-news-research).

### Connected studies and forecast workflows

For `research-study`, KO and XOM have `relationships.json`, `news_feed.json` and
`factors.json` configurations. XOM uses its official issuer RSS, a dated CVX
joint-venture relationship and a local XOM-minus-CVX return composition over
shared stock definitions. See the [XOM study example](../docs/connected_research_study.md#second-stock-example-xom).
These are connected-study configurations, not registered `forecast-workflow`
campaign profiles. A new stock still needs supported public prices and reviewed
source configuration; adding a symbol alone does not establish those inputs.

KO, AAPL and XOM also have `financials.json` mappings for publication-dated business
features. XOM's issuer mapping reads revenue and cash flow and sums the two reported
inventory components. Operating margin remains unavailable. See the
[XOM financial scope](../docs/connected_research_study.md#xom-business-data) for source
selection, successor identity and missing-data limits.

Profiles follow `instruments/<MARKET>/<SYMBOL>/forecast.json`. The current US examples explicitly bind SOXL/XSD/SPY and KO/XLP/SPY as target/industry/market roles. They also declare the supported source, calendar, history, horizon, price units, and availability assumption.

To configure another supported instrument, copy a profile into its symbol directory and edit the explicit roles. Validate it with `forecast-workflow preflight` against a registered campaign. The profile must match the frozen campaign configuration; changing a JSON file does not modify an existing run. Use a new registered workflow identity when inputs change. See the [enhanced workflow guide](../docs/enhanced_forecast_workflow.md) for registration and supported contracts.

The [CALM factor tree](US/CALM/factor_tree.json) is a specification-only factor-discovery ledger, not a runnable forecast profile or development plan. See its [worked example](../docs/calm_factor_discovery_example.md) for the node meanings and unresolved evidence.

## Ignored Data and Reports

Use these destinations for new local exports and reports:

| Path | Contents |
| --- | --- |
| `data/US/<SYMBOL>/<collection-id>/` | Explicitly exported inputs and source evidence |
| `runs/US/<SYMBOL>/<workflow-id>/` | Human-readable reports and other requested output copies |
| `$QUANT_RSI_WORKSPACE/.quant_rsi/` | Shared authoritative registry, campaigns, journals, and bound workflow artifacts |

The first two paths are organization conventions, not automatic storage routing. Select destinations through existing output arguments when available. The guided workflow retains its authoritative collection and execution state in the shared workspace, even when reports are exported elsewhere. All three locations are ignored by Git.

For the campaign registered in the guide:

```bash
export QUANT_RSI_WORKSPACE="$PWD"
quant-rsi forecast-workflow preflight soxl-context-example \
  --campaign-id example-context --profile instruments/US/SOXL/forecast.json
quant-rsi forecast-workflow execute soxl-context-example \
  --campaign-id example-context --profile instruments/US/SOXL/forecast.json \
  --max-fits 10 --output runs/US/SOXL/soxl-context-example
```

Use globally unique workflow and collection IDs across instruments. A report destination does not isolate authority, reset charges, or renew a budget. Keep historical data, workspaces, output paths, and their recorded source environments intact.
