# Six-month training, next-month forecasts

`research-rolling` is a separate calendar-month entry point using the existing
retained daily reader, numerical factor operators, model parameters, provider
transport/cache and account engine. It does not change the existing expanding,
post-selection `research-study --walk-forward-folds` diagnostic.

## SOXX example

Reuse the immutable pre-split collections prepared by the
[SOXX October-cutoff example](soxx_november_2026.md):

```bash
quant-rsi research-rolling SOXX \
  --prices-dir /tmp/soxx-november-2026-inputs/SOXX \
  --related-prices-dir /tmp/soxx-november-2026-inputs \
  --first-month 2026-07 --last-month 2026-11 --train-months 6 \
  --llm codex \
  --price-note 'SOXX: October share basis. The announced 3:1 split starts trading November 5. Divide November forecast prices by 3 for indicative post-split quotes; returns are unchanged by this arithmetic.' \
  --output /tmp/soxx-monthly-rolling
```

Omit `--llm codex` to run only the fixed baseline and simple benchmarks.
`--llm api --llm-model MODEL` uses the existing API-key configuration. There is at
most one LLM decision and two model fits per month: one fixed baseline, and one
adaptive candidate arm. `--model ridge` provides a simple alternative with
training-only standardization. The default is actual LightGBM with the existing
fixed seed and parameters. No simulated LLM or market data is substituted.

The plan is frozen before any calls:

| Price history | Forecast | Trading-session horizon |
|---|---|---:|
| January–June | July 31 | 22 |
| February–July | August 31 | 21 |
| March–August | September 30 | 21 |
| April–September | October 30 | 22 |
| May–October | November 30 | 20 |

All dates above are for 2026. Month ends and horizons come from the hash-verified
published exchange calendar. The command requires complete history for every
requested training window and enough matured daily samples; it does not fetch
new prices, shorten windows, interpolate missing sessions, or assume 20 sessions
per month.

## What each fold learns and observes

Training uses daily origins, not six monthly observations. Its target is the
H-session log return, where H is the length of the upcoming forecast month.
This is horizon-matched daily training, not a separate next-calendar-month-end
label for each historical day. Only labels ending by the current month-end are
admitted. The last H historical origins remain unlabeled; the latest historical
row supplies prediction features. Feature warm-up uses the first 30 rows inside
the six-month window; no earlier prices enter numerical features or training.

Each month performs the following sequence:

1. Slice target and aligned peer prices to that fold's history.
2. Recompute features and fit the fixed baseline on mature labels.
3. Optionally ask the existing LLM transport to retain, replace or withdraw up to
   two candidate expressions, using only supplied eligible news and prior
   matured monthly feedback. Refit the adaptive arm on exactly the same rows.
4. Save `YYYY-MM/prediction.json` without the target outcome.
5. Settle from retained target prices, or mark the month `not_observed`.
6. Make this settled feedback available only to subsequent monthly decisions.

The fixed baseline includes 10-session momentum, log-return volatility and the
10-session relative return against each configured, supplied comparator. The
adaptive arm appends expressions through the existing `DiscoveryFactor` grammar;
it can consult `experimental/candidate_factors` and the executable Qlib adapters.
It cannot execute arbitrary code, invent missing financial inputs, drop training
rows, or select a recipe using the same month's future outcome. Non-finite,
duplicate or unsupported candidates retain the fixed forecast with a visible
rejection reason. A changed recipe is not automatically a better recipe.

`--news-archive PATH` admits only genuine retained items whose receipt and
publication times precede the fold cutoff. No current search or shared study
history is read. Missing news is recorded explicitly; price-only hypotheses are
allowed and identified as such. This is not an expansion of historical business
data coverage. The LLM's pretrained knowledge is not time-filtered.

## Reports and interpretation

`report.html` contains actual training dates, monthly fixed/adaptive/simple
baseline returns, errors on **observed months only**, pending forecasts,
continuous account NAV, costs, Sharpe, drawdown and buy-and-hold comparisons.
`result.json` and each fold retain decisions, parent recipes, feedback references,
model state and a hash of the forecast frozen before settlement.

The fixed and adaptive portfolios each start once at the first origin. Signals
trade at the next observed open and positions carry across months. There is no
monthly account reset or average of fold Sharpes. Pending months cause no fills
or P&L. Forecast close-to-close returns differ from next-open executable returns.
The account remains the existing hypothetical long-only account; it is not an
actual trading track record or a general corporate-action/total-return engine.

November is pending in the October-cutoff example and contributes no accuracy,
realized return or Sharpe. SOXX's [announced 3:1 split](https://www.sec.gov/Archives/edgar/data/1100663/000119312526361162/d133118d497.htm)
requires November quotes to be converted to a consistent share basis. The note
above is presentation context, not historical news injected into model training.

Source collections were fetched later; predictions are reconstructed, not proof
of what the system actually published at those historical month ends. Results
already inspected during development are not a new blind holdout. Four realized
months are too few to establish persistent predictive advantage. Freeze the
protocol before extending observation or comparing research methods.

## Failure and continuation

Use a fresh output directory initially. On a provider failure, successful paid
calls and fits remain in the existing input-bound checkpoint cache and completed
folds remain visible. Repeat the **same command and output path** with `--resume`
to reuse them. Changed code, dependencies, input identities or configuration are
rejected; completed runs cannot be resumed or overwritten. Keep failed outputs
as evidence rather than silently restarting paid research.
