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

## Reviewed split price basis

The default reader rejects unresolved non-dividend corporate actions. For an
already retained target snapshot, `--price-basis-review /path/to/review-bundle`
can explicitly qualify a split-adjusted, distribution-excluding O/C basis.
It never fetches, rescales prices, deletes an action notice, or changes live
execution behavior. Unlisted actions still fail. Related instruments retain
the default action checks.

The bundle's `review.json` follows `ReviewedResearchPriceBasis` in
[`reviewed_price_basis.py`](../src/quant_rsi/research/reviewed_price_basis.py):

- Exact hashes of the collection receipt, raw chart, data contract and events.
- The symbol, currency, `snapshot_research_only` scope, `identity` transformation,
  and a qualification cutoff no later than the first forecast origin.
- Each resolved split's original event ID, action payload hash, effective time
  and numerator/denominator, with its issuer evidence reference.
- Retained issuer, provider-semantics, independent-quote-semantics, quote-image
  and human-review artifacts, each with its hash and source URL where applicable.
- Manually reviewed same-type open/close anchors before and after each resolved
  split. Anchors must be inside the retained history and qualification cutoff;
  their explicit nearest-cent values must match normalized quotes within $0.005.

This is a human-reviewed attestation with automated binding checks. Hashes and
numerical agreement do not interpret documents or prove source independence,
every row's accuracy, historical availability, or a useful prediction. The
report explicitly qualifies the collector's older `raw_currency_per_share`
assumption; the original evidence file remains intact. Original action notices,
review identity and limitations are saved in the result configuration. Review
documents are not added to LLM research context.

The run retains the bundle and cited artifacts under `price_basis_review/`.
Resume rechecks those retained files, so it does not depend on the external
review directory remaining available. Changed evidence or collection bytes
cannot silently reuse previous fits. This O/C-only review cannot be combined
with `--ohlcv-features`: volume and high/low require their own qualification.
Freeze and review the task's source, inputs and execution protocol separately
before treating a qualified snapshot as an experiment.

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
   the configured number of candidate expressions, using only supplied eligible news and prior
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

## Executable news research

Add `--news-research` to the same rolling command:

```bash
quant-rsi research-rolling SOXX \
  --prices-dir /tmp/soxx-november-2026-inputs/SOXX \
  --related-prices-dir /tmp/soxx-november-2026-inputs \
  --first-month 2026-07 --last-month 2026-11 \
  --llm codex --news-research \
  --output /tmp/soxx-monthly-news
```

This acquires date-bounded Google News RSS searches for each calendar month,
starting at the first training window. The SOXX search universe covers
semiconductors, chipmakers, Nvidia, TSMC, chip exports, stock-market news,
the Federal Reserve, Taiwan and the Middle East. Use `--news-query` to
freeze a different sector/company/macro search universe before the run. Results
are selected search results, not comprehensive news coverage. Raw RSS, requests,
actual retrieval timestamps and hashes are retained under `news_history/`.
Transport or parsing failures stop the run and can be retried with `--resume`;
successful searches, decisions and fits are reused.

Before each monthly decision, articles after its cutoff are removed. The LLM can
propose arbitrary evidence-linked `news.*` input recipes: trailing headline counts
or a balance of positive-basket matches minus negative-basket matches, over 1–30
calendar days. Terms must literally match cited headlines. For example, a policy
hypothesis may count retrieved export-restriction headlines; this is not a measure
of actual export revenue. The LLM binds these inputs into the existing candidate
expression grammar. Fixed price factors remain in both model arms.

Each training row independently excludes later publications. Counts use distinct
article IDs; a balance headline matching both baskets contributes zero. Zero
means no match in the retrieved corpus, **not no world event**. News inputs and
candidate expressions must vary on the actual training rows. Declared news inputs
must be used in candidates; a price-only proposal is rejected in news-research
mode. Each candidate is screened independently. Missing evidence, unsupported
recipes, constant inputs and exact duplicate model columns are rejected with
reasons; valid candidates still enter one joint model. If none survive, the fixed
forecast is retained. Correlated but distinct columns are not automatically removed. A retained decision reuses its earlier
recipe and recomputes it on the next window. The report lists actual training
values/ranges, source IDs, hypotheses, coverage and the number of monthly models
that really used news inputs.

The default `--news-information-mode reconstructed` uses reported publication
times of headline versions retrieved today. It preserves today's actual receipts
and **does not claim historical capture or point-in-time article versions**.
Headlines are not replaced by today's mutable full article bodies. Search ranking
may itself reflect later information. LLM pretrained knowledge and retrospective
recipe selection remain additional limitations. This is an explicitly reconstructed
research experiment, not an independent historical validation.

`--news-information-mode retained` additionally requires publication, receipt and
version availability before each decision/row. Newly collected old headlines will
therefore be unavailable for past decisions in this mode. An existing
`--news-archive` uses the original retained-snapshot path and never has its receipt
times relaxed. News discovery may withdraw a hypothesis or fail to produce an
executable factor; the report exposes this instead of relabeling a price factor
as news research.

These inputs are lexical headline measurements, not verified semantic sentiment,
actual revenue/profit/inventory, or a universal measure of political/war risk.
Obtaining dated business measurements remains a distinct data acquisition task.
The new path establishes news → declared non-price input → actual model feature →
monthly forecast → mature feedback, with fixed and simple baselines retained.

News feature windows cannot extend before the fold's history. Unqueried/failed
calendar intervals are unavailable, not zero. Source queries must cover every
calendar day of an admitted numerical window; completed queries with no matching
captured headlines may yield zero. Intermediate expression lags cannot bypass
these unavailable values. For bounded LLM context, at most 12 recent headlines per
month plus incumbent cited evidence are shown; numerical inputs use the full
eligible corpus and the report records omitted-context counts.

## Joint multi-factor pools

Use `--max-candidates 16 --max-news-inputs 32` to let each decision propose up to
16 jointly trained candidates and 32 news input recipes. Supported limits are
1–32 candidates and 1–64 news inputs; defaults preserve two candidates and twice
the candidate limit for inputs. Fixed baseline columns are additional. These are
capacity limits, not guarantees that evidence supports that many factors.

The report records proposed/accepted counts, per-candidate rejection reasons,
actual model columns, and LightGBM split counts/gain or standardized Ridge
coefficients. These describe model use, not reliability or causal importance.
Different expressions of the same topic do not necessarily add information.

To compare pools without collecting a different news corpus, add
`--news-history /tmp/soxx-monthly-news/news_history` alongside `--news-research`.
The acquisition must match the requested symbol, search query and full date range.
Raw RSS hashes, requests, receipts, monthly coverage and parsed items are checked
without network access; verified evidence is copied to the new output directory.
The original files and receipt times remain unchanged. Tampering, missing files
or mismatched requests fail explicitly. Snapshot identity and both limits are
bound into resume configuration; changing them requires a fresh output directory.

For example, extend the command above with `--max-candidates 16
--max-news-inputs 32 --llm-timeout 600 --news-history /tmp/soxx-monthly-news/news_history`
and a fresh output directory. Larger structured proposals can take longer than
the default provider timeout. Failed calls remain recorded; adding features does
not itself establish a forecast advantage.

## Optional retained OHLCV inputs

Add `--ohlcv-features` to a new rolling run to expose the target instrument's
`price.high`, `price.low` and `price.volume` leaves. The adapter reads the already
retained Yahoo response under `--prices-dir`; it does not collect new prices,
modify old collections or expand the fixed baseline. Raw content hashes and
source identity are checked against the collection receipt, and selected sessions
must match the validated open/close history. The raw source identity participates
in resume validation. Changing the option or its inputs requires a fresh run.

These inputs allow the existing expression language to represent, for example:

- Daily range: `(high - low) / open`, also available as the pinned experimental
  `qlib.klen` adapter.
- Relative trading volume: `volume / rolling_mean(volume, 10)`.
- A retrieved-news measure interacted with either of those quantities.

Only the fold's historical prefix reaches expressions. Missing optional values
remain unavailable; zero volume is retained as zero. A missing required feature
rejects its dependent candidate without dropping rows or discarding other usable
candidates. A zero denominator remains missing. The report records per-fold
training coverage, distinct values and forecast availability; model feature usage
shows whether a candidate actually consumed those leaves. This does not activate
unimplemented Qlib correlation/logarithm adapters.

OHLCV describes market trading, not business revenue, chip demand or measured
geopolitical severity. Quote and volume units follow the retained provider
vintage and are not certified original historical share units. Existing corporate
action restrictions still apply. Actual collection timestamps remain unchanged;
these are reconstructed historical inputs, not point-in-time archives.

New inputs do not establish predictive improvement. A subsequent effectiveness
comparison must include a static factor arm using the same OHLCV information,
as well as the existing fixed, unchanged-price and training-mean baselines.
Do not select formulas or months after observing their target errors.

## Measured business inputs

Use a successful `research-study --research-mode discovery` acquisition checkpoint
as a source archive for a new monthly run. Add these options to the rolling command:

```bash
--acquired-history /path/to/study/checkpoints/discovery_acquire_2_0.json \
--max-extractions 6
```

Repeat `--acquired-history` for additional acquisition files. The loader checks
retained output identities and reruns literal source/date/value validation from
the saved articles and extraction records. It recomputes each month's values;
saved feature columns are not authoritative. Input checkpoint hashes without
their original request payload remain provenance, not independently verified
request identities. Only acquisition files are imported, not another study's
forecasts, scores or later decisions. Source files are copied into the new run;
resume uses those copies and rejects changed configuration or source content.

The same monthly decision can supply up to two `data_requests` on a `replace`
action. Only requests referenced by valid proposed candidates are considered:

> News or disclosed evidence → candidate and measurement request → retained-source
> extraction → time alignment and screening → joint model fit → mature feedback.

Existing measurements can be reused without another extraction. A new measurement
must be supported by the archived literal excerpts, including its issuer, unit
and reporting period. For example, a revenue excerpt that also reports inventory
can support a previously unregistered inventory measurement. A measurement absent
from those excerpts remains `needs_data`. The resolver performs no live searches
or page reads; it does not acquire arbitrary missing business data. An issuer's
measurements are not an ETF aggregate, sector total or consensus surprise.

Measurement evidence remains separate from the headline-count corpus. A candidate
must cite an eligible source for each measured input and genuinely depend
numerically on a verified measurement or news input. A name beginning with
`acquired.` is not sufficient. Missing, stale, constant or conflicting inputs
reject their dependent candidates; the frozen rows, labels and fixed baseline
remain unchanged. Unrelated valid candidates can still run. Reports distinguish
training rows, distinct values, source observations and actual model use.

The run-wide extraction limit is 0–6 (default 6), in addition to monthly decision
calls. Failed physical extraction attempts consume this budget. Successful cached
operations are reused during explicit resume; provider failures remain failures.

Date-only disclosures follow the existing next-day 12:00 UTC rule. Retained mode
also respects actual receipt, version and extraction times. Reconstructed mode
does not prove that the current document version existed on the historical date.
Only validated literal excerpt projections reach new extraction prompts, with
original source identity, quote locations and a separate projection hash. This
reduces unrelated prompt material; it is not proof of historical authenticity.
The original article reader does not save raw HTTP bytes, so its content hash
is a retained attestation, not a raw response hash reverified by this adapter.
Literal checks also do not establish correct financial interpretation. Repeating
three quarterly values over many daily rows still provides only three disclosed
states, not many independent business observations or evidence of predictive gain.

## Interpreting mature feedback

### One same-cutoff candidate revision

Add `--candidate-revision` with `--llm codex` or `--llm api` to run:

> Proposal → executable-input screening → training-only structural diagnostics →
> one LLM revision → final screening → joint model fit → frozen forecast.

The preview does not receive a model fit or a target-month outcome. Diagnostics
describe distinct/nonzero training values, variance, matrix rank and correlations
with baseline/other candidate columns. They do not score predictive usefulness,
estimate causal contributions or automatically remove correlated factors. Daily
training labels overlap. The six-month monthly-horizon setting leaves too few
non-overlapping validation intervals to reliably rank a large pool, so this
option does not add per-factor fitting or claim independent validation.

The second decision may retain the whole accepted preview, select an exact subset,
replace it (with valid parent recipe references), or withdraw. For subset selection,
emit `action: "select"`, nonempty unique `selected_recipe_refs` from the current
incumbent, and empty `candidates`/`news_inputs`. Selection reuses the original records,
their original ancestry and only their required news input definitions, in incumbent
order. It still runs the normal coverage and evidence checks. Unknown/duplicate
references, empty selections and mixed selection/replacement payloads are rejected.

Use `replace` for new or modified expressions. Keeping an expression unchanged
does not create a descendant: never add its own recipe reference to its parent
list. Renaming it or changing a separately defined news input does not change the
expression identity. The same selection action is also available between monthly
forecasts, without an extra provider call or fit. An invalid first proposal can be repaired once;
an invalid final proposal explicitly uses the fixed forecast, rather than silently
accepting the earlier pool. There is no loop until a favorable result. The normal
budget is two provider operations and at most two model fits per month: the fixed
model and the final adaptive model. A withdrawal requires only the fixed fit.

`candidate_attempts.json` and `research_revision_context.json` are saved before
the second provider call. Final predictions retain both attempts, and the HTML
report shows their screening/coverage tables beside the existing baselines and
charts. With this option, `research_calls.jsonl` records physical provider attempts,
including failures. Provider failures stop the run. Explicit `--resume` reuses
completed operations and retries the failed operation; that retry is another
recorded attempt, not a fresh free budget. Runtime totals include prior attempts.
Changing this option requires a fresh output directory, like other research
configuration changes. The default remains one decision per month.

A retained real-provider development smoke demonstrated why selection is explicit:
the LLM reduced a 16-factor preview to 12 existing expressions but added each
expression's own reference as a parent. All 12 were correctly rejected. That run
remains a fixed-forecast fallback; adding selection does not rewrite its response,
rerun its models or turn it into successful candidate-training evidence.

Mature monthly feedback includes absolute simple-return errors in percentage
points and paired adaptive-minus-control loss differences. A negative difference
against the fixed model supports a candidate-pool increment on that target; it
does not imply the adaptive forecast beats unchanged price or the training mean.
The research prompt asks for the feedback month, prediction hash and comparator
when explaining a revision. Withdrawal returns the fixed-feature model. These
comparisons do not establish an individual factor's causal contribution, and
unmatured targets never enter the feedback.

## Measured market-risk inputs

For a separate publisher-vintage geopolitical news index, add `--gpr-history`;
see [GPR source, availability rules and setup](geopolitical_risk.md). Its weekly
release alignment differs from the daily VIX/Treasury reconstruction below.

Add `--risk-history /path/to/snapshot` to expose these optional inputs to the LLM:

| Input | Measurement |
|---|---|
| `risk.vix_close` | Cboe VIX closing index level |
| `risk.vix_change_5` | Change in aligned VIX level over five equity sessions |
| `risk.treasury_10y_percent` | US Treasury nominal ten-year par / constant-maturity yield, percent |
| `risk.treasury_10y_change_20` | Change in aligned yield over twenty equity sessions, percentage points |

The Treasury input is a nominal yield, not an inflation-adjusted real yield.
A change of 0.10 percentage points is ten basis points.

These are candidate inputs, not automatic additions to the fixed baseline. A
news-driven candidate can multiply a headline count by measured volatility, for
example; `--news-research` still requires an actual numerical news dependency.
Every accepted candidate uses the same fixed training rows. Missing, stale or
warm-up values remain unavailable and reject a dependent candidate; rows are not
dropped and values are not replaced with zero.

Alignment uses only observations dated strictly before each equity session, with
a maximum seven-calendar-day age. This deliberately avoids assuming a source's
same-day publication time. A later downloaded snapshot can contain revisions;
these are reconstructed historical inputs, not a publication-time vintage archive.
The report records coverage, retrieval dates and source hashes. Fold artifacts
also preserve per-row source dates and change endpoints.

The offline bundle contains `receipts.json` and the corresponding source bytes
under `raw/`. The loader checks HTTP status, official source URLs, byte lengths,
SHA-256 hashes and dated observations. For example, prepare a fresh snapshot
with Python's standard library (this explicitly downloads the sources):

```python
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

root = Path("/tmp/market-risk-snapshot")
(root / "raw").mkdir(parents=True, exist_ok=False)
urls = {"vix.csv": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"}
for year in (2025, 2026):
    urls[f"treasury-{year}.xml"] = (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
        f"?data=daily_treasury_yield_curve&field_tdr_date_value={year}"
    )
receipts = []
for filename, url in urls.items():
    requested = datetime.now(timezone.utc).isoformat()
    with urlopen(Request(url, headers={"User-Agent": "Quant-RSI research"}), timeout=60) as response:
        body = response.read(8 * 1024 * 1024 + 1)
        if len(body) > 8 * 1024 * 1024:
            raise ValueError("Risk response exceeds 8 MiB")
        receipts.append({
            "file": filename, "requested_url": url, "response_url": response.geturl(),
            "status": response.status, "requested_at": requested,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
        })
    (root / "raw" / filename).write_bytes(body)
(root / "receipts.json").write_text(json.dumps(receipts, indent=2))
```

Choose Treasury years covering the requested research period. Source references:
[Cboe VIX historical data](https://www.cboe.com/tradable_products/vix/vix_historical_data/)
and [US Treasury daily interest-rate XML feed](https://home.treasury.gov/treasury-daily-interest-rate-xml-feed).
There is no automatic risk-data acquisition in `research-rolling`.

The first run copies and revalidates the bundle under the output's `risk_history/`
directory and binds its identity to the run configuration. A failed run resumed
with the same command reads that retained copy, so an updated or removed external
snapshot cannot silently change an experiment. Tampering with the retained bundle
fails validation; changing its valid content fails the configuration identity check.
