# SOXX: an October-cutoff November forecast

This fixed task uses the existing discovery CLI and actual LightGBM fits. It
admits 209 daily sessions from January 2 through October 30, 2026 for SOXX and
configured industry/market comparators XSD and SPY. The published exchange
calendar places the twentieth subsequent session on November 30 (including the
Thanksgiving closure and November 27 early close). The example asserts all dates
before making any LLM calls. This is an exploratory price forecast, not a claim
of predictive advantage or a contemporaneously captured historical experiment.

```bash
python examples/soxx_november_study.py \
  --inputs /tmp/soxx-november-2026-inputs \
  --output /tmp/soxx-november-2026-study --llm codex
```

Use a fresh output directory. `--prepare-only` collects/verifies inputs and prints
the exact study command without LLM calls or fits. `--llm api` uses the existing
API configuration. Existing input collections are validated and reused, never
rewritten. A failed study can use the existing `research-resume` command; run
`write_forecast_summary(output)` from this example after successful continuation
if the presentation sidecar was not generated.

## Time and information boundaries

- `collect_daily(..., as_of=aware_datetime)` limits normalized quotes and action
  effective dates to the cutoff while retaining current raw responses and actual
  receipt timestamps. The Yahoo request itself uses bounded `period1`/`period2`
  anchored to that cutoff, rather than a range relative to retrieval time.
  It still rejects missing completed exchange sessions.
  It does not recover a point-in-time provider snapshot or undo adjustments.
- January supplies the feature warm-up; no preceding-year rows enter this task.
  Requested train origins are January 2–August 31 and development origins are
  September 1–October 2. Existing maturity and partition rules remove warm-up
  and overlapping-label rows; inspect `research_windows.json` for actual dates.
- The selected model is reused for the October 30 forecast. Discovery does not
  permit a publication refit, so its reported training-label cutoff is earlier
  than the final available price. This is not a fit on every Jan–Oct label.
- No November prices, labels, feedback, current search or shared study memory
  enter research. No terminal historical score is requested; November is the
  future observation period. Development results remain selection-dependent.
- `--news-archive PATH` can supply genuinely retained historical news. Without
  eligible material, the system records a coverage gap; it must not fabricate
  news or backdate new receipts. A news-driven candidate fit is not guaranteed.
  Retained mode filters supplied evidence, not the LLM's pretrained knowledge.
- XSD/SPY roles are explicitly configured now, not reconstructed SOXX holdings.
  The shared equities catalog and experimental definition survey remain active;
  KO financial variables and SOXL leverage assumptions are not copied.

## Split-aware interpretation

The [August 21 SEC supplement](https://www.sec.gov/Archives/edgar/data/1100663/000119312526361162/d133118d497.htm)
announced a 3:1 SOXX split effective after November 4 close, with split-adjusted
trading beginning November 5. This announcement predates the research cutoff;
the current retrieval is used only for presentation and does not masquerade as
a pre-cutoff captured research article.

This example requires preserved collections received before the split became
effective. Later public OHLC responses may retrospectively adjust historical
prices. A historical date filter alone cannot establish the share basis.

`november_forecast.html` and `november_forecast.json` distinguish:

1. Forecast and reference price on the October 30 share basis.
2. Predicted price return on that consistent basis (not total return).
3. An indicative November nominal quote: forecast divided by three.

The third value is arithmetic, not another model prediction. Do not compare the
October-basis forecast directly with a raw post-split quote or treat the split
as an investment loss. The sidecar links to `report.html` for model/simple
baseline comparisons, strategy/buy-and-hold metrics, Sharpe and development
charts. No November accuracy or realized Sharpe is available yet.

## Older retained collections

The daily adapter also admits official 2024 and 2025 schedules when a requested
historical window or future session horizon requires them. The registered 2024
sources are the [NYSE announcement published in 2023](https://ir.theice.com/press/news-details/2023/NYSE-Group-Announces-2024-2025-and-2026-Holiday-and-Early-Closings-Calendar/default.aspx)
and the [Nasdaq 2024 trading calendar](https://www.nasdaqtrader.com/content/technicalsupport/2024tradingcalendar.pdf).
The Nasdaq PDF was visually inspected: filled orange circles denote closures,
hollow orange circles denote 13:00 ET closes. Its SHA256 is
`5a417265c706abe1e2eaa24e0c1fde1787e79fdd472eca105466a019f8ff1101`;
a changed document is rejected until reviewed. Both exchanges list ten 2024
closures and July 3, November 29 and December 24 early closes, yielding **252**
sessions from January 2 through December 31. A horizon crossing into 2025 also
uses that year's calendar and the separately retained January 9 closure notice.

Each collection retains raw calendar documents, actual retrieval timestamps,
source URLs, source years and content hashes. Historical requests end at an
exclusive whole-second `period2` no later than the cutoff and look back 366 days
for 120 sessions or 732 days for longer histories. Completed-bar admission still
requires the scheduled close plus the 15-minute buffer. Collections without an
`as_of` retain their existing `range=1y`/`range=2y` requests. Existing workspaces
are never rewritten. This extends collection coverage; it does not establish
historical provider vintages, resolve corporate actions, or authorize treating
historical predictions as live observations.
