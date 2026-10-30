# Run a connected stock research study

After the [Linux/Python 3.12 installation](../README.md#installation-and-usage),
run from the repository root with a new output directory outside the checkout:

```bash
quant-rsi research-study KO --output /tmp/ko-study
```

An interactive terminal opens the live research UI automatically; the saved report is
also available at `/tmp/ko-study/report.html`. The default uses Codex CLI saved authentication,
KO's configured relationship profile, registered issuer RSS and public daily
prices. Install/login to Codex first; the existing LightGBM dependency performs
two actual fits. The smaller `research-demo` command remains unchanged.
The default live path is currently configured for KO only; other symbols do not
automatically gain relationship profiles or registered news feeds.

## Live research UI

```bash
quant-rsi research-study KO --news-research --output /tmp/ko-active-study-next
```

The command opens a read-only local browser view. Ten nodes connect market data, company
context, news research, the first hypothesis and factor selection, prediction, feedback,
factor revision, the second prediction and the final report. The active node glows; completed
nodes remain lit. Click a node to inspect its stage outputs, or return to live following.
Outputs appear when a stage finishes; this is not token streaming. Fast stages can complete
between refreshes and remain available in the history.

The task panel follows the current stage. The final node completes only after the research
artifacts are saved, then the page shows the development selection, conclusion and full
prediction/strategy report. Failures remain failures: the page retains previous outputs and
shows the failed stage. Simulated inputs and development-only evidence are labelled.

- `--no-ui` keeps the command terminal-only.
- `--ui` explicitly enables the view, including when output is redirected.
- Noninteractive commands and `--json` default to no UI. `--quiet` only hides terminal stage
  messages; it does not override the UI setting.
- The printed URL binds to `127.0.0.1` with a random per-run path. If a browser cannot open,
  use that URL on the same machine (or forward its port over SSH). Nothing binds publicly.
- The viewer uses existing `runtime.json` and `trace.jsonl` files and cannot submit research
  or trading actions. Closing the browser does not cancel research. Its helper exits after
  two minutes without requests; a completed page retains its results in the browser.
- After completion or a handled failure, open `live.html` directly for the saved view,
  without a server, network access, frontend build or additional dependencies. Long stage
  outputs have a bounded preview; `trace.jsonl` retains the full output.

## Offline example

For a fully offline run after installation, keep real model fitting and replace
the external context and prices:

```bash
quant-rsi research-study KO --output /tmp/ko-study-offline \
  --llm simulated --related simulated --news simulated --prices synthetic
```

## Inputs and controls

Daily prices are collected under `OUTPUT/prices` unless you supply a retained
collection with `--prices-dir PATH`. Populated collections are verified and reused,
not refreshed or overwritten. Freshness is bounded by their receipt. These are
raw USD prices, not dividend-adjusted total returns; unsupported corporate actions
reject the input. See [price details](simulated_research_loop.md#use-collected-daily-prices).

KO's profile supplies CCEP bottling and MNST distribution relationships. Only KO
has a configured RSS feed: related coverage is explicitly partial, with CCEP/MNST
`not_configured`. The default remains configured relationships and RSS; opt-in
[active news research](#active-news-research) adds search, page reads and related
company suggestions. RSS news limits apply per issuer; see
[coverage details](simulated_research_loop.md#use-configured-company-relationships).

| Option | Default and meaning |
| --- | --- |
| `--llm` | `codex`; also `api` or `simulated` |
| `--llm-model`, `--llm-timeout` | Optional Codex model; API model required; timeout 180 seconds per call |
| `--model` | `lightgbm` by default; `ridge` uses a reviewed regularized linear model on the same rows and features |
| `--ridge-alpha` | Positive finite penalty for Ridge, default 1.0; rejected when supplied with LightGBM |
| `--related`, `--news`, `--prices` | `profile`, `issuer-rss`, `daily`; each also accepts `simulated`, except prices use `synthetic` |
| `--news-limit` | 5 items per issuer, range 1–20 |
| `--news-research` | Off by default; enable LLM-led current search, article reading and source-cited related company candidates |
| `--news-archive` | Optional shared snapshot directory; otherwise `OUTPUT/news_archive` |
| `--evaluation-sessions` | 20 historical origins, range 2–20 |
| `--initial-cash` | 10000 USD per strategy/baseline account |
| `--commission-bps`, `--slippage-bps` | 5 basis points each, on each executed side |
| `--threshold` | 0: a predicted five-session log return above this threshold chooses long, otherwise cash |
| `--selection-metric` | `mae` by default (lower is better); optionally `sharpe` (higher is better) for development-only adoption |
| `--quiet` | Hide stderr progress; timing and usage artifacts are still saved |

API mode requires `OPENAI_API_KEY` and `--llm-model MODEL_ID`; see
[LLM prerequisites](simulated_research_loop.md#use-a-real-llm-for-hypotheses).
Service errors stop the study; no silent fallback or automatic resume is claimed.

## Choose the model before research

The default remains LightGBM. To use the simple linear alternative with the same
news, factor revision, strategy evaluation and selection stages:

```bash
quant-rsi research-study KO --model ridge --ridge-alpha 1.0 \
  --output /tmp/ko-ridge-study
```

For an offline check, append `--llm simulated --related simulated --news simulated
--prices synthetic`. Both attempts then perform real Ridge fits on synthetic prices
and retained synthetic news; the report identifies those sources.

The version-1 model configuration is saved before collection and passed to both
hypothesis calls. It records the model name, parameters, preprocessing, target and
fixed mature-label training policy. Neither model type nor penalty changes after
feedback. `--ridge-alpha` must be finite and positive, and is rejected with LightGBM
instead of being silently ignored.

Ridge centers each input using its training mean and scales by its training
population standard deviation; constant columns use scale one. It minimizes
`sum((y - intercept - Z @ coefficients)^2) + alpha * sum(coefficients^2)` for
five-session log returns, with an unpenalized intercept. A NumPy augmented
least-squares solve supports duplicate/constant columns without adding a dependency.
The trace records means, scales, coefficients and intercept in `factor_ids` order.
Evaluation rows and latest forecast inputs never fit the scaler. The selected
attempt's model and scaler produce its latest forecast without a third fit.

LightGBM retains its existing deterministic parameters and 24 boosting rounds,
without scaling. Both model choices retain the same maturity rules, common news
mask and unchanged-price/training-mean controls. Separate CLI runs can receive
different LLM factors: merely running each model once is not a controlled model
comparison. Freeze data, factor columns, origins and information budgets before
making that claim. This completes the alternative-model interface in #39 B; it
does not complete multi-profile, external-engine or independent validation work.

## Active news research

```bash
quant-rsi research-study KO --news-research --output /tmp/ko-active-study
```

The same Codex CLI login used for hypotheses now also supports discovery; there is
no separate search API key. This still uses fresh ephemeral CLI calls, not the
conversation history of a previous Codex session. API mode uses `OPENAI_API_KEY`
and `--llm-model MODEL_ID`; choose a model supporting Responses `web_search` and
structured outputs. Provider/model errors are reported, not silently replaced.
See official [Codex web search](https://learn.chatgpt.com/docs/web-search) and
[Responses web search](https://developers.openai.com/api/docs/guides/tools-web-search).

The prelude makes one discovery request: the LLM chooses a focused query and
returns up to three source URLs. The application reads those public pages, then
asks the LLM to review the extracted text, cite article IDs and suggest related
listed companies. It may request one follow-up search; the second review cannot
extend the loop. There are at most two search requests and six page-read attempts.
The prompts request one native search per request; Codex controls internal tool
calls, so this is not an exact provider-call or billing budget. API searches also
set `max_tool_calls=2`. The actual search-tool metadata is retained; a response
without a recorded search is rejected.

Read failures guide the follow-up query. HTTP 401, 403 and 429 add that exact
normalized hostname to this run's `avoid_hosts`, asking the LLM for alternative
accessible official/primary sources. If a result still uses a denied host, the
application records `skipped_denied_host` without making another page request;
its receipt time, text and hash remain absent. A 404 or timeout does not suppress
the entire hostname. This is a run-scoped exclusion/cooldown, not a permanent
blocklist or an attempt to bypass access restrictions.

Only discovery requests enable web search. Hypothesis and review requests keep
tools disabled; shell, apps and multi-agent tools remain disabled for all Codex
calls. Web pages and search results are treated as untrusted input data.

The reader accepts public HTTP(S) HTML/text pages, validates and pins public DNS
addresses, and revalidates every redirect. Each page is limited to 1 MiB with up
to three redirects and a 15-second socket-operation timeout. The reader strips
navigation/scripts and prefers article/main content, saving at most 16,000 text
characters; LLM context uses at most 8,000 per page. It does not execute JavaScript
or bypass paywalls, and does not assert that extracted text is the complete article.
Failures, truncation, hashes and actual receipt times remain visible. Publication
times are left unknown rather than inferred from search snippets or model memory.

Search summaries and tool actions are kept separately from independently retrieved
page text. Findings and related-company suggestions must cite successfully read
article IDs. Those citations establish which sources were supplied, not automatic
verification of each claim or ticker mapping. Suggestions enter both hypothesis
contexts alongside the existing configured profile, but do not rewrite that
profile, declare measured correlations or automatically add peer price features.
When at least one page is readable, each hypothesis must cite a discovered article
and explain its relevance to factor selection. If all reads fail, the study records
unavailable discovery and can continue using its original RSS/simulated context.

`news_research.json` saves search results, page text and final findings. All search,
read and review stages, including the intermediate follow-up decision, are in
`trace.jsonl`. The HTML/Markdown report shows sources, failures, company suggestions,
and article IDs cited by the initial and revised factor choices. Discovery is
current research context only: articles never enter `news_history.json` or the
receipt-gated historical headline-count matrix.

To exercise this exact path offline:

```bash
quant-rsi research-study KO --news-research --output /tmp/ko-active-offline \
  --llm simulated --related simulated --news simulated --prices synthetic
```

This uses invented search results, articles and a supplier candidate, explicitly
labeled simulated. It completes a follow-up query and changes momentum to trailing
volatility after feedback, with two real LightGBM fits on synthetic data.

## News archive and features

Use the same study command with a shared archive to retain RSS snapshots across runs:

```bash
quant-rsi research-study KO --output /tmp/ko-study-next --news-archive /tmp/ko-news
```

The archive stores content-addressed JSON snapshots, deduplicates identical
content and preserves each source's original `received_at`. Limits are 256 files,
8 MiB total and 512 KiB per file. These recorded receipts are not independently
authenticated timestamps. The run also saves its original snapshot bundle in
`news_history.json`; no additional ledger is created.

The LLM can propose one or two news recipes with `name`, `keywords`, `window_days`
and `description`, identified as `news.<name>_<window_days>`. Each counts distinct
target-issuer headline IDs matching any of up to five case-insensitive literal
substrings (80 characters each) in a 1–30 calendar-day window. It counts the
**retained selected feed corpus**, not all news, summaries or full articles.

A row uses the newest snapshot/source receipt at or before its market-close
cutoff, no more than 24 hours old. Publications must have precise timestamps
inside the window and before the cutoff. Missing, stale, unavailable or date-only
evidence produces a missing value, not zero. A usable corpus with no keyword
matches produces zero. Daily prices use the real exchange-close instants; current
news received after those closes is never backdated.

Before either LLM call, the model freezes a common availability mask for both
attempts and their baselines. Executable news features need at least 12 usable
training rows and two evaluation rows; otherwise news proposals remain `pending`.
Selecting a pending factor is rejected, without silently replacing it with a price
feature. Eligible news factors are optional: either attempt may omit or deactivate
them. Reports distinguish eligibility from actual feature execution.
A first live collection will normally leave historical news factors
pending while price-only research remains usable. If selected news inputs are
missing at the latest origin, the latest forecast is suppressed with a reason.

With both `--prices synthetic` and `--news simulated` and an empty archive, the
offline command creates explicitly invented receipt history and fits actual news
columns. Simulated history is never mixed into a daily-price study. This exercises
the feature path; it does not establish news prediction value on market data.

## What the study does

1. Collect/read context and prices. The LLM proposes one to three numeric recipes
   using `momentum`, `mean_return`, `qlib_roc_v1` or `qlib_ma_v1` with windows 1–30,
   or `log_return_volatility` with windows 2–30.
   Recipes receive IDs such as `research.momentum_5`. News recipes follow the
   availability rules above and are recorded even when pending.
2. Fit the configured model once. All training labels mature strictly before the first
   evaluation origin. Predict the same set of matured five-session outcomes for
   the initial and revised attempts.
3. Report prediction errors against unchanged-price and training-label-mean
   baselines; run the corresponding long/cash strategy and buy-and-hold accounts.
4. Give development feedback to the second LLM call. Retain the proposed,
   activated, deactivated and retained factors in the trace, fit the revised recipe set and
   compare both attempts over the same training/evaluation windows.
   Adopt the revision only on strict improvement of the metric chosen before the run:
   lower price MAE by default, or higher net strategy Sharpe with `--selection-metric sharpe`.
   Ties and unavailable/nonfinite comparisons retain the initial attempt; another
   diagnostic cannot override the chosen metric.
5. In daily-price mode, report a separate latest-bar forecast with an unobserved five-session outcome.
   It uses the **selected** attempt's already fitted model, not a refit including newer matured labels;
   inspect its explicit training cutoff and retained price date.
   If that attempt lacks usable latest inputs, report it unavailable; do not silently
   switch to the other attempt. Both forecasts remain in their original stage records.

Recipes execute bounded operators. The study does not generate Python or execute
arbitrary code. Current news can inform recipe selection; archived headline counts
enter historical feature columns only when their receipt-time evidence is usable.

`research.log_return_volatility_N` adds a study-local numeric recipe with
`op=log_return_volatility`, window 2–30. It computes the unannualized sample standard
deviation (`ddof=1`) of N trailing daily close-to-close log returns, requiring N+1
complete finite positive closes. Incomplete windows remain unavailable; future
closes are not read. This is historical price volatility, not extracted news
sentiment. The protected bounded-engine factor contract is unchanged.

## Read the report

Prediction tables show MAE, RMSE, MAPE and directional agreement with the sample
count. These are historical **development** metrics, not an out-of-sample accuracy
claim: the revised proposal has already seen feedback from these same outcomes.
A smaller error or larger return in the revision does not establish independent gain.

Each attempt's strategy uses close-time signals and next-session open fills.
Positive signals enter long while flat; cash signals exit; unchanged long signals
hold existing units. Positions remain between supplied signals and through the
last forecast target close. Five-session prediction labels therefore differ from
the actual signal-to-signal trade intervals. Buy-and-hold uses the same period,
capital, lot size and costs, buying at the first eligible open. End inventory is
marked at the final close, without an invented terminal sale or fee.
Account quotes are rounded half-even to 12 decimal places to satisfy the existing
account contract; source prices and model inputs are unchanged.

The report includes net return, annualized Sharpe and volatility, maximum drawdown,
turnover, commission/slippage,
ending inventory and an HTML NAV plot from daily account valuations. Orders and
fills are hypothetical, using the existing account helpers; their timestamps do
not prove live or durable submission. Price forecasts and account returns answer
different questions. Read the development comparison separately from the latest
forecast, whose outcome remains `not_observed`.

Sharpe uses the mean of **daily costed simple NAV returns** divided by their sample
standard deviation (`ddof=1`), multiplied by `sqrt(252)`. Annualized volatility is
that standard deviation multiplied by `sqrt(252)`. The risk-free rate is explicitly
assumed to be **0%** for all three accounts, consistent with the zero-interest cash
simulation; it is not a current market rate. These statistics include cash days and
every daily valuation between signals, exclude the initial pre-trade cash mark, and
do not treat overlapping five-session prediction labels as daily returns.

The report displays the actual daily return count (normally 24 for 20 consecutive
forecast origins). The two strategy accounts and buy-and-hold use identical dates
and assumptions. Fewer than two returns, missing/nonpositive NAV, or zero volatility
make Sharpe `null` in JSON and `N/A` in reports, with an explicit status; flat
accounts still have zero volatility. Missing valuations are not silently skipped.
Short-sample annualization is descriptive and does not establish a stable long-term
Sharpe or independent improvement. These metrics also reach the second LLM call
through `feedback.strategy_diagnostics`; they do not establish independent
performance. The explicit
`--selection-metric` option now controls development adoption: it is saved in
`config.json` before collection/fitting and supplied to both hypotheses. The report
and `summary.selection` identify the chosen attempt, scores, factors and decision.
Trying a revised recipe is distinct from accepting it. Selection on these reused
development observations is not independent confirmation, even when the score improves.

## Progress and research cost

The CLI prints each running/completed stage and its elapsed seconds to **stderr**;
`--json` stdout stays machine-readable. `--quiet` suppresses only that display.
`runtime.json` is updated at stage boundaries and when usage arrives, so it also
records the active or failed stage when a run does not finish. It is observational
metadata, not a resume or recovery protocol.

The report lists preparation/research stage durations, total measured research
wall time, started/completed LLM/search/model-stage counts and token usage. Counts
refer to application stages; they do not claim an exact count of provider-internal
tool calls or billing events. Simulated searches remain labeled by the simulation
mode and do not count as real LLM requests. Timing ends before report serialization.

Codex `turn.completed` events and API response usage supply only allowlisted numeric
input, cached-input and output token counts. Cached input is already included in
input totals. Missing fields remain unknown; partial totals are explicitly labeled.
No token estimates, authentication logs, private reasoning or currency costs are
inferred or saved. See the official [Codex JSON event example](https://learn.chatgpt.com/docs/non-interactive-mode).

## Saved output

| File | Contents |
| --- | --- |
| `config.json` | Requested run settings |
| `runtime.json` | Live/final stage timings, attempt counts and reported/unknown token usage; retained on failures |
| `inputs.json` | Price inputs and their provider/evidence metadata |
| `news_history.json` | Original retained news snapshot bundle used by this run |
| `news_research.json` | Optional current discovery: search metadata/results, bounded page text/receipts, final findings and related-company candidates |
| `trace.jsonl` | Completed stage outputs, including proposals and feedback |
| `result.json` | Full structured results and account records |
| `report.md`, `report.html` | Human-readable comparisons; HTML includes NAV visualization |
| `live.html` | Saved interactive workflow view, when UI is enabled; final state and embedded report |
| `ui-server.log` | Local viewer diagnostics, when UI is enabled |
| `failure.json` | Written for failures after run initialization, retaining the completed trace for diagnosis |

Each study needs a fresh directory. Keep outputs and raw collections out of Git;
a failed run is inspectable evidence, not a resumable research campaign.

## Relation to mature engines

[Qlib's workflow](https://github.com/microsoft/qlib/blob/main/docs/introduction/quick.rst)
connects datasets, model training, backtesting and evaluation.
[QuantDinger](https://github.com/OpenByteInc/QuantDinger) presents an integrated
research/strategy/backtest product. They are references for connecting useful
steps into one inspectable flow, not evidence that this study matches their scope
or results. This command uses local Qlib-derived ROC/MA adapters, not an external
Qlib runtime. External-engine workflow integration and comparable performance
studies remain separate work; no Qlib/QuantDinger performance parity is claimed.

## Recorded validation

On 2026-10-29, a price-feature-only live Codex study completed 14 stages, two model fits and 20
matured evaluation origins. It used three KO news items; CCEP/MNST remained
`not_configured`. The revision changed the selected recipes, but development MAE
worsened from `2.270949` to `2.411026`. Strategy net return fell from `2.8682%` to
`2.4878%`, versus buy-and-hold's `3.8996%`. These are retrospective development
results with reused feedback, not independent evidence of gain. This run verifies
the connected workflow while preserving an unfavorable revision outcome.
It did not validate historical news-feature training; that requires usable
receipt-time history rather than a newly fetched current feed.

On 2026-10-30, two further Codex studies each completed 14 stages and two LightGBM
fits. The synthetic study used 128 explicitly invented news snapshots, 46 training
rows and 20 evaluation rows. Codex selected `news.demand_mentions_7` in both
attempts; each fitted matrix contained a price column and an actual news-count
column with provenance. This verifies feature execution, not predictive quality.
The live study used three KO RSS items and retained market prices, but had zero
usable historical news rows. Its proposed results/guidance and dividend factors
remained pending; both fits used two price columns, and the latest forecast stayed
unobserved. It did not train on historical market news.

Also on 2026-10-30, an active-news study using saved Codex CLI authentication,
registered KO RSS and retained real daily prices completed 20 stages: two search
requests, six page-read attempts, two reviews, two hypotheses and two real fits.
One page was readable; five returned HTTP 403 and remained unavailable. Both
hypotheses cited the retrieved article and executed `log_return_volatility_10`.
The initial fit used two columns and the revised fit three. Development MAE
worsened from `2.219264` to `2.516885`; Sharpe fell from `3.795628` to `0.231598`.
The report preserved those negative outcomes and did not fabricate company links
from unreadable pages. A separate Codex review of a successfully retrieved official
CCEP relationship page proposed CCEP with that article's ID; it was a review
interface check, not a discovery result from the full study. API discovery was
validated offline, not with a live API key. None of these runs establish predictive gain.

A subsequent live Codex study on 2026-10-30 exercised explicit adoption and usage
reporting. It completed six LLM stages, two searches and two fits in about 165 seconds
of measured research time. MAE changed from `2.219264` to `2.516885`, so the default
rule retained the initial attempt and its latest forecast rather than the revision.
Three pages were attempted (one readable, two HTTP 403); the follow-up returned no
additional sources instead of repeating those denied hosts. All six LLM stages
reported usage: 150,191 input tokens, including 26,112 cached inputs, and 2,513 output
tokens. These are provider-reported counts, not a monetary-cost estimate or evidence
of predictive advantage. A final offline regression also checks that rendering
failure never announces a completed run.
