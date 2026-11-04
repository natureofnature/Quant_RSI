# Geopolitical news risk as an optional candidate input

Monthly research can expose `risk.gpr_level` to the existing candidate expression
engine. It is the **GPRD daily geopolitical risk index** from Dario Caldara and
Matteo Iacoviello, based on newspaper coverage. It measures a standardized news
frequency, not the physical severity of a war or the causal effect on SOXX.
It is not automatically included in the fixed baseline. The researcher may select
it, combine it with other available inputs, or decline it.

With `--news-research`, a candidate must still use an actual numerical `news.*`
input, for example a retrieved-headline count multiplied by `risk.gpr_level`.
The GPR leaf alone does not satisfy that guard; standalone GPR selection is available
only when news-required mode is off. Adding this source does not relax the guard.

```bash
python -m pip install -e '.[gpr]'
quant-rsi research-rolling SOXX \
  --prices-dir /path/to/verified/SOXX \
  --first-month 2024-07 --last-month 2024-12 \
  --llm codex --news-research --max-candidates 16 --max-news-inputs 32 \
  --gpr-history /path/to/retained-gpr-vintages \
  --output /tmp/soxx-gpr-study
```

The example requires qualified price inputs and a retained GPR directory described
below. It does not bypass corporate-action checks. It is not a record of a completed
SOXX experiment. `xlrd` is optional for users who do not enable GPR; the development
lock includes it for offline parser and integration tests.

## Publication versions, not just observation dates

The [publisher](https://www.matteoiacoviello.com/gpr.htm) supplies daily observations
but updates them weekly, normally Monday with federal-holiday exceptions. Both
recent and older values can be revised. The publisher links to its
[dated archive](https://github.com/iacoviel/iacoviel.github.io/tree/master/gpr_archive_files)
and update log. A current download cannot be treated as the historical value known
on each observation day.

For every equity row, the adapter:

1. Selects the latest retained vintage whose filename date **and original publisher
   Git commit UTC date** precede that row's date. No same-date release is admitted.
   A delayed commit postpones availability even if the filename is earlier.
2. Uses that vintage's latest observation strictly before the equity date.
3. Returns unavailable if the release or observation is more than 14 calendar days
   old, the observation is explicitly missing, or no earlier vintage exists.

This is a conservative date-level availability rule. Git metadata and archive names
are publisher evidence, not cryptographically certified first-publication times.
The data were retrieved now. Do not describe the resulting historical reconstruction
as a contemporaneously captured or strictly point-in-time dataset. No guessed Monday
release or latest-vintage backfill substitutes for a missing archive.

The adapter records the selected vintage, commit, observation date, source hash,
availability and missingness for each row. Future vintages cannot revise an earlier
row. Missing values do not remove samples or alter the baseline's training mask;
an unusable candidate is rejected through the existing candidate screening path.

## Retained directory

Keep `receipts.json` (a list) beside `raw/`, containing the original binary XLS
files. Use filenames `data_gpr_daily_recent_YYYYMMDD.xls`. Each receipt includes:

```json
{
  "file": "data_gpr_daily_recent_20240624.xls",
  "requested_url": "https://raw.githubusercontent.com/iacoviel/iacoviel.github.io/84dfa32d7e6ccd483f5f323d0abc5b53d63da46e/gpr_archive_files/data_gpr_daily_recent_20240624.xls",
  "response_url": "same exact commit-pinned URL as requested_url",
  "status": 200,
  "requested_at": "actual retrieval start timestamp with timezone",
  "received_at": "actual retrieval completion timestamp with timezone",
  "bytes": 3091968,
  "sha256": "SHA-256 of retained original file",
  "release_date": "2024-06-24",
  "commit_sha": "84dfa32d7e6ccd483f5f323d0abc5b53d63da46e",
  "committed_at": "2024-06-24T17:29:42Z"
}
```

The schematic receipt must be populated with actual download metadata; placeholders
are not accepted. Obtain original commit metadata from the publisher repository's
history for that exact archive file, and retain that response with the acquisition
record. Download using the original commit URL rather than mutable `master`.
The loader verifies source identity, filename/date consistency, retrieval ordering,
byte counts and hashes. These checks do not independently authenticate supplied
receipts. Each source is bounded to 8 MiB, with at most 128 vintages per directory.

The study copies its admitted raw files and receipts into its own `gpr_history/`,
binds the normalized bundle to the run identity and uses that retained copy on
resume. Moving or deleting the original directory does not invalidate that copy;
changing retained data is rejected. `--risk-history` for VIX/Treasury remains
independent and may be used together with `--gpr-history`.
The run records the optional XLS reader version and rejects resume if it changes.
Studies without GPR do not require or bind that dependency.

## Evidence and scope

A source qualification on November 4, 2026 compared the publisher's June 17 and
June 24, 2024 files at their original Git commits. Among the 168 common January
1–June 16 observations, 18 GPRD values changed; the earliest affected day was May
18. XLS and Stata agreed exactly for all 176 January–June 24 numeric rows in the
June 24 file. This supports using vintage-aware alignment; it establishes no
forecasting advantage.

Offline tests check actual XLS parsing, revision and delayed-release isolation,
missing and stale data, hash/source rejection, the candidate's actual Ridge input
column, and retained resume. Synthetic engineering fits are not market evidence.
The frozen 2024 SOXX protocol remains unchanged. Adding this optional source does
not establish that RSI outperforms a fixed factor pool or simple baselines.

Source credit: Caldara, Dario and Matteo Iacoviello (2022), *Measuring Geopolitical
Risk*, American Economic Review, 112(4), 1194–1225. The publisher distributes the
data under Creative Commons BY; preserve this attribution and retrieval date.
