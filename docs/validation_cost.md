# Validation cost and feedback

Daily validation now uses a critical tier. PR CI has a **five-minute job limit**
for checkout, dependency installation, lint, tests and artifact upload; GitHub
queue delay is separate. Timeout is failure, not evidence of passing tests.
The full regression and real Qlib integration run daily at 04:00 UTC and on
manual dispatch through [Extended regression](../.github/workflows/extended.yml).
A passing fixture proves engineering behavior, not predictive quality.

## Commands and evidence

Use the pinned Python 3.12 environment installed against this checkout. Workers
clear `PYTHONPATH`; the profiler rejects an interpreter importing another checkout.
Output directories must be fresh and outside the checkout.

```bash
# Daily/PR gate: defaults to critical, with two whole-module shards.
python examples/profile_validation.py --output /tmp/quant-rsi-critical

# Full coverage, including the deferred integration scenarios.
python examples/profile_validation.py --suite full --output /tmp/quant-rsi-full

# An explicit module/node always runs, even when normally deferred.
python examples/profile_validation.py --output /tmp/quant-rsi-focused \
  tests/test_trajectory.py

# Pytest itself remains unfiltered and runs every test.
pytest -q
```

The profiler records the selected suite, exact selectors, deferred module names
and reasons, source identity/hashes, environment, commands, JUnit results, wall
and CPU time, failures and skips. It preserves failure and empty-selection exit
codes and fails if the checkout changes while running. CI retains logs and
profiles for 14 days. Explicit selectors are recorded as `focused` evidence,
not as a complete critical/full run. Two shards keep each module's fixtures in
one process; scheduling hints only affect placement, never which tests run.
`--shards 1` remains available but is not the five-minute PR configuration.

## What remains in the daily gate

All modules except the 19 explicitly listed in
[`EXTENDED_MODULES`](../examples/profile_validation.py) are selected. The gate
continues to include actual model workers and the primary research/discovery UI.

| Invariant | Representative retained modules |
| --- | --- |
| Actual discovery, acquisition, model fit, feedback and revision | `test_study_discovery`, `test_discovery_loop`, `test_candidate_factor_runtime` |
| Future information, source timing and frozen samples | `test_forecast_features`, `test_study_incremental`, `test_study_horizon`, `test_discovery_review_regressions` |
| Budgets, recovery, evidence identities and tampering | `test_controller`, `test_research`, `test_study_resume`, `test_discovery_lineage`, `test_audit_regressions` |
| Account conservation, costs and prospective settlement | `test_execution_account`, `test_protected_accounting`, `test_strategy_replay`, `test_forecast_settlement` |
| Provider process cleanup, CLI and UI isolation | `test_provider_process`, `test_research_study`, `test_study_ui`, `test_validation_profile` |

New modules enter the critical tier automatically. New cases inside a deferred
module remain deferred: put critical regressions in selected modules, or run the
affected extended module explicitly and record its evidence. Selection uses a
reviewed list with reasons, not a runtime threshold or failure history.

## Deferred coverage and its tradeoff

| Extended group | Modules (`test_` prefix omitted) |
| --- | --- |
| Legacy full forecast workflows | `enhanced_forecast_workflow`, `forecast_usability`, `research_workflow`, `forecast_recursive_preparation` |
| Protected C4/C5 packages and trajectories | `local_comparison`, `trajectory`, `trajectory_recovery_guard`, `packages`, `comparison`, `local_packages` |
| Campaign, comparison and observer scenarios | `factor_discovery`, `forecast_campaign`, `observer`, `news_comparison`, `forecast_comparison` |
| Separate intraday execution product | `intraday_second_review`, `intraday_paper`, `intraday_review_regressions`, `intraday_policy` |

These tests are preserved, not equivalent to the smaller gate. In particular,
controller checks do not replace protected C4/C5 end-to-end recovery scenarios.
Regressions unique to extended or real Qlib coverage can now be discovered after
merge by the scheduled workflow. For changes in those paths, run the affected
extended selectors before merging. Manual full validation is available from the
Actions page at the selected branch/ref; its recorded source identity identifies
what was actually tested. Full regression is no longer mandatory on every PR.

## Timing acceptance

The previous full CI on source `e3e3c56`
([41855398994](https://github.com/natureofnature/Quant_RSI/actions/runs/41855398994))
passed 2,038 tests with 12 optional skips in **1,337.52 seconds**. The remaining
critical modules account for about 305 seconds of summed JUnit testcase time in
that run. Dividing by two is only a scheduling estimate; fixture setup, subprocess
startup, imbalance and runner contention still matter.

Accept the new gate only after a real critical run passes below 300 seconds.
Report the test-command wall time and total CI job time (including setup) separately,
alongside counts, skips, selection and source revision. A five-minute timeout alone
does not establish success. Hosted-runner queue time and arbitrary developer hardware
are not a portable latency guarantee. Keep the final-head CI artifacts as evidence;
full-mode collection can be compared without rerunning expensive scenarios merely
to verify the selector. The measurements below are historical optimizations under
the former full-suite PR gate.

## Initial observations

### Whole-module balancing follow-up

The successful calibration run above spent 1,750 and 743 seconds in its two
shards. A subsequent full run (41350144193, which failed one workbench fixture)
spent 2,413 and 810 seconds. Reassigning their recorded module times using the
same hints projects a 27–31% reduction in the longest shard's summed test time.
This is a scheduling projection, not measured wall-time improvement.

The follow-up target is at least 20% lower full-suite profiling wall time against
the parent revision's CI run 41354176687, using the same two-process Linux/Python
3.12 workflow. All tests and required checks remain enabled. Compare actual
per-shard durations and test outcomes as well as the total; hosted-runner
contention is uncontrolled, so one comparison cannot establish a stable speedup.
The final [follow-up CI](https://github.com/natureofnature/Quant_RSI/actions/runs/41355481123)
passed 1,671 core tests (12 optional skips) and 14 Qlib tests. Every parent case
was retained exactly once; one scheduling regression was added. Core wall time
fell from 2,156.72 to 1,480.61 seconds (31.35%); shard times changed from
2,156.70 / 538.04 to 1,480.58 / 1,425.17 seconds. This comparison meets the target.
Child CPU time increased from 2,229.65 to 2,756.98 seconds (23.65%), so it does
not establish reduced total compute cost or a stable wall-time speedup.

### Original fixture-grouping change

Earlier two-process full runs took about 28 minutes but alternated individual
node IDs between processes. That repeats module-scoped real-worker fixtures in
both processes, including local C5, protected comparisons and committed
trajectory setup. The helper keeps whole modules together. A tiny subprocess
regression verifies a shared fixture runs once while both selected tests execute
and a second module runs in the other process; it also checks pytest failure and
empty-selection exit codes. This proves removal of duplicate setup in this
scheduling case, not a particular whole-suite speedup.

Diagnostic samples on source revision `e01cb2e` also found a substantial runtime
cost below replay verification: each sampled path made roughly 147–148 million
calls to `timestamp`, through repeated `visible_revisions`/`knowledge_time`
scans. The local C5 sample completed fixture setup in 519 seconds under cProfile;
the trajectory sample spent most of its time replaying results. These diagnostic
runs were interrupted after approximately nine minutes to bound instrumentation
cost; neither is a passing test run or an uninstrumented latency benchmark.
`python -m cProfile -m pytest` returned process status zero despite pytest reporting
KeyboardInterrupt in this environment, so the diagnostic logs were inspected
and are explicitly excluded from validation evidence. The regular helper invokes
pytest directly and preserves its exit status.

The original #41 production change was limited to a 4,096-entry LRU cache for exact built-in
timestamp strings up to 128 characters. It stores only string-to-UTC-datetime
conversion. Long strings, string subclasses, datetime/custom-timezone inputs,
`trusted_now`, receipt corrections and evidence verification remain uncached.
Tests compare the original parser for cold/warm values and errors, bound the
cache, and exercise changing custom timezones and the live clock. No visibility,
Journal, replay or artifact validity claim is cached.

Before this change, the uninstrumented cold-process C5 two-test workload passed
in 286.88 seconds (221.57 seconds shared setup); the finite-trajectory test passed
in 280.98 seconds (9.50 seconds setup). Both ran concurrently in separate
processes with one thread per numerical library, fresh temporary directories and
uncontrolled OS caches. The pre-change target was at least 20% lower elapsed time
on each identical workload, with the same assertions. These are local
observations, not portable benchmarks. The final comparison is recorded in the
PR; a whole-suite speedup must be measured separately.

For a real factor or strategy change, also record modules edited, manual steps,
hands-on time and time to its final report, before and after. Those end-to-end
research usability measurements and a final-revision whole-suite timing are
still needed; the profiling helper alone does not complete #39's engineering-cost
acceptance.

## Feature-view batching measurement (2026-11-01)

A later sample of the finite-trajectory workload attributed about 30% of parent-process
CPU samples to JSON encoding, mostly repeated hashing of the same snapshot for each
training/prediction session. Repeated label visibility scans accounted for about 19
seconds of a 177.52-second baseline; optimizing that part alone was unlikely to meet
the predeclared 20% elapsed-time target.

The legacy local development path now builds feature views in one synchronous batch.
It hashes the snapshot once per batch, while still recomputing visible revisions,
eligibility and source references for every decision. No snapshot hash or visibility
decision is reused across calls. Training hashes each completed view once for its
instrument rows. Development predictions use the existing lazy model reader within
that one call, as the forecast path already did. The standalone feature-view API
retains its inputs and outputs; forecasts retain their existing training path.

The same unchanged finite-chain test, including actual workers and saturation/replay
assertions, passed before and after:

| Local measurement | Before | After |
| --- | ---: | ---: |
| Elapsed seconds | 177.5246 | 121.9604 |
| Child user CPU seconds | 172.8555 | 116.3706 |
| Child system CPU seconds | 2.1285 | 2.3779 |
| Pytest outcome | 1 passed | 1 passed |
| Source changed during run | false | false |

This is a **31.3% reduction for this workload**, above its 20% target, not a claim
about the complete suite or normal connected-study latency. These were one before/after
pair on the same 16-CPU Linux/Python 3.12 host, separate processes, fresh workspaces,
one numerical thread, no concurrent local study, and uncontrolled OS caches. The baseline
was `c7a9163` (the same tree as `main@f82488c`); the candidate changes feature batching
and model decoding in the local development path. Repeated measurements are needed to
estimate timing variability. Profiling runs were diagnostic and excluded from this pair.

Reproduce the measured workload on each revision with fresh output directories:

```bash
python examples/profile_validation.py --output /tmp/quant-rsi-batch-cost \
  --cache-note 'Record host contention and cache conditions for this run' \
  tests/test_trajectory.py::test_actual_finite_chain_seals_saturation_at_fully_occupied_capacity
```

A separate comparison with the saved original implementations produced byte-identical
canonical outputs for 48 feature views across operators, windows, knowledge bases and
watermarks. Two offline engineering fixtures with actual LightGBM fits also preserved both
34-by-3 training matrices,
labels, training-view/provenance metadata, all 12 development predictions per model,
and account results. Automated regressions cover batch/individual equality, fresh
hashes and visibility after changed input, watermark rejection, output independence,
and one-call versus fresh-per-view model decoding. Full required CI remains the merge
gate; focused timings do not replace it or establish a whole-suite speedup.

## Visible-history indexing measurement (2026-11-02)

Profiling the enhanced two-profile workflow attributed about 48 seconds of a
149-second instrumented run to feature-view construction. History extraction scanned
the entire visible event list separately for every calendar close. The optimization
indexes quotes by session and membership evidence by effective time within each call.
Binary search finds the latest applicable membership time, preserving every tied
record and every conflicting quote. Visibility, revisions and corporate-action
assessment are still recomputed on each call; no evidence-validity cache is introduced.

The unchanged workload was
`tests/test_enhanced_forecast_workflow.py::test_two_profiles_share_one_grant_and_terminal_repeats_are_read_only`.
It prepares KO and SOXL engineering profiles, exercises shared accounting and repeated
terminal operations, and uses an offline worker substitute. These timings do not
measure actual model training or connected-study performance.

| Local measurement | Before (`225cfee`) | After (`5937944`) |
| --- | ---: | ---: |
| Elapsed seconds | 110.2630 | 90.3440 |
| Child user CPU seconds | 94.1539 | 75.3539 |
| Child system CPU seconds | 3.7549 | 4.5136 |
| Pytest outcome | 1 passed | 1 passed |
| Source changed during run | false | false |

The observed **18.1% elapsed-time reduction** exceeds the predeclared 10% target for
this workload. This is one before/after pair on the same Linux/Python 3.12 host with
fresh processes and workspaces, the same dependencies, one numerical thread and no
concurrent local study. Host contention and OS caches were uncontrolled; repeated
measurements would be needed to estimate variability. Instrumented profiling was
excluded from the pair. This is not a whole-suite speedup claim.

Run the measured workload on each installed checkout with a fresh output directory:

```bash
python examples/profile_validation.py --output /tmp/quant-rsi-history-cost \
  --cache-note 'Record host contention and cache conditions for this run' \
  tests/test_enhanced_forecast_workflow.py::test_two_profiles_share_one_grant_and_terminal_repeats_are_read_only
```

A saved-original comparison verified byte-identical canonical outputs for 120 complete
feature views and 20 common masks across both knowledge bases, future-price changes,
missing quotes, delayed availability and shifted calendars. Seven regression cases
also cover equal-time membership evidence, contradictory statuses, delayed membership,
quote conflicts, revision receipt boundaries and absent calendar closes. Full CI was the merge gate for that revision; see the current tier policy above.

## Forecast training-view batching measurement (2026-11-03)

The forecast training path still built each historical feature view separately.
Recursive authorization and replay repeat this path, including for the training-mean
baseline. Each individual view consequently rehashed the entire input snapshot.
Forecast training now uses the existing synchronous batch implementation: it hashes
the snapshot once, still resolves visibility for each decision, and hashes each
completed view once before expanding its instrument rows. Currency, quote basis,
receipt-watermark and objective checks remain in force. No cache survives the call.

A retained engineering input contained 520 invented daily sessions and one instrument,
with a training cutoff at index 360. Seed 7 generated returns as
`r[t] = 0.85 * r[t-5] + Normal(0, 0.006)` (missing initial lags are zero); closes were
`100 * exp(cumsum(r))`, serialized to eight decimal places, with no corporate actions
by construction. This is synthetic data, not an exchange calendar or market result.

Three interleaved calls per implementation compared a saved `347b48c` training function
with the batch implementation in the same Python 3.12 process, on the same preloaded
input and `momentum_3` plan. Numerical libraries used one thread. An unrelated recovery
process was active; host contention and caches were uncontrolled. Timings cover only
`training_view`, excluding input loading, model fitting and complete recursive replay.

| Median per call | Original function | Batched function |
| --- | ---: | ---: |
| Process CPU seconds | 1.1905 | 0.3935 |
| Elapsed seconds | 1.1910 | 0.3937 |

The **66.9% process-CPU reduction** exceeds the predeclared 20% target for this input.
It is not a full-suite or end-to-end workflow speedup claim. Four saved-original
comparisons (`momentum` and `mean_return`, windows 2 and 3) produced identical feature
arrays, labels, metadata and complete training-view bytes. Two actual `momentum_3`
LightGBM fits also produced identical model bytes, predictions and evaluation output;
only their actual fit timestamps differ. These checks do not establish predictive gain.

The focused regressions compare batched training with fresh individual views, require
one snapshot encoding, and cover historical/observed visibility, receipt watermarks,
corporate-action interpretation and currency refusal. Run them with
`pytest -q tests/test_forecast_kernel.py tests/test_forecast_research.py`.
Full Python and Qlib CI were required at that revision; see the current tier policy above.


## Forecast label projection batching (2026-11-03)

After feature batching, forecast training still projected the same snapshot at the
same training cutoff once for every label. The label grid now shares one lazy,
call-local visibility projection. Per-label instrument validation, target maturity,
action assessment, quotes and return calculations are unchanged. Empty or entirely
immature grids do not project events. No projection survives a call, and no caller
can inject prefiltered events to bypass the original visibility checks.

A saved `909dd1c` training function and label implementation were compared with this
change on the same 520-session synthetic input described above. Each implementation
had one warmup, followed by three interleaved rounds of five calls, on one preloaded
`momentum_3` plan. Python 3.12 numerical libraries used one thread. The old frozen
campaign's finalization ran concurrently; this is a component measurement with
uncontrolled host contention, not a complete workflow or test-suite measurement.

| Median per call | Original function | Batched labels |
| --- | ---: | ---: |
| Process CPU seconds | 0.338682 | 0.210792 |
| Elapsed seconds | 0.338768 | 0.210836 |

The **37.8% process-CPU reduction** passes the predeclared 20% target. Eight
saved-original comparisons (`momentum` and `mean_return`, windows 2–5) produced
identical features, labels, metadata and complete training-view bytes. Two actual
standalone `momentum_3` fits produced identical model bytes, predictions and evaluation
output, excluding actual fit-time fields. They are validation fits, not additions to
the frozen campaign's allowance. Model SHA-256:
`0c85737e945cfe844f846769217eff08ebd4eddd67f5380b8f999de3a39c1c2d`.

Regressions cover multiple instruments, actions, immature targets, observed-receipt
visibility, old watermarks, one training projection and fresh calls after input
changes. The focused command remains
`pytest -q tests/test_forecast_kernel.py tests/test_forecast_research.py`.
Local measurement scripts, the before-edit protocol, saved baseline functions and
results are retained outside Git at
`/home/nature/Learning_plan/Quant_RSI_evidence/batch_forecast_label_projections/`.
Full Python and Qlib CI are still required before merge. Neither this measurement
nor the older feature-batching measurement establishes a complete recursive-run
speedup; their percentages must not be added together.
