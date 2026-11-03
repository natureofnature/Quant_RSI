# Validation cost and feedback

The goal of [RFC #39](https://github.com/natureofnature/Quant_RSI/issues/39) is a
usable research loop with measurable engineering cost. A passing fixture proves
engineering behavior, not predictive quality. Keep the full merge checks while
using smaller checks to answer a specific development question.

## Run a measured check

Use the pinned Python 3.12 environment installed against this checkout. Workers
clear `PYTHONPATH`; the profiling command also clears it and rejects an interpreter
that imports another checkout. Run from the repository root:

```bash
# Fast checks for this tooling change; choose modules relevant to your change.
python examples/profile_validation.py --output /tmp/quant-rsi-focused \
  tests/test_validation_profile.py

# Complete regression, optionally using two local processes.
python examples/profile_validation.py --output /tmp/quant-rsi-full --shards 2
```

Output directories must be new and outside the checkout. The command does not
replace a prior report. It records the commit, dirty diff and source/test hashes,
Python/dependency versions, CPU affinity and container limits, thread settings,
cache note, exact commands, per-process durations/exit codes, pytest logs and
JUnit per-test timings. Pytest's slowest durations separate setup from test calls.
The summary fails when any shard fails or the recorded checkout changes during
the run. Freeze code for a run: all package Python files contribute to experiment
runtime identity. Environment/other-process contention and OS caches are not
controlled; use `--cache-note` to record relevant conditions. The runner does not
measure hands-on developer effort or CI queue time.

With no selectors, all repository `test_*.py` and `*_test.py` modules under `tests/`
are included. Explicit file or node selectors produce focused evidence only.
Selectors belonging to the same file stay in one process, even if interleaved on
the command line. If only one module is selected, only one process is needed.
Parallel runs schedule whole modules by estimated duration, placing the longest
first into the least-loaded shard. Thirteen modules have rounded timing hints
from [successful CI 41346746509](https://github.com/natureofnature/Quant_RSI/actions/runs/41346746509);
other modules use three seconds, the rounded mean of modules below 30 seconds in
that run. Hints affect placement only: no tests are omitted and module fixtures
remain together. Single-process runs preserve selector-group order. There is no
cross-run fixture cache. Estimates can become stale, and focused node selections
may take less time than their whole-module hint; compare actual shard durations,
CPU/setup work and wall time before updating the hints.

CI runs the complete suite through the same command with `--shards 2`, keeping
each module in one process. This aims to reduce wall time, not total CPU work.
It retains JSON metadata, JUnit results and logs for 14 days, including on test
failure, and does not upload temporary experiment workspaces. CI queue, installation, lint,
test and artifact-upload durations remain separate job steps. The branch/head
and merge-test revision can differ; report the revision in the profile.

## Choose the smallest useful feedback tier

| Tier | Purpose | Examples | Evidence limit |
| --- | --- | --- | --- |
| Fast numerical/contract | Check formulas, schema bounds and local transitions without worker orchestration | Accounting, feature, factor-layer tests relevant to the edit | Does not prove real worker, recovery or workflow integration |
| Focused integration | Run the affected real-worker or account/workflow path with deterministic offline inputs | Factor-runtime, settlement or workflow tests relevant to the edit | Does not replace unrelated regression coverage |
| Complete frozen revision | Verify compatibility, leakage barriers, budget/conservation, evidence integrity and recovery together | Full `pytest -q`, or the no-selector profiling command | Engineering evidence only; retain skips and failures in the report |

Do not mark broad groups as optional merely because they are slow. Mock fitting
only when fitting is not the behavior under test. Keep actual worker coverage
and the adversarial fixtures for leakage, duplicate charges/fills, account
conservation and artifact integrity. A focused failure should be fixed and rerun
before paying for a complete regression. Run the full suite once on the frozen
final revision, locally or in CI; repeat it for relevant changes or unresolved
failures. A pending CI run is not passing evidence. CI still runs every required
check; local feedback does not disable its gate.

Before optimizing, choose a named workload, resource/cache conditions and a
numerical target. The helper acceptance targets are: shared
module setup exactly once per invocation, all selected tests retained, failure
and no-tests exit codes preserved, and the helper's subprocess regression under
five seconds on the development host. A normal focused-feedback aspiration is
under five minutes; expensive real-workflow exceptions need actual measurements,
not a misleading global timeout. Do not translate local timing to a portable SLA.

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
quote conflicts, revision receipt boundaries and absent calendar closes. Required
full CI remains the merge gate; neither timing nor output equivalence replaces it.

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
The existing full Python and Qlib CI checks remain required before merge.


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
