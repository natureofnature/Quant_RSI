# Two executed feedback-derived policy advances

On 2026-11-03, a frozen `347b48c` runtime completed two consecutive, distinct
schema-4 research-policy changes, selected their executed descendants, sealed the
final model, issued a historical forecast, and passed forecast and trajectory replay.
This supplies the missing bounded execution evidence for [#3](https://github.com/natureofnature/Quant_RSI/issues/3)
and [#1 R2](https://github.com/natureofnature/Quant_RSI/issues/1). It establishes an
engineering capability, not independent predictive improvement or profitability.

## Fixed scenario and observed execution

The retained input has 520 invented daily sessions, including weekends, for `SYNTH`.
Seed 7 generates `r[t] = 0.85 * r[t-5] + Normal(0, 0.006)` with zero missing initial
lags, then `close = 100 * exp(cumsum(r))`, serialized to eight decimal places.
There are no corporate actions by construction. Training ends at index 360; the
objective is five-session price forecasting and the development stride is five.
The lag-five structure deliberately exercises the finite window-expansion grammar.
It is not a market benchmark or a randomly selected predictive task.

The initial factor is `momentum_2`, the initial allowed window range is 2–3, and
the registered expansion domain ends at 5. The schema-4
`terminal_frontier_first_divergence_v1` proposer reads authenticated development
frontier feedback. The campaign permits two proposal stages and two generations.
No LLM, paid data call, arbitrary generated code, or real order was used.

| Selected stage | Policy window cap | Selected factor | Development MAE, percentage points | Actual fits |
| --- | ---: | --- | ---: | ---: |
| Original source | 3 | `mean_return_3` | 1.756959 | 4 |
| First descendant | 4 | `momentum_4` | 1.223997 | 4 parent + 4 candidate |
| Second descendant | 5 | `momentum_5` | 0.988674 | 4 parent + 5 candidate |

Each evaluation has 31 observations. The final attempted `mean_return_5` achieved
1.025221 MAE and was not selected; this negative result remains recorded. The
trajectory sealed at generation 2 with `global_window_bound_reached`.

The two changes are semantic patches to `policy.window_max`, 3→4 and 4→5, followed
by actual LightGBM execution of the newly reachable factors. Four feedback links
were checked against their preceding source/descendant attempt, model and report
artifacts. The second proposal uses the first descendant's genuine feedback.
Ten fits in the unchanged paired prefixes have identical factors, model outputs,
evaluation reports and costs, but distinct actual fit records; they were not reused
as fictitious execution. [PR #110](https://github.com/natureofnature/Quant_RSI/pull/110)
separately checks controlled development-feedback dependence and protected-only
invariance at both caps, with fixed parent state. Those unit tests complement the
execution trace; they do not replace it.

## Costs, failures and recovery

The single registered allowance is 36 attempts / 6,480 worker-seconds. There are
21 proposal reservations, 21 journal attempt rows and 21 confirmed actual fits,
consuming 3,780 registered worker-seconds. All 36 attempts / 6,480 seconds remain
occupied: the unused 15 / 2,700 are irrevocable commitments, not reusable allowance.
The source is charged once; intermediate rereads add no charge. There are no
unreconciled child costs or unconfirmed fit outcomes.

The initial input-precision failure occurred before registration or fitting and is
retained. An intentional interruption exited 130 after five completed fits. A
diagnostic recovery with timed stack dumping then exited 139; its cause is
unresolved and the failure log is preserved. The subsequent plain recovery succeeded
in the same workspace and allowance. All five pre-interruption completed attempt
rows remained identical, and recovery did not redispatch them. The final continuation
exited zero. This was a recovered execution, not an uninterrupted clean run.

Final sealing, forecast issuance, forecast replay and trajectory replay all exited
zero. Before/after attempt rows and shared budget-registry rows are identical across
all four phases. The forecast replay equals the issued result; the terminal replay
equals the original terminal report. Sealing and publication added no fits.

## Selected forecast and evidence identity

The sealed model is the second descendant's selected `momentum_5` model, with the
original five-session objective. It forecasts historical origin `s514` to `s519`:
origin price 9.07631017, forecast 8.65228001 in synthetic `TEST` currency. Both sessions
were already retained before publication. The recorded status is **late**; this is
neither a prospective forecast nor an unseen outcome. No production promotion occurs.

| Artifact | Content reference |
| --- | --- |
| Research input | `992e8d5c6b76884b3801100e06579e822bcfc83e8385a200726236705f724d2b` |
| Final package | `e78b7676bcf0366f83474d4ec92d80ff8dc9cd5701dec363e6a950c1cd76a154` |
| Final checkpoint | `724377205d2ba60b5cb437b04716a0a246c0a41459c4874f9fd8e3e1f5c00aa9` |
| Final model | `4b841099c6bf61684caa825695cbd87a0cbdd32e1e12dbe62f99a32909b91788` |
| Sealed bundle | `f1d9194ca6cc593df953c67ff9612060556cb73650f0d12e8ecc836cbf8eee48` |
| Forecast result | `7e742f7181c844f772d8fcd8cc1ef911ca4f410cf9656989277f07e24f225790` |

Local raw evidence is retained outside Git at
`/home/nature/Learning_plan/Quant_RSI_evidence/recursive_two_advances_347b48c/`.
It includes `protocol.json`, `run_case.py`, `terminal-audit.json`,
`feedback-link-audit.json`, `paired-prefix-audit.json`, `recovery-failure.json`,
`recovery-verification.json`, and `finalization-audit.json`, plus original journals,
artifacts and phase outputs. The final audit records output-file SHA-256 values.
These local paths are not public downloads; the case script uses test helpers and
is not a maintained portable example. The supported operational instructions remain
the [recursive market guide](recursive_market_exploration.md).

## Interpretation and next cost boundary

The first bounded proposer and two-change engineering acceptance are delivered.
General research-program synthesis, calibrated promotion (#2/#4), independent or
prospective effectiveness, and matched-budget research-policy comparisons (#1 R4/R6)
remain separate work. This result does not complete the umbrella roadmap.

Cost remains a practical limitation. Before terminal completion, the old continuation
had already taken over two hours while all 21 recorded worker invocations together
took about 32 seconds. These timings have different scopes and are not a paired
speedup measurement. Repeated proof/replay preparation deserves attention. The frozen
run deliberately predates the subsequent feature and label batching optimizations;
their [component measurements](validation_cost.md) do not establish an end-to-end
speedup for this campaign. Its runtime, manifests and completed evidence were not
rewritten to use newer code.
