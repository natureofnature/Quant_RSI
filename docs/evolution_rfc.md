# RFC: Evidence-led evolution of Quant RSI

Status: proposal for independent review; no implementation or publication approval is implied.
Analysis baseline: `e19524d52b64cd56bc2e01fc0d18856f830963e4`, inspected on 2026-11-01 UTC.

## Decision requested

Adopt the dependency-ordered roadmap below under the existing issue owners. First
resolve a demonstrated connected-study provider lifecycle gap. Then clarify shared
protocol boundaries and establish registered research acceptance before expanding
research-program evolution, factor hierarchies or horizons. Each phase requires its
own implementation proposal and validation; this RFC changes documentation only.

The [architecture specification](architecture.md) remains the target design, and the
[issue ownership audit](issue_status.md) remains the implementation status map.
This proposal does not close existing issues, replace their contracts, authorize
trading, or claim predictive advantage. Valid negative and inconclusive research
results are acceptable outcomes of a correctly implemented protocol.

## Current state and evidence

Paths and line numbers in this section refer to the analysis baseline. The
[baseline source tree](https://github.com/natureofnature/Quant_RSI/tree/e19524d52b64cd56bc2e01fc0d18856f830963e4)
provides an immutable reference if subsequent changes move those lines.

| Boundary | Existing capability and evidence | Remaining qualification |
| --- | --- | --- |
| Bounded Experiment Kernel | [Controller](../src/quant_rsi/core/controller.py), lines 202–255, dispatches bounded workers and passes owner lock descriptors at line 232. [Journal](../src/quant_rsi/core/journal.py), lines 20–57, and [artifact store](../src/quant_rsi/core/artifacts.py), lines 27–71, provide transactional identity and durable content integrity. | These primitives do not establish that every orchestration path has the same ownership contract. |
| Connected study | [Study](../src/quant_rsi/research/study.py), lines 95–100, and [continuation](../src/quant_rsi/research/study_resume.py), lines 16–45 and 115–134, bind inputs and retain successful operation receipts. | Provider processes can outlive ownership or a direct timeout, as described below. Logical fit limits are not money or attempt budgets. |
| Market information | [Market data](../src/quant_rsi/market/market_data.py), lines 447–492, distinguishes observed receipts, documented source availability and assumed historical availability. [As-of qualification](../src/quant_rsi/market/asof.py), lines 19–27, records coverage limits. | Visibility machinery does not prove historical vendor, universe or corporate-action completeness. |
| Research evaluation | [Model intervals](../src/quant_rsi/research/demo_model.py), lines 94–108, separate training, development and reserved terminal origins. Lines 660–703 freeze the selected predictor and qualify terminal results. | [Terminal evaluation](connected_research_study.md#reserved-terminal-evaluation) is retrospective; current news and pretrained LLM knowledge are not historically isolated. Adjacent five-session outcomes overlap. |
| Accounting and execution | [Kernel accounting](../src/quant_rsi/core/accounting.py), lines 99–155, charges costs and liquidates terminal inventory. Connected studies have their own [valuation policy](connected_research_study.md). [Intraday execution](../src/quant_rsi/intraday/live.py), lines 31–67, has separate journal and reconciliation semantics. | Result series and terminal policies need an explicit common protocol before comparison. Standalone intraday capability does not imply controller integration. |
| Protected decisions | [Promotion](../src/quant_rsi/evaluation/promotion.py), lines 21–22, blocks production promotion outside fixtures. [Comparison](../src/quant_rsi/evaluation/comparison.py), lines 139–144, rejects uncalibrated confirmatory preflight. | These are intentional barriers. Engineering integration does not confer calibrated authority. |

### Supported operational finding: provider lifetime exceeds study ownership

The study holds `.study.lock` through `run_lock`, which yields no descriptor for
provider inheritance. The Codex-mode provider in
[llm.py](../src/quant_rsi/research/llm.py), lines 249–268, uses `subprocess.run`
with a timeout and no study ownership inheritance or process-group supervision.
Successful receipts are written only after the operation returns. Consequently,
an owner killed before checkpoint commit can leave a local provider alive while
another process reacquires the study lock. A separate timeout case can leave
descendants of a fake provider alive after the direct process exits.

Atlas reproduced both mechanisms offline with the actual study lock and the
unchanged AST-extracted provider method, substituting deterministic Python argv
for the CLI invocation. Delta independently read and reran both scripts in fresh
scratch; both analysts report exit code 0. The integrated owner-death probe reports
`actual_method_under_actual_run_lock=true`,
`provider_alive_after_study_owner_sigkill=true`, and
`resume_lock_acquired_while_previous_provider_alive=true`. Fixtures were cleaned up.

These checks establish local lifetime and ownership behavior. They do not establish
duplicate real requests, billing loss, checkpoint corruption, predictive harm or
that every installed provider creates descendants. Full CLI imports were blocked
by missing dependencies; no live provider was invoked. The proposed issue is
medium-priority operational reliability work, separate from documented retry and
logical-fit-budget limitations.

### Existing scientific work and duplicate boundaries

The repair checked all eight issue/PR pages (#1–#72) on 2026-11-01 UTC and
retrieved relevant full bodies and comments. Open
[PR #72](https://github.com/natureofnature/Quant_RSI/pull/72) describes an inherited
queue-task lock covering worker startup and server reopening. That is adjacent
work at the queue-to-study boundary; its description does not establish ownership
of the study's provider or bounded descendant cleanup. It is not in the inspected
source baseline. Coordinate the focused lifecycle issue with that PR and recheck
its final implementation before choosing a fix. Closed
[PR #71](https://github.com/natureofnature/Quant_RSI/pull/71) owns study continuation,
while [PR #58](https://github.com/natureofnature/Quant_RSI/pull/58) addresses separate
IBKR order/reader lifecycles. No separate provider-lifecycle issue was identified
in this snapshot; the Manager must refresh this check before publication.

The analysts inspected open issue update paragraphs as well as the local audit.
Keep these owners rather than opening new generic validation issues:

| Existing owner | Residual responsibility |
| --- | --- |
| [#1](https://github.com/natureofnature/Quant_RSI/issues/1), [#39](https://github.com/natureofnature/Quant_RSI/issues/39) | Umbrella acceptance and demonstrated connected-study use cases; closure of a design record does not establish empirical acceptance. |
| [#4](https://github.com/natureofnature/Quant_RSI/issues/4), [#11](https://github.com/natureofnature/Quant_RSI/issues/11), [#12](https://github.com/natureofnature/Quant_RSI/issues/12) | Calibrated inference, older C2/C5 chronological contracts, and applicable robustness/selection diagnostics. |
| [#9](https://github.com/natureofnature/Quant_RSI/issues/9), [#10](https://github.com/natureofnature/Quant_RSI/issues/10) | Registered utility, benchmarks and risk eligibility; descriptive study reports do not satisfy these contracts automatically. |
| [#2](https://github.com/natureofnature/Quant_RSI/issues/2) | Production promotion only after calibrated authority and applicable prerequisites. |
| [#3](https://github.com/natureofnature/Quant_RSI/issues/3) | Actual consecutive feedback-derived research-program changes. |
| [#7](https://github.com/natureofnature/Quant_RSI/issues/7), [#8](https://github.com/natureofnature/Quant_RSI/issues/8) | Multi-horizon objectives and broader learned factor hierarchies beyond current finite compositions. |

Narrow searches for `orphan` and `timeout Codex` returned no issue matches in
Atlas's audit. The task-native issue listing exceeded its output limit; the
connector's open-issue check is not exhaustive closed-issue or open-PR deduplication.
The Manager must recheck related work before opening the lifecycle issue. It has
no issue number yet; a reviewed publication must supply its returned URL rather
than inventing one.

## Analyst discussion and disposition

Atlas asked whether the documented retry and logical-fit caveats made the lifecycle
finding actionable. Delta independently reproduced the probes and supported one
focused issue: the caveats do not supply ownership of a surviving local provider.
Delta challenged a descriptor-only remedy because it does not terminate descendants,
reconcile remote completion, or protect a normal context exit that explicitly
unlocks the shared file description.

This RFC adopts an outcome-based contract: prevent overlapping locally owned
operations after recovery, bound local cleanup, and retain ambiguous remote
completion as unknown. Descriptor inheritance is a possible component, not an
accepted complete design. Atlas explicitly accepted Delta's challenge and the
outcome-based acceptance criteria; Delta confirmed that this resolves the
challenge. Their agreement covers owner death, normal unlock, bounded local
descendant cleanup and unknown remote completion. The implementation choice
remains open; local cleanup does not establish remote completion or cancellation.

## Roadmap and acceptance gates

The order expresses dependencies rather than calendar estimates. No phase promises
positive investment returns. Record capability acceptance, evidence classification
and promotion decision separately, as required by the architecture.

### Phase 0: close the local provider ownership gap

Dependency: final deduplication and acceptance of the focused lifecycle issue.
Scope: connected-study provider ownership, cancellation and continuation evidence.

Acceptance gates:

1. With a direct fake provider active, kill its study owner. Immediate continuation
   must refuse while the prior operation remains owned, or prove local termination
   before redispatch. Cover explicit normal unlock separately from owner death.
2. Time out a fake provider that spawns a descendant. Within a declared cleanup
   deadline, no live fixture process remains, the error is visible, and no success
   receipt exists. Record the deadline in the implementation proposal.
3. Cover ordinary success, cancellation, owner death, timeout, failure around
   checkpoint commit and explicit continuation through the real entry path when
   dependencies are installed. The extracted-method probes remain mechanism evidence.
4. Preserve completed workspaces and recorded identities. Retain attempt evidence
   and an unknown state for ambiguous remote completion; never infer remote
   cancellation or exactly-once billing from local process termination.

### Phase 1: make protocol boundaries explicit before sharing machinery

Dependency: Phase 0 for lifecycle reuse. Documentation and measurement may begin earlier.
Existing owners: #1/#39, with #2's authority boundary preserved.

Deliver one capability matrix for connected studies, forecast campaigns, bounded
C2/C4/C5 execution and intraday paper sessions. For each, identify the entry point,
lock owner, durable identity, receipt/attempt state, timeout semantics, information
cutoff, budget unit, terminal inventory rule and promotion authority.

Acceptance gates:

1. Every supported entry point has a named owner and a documented recovery action;
   unresolved incompatibilities are listed rather than normalized silently.
2. Before extracting a shared primitive, freeze representative legacy workspaces
   and expected reports. Demonstrate unchanged read-only replay under their recorded
   source/runtime, and refusal of changed inputs or incompatible continuation.
3. Any proposed unification has an equivalence test and a measured benefit, such as
   fewer divergent lifecycle paths or lower recovery time. Do not silently rewrite
   old source identities or conflate logical fits with reserved worker slots.

### Phase 2: register research evidence and calibrated decisions

Dependency: stable ownership for new measured runs and explicit Phase 1 protocol
boundaries. Existing owners: #4/#9/#10/#11/#12; #2 remains blocked.

Acceptance gates:

1. Freeze data and receipt identities, universe/action qualifications, factors,
   label horizon/maturity, missingness, chronological folds/embargo, selection
   history, benchmark, costs, objective and eligibility before results are inspected.
2. Under perturbations of terminal labels, selection, eligibility and the selected
   fitted model remain unchanged. Explicitly report unavailable terminal outcomes;
   do not switch models or drop fixed rows to obtain a result.
3. Preregister a minimum useful effect, dependence-aware uncertainty method,
   repeated-selection error budget, sample-size/power gate and predefined cost
   stress. Explain how overlapping outcomes affect the effective sample size.
4. Separate engineering fixtures, development scores, retrospective terminal results
   and prospective evidence in every report. Register a prospective cohort before
   outcomes arrive, preserve receipt clocks and wait for declared label maturity.
5. A valid protocol can return supported, not supported or insufficient evidence.
   Check its decision rules independently; only the applicable calibrated authority
   can satisfy #2's production prerequisite. This RFC grants no promotion authority.

### Phase 3: evaluate broader research evolution within matched budgets

Dependency: Phase 2's frozen evaluation protocol and mature evidence.
Existing owners: #1/#3, then scoped #7/#8 extensions.

Acceptance gates:

1. Execute and retain two consecutive feedback-derived research-program changes
   under #3, including proposals, actual changed mechanisms, failures and outcomes.
   Fixed-policy exploration, lineage fixtures and adapter integration do not suffice.
2. Compare policies using common initial data, factors, origins, missingness and
   information budgets. Report logical fits, actual attempts, tokens/currency,
   wall time and unresolved external costs separately, with predeclared limits.
3. Release the full selection history and matched baseline protocol. Classify
   negative or inconclusive results honestly; do not select the reporting objective
   after seeing the results.
4. Admit horizon/factor extensions only with immutable targets, maturity/eligibility,
   training and resource contracts. Evaluate Qlib or other adapters on common inputs
   and costs before making comparative claims. Production integration requires a
   separate proposal and its own authority review.

## Alternatives and compatibility

| Alternative | Benefit | Cost and decision |
| --- | --- | --- |
| Merge orchestration paths into one engine immediately | Reduces visible duplication. | Risks mixing cost units, historical identities, account policies and protected authority. Defer until equivalence is demonstrated. |
| Keep every path independent indefinitely | Minimizes migration work. | Preserves divergent lifecycle handling and increases recovery burden. Retain distinct protocols but share proven small primitives where appropriate. |
| Expand model/factor search first | Broadens executable choices. | Adds selection and operational expense before evidence gates are settled. Gate broader expansion on Phases 0–2. |
| Narrow primitives plus explicit protocols | Permits incremental reuse with reviewable contracts. | Requires a capability matrix and compatibility fixtures. Preferred approach; each extraction remains a separate scoped change. |

Completed experiments remain bound to their original source/dependency identities.
Use their original environment for continuation or replay. A future migration must
be explicit and independently reviewed; neither this RFC nor a source reorganization
authorizes rewriting manifests. Preserve the reference paper and existing authority
barriers. Follow [version-control guidance](version_control.md) for a temporary-branch
documentation PR targeting `main`.

## Verification record and uncertainty

The analysts used Python 3.12.3 and offline fixtures. Delta reports seven passing
causal-visibility assertions and 380 passing chronological split configurations
(terminal sessions 1–20, evaluation sessions 2–20, horizon 5). These execute unchanged
AST-extracted code; they do not validate the whole model, accounting or CLI. Both
analysts report successful lifecycle probe reruns. Inspected regression tests are
source evidence, not tests executed in this delivery.

Archived analysis records supplied to this RFC are
`reports/fb0f8729-3c91-4e6a-b7da-4f84c8014084/atlas_evidence.md` and
`reports/d9b80615-3c79-49d9-9ceb-428f9348df42/delta_research_evidence.md`.
Probe bundles are
`reports/b266dc77-f172-40a8-a8bf-08333a3a2b9b/atlas_probes.tar.gz` and
`reports/dc59e9c5-212a-48f5-b4f3-2f5477c18c78/delta_probe_results.tar.gz`.
These are delivery artifacts, not files installed by a clone. The Manager should
retain accessible evidence with the publication record; private session paths are
not public reproduction instructions.

The analysts and initial independent review lacked Ruff, pytest, Pydantic and
LightGBM; the three required commands initially exited 127. The repair recovered
the unchanged development lock into disposable Python 3.12.3 scratch from cached
wheels, verified every pinned version and imported the key packages successfully.
No full lint, formatting, pytest or CLI pass is claimed yet. Independent review must
record final-candidate checks, including the complete required suite once on the
frozen revision, or exact environment blockers. Follow [validation cost](validation_cost.md)
and retain verbose output outside version control. Historical CI is not current
candidate validation.

Not verified: live provider descendants, real overlapping remote requests or charges,
prospective predictive advantage, historical vendor completeness, calibrated error
control, an actual optional Qlib environment, remote/GPU execution or live brokerage.
No selected remote environment is required for this documentation proposal.

## Review and publication checklist

Before publication, independently verify source claims, local links, duplicate
boundaries, probe mechanisms, the English issue draft and the documentation-only
diff. Retain the resolved analyst discussion and its evidence in the review record.
The Manager owns final deduplication and publication: open only supported issues,
record returned issue URLs, and submit one reviewed RFC PR. Until those receipts
exist, this document is a candidate and submission is pending. No implementation,
merge, deployment or permission change belongs to this delivery.
