# RFC #119 acceptance evidence

This maps the 29 acceptance items in [RFC #119](https://github.com/natureofnature/Quant_RSI/issues/119)
to the implementation merged by [PR #124](https://github.com/natureofnature/Quant_RSI/pull/124)
and the follow-up in [PR #125](https://github.com/natureofnature/Quant_RSI/pull/125).
The base is `main@907cb20`, whose file tree is identical to reviewed `7cebbc0`
and scan-trigger commit `2addcf7`. The follow-up runtime is `a4431c6`, source
subtree `e07947e8f6bb69d98d0ce222e39bd3a2fbc418a4`. This is an evidence index,
not a claim that historical observations were repeated on the new source.

**The bounded news-information gate now has a passing witness (E7).** The earlier
headline-only observation remains inconclusive (E3). RFC closure also requires
final-revision Python/Qlib checks and independent review in PR #125; neither
this witness nor the historical development results establish predictive value.

## Evidence records and scope

Repository links below resolve against this checkout. Local retained artifacts
are not checked into Git and are not public downloadable datasets. The following
locations identify the original evidence for an operator with access to this
workspace; the linked [implementation record](news_driven_discovery.md#evidence-and-completion-status)
provides the public summary.

| Record | Identity and location | What it establishes |
| --- | --- | --- |
| E1: real two-candidate trajectory | Runtime `2d196573db7811fe3dc1ecbf04f35531b1ed1495`; `/home/nature/Learning_plan/Quant_RSI_evidence/rfc119_delivery/rfc119-ko-real-04/` and sibling `real04-receipt.json` | Real Codex, initial RSS evidence, targeted public-source acquisition/extraction and LightGBM. H1 computes a five-session change in the price/mix-minus-volume gap; actual feedback precedes H2's persistent-gap expression. Two acquired inputs, three fits, four decisions, **zero survey actions**. This does not witness a separate real survey-analysis call. Baseline MAE 2.5450; H1 2.5450; H2 2.5586; baseline retained. Only two quarterly disclosures supply the numerical inputs. H2 consumes H1's Memory but predates explicit parent fields. |
| E2: unsuccessful real attempts | Sibling `rfc119-ko-real-01/`, `rfc119-ko-real-02/`, `rfc119-ko-real-03/` | Baseline-only attempts, extraction/date problems and a decision timeout remain preserved. They do not count as successful discovery witnesses. |
| E3: initial news ablation | `/home/nature/Learning_plan/Quant_RSI_evidence/rfc119_acceptance/news-pair/`; plan `236abe671bdf4325354ae0a5179ea343adfaefa10eb04d0ecc79b7cd897ee1fa`, created 2026-11-04 09:34 UTC | Both real Codex arms sought a pricing/volume gap despite different citations. `assessment.json` records **inconclusive**, including redundant evidence, pretrained knowledge and randomness as alternatives. |
| E4: feedback intervention | `/home/nature/Learning_plan/Quant_RSI_evidence/rfc119_acceptance/feedback-pair/`; plan `9bab8810deda55a78455d830f2291777d490f1d0ac723fe0ea7b04ac82eb86a1`, created 2026-11-04 09:37 UTC | Real Codex decisions under explicitly synthetic feedback meet the declared qualitative action-sensitivity criterion: favorable feedback retains pricing/volume investigation; unfavorable feedback motivates a currency-impact change. One pair is not statistical evidence of causality or predictive gain. |
| E5: actual browser acceptance | `/home/nature/Learning_plan/Quant_RSI_evidence/rfc119_browser/`: `source_identity.json`, `acceptance.json`, screenshots, `real04-browser.json` | Chromium 156 exercised submission, progress, history/settings reuse and report lineage with simulated providers and real LightGBM. Source identity records the actual file hashes and base `2d19657`, not a later review-fix rerun. The separately inspected real report has SHA-256 `78d7ab520162cbb697b10e5b36b6e828b46886941c0d115fda33e7ee9ea21d27`; desktop/mobile overflow and browser error checks passed. |
| E6: reviewed-tree CI | [Python/Qlib CI run 41836459542](https://github.com/natureofnature/Quant_RSI/actions/runs/41836459542), reviewed `7cebbc0`; [CodeQL run 41841617768](https://github.com/natureofnature/Quant_RSI/actions/runs/41841617768), identical-tree `2addcf7` | Required full Python and Qlib checks passed for the merged file tree. Earlier local **2001 passed, 12 skipped** belongs to `2115909`; earlier **14 Qlib passed** belongs to the `2d19657` runtime. Neither historical local count is relabeled as a rerun after review fixes. |
| E7: news-body ablation | Runtime `a4431c6`; `/home/nature/Learning_plan/Quant_RSI_evidence/rfc119_acceptance/news-body-pairs-v2/`; plan `75d54c7eff80805ddd315eb24d33bf6ae2ea719e6967989719932440fb23a0cd` | Two preregistered pairs / four real Codex decisions, no retries or failures. Independent advisor assessment: **pass** against the declared qualitative criterion. Both original arms request volume and price/mix measurements; both removal arms request dividend amounts instead. No acquisition or training was executed in this diagnostic. |
| E8: follow-up regression/browser evidence | Runtime `a4431c6`; sibling `lineage-study/` and `lineage-browser/acceptance.json`; report SHA-256 `f1ab144c57d38b24179a796facfe713a13e5b2f7706e355c9a3115b91ea05f04` | **137 relevant tests passed**, including 18 lineage cases; Ruff passed. Actual Chromium 156 displayed explicit parent references at 1440px and 390px with no page overflow/errors. The study used simulated providers/prices and three actual LightGBM fits. Final full-suite/Qlib results are recorded in PR #125 checks and validation description. |

The paired records retain `plan.json`, both effective inputs, every output,
`results.json` and the explicit assessment. They are separate from E1's authentic
measured Memory. No synthetic intervention feedback entered E1.

## Explicit candidate revisions in the follow-up

The [candidate contract](../src/quant_rsi/research/discovery_factors.py) now accepts
`parent_recipe_refs`: at most eight unique SHA-256 recipe identities describing
research revisions or replacements, not feature dependencies. The semantic recipe
identity still depends only on the expression and input bindings. Empty parent
metadata is omitted from executable definitions so legacy definitions retain
their serialization.

The [outer loop](../src/quant_rsi/research/discovery_loop.py) constructs the visible
candidate index from admitted Memory before each decision. A parent must already
be visible; fabricated, ineligible and same-batch forward references cannot
authorize acquisition or candidate fitting. New revision decisions must declare
parents. Every interpretation preserves the accepted/rejected lineage, including
duplicate numerical proposals that do not retrain. The
[report](../src/quant_rsi/research/discovery_report.py) displays full recipe and
parent references and their status. Older records remain **Unspecified (legacy)**;
the report does not infer parents from chronology or rewrite E1.

The [18 lineage regression cases](../tests/test_discovery_lineage.py) passed
locally: precise parent selection, duplicate proposals with distinct parents,
ineligible references, report compatibility and provider contracts. Invalid
provider proposals, including malformed fields and missing expression/input
shapes, persist as diagnostics that reach the next decision and final report;
genuine transport errors still fail explicitly. The
[connected LightGBM fixture](../tests/test_study_discovery.py) additionally asserts
that H2's parent is H1's semantic recipe identity, that the edge survives into the
actually trained feature definition, and that it appears in the report. Its
providers and data are simulated; this is not a new real-service parent-link
witness. E8 adds actual browser validation of the new report. Full final-revision
validation is tracked on PR #125.

## News-body paired observation (E7)

The new plan was independently reviewed and frozen before calling Codex. It uses
`gpt-6-astra` with Codex CLI `0.166.0`, exactly two pairs in alternating order,
stateless tool-free calls, and identical question, policy, initial definitions,
resource limits and requested date ranges. Memory, prior generated candidates,
survey results and derived context are empty in both arms. Only evidence differs:
the original receives the entire saved extracted Q2 issuer body plus a dividend
headline; the intervention receives that same dividend headline alone. The
harness now retains the fixed resource/time fields instead of silently dropping them.

The article selection rule was fixed before outputs: use the later quarterly
release from real04's first acquisition bundle, without selecting sentences.
The original HTTP-response hash is distinct from the extracted-text SHA-256
`7a7dd567bc101355957e4df7ef83cc14f2e8be7525dfcbcd6acb7be7ec44d405`.
This is retained extracted public-page text, not verified complete HTML, and the
observation is retrospective. The earlier draft was never executed; its status
records that the final plan superseded it after the parent-reference schema froze.

| Pair | Original arm | Removal arm | Independent disposition |
| --- | --- | --- | --- |
| 0 | Cites reported volume growth of 5% and price/mix growth of 2%; requests both operating series; proposes their difference. | Requests declared quarterly dividend per share; proposes dividend divided by price, explicitly without assuming a payout change. | Pass: both requested variables and expression semantics change. |
| 1 | Again requests volume and price/mix growth and proposes their difference, citing the same reported facts. | Again requests dividend per share and proposes quarterly dividend divided by price. | Pass: same substantive evidence-consistent change. |

All four arms completed. Input hashes, source text and output provenance were
independently checked; `advisor-assessment.json` and the harness-bound
`assessment.json` retain the reasoning. `results.json` preserves the harness's
original pending-review state rather than silently rewriting execution records.
The predeclared rule required both pairs to qualify; changing only citations,
wording or uncertainty could not pass. No calls, articles or prompt revisions were
added after observing outputs. E3 remains a separate inconclusive experiment.

This is an exploratory follow-up, not an unseen independent confirmation. Two
pairs cannot rule out stochastic generation or pretrained company knowledge;
removing a whole article also changes topic salience and context length. No data
request was executed in these pairs: some queries reach the length limit, and
earlier source coverage would still need verification. The separate E1 record
establishes acquisition/fitting/feedback execution. E7 establishes only the bounded
information-use witness, not successful acquisition for these new requests,
general news sensitivity, profitability or statistical causal influence.

## Checklist mapping

Item numbers preserve the order of the RFC's acceptance checklist. **Supported**
means the stated implementation/test or execution evidence exists at the assessed
revision; the evidence column identifies whether it is an offline fixture or a
real service/browser observation. It does not imply statistical research efficacy.

| # | Acceptance requirement | Evidence and implementation | Status / boundary |
| --- | --- | --- | --- |
| 1 | Bounded experiment batches return diagnostics; outer loop owns replanning | [Discovery loop](../src/quant_rsi/research/discovery_loop.py), [runner](../src/quant_rsi/research/discovery_engine.py); `test_zero_fit_diagnostic_does_not_exhaust_research` in [loop tests](../tests/test_discovery_loop.py) | Supported; missing inputs can return before any fit. |
| 2 | Persist outcomes/failures before next decision; record consumed Memory; resume handoff | `test_feedback_is_on_disk_before_next_decision_and_resume`, `test_failed_fit_record_survives_before_error` in [loop tests](../tests/test_discovery_loop.py); [connected resume tests](../tests/test_study_discovery.py); E1 | Supported; interpretation records remain separate from experiment facts. |
| 3 | G1 excludes future/revised/untimed evidence and ineligible Memory/tool/cache content from retained prompts | [Record guards](../src/quant_rsi/research/discovery_contracts.py), [source gates](../src/quant_rsi/research/discovery_sources.py); `test_retained_ineligible_article_is_never_supplied_to_llm`, `test_retained_headline_does_not_fetch_or_extract_missing_body` in [source tests](../tests/test_discovery_sources.py); [restoration tests](../tests/test_discovery_source_restore.py) | Supported; retained cross-study retrieval is conservatively disabled, not historically reconstructed. |
| 4 | G2 checks each prediction origin, future feature availability, label maturity and preprocessing ownership | `guard_numerical` and `test_numerical_gate_checks_each_origin_and_training_labels` in [contracts](../src/quant_rsi/research/discovery_contracts.py) / [loop tests](../tests/test_discovery_loop.py); source-bound alignment in [runner](../src/quant_rsi/research/discovery_engine.py); [existing feature](../tests/test_forecast_features.py) and [rolling-window tests](../tests/test_study_walk_forward.py) | Supported through shared guards and existing model paths; the helper's preprocessing rejection test does not introduce a new learned-preprocessing engine. |
| 5 | G3 excludes terminal data from active Memory; guards retain rule/version and reject authority changes | [Guard reports](../src/quant_rsi/research/discovery_contracts.py); `test_temporal_guard_excludes_source_versions_and_terminal_contents` in [loop tests](../tests/test_discovery_loop.py); [publication-feedback regression](../tests/test_discovery_review_regressions.py) | Supported; `schema_version`, gate and rule records accompany decisions. |
| 6 | Executed versioned policy, permitted deterministic patch, parent link and protected-field rejection | `research_policy` / `patch_policy` in [contracts](../src/quant_rsi/research/discovery_contracts.py); `test_policy_patch_changes_consumed_configuration_and_rejects_authority` in [loop tests](../tests/test_discovery_loop.py); [source policy/cache tests](../tests/test_discovery_sources.py) | Supported for explicit deterministic patches; autonomous policy evolution is not delivered. |
| 7 | Preregistered news ablation, cleanup, bounded real pairs and evidence/action criteria | E3 preserves the inconclusive headline-only result; E7 preserves both complete body-removal pairs, plan/input hashes and independent assessment; [harness and fixture checks](../tests/test_discovery_acceptance.py) | Supported by E7 against its declared qualitative criterion. No statistical or predictive-gain claim. |
| 8 | Parameterized temporal DSL computes fitted values with warm-up/future isolation | [Expression evaluator](../src/quant_rsi/factors/expressions.py), [factor tests](../tests/test_discovery_factors.py); E1 H1's lag-5 gap change | Supported; E1 has an actual numerical fitted temporal feature. |
| 9 | Authentic H1 → persisted feedback → H2 across two outer iterations | E1 `discovery/result.json`, checkpoints and `real04-receipt.json`; [connected lineage test](../tests/test_study_discovery.py) | Supported by real services, not a scripted second answer. H2 was worse than baseline. |
| 10 | Separate counterfactual-feedback fixture and real paired observation | [Synthetic-context tests](../tests/test_discovery_acceptance.py), E4 | Supported at the declared qualitative single-pair level; no statistical sensitivity claim. |
| 11 | Four-fit allowance when a new baseline is needed; one shared fit budget | `test_new_inputs_require_new_baseline_and_no_global_ranking`, `test_no_partial_linked_pair_when_budget_insufficient` in [loop tests](../tests/test_discovery_loop.py); [two-group connected test](../tests/test_discovery_history.py) | Supported; baseline and failed-fit charges are retained. |
| 12 | Detailed experimental definitions are retrievable with provenance/status | `survey_catalog` / `retrieve_candidates` in [factor layer](../src/quant_rsi/research/discovery_factors.py); `test_all_survey_records_have_provenance_and_explicit_capabilities` in [factor tests](../tests/test_discovery_factors.py) | Supported for all **32 reviewed definitions**; the **341 raw inventory entries** are not executable factors. |
| 13 | Existing two Qlib adapters plus an additional numerical recipe | [Adapters](../src/quant_rsi/research/discovery_factors.py), [connected runner initialization](../src/quant_rsi/research/discovery_engine.py); `test_existing_and_new_survey_adapters_execute_without_upstream_imports` in [factor tests](../tests/test_discovery_factors.py) | Supported numerical execution for ROC, MA and new KMID; this is three surveyed recipes, not 32 executable recipes. |
| 14 | New agent-created DSL expression binds inputs and fits without hand-added catalog entry | E1; [decision contracts](../src/quant_rsi/research/discovery_sources.py), [factor binding](../src/quant_rsi/research/discovery_factors.py), [connected tests](../tests/test_study_discovery.py) | Supported; price/mix and volume inputs were acquired, then consumed by the fitted candidate. |
| 15 | Semantic novelty and real two-candidate feedback trajectory | E1; `test_recipe_identity_ignores_names_but_tracks_operator_and_input_changes` in [factor tests](../tests/test_discovery_factors.py), [normalized duplicate tests](../tests/test_discovery_loop.py) | Supported for expressions distinct from the starting catalog; no claim of novelty relative to all quantitative literature. |
| 16 | Surveyed/derived leaves compose; unavailable/noncausal recipes receive diagnostics | [Factor binding and capability catalog](../src/quant_rsi/research/discovery_factors.py); [reference-only, missing-input and operator tests](../tests/test_discovery_factors.py) | Supported within implemented operators/adapters; arbitrary Python/plugin generation is outside this slice. |
| 17 | Numerical, parameter, missingness, temporal and identity compatibility tests | [Factor tests](../tests/test_discovery_factors.py), including `test_v1_serialization_and_identity_are_unchanged`; [source alignment tests](../tests/test_discovery_sources.py) | Supported offline; existing serialized recipe identities remain valid. |
| 18 | Entry input is a symbol and research question, without prescribed factor answer | [Existing study integration](../src/quant_rsi/research/study_discovery.py), [CLI examples](news_driven_discovery.md); E1 retained `config.json` / decision contexts | Supported; acceptance does not require preselecting a factor ID in the user request. |
| 19 | Missing-input fixture calls acquisition, computes features and runs actual model | `test_discovery_acquired_lineage_real_fits_and_frozen_rows` in [connected tests](../tests/test_study_discovery.py) | Supported with simulated providers/data and real LightGBM; establishes wiring only. |
| 20 | Same real candidate links news, missing-input acquisition, actual feature fit and feedback; explicit parent references across revisions | E1 `discovery/result.json`, source bundles, fit receipts and `real04-receipt.json`; follow-up [lineage tests](../tests/test_discovery_lineage.py), [connected trained-definition assertion](../tests/test_study_discovery.py), [report projection](../src/quant_rsi/research/discovery_report.py) | E1 supports the same-candidate real execution path and consumed H1 feedback, but has no explicit parent field. The follow-up supplies explicit parent contracts/persistence and a connected real-model fixture; it does not retrofit or relabel E1. |
| 21 | Unsupported/unavailable candidates remain visible without price substitution | `test_missing_input_still_reports_original_baseline` in [loop tests](../tests/test_discovery_loop.py), `test_discovery_missing_candidates_finishes_baseline` in [connected tests](../tests/test_study_discovery.py); E2 | Supported; a baseline-only outcome is explicit and is not called discovery success. |
| 22 | Separate discovery/dedup/screen/acquisition/execution/fit counts | [Loop accounting](../src/quant_rsi/research/discovery_loop.py), [report](../src/quant_rsi/research/discovery_report.py), [live-state tests](../tests/test_discovery_workbench.py); E1 | Supported; action and fit limits are separate, not a two-candidate discovery cap. |
| 23 | Replaceable analysis/acquisition adapters and supported new expressions without orchestrator edits | [Sources](../src/quant_rsi/research/discovery_sources.py), [study adapter](../src/quant_rsi/research/study_discovery.py), [runner](../src/quant_rsi/research/discovery_engine.py); [source tests](../tests/test_discovery_sources.py) and E1 | Supported within the documented DSL and data-request contracts. |
| 24 | Controlled second run receives eligible prior experiment/acquisition evidence | [Prior-memory projection](../src/quant_rsi/research/study_discovery.py); [measured-feedback regression](../tests/test_discovery_review_regressions.py), [incomparable-history integration](../tests/test_discovery_history.py) | Supported in retrospective mode, including runs with no selected forecast. Retained cross-study reuse remains unavailable. |
| 25 | Frozen samples survive acquisition; changed domains need matched baselines/shared budget | [Runner](../src/quant_rsi/research/discovery_engine.py), [loop linked-experiment tests](../tests/test_discovery_loop.py), [connected frozen-row test](../tests/test_study_discovery.py) | Supported; incompatible groups have no automatic global winner. |
| 26 | Bounded invalid/no-op/missing paths; resume preserves work and identities | [Loop regressions](../tests/test_discovery_loop.py), [connected fit-reuse tests](../tests/test_study_discovery.py), [article restoration](../tests/test_discovery_source_restore.py) | Supported; completed fits and surveyed article bodies are reused; failed calls retain charges. |
| 27 | Future/late/immature/terminal boundaries survive linked experiments | [Guards](../src/quant_rsi/research/discovery_contracts.py), [source admission](../src/quant_rsi/research/discovery_sources.py), [runner alignment](../src/quant_rsi/research/discovery_engine.py); [loop](../tests/test_discovery_loop.py), [source](../tests/test_discovery_sources.py) and [publication isolation tests](../tests/test_discovery_review_regressions.py) | Supported by shared cutoff/label checks; an experiment ID does not grant new information access. |
| 28 | UI/report shows lineage and separates incompatible groups; publication follows selection rule | [Report](../src/quant_rsi/research/discovery_report.py), [workbench tests](../tests/test_discovery_workbench.py), [unselected history test](../tests/test_discovery_history.py); E5 and E8 | Supported; absent global selection produces research history without a forecast publication. E8 checks the new revision table; E5 retains its earlier workbench identity. |
| 29 | Required Python/Qlib validation and actual browser acceptance | E6 for the original merged tree; E5/E8 for browser interaction; final follow-up checks and validation evidence in [PR #125](https://github.com/natureofnature/Quant_RSI/pull/125) | Historical and follow-up revisions remain separate. Final-revision required checks must pass before merge/closure. |

## Completion decision

E7 satisfies the remaining bounded news-information observation. Close #119 only
after PR #125's final-revision required checks and independent review pass, with
this checklist mapping retained. Preserve E3 and all prior failures. The decision
concerns the specified discovery capability and interfaces, not profitable forecasts.

The current execution evidence already supports the two requested capabilities:
clear module/loop ownership and news-to-new-candidate-to-acquired-data-to-fit
execution. Broad business-series coverage, autonomous operator implementation,
retained-mode cross-study Memory, policy self-evolution and predictive superiority
remain separate future work. They should not be silently added as requirements
to finish this bounded delivery, or advertised as completed by it.
