# Open-issue audit and delivery ownership

This audit reviewed all 16 open issues against `main@802c62c` on 2026-10-30,
including the merged connected-study work through PR #56. The accompanying PR
adds a reviewed Ridge alternative to that same study entry point. It does not
change protected evaluation, historical runs or the registered C2/C5 contracts.

Issue closure below means the stated deliverable is complete. Planning acceptance,
an executable engineering slice and evidence of research effectiveness are different
outcomes. Residual requirements keep their existing owners; none is silently dropped.
The closure recommendations take effect only after review, required CI and merge.

## Disposition of every open issue

| Issue | Disposition | Delivered evidence and remaining requirement |
| --- | --- | --- |
| [#1](https://github.com/natureofnature/Quant_RSI/issues/1) | Keep open | Forecasting and the stock/news/LLM/factor loop work. Two consecutive executed research-program changes, independent/prospective acceptance and matched-budget research studies remain in R2/R4/R6. |
| [#2](https://github.com/natureofnature/Quant_RSI/issues/2) | Keep open | Production registry promotion remains blocked. Study-level development selection in #56 is not protected production promotion. Requires #4 first. |
| [#3](https://github.com/natureofnature/Quant_RSI/issues/3) | Keep open; first bounded proposer delivered | The registered schema-4 proposer reads authenticated development frontier feedback and emits a versioned `policy.window_max` patch; #17 records one executed selected-descendant advance. The issue's first-version evidence-to-policy mechanism exists, but its linked #1 R2 acceptance of two consecutive executed advances remains unaccepted. Retain this owner with #39 D rather than closing broader program evolution on one advance. |
| [#4](https://github.com/natureofnature/Quant_RSI/issues/4) | Keep open | Confirmatory mode is still blocked as uncalibrated. Development MAE/Sharpe selection supplies no dependence-aware inference or repeated-comparison error budget. |
| [#5](https://github.com/natureofnature/Quant_RSI/issues/5) | Keep open | The newer discovery frontier explores additional candidates, but the issue explicitly concerns the older C2 neighbor policy. Its plateau/restart rule is not implemented. Do not count outer cap expansion or an LLM factor revision as that fix. |
| [#6](https://github.com/natureofnature/Quant_RSI/issues/6) | Close the real-data objective connection | Explicit market preparation feeds retained public daily inputs into the original recursive forecast worker and cumulative authority; #17 records a completed market execution, selected model and replay. Historical availability remains an assumption, its forecast was late, and no alpha or independent validation was established. |
| [#7](https://github.com/natureofnature/Quant_RSI/issues/7) | Keep open | Five-session horizons remain fixed. A weighted multi-horizon task and maturity/mask/allowance rules are not implemented. |
| [#8](https://github.com/natureofnature/Quant_RSI/issues/8) | Keep open | Bounded deterministic multi-leaf expressions exist. Learned intermediate predictors, general causal hierarchies and corresponding evidence/training contracts do not. |
| [#9](https://github.com/natureofnature/Quant_RSI/issues/9) | Keep open | Connected reports include costed target buy-and-hold. The older registered research utility still uses its cash comparator; a predeclared investable-index primary utility is a distinct change. |
| [#10](https://github.com/natureofnature/Quant_RSI/issues/10) | Keep open | Study reports include daily-return Sharpe and #56 permits preselected Sharpe adoption. The requested C2/C5 risk-adjusted utility and explicit risk eligibility gates remain unimplemented. |
| [#11](https://github.com/natureofnature/Quant_RSI/issues/11) | Keep open | Forecast comparison/discovery uses common chronological folds. The older C2/C5 split contracts and separate final holdout are not replaced by those interfaces. |
| [#12](https://github.com/natureofnature/Quant_RSI/issues/12) | Keep open | Complete trial histories and descriptive risk metrics exist. Applicable overfitting diagnostics, dependence-aware uncertainty and predefined cost stress still need a scoped protocol; ordinary Sharpe alone is not a correction for multiple trials. |
| [#24](https://github.com/natureofnature/Quant_RSI/issues/24) | Close as architecture/integration specification | README diagrams, terminology, capability matrix, current AGENTS instructions and executable bounded source/factor slices are maintained. Implementation/scientific residuals remain in #1, #2, #3, #4, #5, #8 and #39, rather than being marked completed by a diagram. |
| [#28](https://github.com/natureofnature/Quant_RSI/issues/28) | Close its selected first A/B/C slice | KO retained-corpus search → versioned evidence-backed title extraction → seven-day count → paired B0/N1 worker, with common masks, allowance/recovery and explicit missingness, is implemented. Current web search is a separate study-context path. General quantity/full-text historical features remain in #8/#39 B, and confirmation in #4. |
| [#39](https://github.com/natureofnature/Quant_RSI/issues/39) | Keep open | V0 and real service replacements are delivered. Ridge supplies the requested simple alternative through the product loop. Broader input/data coverage, two real instrument configurations, remaining real-use/cost acceptance, compatible Qlib workflow comparison and prospective/research studies remain. |
| [#40](https://github.com/natureofnature/Quant_RSI/issues/40) | Close as planning artifact | Positioning, G0–G4 order, comparison requirements and build/integrate/defer boundaries are specified. Delivery and investment decisions remain tracked by #1/#39 and the specific open gaps above; no competitive advantage or completed G1–G4 is claimed. |

## Evidence for completed and partially delivered slices

- **Evidence-derived policy and market objective (#3/#6):**
  [recursive market preparation and policy](recursive_market_exploration.md),
  [forecast research contracts](forecast_recursive_research.md),
  [merged #17 validation record](https://github.com/natureofnature/Quant_RSI/pull/17),
  and [market integration regressions](../tests/test_forecast_recursive_market.py).
  The recorded frozen historical run used ten registered attempts and 1,800 reserved
  worker-seconds, advanced once and then sealed with no further patch. Its original
  missing evidence was restored explicitly before late issuance; the failed command
  and qualified outcome remain in the original record. This is not a new run on
  today's revision or proof of two consecutive advances.
- **Architecture (#24):** [current overview](../README.md#architecture),
  [factor layers](factor_layers.md), [bounded discovery](factor_discovery.md),
  [workflow guide](research_workflow.md) and [AGENTS](../AGENTS.md).
- **First news-factor slice (#28):** [frozen scope and rules](news_factor_slice.md),
  [typed extraction](../src/quant_rsi/factors/news.py),
  [evidence/recovery tests](../tests/test_news_factor.py) and
  [paired worker tests](../tests/test_news_comparison.py). A boolean headline task
  neither extracts earnings quantities nor supplies missing historical coverage.
- **Connected study (#39):** [one-entry-point guide](connected_research_study.md)
  and merged #46–#56. Model choice, factor revision and selected forecasts share
  the existing training masks and accounts; they do not migrate old campaign contracts.

## Next implementation boundary

Use the connected study for small, explicit model/factor experiments. Select the
model, its parameters and adoption metric before observing results. Adding Ridge
does not automatically compare it with LightGBM: separately invoked runs can choose
different LLM factors and must not be called a controlled model benchmark. A fair
comparison must also freeze common data, factors, origins and information budgets.

Prioritize a demonstrated use-case gap from #39 before broadening the old search
or trading contracts. Independent evidence and actual elapsed prospective outcomes
cannot be manufactured by more fixtures or by closing planning RFCs.
