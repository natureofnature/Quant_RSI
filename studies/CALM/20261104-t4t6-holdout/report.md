# CALM · Connected research study

Connected research study with public_daily_prices prices and real LightGBM training at a 6-session horizon. Hypotheses: cursor_handoff; news: not_configured; relationships: not_configured. All attempts use the same retrospective development window; no independent gain is established. Inspect source coverage and the separate unobserved latest forecast.

## Research question

Which combinations improve the t+4, t+5 and t+6 average-close forecast when 2026-04-27 through 2026-07-30 is held out as the test?

## Selected research hypothesis

Round 42 uses 17 complete inputs plus research.log_return_volatility_30. The label is the log ratio of the mean close on sessions t+4, t+5 and t+6 to today's close. Sessions from 2026-04-27 through 2026-07-30 are held out for the test and are not used to choose factors.

## Selected question map

Claims, rationale and limitations are the researcher's assessment. Execution is checked against this attempt's recorded model inputs. Inputs fitted does not mean the claim was confirmed or caused returns.

### These inputs are finite on the training and development sessions before 2026-04-27. This round changes which of them enter the t+4, t+5 and t+6 average-close forecast.

**Execution:** inputs fitted

**Bound factors:** bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel

**Rationale (LLM):** Round 42 rotates the complete feed, energy, rate, drought, calendar, and filing inputs. Size 17.

**Limitations (LLM):**

- Egg-price and NASS series still have empty sessions.
- A lower development error on this window is not confirmation until the reserved sessions are scored.

## Scope of the answer

The question guides evidence discovery and factor hypotheses. The results below evaluate the selected factors under the configured forecast objective; they do not establish a causal answer or measure business variables absent from the inputs.

## Conclusion

CALM: completed 50 fits and 49 revision attempts at a 6-session horizon; stopped: round_limit. Final trial evaluated over 60 development observations. Price MAE: initial 2.6841, revised 2.4904, unchanged-price baseline 2.3229. The revision decreased MAE by 0.1938. Selected round 42 by strict mae improvement on the same development window; ties and missing metrics retain the incumbent. Selected round 42: price MAE 2.4003, net return -0.15%, Sharpe 0.0375. Net strategy return: initial -15.31%, revised -2.35%, buy-and-hold -3.99%. Annualized Sharpe: initial -2.5575, revised -0.6860, buy-and-hold -0.3725 (65 daily returns; 252 sessions/year; risk-free rate 0%). Short-sample Sharpe is descriptive, not a reliable long-term estimate. These are retrospective development results after factor selection and feedback, not independent accuracy or evidence of improvement. Strategy results are hypothetical and include explicit costs. The latest forecast, when present, has no observed outcome. Reserved terminal evaluation (60 observations), same fitted round 42, no refit: price MAE 2.1046, unchanged-price MAE 2.0883; net return 9.73%, Sharpe 1.7468, buy-and-hold 10.46%. Reserved from research feedback and selection, but retrospective. Current news and pretrained LLM knowledge are not historically isolated. This is not prospective validation or calibrated evidence of predictive gain. Financial-feature diagnostic: 1 additional fits outside the research fit budget; does not alter selection.

## Attempt selection

Selected round 42 by strict mae improvement on the same development window; ties and missing metrics retain the incumbent. Latest forecast uses the selected fitted model; no refit. This is development selection after feedback, not independent validation.

| Preselected metric | Initial | Selected score | Selected attempt | Selected factors |
| --- | --- | --- | --- | --- |
| mae (lower is better) | 2.6841 | 2.4003 | round_42 | research.log_return_volatility_30, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel |

## What ran

50 research model fits and a fixed development window. Results include failed or missing source coverage. No orders were sent.

## Research rounds

Stopped: round_limit. Fit allowance: 50; selected round: 42. Every trial uses development data only.

| Round | Factors | Price MAE | Net return | Sharpe | Accepted |
| --- | --- | --- | --- | --- | --- |
| 1 | research.log_return_volatility_10, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr | 2.6841 | -15.3139% | -2.5575 | True |
| 2 | research.log_return_volatility_20, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought | 2.6265 | -1.7240% | -0.5805 | True |
| 3 | research.log_return_volatility_30, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter | 2.6708 | -4.6227% | -0.9859 | False |
| 4 | research.mean_return_5, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving | 2.5658 | 4.6947% | 0.8639 | True |
| 5 | research.mean_return_10, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy | 2.4741 | -8.2237% | -1.0727 | True |
| 6 | research.mean_return_20, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin | 2.6941 | -3.1862% | -1.2159 | False |
| 7 | research.momentum_5, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy | 2.7785 | -3.1862% | -1.2159 | False |
| 8 | research.momentum_10, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy | 2.7185 | -3.1862% | -1.2159 | False |
| 9 | research.momentum_20, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy | 2.9162 | -3.1862% | -1.2159 | False |
| 10 | research.qlib_roc_v1_10, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy | 2.6739 | -3.1862% | -1.2159 | False |
| 11 | research.qlib_roc_v1_20, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum | 2.4775 | 2.6823% | 1.8718 | False |
| 12 | research.qlib_ma_v1_10, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn | 2.4211 | 0.0000% | N/A | True |
| 13 | research.qlib_ma_v1_20, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal | 2.4654 | -0.5370% | -0.7428 | False |
| 14 | research.log_return_volatility_10, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr | 2.7077 | -5.7656% | -3.4213 | False |
| 15 | research.log_return_volatility_20, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought | 2.7150 | 0.0000% | N/A | False |
| 16 | research.log_return_volatility_30, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter | 2.9928 | 0.0000% | N/A | False |
| 17 | research.mean_return_5, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving | 2.4483 | 3.8762% | 0.7800 | False |
| 18 | research.mean_return_10, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy | 2.4354 | -5.3912% | -1.0272 | False |
| 19 | research.mean_return_20, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin | 2.7115 | -3.1862% | -1.2159 | False |
| 20 | research.momentum_5, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy | 2.5394 | -3.2956% | -0.6397 | False |
| 21 | research.momentum_10, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy | 2.5564 | -3.1862% | -1.2159 | False |
| 22 | research.momentum_20, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy | 2.6368 | -0.7211% | -0.4567 | False |
| 23 | research.qlib_roc_v1_10, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy | 2.7321 | -4.1382% | -2.5346 | False |
| 24 | research.qlib_roc_v1_20, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum | 2.9120 | -3.1862% | -1.2159 | False |
| 25 | research.qlib_ma_v1_10, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn | 2.6365 | -3.1862% | -1.2159 | False |
| 26 | research.qlib_ma_v1_20, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal | 2.6988 | 0.0000% | N/A | False |
| 27 | research.log_return_volatility_10, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel | 2.5553 | 2.5953% | 0.6120 | False |
| 28 | research.log_return_volatility_20, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr | 2.5068 | 0.8555% | 0.2635 | False |
| 29 | research.log_return_volatility_30, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought | 2.8408 | 0.0000% | N/A | False |
| 30 | research.mean_return_5, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter | 2.8740 | 0.0000% | N/A | False |
| 31 | research.mean_return_10, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving | 2.4820 | -2.1787% | -0.6203 | False |
| 32 | research.mean_return_20, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy | 2.7052 | -3.1862% | -1.2159 | False |
| 33 | research.momentum_5, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin | 2.4675 | -1.8493% | -0.3307 | False |
| 34 | research.momentum_10, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy | 2.4622 | -1.8493% | -0.3307 | False |
| 35 | research.momentum_20, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy | 2.6684 | -1.7240% | -0.5805 | False |
| 36 | research.qlib_roc_v1_10, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy | 2.5797 | 0.0000% | N/A | False |
| 37 | research.qlib_roc_v1_20, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy | 2.6518 | 3.0415% | 2.3273 | False |
| 38 | research.qlib_ma_v1_10, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum | 2.6181 | -0.7130% | -0.2728 | False |
| 39 | research.qlib_ma_v1_20, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn | 2.5914 | 0.0000% | N/A | False |
| 40 | research.log_return_volatility_10, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas | 2.4985 | -1.8493% | -0.3307 | False |
| 41 | research.log_return_volatility_20, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn | 2.4167 | 2.0299% | 0.5117 | True |
| 42 | research.log_return_volatility_30, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel | 2.4003 | -0.1490% | 0.0375 | True |
| 43 | research.mean_return_5, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr | 2.4483 | 3.8762% | 0.7800 | False |
| 44 | research.mean_return_10, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought | 2.4820 | -2.1787% | -0.6203 | False |
| 45 | research.mean_return_20, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter | 2.7052 | -3.1862% | -1.2159 | False |
| 46 | research.momentum_5, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving | 2.4675 | -1.8493% | -0.3307 | False |
| 47 | research.momentum_10, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy | 2.4622 | -1.8493% | -0.3307 | False |
| 48 | research.momentum_20, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin | 2.5110 | -3.1862% | -1.2159 | False |
| 49 | research.qlib_roc_v1_10, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy | 2.5362 | -3.1862% | -1.2159 | False |
| 50 | research.qlib_roc_v1_20, financial.operating_cashflow_yoy, financial.free_cashflow_yoy, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy | 2.4904 | -2.3528% | -0.6860 | False |

## Continuing the previous research

History at this run's start (latest 50 receipts). Only the latest 20 settled outcomes are supplied as outcome feedback; pending forecasts are not errors. Providing feedback does not prove the LLM used it. Factor changes compare this run's selected model with the previous run's selected model, not their predictive quality.

## First run

No previous outcome feedback in this history.

## Forecast history and later observations

Individual forecast receipts, not a preregistered cohort or independent confirmation. Later public collections are matched by provider, instrument and frozen calendar; they are not receipt-prefix extensions of a market registry. Strategy metrics replay the frozen signal on later received bars with explicit costs; they are hypothetical, not durable paper orders or broker fills.

| Issued | Target | Round | Status | Predicted | Actual | Price error | Net return | Sharpe | Buy-and-hold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-04T06:38:30.307578+00:00 | N/A | 42 | unavailable | N/A | N/A | N/A | N/A | N/A | N/A |

## Instrument factor profile

Ownership and exact source bindings are retained. Availability is fixed on training/development rows; inclusion is not selection. Unsupported business/text sources remain pending, never target-price substitutes.

| Factor | Definition | Scope / owner | Source / inputs | Availability |
| --- | --- | --- | --- | --- |
| bound.price_momentum | equity.price_momentum_5 | shared / equities | target_daily_prices / CALM / close | complete_common_sample |
| bound.announced_events | equity.announced_events | shared / equities | issuer_announcements / CALM / text | text_extraction_not_executable_in_this_profile |
| bound.egg_price | calm.egg_price_momentum_5 | instrument / US/CALM | usda_ams_egg_reports / US shell eggs / wholesale_price | retained_series_not_supplied |
| bound.confirmed_layer_culls | calm.confirmed_layer_culls | instrument / US/CALM | confirmed_hpai_reports / US layers / text | text_extraction_not_executable_in_this_profile |
| bound.table_egg_layers | calm.table_egg_layers_momentum_20 | instrument / US/CALM | nass_chickens_eggs / US table-egg layers / inventory | missing_bound_input_on_fixed_sample |
| bound.egg_type_pullets | calm.egg_type_pullets_momentum_20 | instrument / US/CALM | nass_chickens_eggs / US egg-type pullets / placements | missing_bound_input_on_fixed_sample |
| bound.egg_type_chicks_hatched | calm.egg_type_chicks_hatched_momentum_20 | instrument / US/CALM | nass_chickens_eggs / US egg-type chicks hatched / hatched | missing_bound_input_on_fixed_sample |
| bound.corn | calm.corn_momentum_5 | instrument / US/CALM | egg_catalog / US / corn | complete_common_sample |
| bound.meal | calm.meal_momentum_5 | instrument / US/CALM | egg_catalog / US / meal | complete_common_sample |
| bound.feed_index | calm.feed_index_momentum_5 | instrument / US/CALM | egg_catalog / US / feed_index | complete_common_sample |
| bound.gas | calm.gas_momentum_5 | instrument / US/CALM | egg_catalog / US / gas | complete_common_sample |
| bound.mxn | calm.mxn_momentum_5 | instrument / US/CALM | egg_catalog / US / mxn | complete_common_sample |
| bound.diesel | calm.diesel_momentum_5 | instrument / US/CALM | egg_catalog / US / diesel | complete_common_sample |
| bound.effr | calm.effr_momentum_5 | instrument / US/CALM | egg_catalog / US / effr | complete_common_sample |
| bound.drought | calm.drought_momentum_20 | instrument / US/CALM | egg_catalog / US / drought | complete_common_sample |
| bound.easter | calm.easter_momentum_20 | instrument / US/CALM | egg_catalog / US / easter | complete_common_sample |
| bound.thanksgiving | calm.thanksgiving_momentum_20 | instrument / US/CALM | egg_catalog / US / thanksgiving | complete_common_sample |

## Reserved terminal evaluation

Reserved from research feedback and selection, but retrospective. Current news and pretrained LLM knowledge are not historically isolated. This is not prospective validation or calibrated evidence of predictive gain.

## Frozen selection

Round 42; 60 reserved observations; status: completed. The same fitted model is reused without refitting.

| Terminal forecast | Price MAE | Price RMSE | Observations |
| --- | --- | --- | --- |
| Selected model | 2.1046 | 2.6213 | 60 |
| unchanged_price | 2.0883 | 2.5779 | 60 |
| training_label_mean | 2.1533 | 2.6952 | 60 |

| Terminal account | Net return | Sharpe | Max drawdown | Daily returns |
| --- | --- | --- | --- | --- |
| strategy | 9.7274% | 1.7468 | -6.9171% | 65 |
| buy_and_hold | 10.4596% | 1.2909 | -6.8235% | 65 |

## Terminal account period

{'start': '2026-04-27', 'end': '2026-07-30'}. Same account costs and policy as development, independently initialized. Sharpe is annualized using 252 sessions/year and zero risk-free rate; these short samples are descriptive. Do not join these NAV curves.

| Terminal origin | Target | Observed | Predicted | Unchanged price |
| --- | --- | --- | --- | --- |
| 2026-04-27 | 2026-05-05 | 75.9567 | 76.7471 | 77.7800 |
| 2026-04-28 | 2026-05-06 | 75.8900 | 77.2448 | 77.2800 |
| 2026-04-29 | 2026-05-07 | 76.2200 | 75.0358 | 75.0700 |
| 2026-04-30 | 2026-05-08 | 75.9600 | 77.2248 | 77.2600 |
| 2026-05-01 | 2026-05-11 | 75.7600 | 76.3493 | 76.2500 |
| 2026-05-04 | 2026-05-12 | 76.8067 | 74.3162 | 74.7500 |
| 2026-05-05 | 2026-05-13 | 77.9467 | 77.6513 | 76.8700 |
| 2026-05-06 | 2026-05-14 | 78.8000 | 76.6882 | 76.0500 |
| 2026-05-07 | 2026-05-15 | 78.1333 | 76.3756 | 75.7400 |
| 2026-05-08 | 2026-05-18 | 77.5000 | 76.7285 | 76.0900 |
| 2026-05-11 | 2026-05-19 | 77.6867 | 76.0831 | 75.4500 |
| 2026-05-12 | 2026-05-20 | 78.1467 | 79.0099 | 78.8800 |
| 2026-05-13 | 2026-05-21 | 77.9167 | 78.5500 | 79.5100 |
| 2026-05-14 | 2026-05-22 | 77.3967 | 78.1385 | 78.0100 |
| 2026-05-15 | 2026-05-26 | 76.6433 | 77.6290 | 76.8800 |
| 2026-05-18 | 2026-05-27 | 76.3467 | 77.3127 | 77.6100 |
| 2026-05-19 | 2026-05-28 | 76.1733 | 78.8787 | 78.5700 |
| 2026-05-20 | 2026-05-29 | 75.7467 | 78.4296 | 78.2600 |
| 2026-05-21 | 2026-06-01 | 75.4500 | 76.8769 | 76.9200 |
| 2026-05-22 | 2026-06-02 | 74.7967 | 77.1769 | 77.0100 |
| 2026-05-26 | 2026-06-03 | 75.0067 | 76.1647 | 76.0000 |
| 2026-05-27 | 2026-06-04 | 74.7800 | 75.9872 | 76.0300 |
| 2026-05-28 | 2026-06-05 | 75.1533 | 76.7906 | 76.4900 |
| 2026-05-29 | 2026-06-08 | 75.4833 | 75.0136 | 74.7200 |
| 2026-06-01 | 2026-06-09 | 76.5600 | 75.4353 | 75.1400 |
| 2026-06-02 | 2026-06-10 | 77.7000 | 74.8229 | 74.5300 |
| 2026-06-03 | 2026-06-11 | 78.8767 | 76.0946 | 75.3500 |
| 2026-06-04 | 2026-06-12 | 79.0133 | 74.5964 | 74.4600 |
| 2026-06-05 | 2026-06-15 | 79.1800 | 75.7886 | 75.6500 |
| 2026-06-08 | 2026-06-16 | 78.9000 | 76.3865 | 76.3400 |
| 2026-06-09 | 2026-06-17 | 78.9667 | 76.7399 | 77.6900 |
| 2026-06-10 | 2026-06-18 | 78.3533 | 78.1030 | 79.0700 |
| 2026-06-11 | 2026-06-22 | 77.3333 | 78.6688 | 79.8700 |
| 2026-06-12 | 2026-06-23 | 77.7567 | 77.4500 | 78.1000 |
| 2026-06-15 | 2026-06-24 | 78.1900 | 78.9078 | 79.5700 |
| 2026-06-16 | 2026-06-25 | 78.9467 | 79.4608 | 79.0300 |
| 2026-06-17 | 2026-06-26 | 79.1667 | 78.9866 | 78.3000 |
| 2026-06-18 | 2026-06-29 | 79.2700 | 77.7300 | 77.7300 |
| 2026-06-22 | 2026-06-30 | 80.0433 | 76.1399 | 75.9700 |
| 2026-06-23 | 2026-07-01 | 80.1167 | 79.7153 | 79.5700 |
| 2026-06-24 | 2026-07-02 | 81.6200 | 79.1743 | 79.0300 |
| 2026-06-25 | 2026-07-06 | 83.1633 | 78.5207 | 78.2400 |
| 2026-06-26 | 2026-07-07 | 84.8167 | 80.0239 | 80.2300 |
| 2026-06-29 | 2026-07-08 | 85.2267 | 78.4650 | 79.3400 |
| 2026-06-30 | 2026-07-09 | 85.2800 | 81.2332 | 80.5600 |
| 2026-07-01 | 2026-07-10 | 86.2567 | 80.7664 | 80.4500 |
| 2026-07-02 | 2026-07-13 | 85.7000 | 82.6894 | 83.8500 |
| 2026-07-06 | 2026-07-14 | 84.9633 | 83.4790 | 85.1900 |
| 2026-07-07 | 2026-07-15 | 83.9700 | 83.3546 | 85.4100 |
| 2026-07-08 | 2026-07-16 | 85.5933 | 83.2695 | 85.0800 |
| 2026-07-09 | 2026-07-17 | 87.3967 | 84.4088 | 85.3500 |
| 2026-07-10 | 2026-07-20 | 88.4167 | 87.3025 | 88.3400 |
| 2026-07-13 | 2026-07-21 | 88.0733 | 83.3777 | 83.4100 |
| 2026-07-14 | 2026-07-22 | 87.8433 | 83.8434 | 83.1400 |
| 2026-07-15 | 2026-07-23 | 87.7500 | 85.7185 | 85.3600 |
| 2026-07-16 | 2026-07-24 | 88.7733 | 87.0528 | 88.2800 |
| 2026-07-17 | 2026-07-27 | 89.0600 | 89.2627 | 88.5500 |
| 2026-07-20 | 2026-07-28 | 90.7500 | 87.1908 | 88.4200 |
| 2026-07-21 | 2026-07-29 | 91.8100 | 85.8964 | 87.2500 |
| 2026-07-22 | 2026-07-30 | 91.2633 | 87.3796 | 87.8600 |

| Component | Provider |
| --- | --- |
| Relationships | not_configured |
| News | not_configured |
| Hypotheses | cursor_handoff |
| Prices | public_daily_prices |
| Model | lightgbm |
| Execution | historical_development_hypothetical_account |

## Model configuration

Chosen before research and shared by both attempts. Preprocessing is fitted only on training rows; neither model choice nor hyperparameters are searched in this run.

| Setting | Value |
| --- | --- |
| schema_version | 1 |
| name | "lightgbm" |
| parameters | {"objective": "regression", "verbosity": -1, "num_threads": 1, "deterministic": true, "force_col_wise": true, "num_leaves": 7, "min_data_in_leaf": 5, "learning_rate": 0.05, "feature_pre_filter": false, "seed": 7, "num_boost_round": 24} |
| preprocessing | "none" |
| objective | "regression" |
| training_policy | "fixed_mature_label_window_shared_between_attempts" |
| target | "log_mean_close_t_plus_4_5_6_over_close" |

## News and relationships

News was supplied as hypothesis context; no news column was fitted.

## Round 1: news → hypothesis → fitted inputs

Round 1 uses 8 complete inputs plus research.log_return_volatility_10. The label is the log ratio of the mean close on sessions t+4, t+5 and t+6 to today's close. Sessions from 2026-04-27 through 2026-07-30 are held out for the test and are not used to choose factors. Citations support the LLM's hypothesis, not a measured contribution to prediction. Current article text is research context only. Fitted news features use retained issuer snapshots available at each historical cutoff; unavailable history is not a zero signal.

| Cited source ID | Title | Role |
| --- | --- | --- |

| Factor | Input | Execution | Reason |
| --- | --- | --- | --- |
| research.log_return_volatility_10 | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.price_momentum | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.corn | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.meal | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.feed_index | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.gas | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.mxn | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.diesel | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.effr | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.announced_events | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.egg_price | positive_price_series | unavailable | retained_series_not_supplied |
| bound.confirmed_layer_culls | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.table_egg_layers | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_pullets | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_chicks_hatched | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |

### These inputs are finite on the training and development sessions before 2026-04-27. This round changes which of them enter the t+4, t+5 and t+6 average-close forecast.

**Execution:** inputs fitted

**Bound factors:** bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr

**Rationale (LLM):** Round 1 rotates the complete feed, energy, rate, drought, calendar, and filing inputs. Size 8.

**Limitations (LLM):**

- Egg-price and NASS series still have empty sessions.
- A lower development error on this window is not confirmation until the reserved sessions are scored.

## Round 2: news → hypothesis → fitted inputs

Round 2 uses 8 complete inputs plus research.log_return_volatility_20. The label is the log ratio of the mean close on sessions t+4, t+5 and t+6 to today's close. Sessions from 2026-04-27 through 2026-07-30 are held out for the test and are not used to choose factors. Citations support the LLM's hypothesis, not a measured contribution to prediction. Current article text is research context only. Fitted news features use retained issuer snapshots available at each historical cutoff; unavailable history is not a zero signal.

| Cited source ID | Title | Role |
| --- | --- | --- |

| Factor | Input | Execution | Reason |
| --- | --- | --- | --- |
| research.log_return_volatility_20 | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.corn | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.meal | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.feed_index | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.gas | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.mxn | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.diesel | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.effr | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.drought | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.announced_events | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.egg_price | positive_price_series | unavailable | retained_series_not_supplied |
| bound.confirmed_layer_culls | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.table_egg_layers | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_pullets | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_chicks_hatched | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |

### These inputs are finite on the training and development sessions before 2026-04-27. This round changes which of them enter the t+4, t+5 and t+6 average-close forecast.

**Execution:** inputs fitted

**Bound factors:** bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought

**Rationale (LLM):** Round 2 rotates the complete feed, energy, rate, drought, calendar, and filing inputs. Size 8.

**Limitations (LLM):**

- Egg-price and NASS series still have empty sessions.
- A lower development error on this window is not confirmation until the reserved sessions are scored.

## Round 3: news → hypothesis → fitted inputs

Round 3 uses 8 complete inputs plus research.log_return_volatility_30. The label is the log ratio of the mean close on sessions t+4, t+5 and t+6 to today's close. Sessions from 2026-04-27 through 2026-07-30 are held out for the test and are not used to choose factors. Citations support the LLM's hypothesis, not a measured contribution to prediction. Current article text is research context only. Fitted news features use retained issuer snapshots available at each historical cutoff; unavailable history is not a zero signal.

| Cited source ID | Title | Role |
| --- | --- | --- |

| Factor | Input | Execution | Reason |
| --- | --- | --- | --- |
| research.log_return_volatility_30 | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.meal | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.feed_index | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.gas | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.mxn | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.diesel | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.effr | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.drought | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.easter | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.announced_events | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.egg_price | positive_price_series | unavailable | retained_series_not_supplied |
| bound.confirmed_layer_culls | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.table_egg_layers | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_pullets | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_chicks_hatched | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |

### These inputs are finite on the training and development sessions before 2026-04-27. This round changes which of them enter the t+4, t+5 and t+6 average-close forecast.

**Execution:** inputs fitted

**Bound factors:** bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter

**Rationale (LLM):** Round 3 rotates the complete feed, energy, rate, drought, calendar, and filing inputs. Size 8.

**Limitations (LLM):**

- Egg-price and NASS series still have empty sessions.
- A lower development error on this window is not confirmation until the reserved sessions are scored.

## Round 4: news → hypothesis → fitted inputs

Round 4 uses 8 complete inputs plus research.mean_return_5. The label is the log ratio of the mean close on sessions t+4, t+5 and t+6 to today's close. Sessions from 2026-04-27 through 2026-07-30 are held out for the test and are not used to choose factors. Citations support the LLM's hypothesis, not a measured contribution to prediction. Current article text is research context only. Fitted news features use retained issuer snapshots available at each historical cutoff; unavailable history is not a zero signal.

| Cited source ID | Title | Role |
| --- | --- | --- |

| Factor | Input | Execution | Reason |
| --- | --- | --- | --- |
| research.mean_return_5 | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.feed_index | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.gas | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.mxn | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.diesel | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.effr | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.drought | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.easter | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.thanksgiving | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.announced_events | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.egg_price | positive_price_series | unavailable | retained_series_not_supplied |
| bound.confirmed_layer_culls | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.table_egg_layers | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_pullets | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_chicks_hatched | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |

### These inputs are finite on the training and development sessions before 2026-04-27. This round changes which of them enter the t+4, t+5 and t+6 average-close forecast.

**Execution:** inputs fitted

**Bound factors:** bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving

**Rationale (LLM):** Round 4 rotates the complete feed, energy, rate, drought, calendar, and filing inputs. Size 8.

**Limitations (LLM):**

- Egg-price and NASS series still have empty sessions.
- A lower development error on this window is not confirmation until the reserved sessions are scored.

## Round 5: news → hypothesis → fitted inputs

Round 5 uses 8 complete inputs plus research.mean_return_10. The label is the log ratio of the mean close on sessions t+4, t+5 and t+6 to today's close. Sessions from 2026-04-27 through 2026-07-30 are held out for the test and are not used to choose factors. Citations support the LLM's hypothesis, not a measured contribution to prediction. Current article text is research context only. Fitted news features use retained issuer snapshots available at each historical cutoff; unavailable history is not a zero signal.

| Cited source ID | Title | Role |
| --- | --- | --- |

| Factor | Input | Execution | Reason |
| --- | --- | --- | --- |
| research.mean_return_10 | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.gas | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.mxn | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.diesel | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.effr | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.drought | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.easter | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.thanksgiving | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| financial.revenue_yoy | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| bound.announced_events | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.egg_price | positive_price_series | unavailable | retained_series_not_supplied |
| bound.confirmed_layer_culls | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.table_egg_layers | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_pullets | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_chicks_hatched | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |

### These inputs are finite on the training and development sessions before 2026-04-27. This round changes which of them enter the t+4, t+5 and t+6 average-close forecast.

**Execution:** inputs fitted

**Bound factors:** bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy

**Rationale (LLM):** Round 5 rotates the complete feed, energy, rate, drought, calendar, and filing inputs. Size 8.

**Limitations (LLM):**

- Egg-price and NASS series still have empty sessions.
- A lower development error on this window is not confirmation until the reserved sessions are scored.

## Round 6: news → hypothesis → fitted inputs

Round 6 uses 8 complete inputs plus research.mean_return_20. The label is the log ratio of the mean close on sessions t+4, t+5 and t+6 to today's close. Sessions from 2026-04-27 through 2026-07-30 are held out for the test and are not used to choose factors. Citations support the LLM's hypothesis, not a measured contribution to prediction. Current article text is research context only. Fitted news features use retained issuer snapshots available at each historical cutoff; unavailable history is not a zero signal.

| Cited source ID | Title | Role |
| --- | --- | --- |

| Factor | Input | Execution | Reason |
| --- | --- | --- | --- |
| research.mean_return_20 | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.mxn | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.diesel | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.effr | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.drought | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.easter | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.thanksgiving | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| financial.revenue_yoy | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| financial.operating_margin | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| bound.announced_events | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.egg_price | positive_price_series | unavailable | retained_series_not_supplied |
| bound.confirmed_layer_culls | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.table_egg_layers | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_pullets | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_chicks_hatched | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |

### These inputs are finite on the training and development sessions before 2026-04-27. This round changes which of them enter the t+4, t+5 and t+6 average-close forecast.

**Execution:** inputs fitted

**Bound factors:** bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin

**Rationale (LLM):** Round 6 rotates the complete feed, energy, rate, drought, calendar, and filing inputs. Size 8.

**Limitations (LLM):**

- Egg-price and NASS series still have empty sessions.
- A lower development error on this window is not confirmation until the reserved sessions are scored.

## Round 7: news → hypothesis → fitted inputs

Round 7 uses 8 complete inputs plus research.momentum_5. The label is the log ratio of the mean close on sessions t+4, t+5 and t+6 to today's close. Sessions from 2026-04-27 through 2026-07-30 are held out for the test and are not used to choose factors. Citations support the LLM's hypothesis, not a measured contribution to prediction. Current article text is research context only. Fitted news features use retained issuer snapshots available at each historical cutoff; unavailable history is not a zero signal.

| Cited source ID | Title | Role |
| --- | --- | --- |

| Factor | Input | Execution | Reason |
| --- | --- | --- | --- |
| research.momentum_5 | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.diesel | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.effr | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.drought | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.easter | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.thanksgiving | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| financial.revenue_yoy | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| financial.operating_margin | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| financial.operating_margin_yoy | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| bound.announced_events | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.egg_price | positive_price_series | unavailable | retained_series_not_supplied |
| bound.confirmed_layer_culls | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.table_egg_layers | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_pullets | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_chicks_hatched | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |

### These inputs are finite on the training and development sessions before 2026-04-27. This round changes which of them enter the t+4, t+5 and t+6 average-close forecast.

**Execution:** inputs fitted

**Bound factors:** bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy

**Rationale (LLM):** Round 7 rotates the complete feed, energy, rate, drought, calendar, and filing inputs. Size 8.

**Limitations (LLM):**

- Egg-price and NASS series still have empty sessions.
- A lower development error on this window is not confirmation until the reserved sessions are scored.

## Round 8: news → hypothesis → fitted inputs

Round 8 uses 8 complete inputs plus research.momentum_10. The label is the log ratio of the mean close on sessions t+4, t+5 and t+6 to today's close. Sessions from 2026-04-27 through 2026-07-30 are held out for the test and are not used to choose factors. Citations support the LLM's hypothesis, not a measured contribution to prediction. Current article text is research context only. Fitted news features use retained issuer snapshots available at each historical cutoff; unavailable history is not a zero signal.

| Cited source ID | Title | Role |
| --- | --- | --- |

| Factor | Input | Execution | Reason |
| --- | --- | --- | --- |
| research.momentum_10 | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.effr | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.drought | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.easter | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.thanksgiving | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| financial.revenue_yoy | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| financial.operating_margin | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| financial.operating_margin_yoy | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| financial.inventory_yoy | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| bound.announced_events | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.egg_price | positive_price_series | unavailable | retained_series_not_supplied |
| bound.confirmed_layer_culls | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.table_egg_layers | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_pullets | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_chicks_hatched | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |

### These inputs are finite on the training and development sessions before 2026-04-27. This round changes which of them enter the t+4, t+5 and t+6 average-close forecast.

**Execution:** inputs fitted

**Bound factors:** bound.effr, bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy

**Rationale (LLM):** Round 8 rotates the complete feed, energy, rate, drought, calendar, and filing inputs. Size 8.

**Limitations (LLM):**

- Egg-price and NASS series still have empty sessions.
- A lower development error on this window is not confirmation until the reserved sessions are scored.

## Round 9: news → hypothesis → fitted inputs

Round 9 uses 8 complete inputs plus research.momentum_20. The label is the log ratio of the mean close on sessions t+4, t+5 and t+6 to today's close. Sessions from 2026-04-27 through 2026-07-30 are held out for the test and are not used to choose factors. Citations support the LLM's hypothesis, not a measured contribution to prediction. Current article text is research context only. Fitted news features use retained issuer snapshots available at each historical cutoff; unavailable history is not a zero signal.

| Cited source ID | Title | Role |
| --- | --- | --- |

| Factor | Input | Execution | Reason |
| --- | --- | --- | --- |
| research.momentum_20 | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.drought | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.easter | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.thanksgiving | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| financial.revenue_yoy | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| financial.operating_margin | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| financial.operating_margin_yoy | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| financial.inventory_yoy | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| financial.operating_cashflow_yoy | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| bound.announced_events | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.egg_price | positive_price_series | unavailable | retained_series_not_supplied |
| bound.confirmed_layer_culls | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.table_egg_layers | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_pullets | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_chicks_hatched | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |

### These inputs are finite on the training and development sessions before 2026-04-27. This round changes which of them enter the t+4, t+5 and t+6 average-close forecast.

**Execution:** inputs fitted

**Bound factors:** bound.drought, bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy

**Rationale (LLM):** Round 9 rotates the complete feed, energy, rate, drought, calendar, and filing inputs. Size 8.

**Limitations (LLM):**

- Egg-price and NASS series still have empty sessions.
- A lower development error on this window is not confirmation until the reserved sessions are scored.

## Round 10: news → hypothesis → fitted inputs

Round 10 uses 8 complete inputs plus research.qlib_roc_v1_10. The label is the log ratio of the mean close on sessions t+4, t+5 and t+6 to today's close. Sessions from 2026-04-27 through 2026-07-30 are held out for the test and are not used to choose factors. Citations support the LLM's hypothesis, not a measured contribution to prediction. Current article text is research context only. Fitted news features use retained issuer snapshots available at each historical cutoff; unavailable history is not a zero signal.

| Cited source ID | Title | Role |
| --- | --- | --- |

| Factor | Input | Execution | Reason |
| --- | --- | --- | --- |
| research.qlib_roc_v1_10 | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.easter | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| bound.thanksgiving | positive_price_series | fitted | Present in this completed fit's recorded feature columns |
| financial.revenue_yoy | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| financial.operating_margin | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| financial.operating_margin_yoy | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| financial.inventory_yoy | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| financial.operating_cashflow_yoy | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| financial.free_cashflow_yoy | publication_dated_financials | fitted | Present in this completed fit's recorded feature columns |
| bound.announced_events | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.egg_price | positive_price_series | unavailable | retained_series_not_supplied |
| bound.confirmed_layer_culls | text | unavailable | text_extraction_not_executable_in_this_profile |
| bound.table_egg_layers | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_pullets | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |
| bound.egg_type_chicks_hatched | positive_price_series | unavailable | missing_bound_input_on_fixed_sample |

### These inputs are finite on the training and development sessions before 2026-04-27. This round changes which of them enter the t+4, t+5 and t+6 average-close forecast.

**Execution:** inputs fitted

**Bound factors:** bound.easter, bound.thanksgiving, financial.revenue_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.free_cashflow_yoy

**Rationale (LLM):** Round 10 rotates the complete feed, energy, rate, drought, calendar, and filing inputs. Size 8.

**Limitations (LLM):**

- Egg-price and NASS series still have empty sessions.
- A lower development error on this window is not confirmation until the reserved sessions are scored.

## Relationship context

No relationship profile for CALM; configure instruments/US/CALM/relationships.json

| Symbol / scope | News | Published | Source |
| --- | --- | --- | --- |

## Initial hypothesis

Round 1 uses 8 complete inputs plus research.log_return_volatility_10. The label is the log ratio of the mean close on sessions t+4, t+5 and t+6 to today's close. Sessions from 2026-04-27 through 2026-07-30 are held out for the test and are not used to choose factors.

## Revision after feedback

Round 2 uses 8 complete inputs plus research.log_return_volatility_20. The label is the log ratio of the mean close on sessions t+4, t+5 and t+6 to today's close. Sessions from 2026-04-27 through 2026-07-30 are held out for the test and are not used to choose factors.

| Factor change | IDs |
| --- | --- |
| activated | bound.drought, bound.easter, bound.thanksgiving, financial.free_cashflow_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.revenue_yoy, research.qlib_roc_v1_20 |
| deactivated | research.log_return_volatility_10 |
| retained | bound.corn, bound.diesel, bound.effr, bound.feed_index, bound.gas, bound.meal, bound.mxn, bound.price_momentum |

| Round | Executed factor | Recipe |
| --- | --- | --- |
| Initial | research.log_return_volatility_10 | {"op": "log_return_volatility", "window": 10} |
| Initial | bound.price_momentum | {"op": "momentum", "window": 5} |
| Initial | bound.corn | {"op": "momentum", "window": 5} |
| Initial | bound.meal | {"op": "momentum", "window": 5} |
| Initial | bound.feed_index | {"op": "momentum", "window": 5} |
| Initial | bound.gas | {"op": "momentum", "window": 5} |
| Initial | bound.mxn | {"op": "momentum", "window": 5} |
| Initial | bound.diesel | {"op": "momentum", "window": 5} |
| Initial | bound.effr | {"op": "momentum", "window": 5} |
| Revised | research.log_return_volatility_20 | {"op": "log_return_volatility", "window": 20} |
| Revised | bound.corn | {"op": "momentum", "window": 5} |
| Revised | bound.meal | {"op": "momentum", "window": 5} |
| Revised | bound.feed_index | {"op": "momentum", "window": 5} |
| Revised | bound.gas | {"op": "momentum", "window": 5} |
| Revised | bound.mxn | {"op": "momentum", "window": 5} |
| Revised | bound.diesel | {"op": "momentum", "window": 5} |
| Revised | bound.effr | {"op": "momentum", "window": 5} |
| Revised | bound.drought | {"op": "momentum", "window": 20} |

## Model input usage

Used means an input entered this attempt's numerical path. Available but unselected peers and context-only news are distinct from executed features. Missing peer windows do not drop samples or become zero. Related prices retain the source adapter's retrospective availability qualifications; current relationship discovery is not historical or causal evidence.

| Attempt | Source | Status | Executed factors | Reason |
| --- | --- | --- | --- | --- |
| Initial | target_prices | used | research.log_return_volatility_10, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr | target_labels_and_reference_prices |
| Initial | publication_dated_financials | not_selected | none | Retrospective reconstruction from currently retained SEC facts, not a complete historical vendor archive. A filing date without an acceptance time becomes usable at 05:00 UTC on the following day. Only directly reported 70–110 day quarters in USD are used; no annual income values are silently converted. Cash-flow facts retain their reported 3/6/9/12-month durations and compare only matching year-over-year periods. Revisions are selected as of each cutoff. Inventory uses reported balance-sheet instants and comparable prior-year dates, not cash-flow inventory changes or the previous fiscal year-end by default. |
| Initial | archived_headlines | unavailable | none |  |
| Initial | current_news | context_only | none | hypothesis_context_not_backdated_training_data |
| Revised | target_prices | used | research.qlib_roc_v1_20, bound.price_momentum, bound.corn, bound.meal, bound.feed_index, bound.gas, bound.mxn, bound.diesel, bound.effr, bound.drought, bound.easter, bound.thanksgiving | target_labels_and_reference_prices |
| Revised | publication_dated_financials | used | financial.free_cashflow_yoy, financial.inventory_yoy, financial.operating_cashflow_yoy, financial.operating_margin, financial.operating_margin_yoy, financial.revenue_yoy | Retrospective reconstruction from currently retained SEC facts, not a complete historical vendor archive. A filing date without an acceptance time becomes usable at 05:00 UTC on the following day. Only directly reported 70–110 day quarters in USD are used; no annual income values are silently converted. Cash-flow facts retain their reported 3/6/9/12-month durations and compare only matching year-over-year periods. Revisions are selected as of each cutoff. Inventory uses reported balance-sheet instants and comparable prior-year dates, not cash-flow inventory changes or the previous fiscal year-end by default. |
| Revised | archived_headlines | unavailable | none |  |
| Revised | current_news | context_only | none | hypothesis_context_not_backdated_training_data |

## News factor execution

No issuer RSS feed was configured, so no news column was built. News was supplied as hypothesis context; no news column was fitted. Original receipt times are preserved source records, not independently authenticated clocks. Real price cutoffs use the retained exchange calendar. Both attempts and prediction baselines share the same preselected sample mask. Keyword windows use calendar days; zero means no matching retained title, not absence of all issuer news.

| Coverage | Count |
| --- | --- |

| Proposed factor | Keywords | Days | State | Reuse until |
| --- | --- | --- | --- | --- |

## Prediction comparison

2026-01-21 to 2026-04-16: overlapping mean close at t+4, t+5 and t+6 labels, retrospective development observations. MAPE is price error percent; direction hit rate is a sample frequency, not confidence or independent accuracy. Current news and feedback influenced recipe selection. No independent holdout. Where news coverage removes dates, accounts hold prior units until the next supplied signal; missing dates do not imply a fresh daily forecast.

| Arm | N | Price MAE | Price RMSE | MAPE | Direction hits |
| --- | --- | --- | --- | --- | --- |
| Initial | 60 | 2.6841 | 3.5221 | 3.3006% | 45.0000% |
| Revised | 60 | 2.4904 | 3.2169 | 3.0546% | 46.6667% |
| unchanged_price | 60 | 2.3229 | 3.2367 | 2.8534% | 0.0000% |
| training_label_mean | 60 | 2.3392 | 3.2101 | 2.8698% | 46.6667% |

## Costed strategy comparison

Hypothetical long/cash policy: a forecast log return above the configured threshold targets the next open; unchanged long signals keep existing units. Buy-and-hold uses the same capital, dates and costs. Inventory is marked at the final close without assumed exit fees. Five-session forecast and execution intervals differ. Returns exclude dividends. Sharpe and volatility use consecutive costed daily NAV returns, including cash days and excluding the initial pre-trade mark. Annualization assumes 252 sessions/year, a 0% risk-free rate and sample standard deviation (ddof=1). These short development samples do not establish reliable long-term Sharpe. N/A means the statistic cannot be calculated; inspect the status below.

| Account | Net return | Sharpe (ann.) | Volatility (ann.) | Max drawdown | Daily N |
| --- | --- | --- | --- | --- | --- |
| Initial strategy | -15.3139% | -2.5575 | 24.0272% | -16.6611% | 65 |
| Revised strategy | -2.3528% | -0.6860 | 12.3536% | -4.9612% | 65 |
| Buy-and-hold | -3.9867% | -0.3725 | 30.2068% | -16.3919% | 65 |

| Account | Commission | Slippage | Turnover | Risk statistic status |
| --- | --- | --- | --- | --- |
| Initial strategy | 65.91 | 65.89 | 13.1802 | available |
| Revised strategy | 24.49 | 24.48 | 4.8979 | available |
| Buy-and-hold | 4.96 | 4.96 | 0.9919 | available |

## Latest forecast

No latest forecast was published. The selected round_42 fitted model and completed development results remain retained.

## Forecast unavailable

target_session_beyond_retained_calendar

## Financial business factors

Retrospective reconstruction from currently retained SEC facts, not a complete historical vendor archive. A filing date without an acceptance time becomes usable at 05:00 UTC on the following day. Only directly reported 70–110 day quarters in USD are used; no annual income values are silently converted. Cash-flow facts retain their reported 3/6/9/12-month durations and compare only matching year-over-year periods. Revisions are selected as of each cutoff. Inventory uses reported balance-sheet instants and comparable prior-year dates, not cash-flow inventory changes or the previous fiscal year-end by default.

| Factor | Use in selected model | Missing fixed rows | Definition |
| --- | --- | --- | --- |
| financial.revenue_yoy | Selected | 0 | Quarterly revenue / comparable prior-year revenue - 1 |
| financial.operating_margin | Selected | 0 | Quarterly operating income / revenue |
| financial.operating_margin_yoy | Selected | 0 | Quarterly operating margin minus comparable prior-year margin |
| financial.inventory_yoy | Selected | 0 | Reported inventory / comparable prior-year inventory - 1 |
| financial.operating_cashflow_yoy | Selected | 0 | Reported cumulative operating cash flow change / absolute prior-year value; matching 3/6/9/12-month periods |
| financial.free_cashflow_yoy | Selected | 0 | Reported cumulative (operating cash flow - capital expenditure) change / absolute prior-year value; matching periods, not a single-quarter estimate |

## Financial variation on research rows

Counts use the frozen research rows only. Repeated daily values are not independent financial observations; distinct values are not a count of filings. A constant training feature cannot explain differences between training rows. Coverage does not establish predictive value.

| Factor | Available training rows | Distinct training values | Distinct development values |
| --- | --- | --- | --- |
| financial.revenue_yoy | 211 / 211 | 4 | 2 |
| financial.operating_margin | 211 / 211 | 4 | 2 |
| financial.operating_margin_yoy | 211 / 211 | 4 | 2 |
| financial.inventory_yoy | 211 / 211 | 5 | 2 |
| financial.operating_cashflow_yoy | 211 / 211 | 5 | 2 |
| financial.free_cashflow_yoy | 211 / 211 | 5 | 2 |

## Financial source

sec_companyfacts

| Source | Status | Reason |
| --- | --- | --- |
| sec_companyfacts | available |  |

| Unavailable statement | Reason |
| --- | --- |

## With / without financial factors

One additional diagnostic fit on identical training/development rows and model settings; only financials columns are removed. This result does not alter factor/model selection or the latest forecast. Retrospective development evidence, not independent gain.

## Financial diagnostic cost

1 additional fits.

| Model inputs | Price MAE | Net return | Sharpe |
| --- | --- | --- | --- |
| Selected + financials | 2.400262528307733 | -0.00149 | 0.03748316209413944 |
| Same model settings, financials removed | 2.4216529797517152 | 0.058963 | 1.0952943597141402 |

## Engine and limits

Qlib-style local price adapters; no Qlib or QuantDinger execution. This is a research run, not a broker deployment or a registered prospective study. Missing partner news is not zero news. Definitions, full quotes, predictions, orders and valuations are retained locally.

| Setting | Value |
| --- | --- |
| command | research-study |
| symbol | CALM |
| catalog | None |
| json | False |
| llm | cursor |
| llm_model | None |
| llm_timeout | 900.0 |
| cursor_dir | /tmp/qrsi_calm_egg/cursor-llm10 |
| related | profile |
| news | issuer-rss |
| news_limit | 5 |
| prices | daily |
| prices_dir | /tmp/qrsi_calm_egg/prices |
| news_snapshot | None |
| history_dir | /tmp/qrsi_calm_egg/history10 |
| factor_root | /workspace/wtk/Quant_RSI_pr/instruments |
| related_prices | True |
| related_prices_dir | None |
| financials | True |
| financials_dir | None |
| financials_source | sec |
| check_inputs | False |
| external_drivers | False |
| external_drivers_dir | None |
| external_drivers_cache_dir | None |
| model | lightgbm |
| compare_qlib | False |
| ridge_alpha | None |
| output | /tmp/qrsi_calm_egg/study10 |
| question | Which combinations improve the t+4, t+5 and t+6 average-close forecast when 2026-04-27 through 2026-07-30 is held out as the test? |
| news_research | False |
| news_archive | None |
| evaluation_sessions | 60 |
| historical_sessions | 445 |
| walk_forward_folds | 0 |
| horizon_sessions | 5 |
| label_endpoint | 6 |
| label_average | 3 |
| sample_through | 2026-07-30 |
| terminal_sessions | 60 |
| rounds | 50 |
| max_fits | 50 |
| patience | None |
| initial_cash | 10000 |
| commission_bps | 5 |
| slippage_bps | 5 |
| threshold | 0.0 |
| selection_metric | mae |
| quiet | False |
| ui | False |
| model_configuration | {'schema_version': 1, 'name': 'lightgbm', 'parameters': {'objective': 'regression', 'verbosity': -1, 'num_threads': 1, 'deterministic': True, 'force_col_wise': True, 'num_leaves': 7, 'min_data_in_leaf': 5, 'learning_rate': 0.05, 'feature_pre_filter': False, 'seed': 7, 'num_boost_round': 24}, 'preprocessing': 'none', 'objective': 'regression', 'training_policy': 'fixed_mature_label_window_shared_between_attempts', 'target': 'log_mean_close_t_plus_4_5_6_over_close'} |
| research_rounds | 50 |
| fit_budget | 50 |
| comparison_fit_budget | 1 |
| walk_forward_fit_budget | 0 |
| generated_recipes | bounded_price_news_and_business_expressions |

## Run time and usage

Measured wall time: 29.22 seconds. Counts describe the current execution attempt, excluding reused operations. Earlier attempts are listed separately. Token counts are only what the provider reported; missing usage is unknown, not zero. Cached input tokens are included in input totals. No monetary cost is estimated.

| Operation count | Value |
| --- | --- |
| llm_requests_started | 50 |
| llm_requests_completed | 50 |
| search_requests_started | 0 |
| search_requests_completed | 0 |
| model_stages_started | 50 |
| model_stages_completed | 50 |
| comparison_model_stages_started | 1 |
| comparison_model_stages_completed | 1 |

| Usage coverage | Input tokens | Cached input tokens | Output tokens |
| --- | --- | --- | --- |
| unavailable | not reported | not reported | not reported |

| Stage | Status | Seconds |
| --- | --- | --- |
| load_prices | completed | 0.0741 |
| observe_prior_forecasts | completed | 0.0071 |
| collect_news_context | completed | 0.0041 |
| load_financials | completed | 0.9755 |
| check_inputs | completed | 0.0007 |
| load_related_prices | completed | 0.0020 |
| load_bound_supply | completed | 0.2309 |
| target | completed | 0.0015 |
| related_stocks | completed | 0.0009 |
| news | completed | 0.0008 |
| factor_library | completed | 0.0013 |
| initial_hypothesis | completed | 0.4078 |
| initial_factor_library | completed | 0.0017 |
| initial_factors | completed | 0.0012 |
| first_prediction | completed | 0.5207 |
| round_1_decision | completed | 0.0012 |
| feedback | completed | 0.0044 |
| revised_hypothesis | completed | 0.4065 |
| revised_factor_library | completed | 0.0017 |
| revised_factors | completed | 0.0013 |
| second_prediction | completed | 0.0743 |
| round_2_decision | completed | 0.0012 |
| round_3_feedback | completed | 0.0034 |
| round_3_hypothesis | completed | 0.4068 |
| round_3_factor_library | completed | 0.0018 |
| round_3_factors | completed | 0.0013 |
| round_3_prediction | completed | 0.0912 |
| round_3_decision | completed | 0.0012 |
| round_4_feedback | completed | 0.0034 |
| round_4_hypothesis | completed | 0.4071 |
| round_4_factor_library | completed | 0.0018 |
| round_4_factors | completed | 0.0013 |
| round_4_prediction | completed | 0.1019 |
| round_4_decision | completed | 0.0013 |
| round_5_feedback | completed | 0.0039 |
| round_5_hypothesis | completed | 0.4074 |
| round_5_factor_library | completed | 0.0019 |
| round_5_factors | completed | 0.0013 |
| round_5_prediction | completed | 0.0901 |
| round_5_decision | completed | 0.0013 |
| round_6_feedback | completed | 0.0038 |
| round_6_hypothesis | completed | 0.4076 |
| round_6_factor_library | completed | 0.0019 |
| round_6_factors | completed | 0.0013 |
| round_6_prediction | completed | 0.0743 |
| round_6_decision | completed | 0.0052 |
| round_7_feedback | completed | 0.0040 |
| round_7_hypothesis | completed | 0.4081 |
| round_7_factor_library | completed | 0.0020 |
| round_7_factors | completed | 0.0014 |
| round_7_prediction | completed | 0.1015 |
| round_7_decision | completed | 0.0013 |
| round_8_feedback | completed | 0.0040 |
| round_8_hypothesis | completed | 0.4079 |
| round_8_factor_library | completed | 0.0020 |
| round_8_factors | completed | 0.0014 |
| round_8_prediction | completed | 0.0687 |
| round_8_decision | completed | 0.0014 |
| round_9_feedback | completed | 0.0046 |
| round_9_hypothesis | completed | 0.4081 |
| round_9_factor_library | completed | 0.0020 |
| round_9_factors | completed | 0.0014 |
| round_9_prediction | completed | 0.0685 |
| round_9_decision | completed | 0.0014 |
| round_10_feedback | completed | 0.0044 |
| round_10_hypothesis | completed | 0.4084 |
| round_10_factor_library | completed | 0.0021 |
| round_10_factors | completed | 0.0014 |
| round_10_prediction | completed | 0.0697 |
| round_10_decision | completed | 0.0014 |
| round_11_feedback | completed | 0.0045 |
| round_11_hypothesis | completed | 0.4086 |
| round_11_factor_library | completed | 0.0021 |
| round_11_factors | completed | 0.0015 |
| round_11_prediction | completed | 0.0718 |
| round_11_decision | completed | 0.0015 |
| round_12_feedback | completed | 0.0049 |
| round_12_hypothesis | completed | 0.4091 |
| round_12_factor_library | completed | 0.0021 |
| round_12_factors | completed | 0.0015 |
| round_12_prediction | completed | 0.0684 |
| round_12_decision | completed | 0.0017 |
| round_13_feedback | completed | 0.0045 |
| round_13_hypothesis | completed | 0.4097 |
| round_13_factor_library | completed | 0.0022 |
| round_13_factors | completed | 0.0015 |
| round_13_prediction | completed | 0.0708 |
| round_13_decision | completed | 0.0015 |
| round_14_feedback | completed | 0.0045 |
| round_14_hypothesis | completed | 0.4096 |
| round_14_factor_library | completed | 0.0022 |
| round_14_factors | completed | 0.0018 |
| round_14_prediction | completed | 0.0862 |
| round_14_decision | completed | 0.0027 |
| round_15_feedback | completed | 0.0050 |
| round_15_hypothesis | completed | 0.4106 |
| round_15_factor_library | completed | 0.0023 |
| round_15_factors | completed | 0.0018 |
| round_15_prediction | completed | 0.0749 |
| round_15_decision | completed | 0.0016 |
| round_16_feedback | completed | 0.0048 |
| round_16_hypothesis | completed | 0.4105 |
| round_16_factor_library | completed | 0.0022 |
| round_16_factors | completed | 0.0017 |
| round_16_prediction | completed | 0.0737 |
| round_16_decision | completed | 0.0016 |
| round_17_feedback | completed | 0.0049 |
| round_17_hypothesis | completed | 0.4106 |
| round_17_factor_library | completed | 0.0024 |
| round_17_factors | completed | 0.0018 |
| round_17_prediction | completed | 0.1005 |
| round_17_decision | completed | 0.0018 |
| round_18_feedback | completed | 0.0051 |
| round_18_hypothesis | completed | 0.4104 |
| round_18_factor_library | completed | 0.0023 |
| round_18_factors | completed | 0.0018 |
| round_18_prediction | completed | 0.1073 |
| round_18_decision | completed | 0.0017 |
| round_19_feedback | completed | 0.0053 |
| round_19_hypothesis | completed | 0.4108 |
| round_19_factor_library | completed | 0.0024 |
| round_19_factors | completed | 0.0021 |
| round_19_prediction | completed | 0.0712 |
| round_19_decision | completed | 0.0017 |
| round_20_feedback | completed | 0.0057 |
| round_20_hypothesis | completed | 0.4142 |
| round_20_factor_library | completed | 0.0025 |
| round_20_factors | completed | 0.0019 |
| round_20_prediction | completed | 0.0944 |
| round_20_decision | completed | 0.0017 |
| round_21_feedback | completed | 0.0055 |
| round_21_hypothesis | completed | 0.4118 |
| round_21_factor_library | completed | 0.0025 |
| round_21_factors | completed | 0.0019 |
| round_21_prediction | completed | 0.0697 |
| round_21_decision | completed | 0.0018 |
| round_22_feedback | completed | 0.0056 |
| round_22_hypothesis | completed | 0.4147 |
| round_22_factor_library | completed | 0.0038 |
| round_22_factors | completed | 0.0030 |
| round_22_prediction | completed | 0.1549 |
| round_22_decision | completed | 0.0018 |
| round_23_feedback | completed | 0.0058 |
| round_23_hypothesis | completed | 0.4117 |
| round_23_factor_library | completed | 0.0026 |
| round_23_factors | completed | 0.0020 |
| round_23_prediction | completed | 0.0723 |
| round_23_decision | completed | 0.0018 |
| round_24_feedback | completed | 0.0059 |
| round_24_hypothesis | completed | 0.4119 |
| round_24_factor_library | completed | 0.0026 |
| round_24_factors | completed | 0.0020 |
| round_24_prediction | completed | 0.0729 |
| round_24_decision | completed | 0.0020 |
| round_25_feedback | completed | 0.0064 |
| round_25_hypothesis | completed | 0.4122 |
| round_25_factor_library | completed | 0.0026 |
| round_25_factors | completed | 0.0021 |
| round_25_prediction | completed | 0.0726 |
| round_25_decision | completed | 0.0019 |
| round_26_feedback | completed | 0.0062 |
| round_26_hypothesis | completed | 0.4125 |
| round_26_factor_library | completed | 0.0026 |
| round_26_factors | completed | 0.0020 |
| round_26_prediction | completed | 0.0695 |
| round_26_decision | completed | 0.0019 |
| round_27_feedback | completed | 0.0063 |
| round_27_hypothesis | completed | 0.4130 |
| round_27_factor_library | completed | 0.0026 |
| round_27_factors | completed | 0.0022 |
| round_27_prediction | completed | 0.1018 |
| round_27_decision | completed | 0.0019 |
| round_28_feedback | completed | 0.0062 |
| round_28_hypothesis | completed | 0.4131 |
| round_28_factor_library | completed | 0.0027 |
| round_28_factors | completed | 0.0023 |
| round_28_prediction | completed | 0.0952 |
| round_28_decision | completed | 0.0024 |
| round_29_feedback | completed | 0.0072 |
| round_29_hypothesis | completed | 0.4146 |
| round_29_factor_library | completed | 0.0027 |
| round_29_factors | completed | 0.0022 |
| round_29_prediction | completed | 0.0744 |
| round_29_decision | completed | 0.0020 |
| round_30_feedback | completed | 0.0067 |
| round_30_hypothesis | completed | 0.4132 |
| round_30_factor_library | completed | 0.0027 |
| round_30_factors | completed | 0.0023 |
| round_30_prediction | completed | 0.0715 |
| round_30_decision | completed | 0.0020 |
| round_31_feedback | completed | 0.0070 |
| round_31_hypothesis | completed | 0.4137 |
| round_31_factor_library | completed | 0.0029 |
| round_31_factors | completed | 0.0029 |
| round_31_prediction | completed | 0.0883 |
| round_31_decision | completed | 0.0021 |
| round_32_feedback | completed | 0.0076 |
| round_32_hypothesis | completed | 0.4142 |
| round_32_factor_library | completed | 0.0028 |
| round_32_factors | completed | 0.0023 |
| round_32_prediction | completed | 0.0728 |
| round_32_decision | completed | 0.0021 |
| round_33_feedback | completed | 0.0071 |
| round_33_hypothesis | completed | 0.4141 |
| round_33_factor_library | completed | 0.0028 |
| round_33_factors | completed | 0.0024 |
| round_33_prediction | completed | 0.0926 |
| round_33_decision | completed | 0.0022 |
| round_34_feedback | completed | 0.0071 |
| round_34_hypothesis | completed | 0.4141 |
| round_34_factor_library | completed | 0.0029 |
| round_34_factors | completed | 0.0024 |
| round_34_prediction | completed | 0.0929 |
| round_34_decision | completed | 0.0023 |
| round_35_feedback | completed | 0.0074 |
| round_35_hypothesis | completed | 0.4144 |
| round_35_factor_library | completed | 0.0029 |
| round_35_factors | completed | 0.0024 |
| round_35_prediction | completed | 0.0710 |
| round_35_decision | completed | 0.0022 |
| round_36_feedback | completed | 0.0075 |
| round_36_hypothesis | completed | 0.4152 |
| round_36_factor_library | completed | 0.0030 |
| round_36_factors | completed | 0.0033 |
| round_36_prediction | completed | 0.0709 |
| round_36_decision | completed | 0.0022 |
| round_37_feedback | completed | 0.0076 |
| round_37_hypothesis | completed | 0.4149 |
| round_37_factor_library | completed | 0.0030 |
| round_37_factors | completed | 0.0024 |
| round_37_prediction | completed | 0.0746 |
| round_37_decision | completed | 0.0023 |
| round_38_feedback | completed | 0.0080 |
| round_38_hypothesis | completed | 0.4150 |
| round_38_factor_library | completed | 0.0031 |
| round_38_factors | completed | 0.0025 |
| round_38_prediction | completed | 0.0736 |
| round_38_decision | completed | 0.0026 |
| round_39_feedback | completed | 0.0078 |
| round_39_hypothesis | completed | 0.4153 |
| round_39_factor_library | completed | 0.0030 |
| round_39_factors | completed | 0.0025 |
| round_39_prediction | completed | 0.0711 |
| round_39_decision | completed | 0.0023 |
| round_40_feedback | completed | 0.0080 |
| round_40_hypothesis | completed | 0.4155 |
| round_40_factor_library | completed | 0.0030 |
| round_40_factors | completed | 0.0026 |
| round_40_prediction | completed | 0.0978 |
| round_40_decision | completed | 0.0023 |
| round_41_feedback | completed | 0.0081 |
| round_41_hypothesis | completed | 0.4160 |
| round_41_factor_library | completed | 0.0030 |
| round_41_factors | completed | 0.0026 |
| round_41_prediction | completed | 0.1025 |
| round_41_decision | completed | 0.0023 |
| round_42_feedback | completed | 0.0087 |
| round_42_hypothesis | completed | 0.4191 |
| round_42_factor_library | completed | 0.0031 |
| round_42_factors | completed | 0.0026 |
| round_42_prediction | completed | 0.1030 |
| round_42_decision | completed | 0.0416 |
| round_43_feedback | completed | 0.0086 |
| round_43_hypothesis | completed | 0.4183 |
| round_43_factor_library | completed | 0.0034 |
| round_43_factors | completed | 0.0027 |
| round_43_prediction | completed | 0.1716 |
| round_43_decision | completed | 0.0027 |
| round_44_feedback | completed | 0.0089 |
| round_44_hypothesis | completed | 0.4189 |
| round_44_factor_library | completed | 0.0032 |
| round_44_factors | completed | 0.0028 |
| round_44_prediction | completed | 0.0890 |
| round_44_decision | completed | 0.0025 |
| round_45_feedback | completed | 0.0093 |
| round_45_hypothesis | completed | 0.4189 |
| round_45_factor_library | completed | 0.0032 |
| round_45_factors | completed | 0.0028 |
| round_45_prediction | completed | 0.0746 |
| round_45_decision | completed | 0.0032 |
| round_46_feedback | completed | 0.0092 |
| round_46_hypothesis | completed | 0.4186 |
| round_46_factor_library | completed | 0.0032 |
| round_46_factors | completed | 0.0028 |
| round_46_prediction | completed | 0.0921 |
| round_46_decision | completed | 0.0024 |
| round_47_feedback | completed | 0.0092 |
| round_47_hypothesis | completed | 0.4190 |
| round_47_factor_library | completed | 0.0033 |
| round_47_factors | completed | 0.0028 |
| round_47_prediction | completed | 0.0930 |
| round_47_decision | completed | 0.0025 |
| round_48_feedback | completed | 0.0094 |
| round_48_hypothesis | completed | 0.4190 |
| round_48_factor_library | completed | 0.0033 |
| round_48_factors | completed | 0.0029 |
| round_48_prediction | completed | 0.0720 |
| round_48_decision | completed | 0.0025 |
| round_49_feedback | completed | 0.0099 |
| round_49_hypothesis | completed | 0.4197 |
| round_49_factor_library | completed | 0.0035 |
| round_49_factors | completed | 0.0030 |
| round_49_prediction | completed | 0.0744 |
| round_49_decision | completed | 0.0027 |
| round_50_feedback | completed | 0.0097 |
| round_50_hypothesis | completed | 0.4260 |
| round_50_factor_library | completed | 0.0035 |
| round_50_factors | completed | 0.0029 |
| round_50_prediction | completed | 0.0801 |
| round_50_decision | completed | 0.0026 |
| terminal_evaluation | completed | 0.1038 |
| summary | completed | 0.0038 |
| financial_comparison | completed | 0.0963 |

## Full evidence

See `result.json`, `inputs.json`, `news_history.json`, `config.json` and `trace.jsonl` in this directory. Reading these files makes no API calls.

[Run time and usage](runtime.json)
