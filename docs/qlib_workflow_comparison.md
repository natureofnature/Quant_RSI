# Compare an actual Qlib workflow on the selected study inputs

`research-study --compare-qlib` runs the upstream Qlib workflow after native
research and selection. It uses actual `StaticDataLoader`, `DataHandlerLP`,
`DatasetH`, `LGBModel.fit/predict` and a local MLflow recorder. It is separate from
the older [operator-only conformance example](qlib_operator_comparison.md).

## Optional environment

Use Linux and Python 3.12 from the repository root. Keep the normal development
environment unchanged; Qlib brings additional dependencies including MLflow and
Pandas. The optional requirements pin Qlib 0.9.7, MLflow 3.17.2 and Pandas 2.3.3
alongside the native numerical/test pins.

```bash
python3.12 -m venv /tmp/quant-rsi-qlib-env
source /tmp/quant-rsi-qlib-env/bin/activate
python -m pip install -r examples/requirements-qlib-workflow.txt
python -m pip install --no-deps -e .

quant-rsi research-study XOM --news-research --compare-qlib \
  --output /tmp/xom-qlib-study
```

Codex authentication and public service access follow the
[connected-study guide](connected_research_study.md). Use a new output directory.
Missing Qlib dependencies or `--model ridge` fail before news collection or LLM
calls. This first comparison supports LightGBM only.

For an offline demonstration with actual native and Qlib fits:

```bash
quant-rsi research-study KO --llm simulated --news simulated --related simulated \
  --prices synthetic --compare-qlib --no-ui --output /tmp/qlib-offline-study
pytest -q tests/test_study_qlib.py tests/test_qlib_operator_comparison.py
```

Normal CI runs the core regression suite without Qlib and explicitly skips the
optional upstream integration cases. Run the command above in the optional
environment to exercise the real upstream model and recorder; mocks do not
satisfy this check.

## What is held equal

The selected native fit supplies its exact feature order, feature values,
mature training labels, and development/terminal session lists. Current news
and peer features keep their existing eligibility; no second feature pipeline
recomputes or substitutes inputs. Qlib gets evaluation features without their
labels. No Qlib preprocessing, validation segment or early stopping changes the
frozen model configuration: both fits use 24 boosting rounds and seed 7. The
adapter checks Qlib's prepared rows, columns and training labels before fitting.

The selected native model remains frozen. Qlib receives one additional fit,
declared as `comparison_fit_budget: 1`, counted separately from the research
`--max-fits` allowance. For the default two-round study this is three fits total.
Comparison results never feed back into native selection or replace its latest
forecast. Disabled or unavailable terminal rows remain unavailable; the adapter
does not drop them to manufacture a comparison. A later comparison failure
leaves the study failed with its completed native stages retained in the trace.

Both engines' forecasts enter the same Quant RSI account evaluator with the
same next-open policy, costs, cash, threshold and buy-and-hold benchmark. The
HTML report and final UI node show forecast MAE, net return, Sharpe and drawdown.
The complete account histories remain in `qlib-comparison/comparison.json`.

`qlib-comparison/inputs.json` retains the common numerical inputs and hashes of
the native prediction and market prices. `model.txt` retains the upstream fitted
booster. `mlflow.db` and `artifacts/` hold local workflow tracking; no remote
tracking service is required. Recorded package versions and the input hash
appear in the comparison report. Existing completed study directories are not
rewritten; run a fresh study to create a comparison.

## Interpretation

This is a workflow semantics comparison of the same selected LightGBM problem.
Matching forecasts are expected when the adapters preserve its semantics;
agreement is reported with absolute tolerances of `1e-10` for log returns and
`1e-8` for prices. Differences are reported rather than treated as an automatic
win. Reusing factors selected by the native research loop is not a matched-budget
comparison of research policies or independent predictive evidence.

This does not run Qlib's portfolio/backtest engine, Alpha158/360, a cross-sectional
benchmark, or an autonomous Qlib researcher. Both account reports use the common
native execution adapter. It provides the first actual upstream workflow path
on which broader comparisons can be built, without changing the study's existing
historical-data and LLM-knowledge limitations.

Upstream implementations: [data loader](https://github.com/microsoft/qlib/blob/v0.9.7/qlib/data/dataset/loader.py),
[data handler](https://github.com/microsoft/qlib/blob/v0.9.7/qlib/data/dataset/handler.py),
[LightGBM model](https://github.com/microsoft/qlib/blob/v0.9.7/qlib/contrib/model/gbdt.py).
