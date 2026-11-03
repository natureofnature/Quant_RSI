# Usability alignment with QuantDinger

## Evidence and scope

Reviewed on 2026-11-03 against Quant RSI `463bb7b` and
[QuantDinger `e75f0cf`](https://github.com/OpenByteInc/QuantDinger/tree/e75f0cfa9cd47de9259844fefbf20c2fa1464d45).
This is a workflow and source comparison, not a product performance benchmark.
QuantDinger was not deployed for this audit. Its desktop/mobile clients are shipped
from separate repositories as documented in its
[repository layout](https://github.com/OpenByteInc/QuantDinger/blob/e75f0cfa9cd47de9259844fefbf20c2fa1464d45/README.md#repository-layout).
Claims below distinguish documented behavior from backend code inspected.

The useful reference is a connected journey: configure a service, inspect data,
start from a runnable example, understand results, and continue working without
reconstructing terminal commands. Quant RSI already has the study loop, reports,
serial queue, templates, observation tasks and recovery. This effort improves
those existing paths. It does not require replacing the engine or adding an
execution platform before the research workbench becomes easier to use.

## Findings

| User task | QuantDinger evidence | Quant RSI at the comparison revision | Direction |
| --- | --- | --- | --- |
| First successful run | The [30-minute guide](https://github.com/OpenByteInc/QuantDinger/blob/e75f0cfa9cd47de9259844fefbf20c2fa1464d45/docs/getting-started/QUICKSTART_30_MINUTES.md) gives an ordered setup/data/research/backtest journey. | A working browser form, but users choose many settings before seeing any result. No built-in starter studies. | An offline example first, then public-data and business-research starters, all using the same queue. |
| Configure and inspect services | The guide describes AI settings and connection testing; [settings routes](https://github.com/OpenByteInc/QuantDinger/blob/e75f0cfa9cd47de9259844fefbf20c2fa1464d45/backend_api_python/app/routes/settings.py) implement saved settings and selected service connection checks. | Provider choice exists; keys and CLI setup stay in the server environment and failures may appear only after enqueueing. | Show configuration status and clear next actions. Distinguish local presence from a verified connection. Follow with explicit connection checks and convenient configuration. |
| Set execution assumptions | The [backtest guide](https://github.com/OpenByteInc/QuantDinger/blob/e75f0cfa9cd47de9259844fefbf20c2fa1464d45/docs/product/BACKTEST_CENTER.md) exposes range, capital, fees, slippage and comparable benchmarks. [Preset code](https://github.com/OpenByteInc/QuantDinger/blob/e75f0cfa9cd47de9259844fefbf20c2fa1464d45/backend_api_python/app/services/backtest_presets.py) preserves explicit user overrides. | The CLI has capital, cost and signal-threshold options; the browser cannot set them. | Carry these controls through templates, frozen tasks, reports and evidence-change checks. Show estimated fit counts before starting. |
| Understand the result | The [research guide](https://github.com/OpenByteInc/QuantDinger/blob/e75f0cfa9cd47de9259844fefbf20c2fa1464d45/docs/product/AI_RESEARCH.md) prioritizes data time, identity, evidence and missing inputs; its backtest guide prioritizes risk, ledger and benchmark. | Detailed per-run reports exist, including risk and costs. The overview is a dense cross-task table. | Add focused, compatible-run comparison, input coverage and a clear path to the supporting report/ledger. Do not rank incompatible stocks or periods as if MAE units matched. |
| Continue after a result or error | The documented [strategy workflow](https://github.com/OpenByteInc/QuantDinger/blob/e75f0cfa9cd47de9259844fefbf20c2fa1464d45/docs/product/STRATEGY_WORKFLOW.md) links validation, versioned results and next actions. | Templates and Continue study work; errors are mainly exception text and repeating a task requires reconstructing settings or finding its template. | Support editable repeat-from-task settings, actionable failure explanations, and clear distinctions between retrying retained work and starting a fresh study. |

## Acceptance journey and delivery order

### 1. Guided study launch — first implementation

- Open the existing workbench and apply a named starter without creating a task.
- Run the offline starter with no API key or network dependency; produce an actual
  fitted model, factor revision, costed strategy report and history entry.
- Inspect local provider/model configuration without revealing keys, making paid
  calls, or labeling mere credential presence as successful connectivity.
- See the stock count, research/extra fit ceilings, capital and per-side cost rates
  before enqueueing. Keep advanced settings available in a collapsed section.
- Modify capital/costs/threshold, save a template, queue it and reopen the workspace.
  The exact assumptions must reach the worker and 1×/2×/3× cost reports. Invalid
  values must fail before queue creation; cost changes must not be skipped as
  unchanged evidence.
- Preserve old CLI callers, templates and study recovery behavior.

This slice does not complete the overall usability goal. Automated form and HTTP
checks are not browser visual acceptance, and local configuration checks are not
connection tests.

### 2. Results and repeated use — implementation

- Choose retained studies and compare only compatible instruments, horizons,
  periods and policies, explaining any incompatibility rather than hiding it.
- Lead with input coverage and usable-result status, then development/terminal
  metrics, Sharpe, drawdown, costs and benchmark; link detailed evidence.
- Start an editable new study from an old task without mutating its artifacts.
- Show next actions for missing service configuration, missing data and resumable
  failures; make fresh-run versus retained-work recovery explicit.
The workbench now provides **Use settings** on research tasks. It reads the
retained configuration, opens an editable form, and never automatically queues a
run or enables a schedule. **Queue research** creates a fresh task; old artifacts
stay unchanged. Existing unchanged-evidence controls still apply. This takes two
explicit actions to repeat settings, with no parameter re-entry or terminal
command once the workbench is open. Continue study remains the separate operation
for retrying retained work. Runtime-change errors direct users to a fresh study.

**Compare two retained studies** shows the settings, selected factors, retained
input counts and development/terminal metrics. It checks the symbol, input mode,
horizon, information mode, forecast observations, actual account price path,
metric units and complete execution policy before computing right-minus-left
values. Missing evidence or incompatible results produce reasons and no numerical
differences. Undefined risk metrics remain unavailable. Retained input counts are
not an automatic data-quality approval, and matched accounts do not establish
independent validation or fair research-budget matching.

HTTP and real offline worker tests cover these paths, including byte-for-byte
preservation of the original study after copying settings and completing a new
run. Full-script Node tests cover editable reuse, comparison rendering and failure
actions. Browser visual acceptance remains outstanding when no browser is available.

### 3. Service and data setup

Implemented: browser setup for Codex/OpenAI API/DeepSeek, workspace model defaults
and optional private API keys, and explicit connection tests on the existing
serial queue. Keys are saved in a mode-0600 file, never returned by the HTTP API,
and injected only into new workers without changing the server environment.
Removing a workspace key falls back to a configured environment key. Codex keeps
its CLI session. Saved model defaults are applied explicitly to the study form.

Connection tests issue one tool-free call through the existing transport, with a
60-second provider timeout, no automatic retry, and no data collection or fits.
The UI labels possible service usage and shows timestamps, sanitized outcomes and
reported tokens. A successful test is not a permanent availability guarantee.
The existing **Check inputs** action is the next setup step before real research;
its report retains source, freshness, coverage and unavailable-factor diagnostics.

Local acceptance on November 3, 2026 used the real Codex session and loopback HTTP
routes at `63faae1`. An unavailable model failed as intended; a subsequent default
model connection test succeeded. A KO input check completed in about 83 seconds
without LLM calls or fits. The public-data study then completed in about 227 seconds:
six model-service calls (including two searches), two LightGBM fits, and a business
increment adding `business.inventory_revenue_growth_spread` to the fixed baseline.
The historical settings route returned the completed task's configuration without
queuing another study. Evidence was retained outside the checkout; this was not a
browser interaction test or a deployed QuantDinger comparison.

The run establishes workflow execution, not predictive quality. Development MAE
changed from 2.8231 to 2.8227, while unchanged-price MAE was 1.0460. Inspection of
these retained results exposed Decimal-string account metrics being displayed as
unavailable; the comparison reader now converts finite numeric strings, with
regression checks for net return, drawdown and buy-and-hold values. Undefined risk
statistics remain unavailable. The paid research was not repeated for this display
fix. Browser visual acceptance remains separate from the full-script DOM tests;
no browser surface, including the in-app browser, was available in this session.

Keep this list tied to observed friction. Broader portfolio construction, live
execution and SaaS operations are separate capabilities; their existence in
QuantDinger is not evidence that adding them would improve this research journey.
