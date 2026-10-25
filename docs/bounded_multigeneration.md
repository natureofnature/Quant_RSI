# Bounded multigeneration development

The explicit version 2 profile permits at most two experimental package changes under one
registered authority. It expands `policy.window_max` by exactly one, within a global domain
registered before gen0. This changes the reachable factor set. It is a development experiment,
not an independent improvement estimate, release decision, or C5 evaluation. The implementation
still requires the same-source real-chain and interruption acceptance evidence.

## Register before running gen0

Prepare `examples/bounded_root_research.json.template` using an explicitly imported local snapshot
and account rules. Replace every `SET_...` value. Register canonical absolute workspace and root
source paths, total attempts/seconds, stage/generation limits, global window bounds and initial
cap. The source grant must fit the total. The allowed caps are `{U0, U0+1, U0+2}` intersected with
the global domain. `window_first`, the baseline, all other policy fields and reset budgets stay fixed.

```bash
export QUANT_RSI_WORKSPACE=/absolute/registered/workspace
quant-rsi research run root-plan.json --output /absolute/registered/source --steps 1
quant-rsi research resume /absolute/registered/source --steps 1
quant-rsi research replay /absolute/registered/source
```

The first creation durably reserves one root campaign before a worker can start. Recover an
unfinished creation with the same plan and path. Copies, alternate source paths and changed
registrations cannot acquire another grant. Source research may pause at a settled checkpoint;
pending requests or unresolved worker charges must be reconciled before handoff.

## Hand off and advance

Fill `examples/bounded_multigeneration.json.template` with the actual root checkpoint and exactly
the preregistered limits. Use the existing trajectory CLI:

```bash
quant-rsi trajectory init trajectory-plan.json
quant-rsi trajectory resume --steps 1
quant-rsi trajectory status
quant-rsi trajectory replay
```

Initialization verifies the actual source before committing the irreversible `HANDOFF` record.
The original root then refuses further dispatch. An interrupted handoff resumes the same identity
and checkpoint; it does not register another source. Descendants run only through trajectory
resume with the existing trajectory → C4 → C2 owner capabilities. Readonly copies never dispatch.

Each patch requires the current source's final selected eligible improvement from the same
operator at `U−1` to `U`, strictly above the fixed threshold. Its actual request prefix through
that selected observation must remain structurally shared, and the reset grant must fund that
prefix plus a new full-timeout trial. Failed or uncharged historical requests still count toward
this conservative opportunity check. Unknown reset outcomes are not predicted. Later requests
use the preceding actual candidate receipt and head revision, with the full ancestor closure.

## Costs and stopping

The root's actual cost is charged once. Each C4 stage irreversibly reserves both complete reset
grants; confirmed consumption uses that stage's latest cumulative actual receipts. Reading a
candidate as the next source adds no second historical charge. New reset workers do incur charges.
Unused old commitment never funds another stage.

Unsupported feedback, insufficient reset opportunity, global window bounds, registered cap
bounds, stage/generation limits, outer capacity and required-operation failure retain distinct
outcomes. A bound does not mean every factor was searched. Valid paired development results may
advance an experimental head with zero or negative paired delta; this is not promotion.

Replay uses frozen task/domain, registration/handoff, source, model, mask, lineage, CAS and terminal
evidence. It requires neither the original CSV/registry nor mutable child paths. Missing committed
evidence refuses without reconstruction or training. Legacy version 1 trajectories retain their
finite order-change behavior; C5 explicitly rejects this new profile.
