"""Generate/register only the two approved causal engineering fixtures; never fit a model."""

import argparse
import csv
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from quant_rsi import controller, local_research, market_data
from quant_rsi.artifacts import canonical, digest
from quant_rsi.market_contracts import CSV_FIELDS, AccountRules, DataContract
from quant_rsi.researcher import ResearchPlan
from quant_rsi.schemas import FactorSpec

SYMBOLS = tuple(f"ENGINEERING_{i:02d}" for i in range(8))


def prices(innovations):
    opening = np.zeros((260, 8))
    closing = np.zeros_like(opening)
    signals = np.zeros_like(opening)
    for t in range(260):
        lag = signals[t - 2] if t >= 2 else np.zeros(8)
        opening[t] = (
            100.0 if t == 0 else opening[t - 1] * np.exp(0.006 * lag + 0.004 * innovations[t, :, 0])
        )
        closing[t] = opening[t] * np.exp(0.003 * innovations[t, :, 1])
        if t >= 5:
            signals[t] = np.tanh(np.log(closing[t] / closing[t - 5]) / (0.01 * np.sqrt(5)))
    return opening, closing, signals


def configuration(seed):
    start = datetime(2020, 1, 1, 9, tzinfo=timezone.utc)
    sessions = [
        dict(
            session_id=f"s{i}",
            open_at=(start + timedelta(days=i)).isoformat(),
            close_at=(start + timedelta(days=i, hours=7)).isoformat(),
        )
        for i in range(260)
    ]
    contract = DataContract.model_validate_json(
        canonical(
            dict(
                schema_version=1,
                source_id=f"causal_engineering_seed_{seed}",
                provider_statement="Artificial causal engineering data; explicit complete constant "
                "membership and no corporate actions. Not a provider or market claim.",
                currency="TEST",
                price_basis="raw",
                frequency="daily_equities",
                calendar=dict(calendar_id="fictional_daily_260", timezone="UTC", sessions=sessions),
                instruments=list(SYMBOLS),
                knowledge_basis="assumed_historical_availability",
                availability_evidence_ref=None,
                availability_claim="explicit_assumption",
                historical_assumption="available_at_event_time",
                universe_provenance="source_documented_not_independently_verified",
                corporate_action_coverage="source_declares_complete_notices",
                columns={name: name for name in sorted(CSV_FIELDS)},
            )
        )
    )
    side = dict(commission_bps="1", slippage_bps="1", tax_bps="0")
    account = AccountRules.model_validate_json(
        canonical(
            dict(
                currency="TEST",
                currency_precision=2,
                rounding_policy="half_even_per_component",
                initial_cash="100000",
                cash_settlement="immediate_simulation",
                inventory_unlock="immediate_simulation",
                sizing="long_only_topk",
                allocation="0.8",
                max_positions=2,
                lot_size=1,
                cost_policy="linear_side_bps",
                buy_costs=side,
                sell_costs=side,
                cost_claim="explicit_observation_assumption",
                fill_policy="next_registered_open_market",
                fill_cap_units={symbol: 100000 for symbol in SYMBOLS},
                funding_policy="pre_funded_no_unfilled_sale_credit",
                valuation="close_fresh_quote_only",
                benchmark="cash_zero_interest",
                action_policy="notice_suspend_unresolved",
                termination="preserve_inventory",
                stop_policy="cancel_unelapsed_targets",
                max_drawdown="0.5",
            )
        )
    )
    return contract, account


def write_once(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"frozen preparation conflict: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(raw)


def prepare(seed, workspace):
    if seed not in {17, 29} or not workspace.is_absolute():
        raise ValueError("only the registered seeds and absolute workspace paths are supported")
    workspace = workspace.resolve()
    packet = workspace / "packet"
    packet.mkdir(parents=True, exist_ok=True)
    innovations = np.random.Generator(np.random.PCG64(seed)).standard_normal((260, 8, 2))
    opening, closing, signals = prices(innovations)
    future = innovations.copy()
    future[200:] += 1.0
    changed = prices(future)
    assert np.array_equal(opening[:200], changed[0][:200])
    assert np.array_equal(closing[:200], changed[1][:200])
    labels = opening[2:] / opening[1:-1] - 1
    expected = np.exp(0.006 * signals[:-2] + 0.004 * innovations[2:, :, 0]) - 1
    assert np.allclose(labels, expected, rtol=0, atol=5e-15)
    contract, account = configuration(seed)
    write_once(packet / "contract.json", canonical(contract.model_dump(mode="json")))
    write_once(packet / "account.json", canonical(account.model_dump(mode="json")))
    # Explicit time-major, then instrument-major CSV order; no signal/label feature columns.
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(contract.columns.values()), lineterminator="\n")
    writer.writeheader()
    for symbol in SYMBOLS:
        writer.writerow(
            dict(
                event_id=f"{symbol}.membership",
                revision=0,
                kind="UNIVERSE_STATUS",
                instrument=symbol,
                session_id=None,
                event_at=contract.calendar.sessions[0].open_at.isoformat(),
                source_available_at=contract.calendar.sessions[0].open_at.isoformat(),
                status="active",
                price=None,
                action=None,
                quality="valid",
            )
        )
    for t, session in enumerate(contract.calendar.sessions):
        for j, symbol in enumerate(SYMBOLS):
            for kind, stamp, price in [
                ("OPEN_QUOTE", session.open_at, opening[t, j]),
                ("CLOSE_QUOTE", session.close_at, closing[t, j]),
            ]:
                writer.writerow(
                    dict(
                        event_id=f"{symbol}.{kind}.{t}",
                        revision=0,
                        kind=kind,
                        instrument=symbol,
                        session_id=session.session_id,
                        event_at=stamp.isoformat(),
                        source_available_at=stamp.isoformat(),
                        status=None,
                        price=f"{price:.10f}",
                        action=None,
                        quality="valid",
                    )
                )
    write_once(packet / "invented.csv", output.getvalue().encode())
    os.environ["QUANT_RSI_WORKSPACE"] = str(workspace)
    imported = market_data.ingest(contract, packet / "invented.csv", request_id=f"causal-{seed}")
    write_once(packet / "import_receipt.json", canonical(imported))
    root = dict(
        schema_version=2,
        plan_type="development_research",
        development=dict(
            schema_version=1,
            plan_type="development",
            training_mode="M0",
            data=dict(
                source="local_event_snapshot",
                sessions=260,
                instruments=8,
                train_cutoff=160,
                seed=0,
                snapshot_ref=imported["snapshot_ref"],
                decision_delay_seconds=60,
            ),
            portfolio=account.model_dump(mode="json"),
            budget=dict(max_attempts=8, worker_timeout_seconds=20, max_worker_seconds=160),
            annualization=252,
            primary_metric="annualized_mean_net_active_return",
            min_development_sessions=10,
            baseline="momentum_2",
            proposals=[
                dict(
                    name="momentum_2",
                    hypothesis="Frozen two-session engineering baseline",
                    factor=dict(op="momentum", window=2),
                )
            ],
        ),
        policy=dict(
            algorithm="neighbor_search_v1",
            window_min=2,
            window_max=3,
            min_improvement=0.0,
            max_no_progress=3,
            neighbor_order="window_first",
        ),
        neighbor_domain=dict(
            kind="RegisteredNeighborDomain",
            schema_version=1,
            operators=["momentum", "mean_return"],
            window_min=2,
            window_max=5,
            initial_window_max=3,
            cap_step=1,
            max_cap_advances=2,
        ),
        trajectory_registration=dict(
            kind="TrajectoryRegistration",
            schema_version=1,
            workspace=str(workspace),
            source_directory=str(workspace / "source"),
            worker_attempts=40,
            registered_worker_seconds=800,
            max_proposal_stages=2,
            max_generations=2,
        ),
    )
    plan = ResearchPlan.model_validate_json(canonical(root))
    write_once(packet / "root_plan.json", canonical(plan.model_dump(mode="json")))
    write_once(packet / "runtime.json", canonical(controller.runtime_identity()))
    write_once(packet / "generator.py", Path(__file__).read_bytes())
    data = local_research.LocalResearchData.from_json(
        local_research.export_local_data(plan.development.data)
    )
    checks = []
    for window in (2, 3, 4, 5):
        started = time.monotonic()
        x, y, metadata, mask = local_research.training_view(
            data, plan.development, FactorSpec(op="momentum", window=window)
        )
        assert x.shape == (8 * (159 - window), 3) and y.shape == (8 * (159 - window),)
        write_once(packet / f"mask_w{window}.json", canonical(mask))
        checks.append(
            dict(
                window=window,
                rows=len(x),
                columns=x.shape[1],
                **metadata,
                feature_values_ref=digest(canonical(x.tolist())),
                labels_ref=digest(canonical(y.tolist())),
            )
        )
        print(
            f"MASK_DONE seed={seed} window={window} rows={len(x)} "
            f"seconds={time.monotonic() - started:.3f}",
            flush=True,
        )
    write_once(
        packet / "no_fit_checks.json",
        canonical(
            dict(
                future_innovation_prefix_equal=True,
                causal_target_identity=True,
                model_fits=0,
                mask_checks=checks,
                no_signal_or_target_csv_columns=True,
            )
        ),
    )
    write_once(
        packet / "recipe.json",
        canonical(
            dict(
                raw_seed=seed,
                generator="numpy.random.Generator(PCG64)",
                numpy_version=np.__version__,
                draw_shape=[260, 8, 2],
                draw_order="C order: session, instrument, [z,e]",
                initial_open="O_0=100 exactly; z_0 drawn but unused",
                warmup_signal="s_t=0 for t<5",
                open_rule="O_t=O_(t-1)*exp(.006*s_(t-2)+.004*z_t) for t>=1; missing signal is zero",
                close_rule="C_t=O_t*exp(.003*e_t)",
                signal_rule="tanh(log(C_t/C_(t-5))/(.01*sqrt(5))) for t>=5",
                csv_price_precision="10 fixed decimal places; recursion uses unrounded float64",
                symbols=list(SYMBOLS),
                sessions=260,
                timezone="UTC",
                first_open="2020-01-01T09:00:00+00:00",
                cadence="fictional consecutive calendar days, 09:00 open / 16:00 close",
                knowledge="assumed historical availability; "
                "actual first_seen is the real import receipt",
                controls="same generator/account/model/domain/budgets across seeds; "
                "no fitting in preparation",
            )
        ),
    )
    manifest = {
        str(p.relative_to(packet)): digest(p.read_bytes())
        for p in sorted(packet.iterdir())
        if p.is_file() and p.name != "manifest.json"
    }
    write_once(
        packet / "manifest.json",
        canonical(dict(files=manifest, fingerprint=digest(canonical(manifest)))),
    )
    print(
        f"PREPARED seed={seed} packet={packet} fingerprint={digest(canonical(manifest))}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int, choices=(17, 29))
    parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.seed, args.workspace)
