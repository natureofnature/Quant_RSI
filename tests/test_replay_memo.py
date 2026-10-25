"""Real API/lock/ledger interfaces with explicitly synthetic no-fit ledger fixtures."""

import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import pytest
from replay_memo_helpers import unit_ledger

from quant_rsi import replay_memo, trajectory, trajectory_delegation, trajectory_multigen
from quant_rsi.trajectory_state import TrajectoryLedger, locked_trajectory, require_owner


@pytest.fixture
def no_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("no-fit scope tests attempted fitting or worker dispatch")

    monkeypatch.setattr(lgb, "train", forbidden)
    monkeypatch.setattr(lgb.LGBMModel, "fit", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.mark.parametrize("failure", [None, "first_summary", "step", "final_summary"])
def test_real_api_scope_lifecycle(tmp_path, monkeypatch, no_fit, failure):
    root, _ = unit_ledger(tmp_path)
    events, scopes, ledgers = [], [], []

    class TracedLedger(TrajectoryLedger):
        def __init__(self, *args, **kwargs):
            assert replay_memo._ACTIVE.get() is None
            super().__init__(*args, **kwargs)
            require_owner(self.root, self.owner_fd)
            events.append("ledger_open")
            ledgers.append(self)

        def __exit__(self, *args):
            assert replay_memo._ACTIVE.get() is None
            require_owner(self.root, self.owner_fd)
            super().__exit__(*args)
            events.append("ledger_closed")

    original_scope = replay_memo.CalculationScope

    class TracedScope(original_scope):
        def __enter__(self):
            require_owner(root, ledgers[-1].owner_fd)
            result = super().__enter__()
            self.pools["training_view"].put("cleanup_marker", b"not a result")
            scopes.append(self)
            events.append("scope_enter")
            return result

        def __exit__(self, *args):
            super().__exit__(*args)
            assert all(not pool.values and pool.bytes == 0 for pool in self.pools.values())
            events.append("scope_exit")

    summaries = []

    def summary(authority):
        phase = "first_summary" if not summaries else "final_summary"
        assert replay_memo._ACTIVE.get() is scopes[0]
        require_owner(root, authority.owner_fd)
        authority.verify_chain()
        summaries.append(phase)
        events.append(phase)
        if failure == phase:
            raise RuntimeError(phase)
        return {"unit_summary": phase}

    def step(authority, owner_fd):
        assert replay_memo._ACTIVE.get() is scopes[0]
        require_owner(root, owner_fd)
        events.append("step")
        if failure == "step":
            raise RuntimeError("step")

    monkeypatch.setattr(trajectory, "TrajectoryLedger", TracedLedger)
    monkeypatch.setattr(replay_memo, "CalculationScope", TracedScope)
    monkeypatch.setattr(trajectory, "summarize", summary)
    monkeypatch.setattr(trajectory_multigen, "resume_step", step)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            trajectory.resume_trajectory()
    else:
        assert trajectory.resume_trajectory() == {"unit_summary": "final_summary"}
        assert events == [
            "ledger_open",
            "scope_enter",
            "first_summary",
            "step",
            "final_summary",
            "scope_exit",
            "ledger_closed",
        ]
    assert events[-2:] == ["scope_exit", "ledger_closed"]
    assert replay_memo._ACTIVE.get() is None and scopes[0].closed
    with pytest.raises(sqlite3.ProgrammingError):
        ledgers[0].connection.execute("SELECT 1")
    with locked_trajectory() as (again, fd):
        require_owner(again, fd)


def test_legacy_api_stays_uncached_and_steps_validate_first(tmp_path, monkeypatch, no_fit):
    unit_ledger(tmp_path, version=1)

    def forbidden_scope(*args, **kwargs):
        pytest.fail("legacy or rejected steps must not create a scope")

    monkeypatch.setattr(replay_memo, "CalculationScope", forbidden_scope)
    monkeypatch.setattr(trajectory, "summarize", lambda authority: {"legacy": True})
    monkeypatch.setattr(trajectory, "reserve_stage", lambda authority: None)
    assert trajectory.resume_trajectory() == {"legacy": True}
    with pytest.raises(ValueError):
        trajectory.resume_trajectory(steps=0)


def test_actual_api_busy_lock_precedes_scope(tmp_path, monkeypatch, no_fit):
    unit_ledger(tmp_path)

    def forbidden_scope(*args, **kwargs):
        pytest.fail("busy authority must fail before scope creation")

    monkeypatch.setattr(replay_memo, "CalculationScope", forbidden_scope)
    with locked_trajectory():
        with pytest.raises(ValueError, match="authority is busy"):
            trajectory.resume_trajectory()
    assert replay_memo._ACTIVE.get() is None
    with locked_trajectory() as (root, fd):
        require_owner(root, fd)


def test_scope_constructor_failure_releases_real_ledger_and_lock(tmp_path, monkeypatch, no_fit):
    unit_ledger(tmp_path)
    opened = []

    class TracedLedger(TrajectoryLedger):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)

        def __exit__(self, *args):
            require_owner(self.root, self.owner_fd)
            assert replay_memo._ACTIVE.get() is None
            super().__exit__(*args)

    def broken_scope(*args, **kwargs):
        raise RuntimeError("scope constructor sentinel")

    monkeypatch.setattr(trajectory, "TrajectoryLedger", TracedLedger)
    monkeypatch.setattr(replay_memo, "CalculationScope", broken_scope)
    with pytest.raises(RuntimeError, match="scope constructor sentinel"):
        trajectory.resume_trajectory()
    assert len(opened) == 1 and replay_memo._ACTIVE.get() is None
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].connection.execute("SELECT 1")
    with locked_trajectory() as (root, fd):
        require_owner(root, fd)


@pytest.mark.parametrize("missing", [False, True])
def test_original_ledger_rejects_bad_cas_before_scope(tmp_path, monkeypatch, no_fit, missing):
    root, anchor = unit_ledger(tmp_path)
    path = root / "artifacts" / anchor["program_ref"]
    if missing:
        path.unlink()
    else:
        path.write_bytes(b"corrupt isolated unit CAS")

    def forbidden_scope(*args, **kwargs):
        pytest.fail("corrupt/missing ledger input must fail before scope creation")

    monkeypatch.setattr(replay_memo, "CalculationScope", forbidden_scope)
    with pytest.raises((ValueError, FileNotFoundError)):
        trajectory.resume_trajectory()
    assert replay_memo._ACTIVE.get() is None
    with locked_trajectory() as (again, fd):
        require_owner(again, fd)


def test_copied_authority_rejected_by_actual_v2_step(tmp_path, monkeypatch, no_fit):
    original = tmp_path / "original"
    unit_ledger(original)
    copied = tmp_path / "copied"
    shutil.copytree(original, copied)
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(copied))
    # Source evidence is intentionally absent; isolate the real copied-path guard.
    monkeypatch.setattr(trajectory, "summarize", lambda authority: {"unit": "scope only"})
    with pytest.raises(ValueError, match="copied or relocated trajectory cannot dispatch"):
        trajectory.resume_trajectory()
    assert replay_memo._ACTIVE.get() is None


def test_actual_fd_and_current_grant_rejections_inside_scope(tmp_path, monkeypatch, no_fit):
    root, anchor = unit_ledger(tmp_path)
    child = root / "children" / "unit"
    context = dict(
        kind="TrajectoryC4Context",
        authority_root=str(root),
        authority_id=anchor["authority_id"],
        authorization_ref="a" * 64,
    )
    with locked_trajectory() as (_, owner_fd), TrajectoryLedger(root, owner_fd=owner_fd) as ledger:
        with replay_memo.CalculationScope({"fixture": "not research evidence"}):
            with (tmp_path / "foreign_fd").open("w+") as foreign:
                with pytest.raises(ValueError, match="actual owner lock"):
                    trajectory_delegation.authorize_package(
                        child, foreign.fileno(), context=context
                    )
            with pytest.raises(ValueError, match="active parent operation"):
                trajectory_delegation.authorize_package(child, owner_fd, context=context)

            def reserve_unit(state):
                state["stages"] = [
                    dict(
                        ordinal=0,
                        authorization_ref="b" * 64,
                        capacity=dict(worker_attempts=0, registered_worker_seconds=0),
                        consumed=dict(worker_attempts=0, registered_worker_seconds=0),
                    )
                ]
                state["pending"] = dict(
                    kind="package_round",
                    authorization_ref="b" * 64,
                    steps=1,
                    stage=0,
                    child_campaign_id="unit-only",
                )

            ledger.update("unit_zero_capacity_pending", reserve_unit)
            before = (root / "control.sqlite").read_bytes()
            with pytest.raises(ValueError, match="active parent operation"):
                trajectory_delegation.authorize_package(child, owner_fd, context=context)
            assert (root / "control.sqlite").read_bytes() == before


def test_supported_clean_interpreter_and_exec_boundary(tmp_path):
    """A no-fit subprocess probe is not a business worker or FD acceptance run."""
    output = tmp_path / "clean_probe.log"
    command = [sys.executable, str(Path(__file__).with_name("replay_memo_probe.py")), str(tmp_path)]
    with output.open("w") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    assert completed.returncode == 0, f"clean interpreter probe failed: {output}"
