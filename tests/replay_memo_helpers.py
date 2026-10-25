"""Synthetic ledger structure only; no research evidence, fitting or worker grants."""

import os

from quant_rsi import controller
from quant_rsi.artifacts import ArtifactStore, canonical, digest
from quant_rsi.trajectory_state import anchor_path, initialize_ledger, locked_trajectory


def unit_ledger(workspace, *, version=2):
    os.environ["QUANT_RSI_WORKSPACE"] = str(workspace)
    with locked_trajectory(bootstrap=True) as (root, owner_fd):
        store = ArtifactStore(root / "artifacts")
        head = dict(semantic_ref=digest(b"unit semantic"), package_ref=digest(b"unit package"))
        program = dict(
            kind="DevelopmentTrajectoryProgram",
            schema_version=version,
            authority_id="unit-authority",
            authority_root=str(root),
            runtime_ref=store.put_json(controller.runtime_identity()),
            initial_head=head,
            initial_costs=dict(worker_attempts=0, registered_worker_seconds=0),
            input=dict(
                schema_version=version,
                name="scope lifecycle unit ledger, not research evidence",
                source_directory=str(workspace / "unused_source"),
                source_checkpoint_ref=digest(b"unbacked unit source"),
                state_policy="reset",
                max_proposal_stages=2,
                max_generations=2 if version == 2 else 1,
                budget=dict(worker_attempts=0, registered_worker_seconds=0),
            ),
        )
        anchor = dict(
            kind="TrajectoryAuthority",
            authority_id=program["authority_id"],
            program_ref=store.put_json(program),
        )
        state = dict(
            authority_id=anchor["authority_id"],
            program_ref=anchor["program_ref"],
            head=head,
            generation=0,
            revision=0,
            semantic_history=[head["semantic_ref"]],
            stages=[],
            pending=None,
            status="ACTIVE",
            terminal_ref=None,
            decisions={},
        )
        initialize_ledger(root, anchor, state, owner_fd=owner_fd)
        anchor_path(root).write_bytes(canonical(anchor))
    return root, anchor
