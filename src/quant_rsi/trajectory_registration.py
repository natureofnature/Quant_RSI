"""One gen0 identity and monotonic handoff, using the existing trajectory owner lock."""

import uuid
from contextlib import contextmanager
from functools import wraps

from quant_rsi import controller
from quant_rsi.artifacts import ArtifactStore, atomic_write, canonical, decode, digest
from quant_rsi.researcher import ResearchPlan
from quant_rsi.task_definition import registration_ref, require_disjoint_source
from quant_rsi.trajectory_state import (
    TrajectoryPlan,
    anchor_path,
    locked_trajectory,
    require_owner,
    trajectory_root,
)


def registration_path():
    return trajectory_root().parent / "trajectory_registration.json"


def identity_definition(plan, campaign_id):
    registration = plan.trajectory_registration
    return dict(
        kind="RegisteredGen0Identity",
        schema_version=1,
        registration_ref=registration_ref(plan),
        research_plan_ref=digest(canonical(plan.model_dump(mode="json"))),
        runtime_ref=digest(canonical(controller.runtime_identity())),
        campaign_id=campaign_id,
        workspace=str(registration.workspace),
        source_directory=str(registration.source_directory),
        registration=registration.model_dump(mode="json"),
    )


def require_root_identity(plan, root, campaign_id, owner_fd):
    require_owner(trajectory_root(), owner_fd)
    registration = plan.trajectory_registration
    if (
        registration is None
        or plan.package_lineage is not None
        or registration.workspace.resolve() / ".quant_rsi" / "trajectory" != trajectory_root()
        or root.resolve() != registration.source_directory.resolve()
    ):
        raise ValueError("registered gen0 belongs to another source/workspace")
    record = decode(registration_path().read_bytes())
    if (
        record["kind"] != "TrajectoryRootRegistration"
        or record["state"] != "ROOT_ACTIVE"
        or record["handoff"] is not None
        or anchor_path(trajectory_root()).exists()
        or (trajectory_root().parent / "trajectory_initialization.json").exists()
        or record["identity"] != identity_definition(plan, campaign_id)
    ):
        raise ValueError("registered gen0 is handed off or has a different immutable identity")
    return record["identity"]


@contextmanager
def root_execution(plan, root, campaign_id, owner_fd, *, create):
    """Acquire trajectory before C2; its FD is inherited by any real root worker."""
    registration = plan.trajectory_registration
    require_disjoint_source(root, trajectory_root())
    if (
        registration.workspace.resolve() / ".quant_rsi" / "trajectory" != trajectory_root()
        or root.resolve() != registration.source_directory.resolve()
        or plan.package_lineage is not None
    ):
        raise ValueError("registered gen0 belongs to another source/workspace")
    if owner_fd is None:
        if not create and not registration_path().is_file():
            raise ValueError("registered gen0 identity is missing")
        with locked_trajectory(bootstrap=True) as (_, owned):
            with root_execution(plan, root, campaign_id, owned, create=create) as result:
                yield result
        return
    require_owner(trajectory_root(), owner_fd)
    path = registration_path()
    if not path.exists():
        if not create:
            raise ValueError("registered gen0 identity is missing")
        if (root / "manifest.json").exists() or (root / "research_initialization.json").exists():
            raise ValueError("existing gen0 cannot acquire a replacement registration")
        if (
            anchor_path(trajectory_root()).exists()
            or (trajectory_root().parent / "trajectory_initialization.json").exists()
        ):
            raise ValueError("existing trajectory cannot acquire a replacement root registration")
        campaign_id = campaign_id or uuid.uuid4().hex
        record = dict(
            kind="TrajectoryRootRegistration",
            state="ROOT_ACTIVE",
            identity=identity_definition(plan, campaign_id),
            handoff=None,
        )
        atomic_write(path, canonical(record))
    else:
        record = decode(path.read_bytes())
        campaign_id = campaign_id or record["identity"]["campaign_id"]
    require_root_identity(plan, root, campaign_id, owner_fd)
    yield campaign_id, owner_fd


def registered_creation(function):
    @wraps(function)
    def wrapped(plan, root, **kwargs):
        if plan.schema_version == 2 and kwargs.get("launch_context") is None:
            with root_execution(
                plan,
                root,
                kwargs.get("campaign_id"),
                kwargs.get("trajectory_owner_fd"),
                create=True,
            ) as (identity, fd):
                return function(
                    plan, root, **(kwargs | dict(campaign_id=identity, trajectory_owner_fd=fd))
                )
        return function(plan, root, **kwargs)

    return wrapped


def registered_resume(function):
    @wraps(function)
    def wrapped(root, **kwargs):
        saved = persisted_root_plan(root)
        if saved is not None:
            plan, campaign_id = saved
            with root_execution(
                plan, root, campaign_id, kwargs.get("trajectory_owner_fd"), create=False
            ) as (_, fd):
                return function(root, **(kwargs | dict(trajectory_owner_fd=fd)))
        return function(root, **kwargs)

    return wrapped


def persisted_root_plan(root):
    """Read only enough persisted identity to enter the correct lock hierarchy."""
    marker = root / "research_initialization.json"
    if marker.exists():
        saved = decode(marker.read_bytes())
        if saved.get("launch_context") is not None:
            return None
        plan = ResearchPlan.model_validate_json(canonical(saved["plan"]))
        return (plan, saved["campaign_id"]) if plan.schema_version == 2 else None
    path = root / "manifest.json"
    if path.exists():
        manifest = decode(path.read_bytes())
        if "launch_context_ref" in manifest:
            return None
        store = ArtifactStore(root / "artifacts")
        plan = ResearchPlan.model_validate_json(store.get(manifest["research_plan_ref"]))
        return (plan, manifest["campaign_id"]) if plan.schema_version == 2 else None
    return None


def match_trajectory_input(plan, registration):
    if (
        plan.source_directory.resolve() != registration.source_directory.resolve()
        or plan.budget.worker_attempts != registration.worker_attempts
        or plan.budget.registered_worker_seconds != registration.registered_worker_seconds
        or plan.max_proposal_stages != registration.max_proposal_stages
        or plan.max_generations != registration.max_generations
    ):
        raise ValueError("trajectory input differs from gen0's preregistered limits/source")


def verify_root_manifest(plan, store, manifest):
    if plan.schema_version == 2 and "launch_context_ref" not in manifest:
        if store.json(manifest["root_registration_identity_ref"]) != identity_definition(
            plan, manifest["campaign_id"]
        ):
            raise ValueError("gen0 manifest differs from its immutable registration identity")


def begin_handoff(plan, owner_fd):
    """ROOT_ACTIVE -> HANDOFF is the sole authoritative dispatch cutover."""
    from quant_rsi import research
    from quant_rsi.trajectory import substantive
    from quant_rsi.trajectory_evidence import capture_research

    require_owner(trajectory_root(), owner_fd)
    require_disjoint_source(plan.source_directory, trajectory_root())
    marker = trajectory_root().parent / "trajectory_initialization.json"
    if not marker.exists() and trajectory_root().exists() and any(trajectory_root().iterdir()):
        raise ValueError("trajectory data exists without its initialization authority")
    path = registration_path()
    record = decode(path.read_bytes())
    inputs = substantive(plan.model_dump(mode="json"))
    if record["state"] == "HANDOFF":
        if (
            record["handoff"]["input"] != inputs
            or record["identity"]["workspace"] != str(trajectory_root().parent.parent)
            or record["identity"]["runtime_ref"] != digest(canonical(controller.runtime_identity()))
        ):
            raise ValueError("registered handoff input/runtime/workspace conflict")
        return record
    source = plan.source_directory.resolve()
    with controller.controller_lock(source):
        journal, store, manifest, source_plan, data = research.open_research(source)
        try:
            if source_plan.development.data.source != "local_event_snapshot":
                raise ValueError("registered handoff requires an actual local C2 root")
            require_root_identity(source_plan, source, manifest["campaign_id"], owner_fd)
            match_trajectory_input(plan, source_plan.trajectory_registration)
            if store.json(manifest["root_registration_identity_ref"]) != record["identity"]:
                raise ValueError("handoff source lacks its original gen0 registration")
        finally:
            journal.close()
    # The outer trajectory lock remains held; capture acquires the C2 lock exactly once.
    receipt_ref, receipt = capture_research(
        source, None, publish=None, initial=True, expected_checkpoint=plan.source_checkpoint_ref
    )
    record = record | dict(
        state="HANDOFF",
        handoff=dict(
            authority_id=uuid.uuid4().hex,
            input=inputs,
            checkpoint_ref=receipt["checkpoint_ref"],
            source_manifest_ref=receipt["manifest_ref"],
            source_receipt_ref=receipt_ref,
            costs=receipt["costs"],
        ),
    )
    atomic_write(path, canonical(record))
    return record


def verify_frozen_handoff(program, initial, store):
    """Readonly verification uses only frozen CAS, never the original registration path."""
    from quant_rsi.trajectory import substantive

    record = store.json(program["root_registration_ref"])
    source_plan = ResearchPlan.model_validate_json(store.get(initial["research_plan_ref"]))
    manifest = store.json(initial["manifest_ref"])
    match_trajectory_input(
        TrajectoryPlan.model_validate_json(canonical(program["input"])),
        source_plan.trajectory_registration,
    )
    if (
        record["state"] != "HANDOFF"
        or program["authority_root"]
        != str(source_plan.trajectory_registration.workspace / ".quant_rsi" / "trajectory")
        or record["identity"] != identity_definition(source_plan, initial["campaign_id"])
        or store.json(manifest["root_registration_identity_ref"]) != record["identity"]
        or record["handoff"]
        != dict(
            authority_id=program["authority_id"],
            input=substantive(program["input"]),
            checkpoint_ref=initial["checkpoint_ref"],
            source_manifest_ref=initial["manifest_ref"],
            source_receipt_ref=program["initial_head"]["research_receipt_ref"],
            costs=initial["costs"],
        )
    ):
        raise ValueError("trajectory frozen root registration/handoff mismatch")
