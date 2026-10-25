"""Canonical task identity, including an explicitly registered finite search domain."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from quant_rsi.artifacts import canonical, digest
from quant_rsi.contracts_base import StrictModel


class RegisteredNeighborDomain(StrictModel):
    kind: Literal["RegisteredNeighborDomain"]
    schema_version: Literal[1]
    operators: tuple[Literal["momentum"], Literal["mean_return"]]
    window_min: int = Field(strict=True, ge=1, le=30)
    window_max: int = Field(strict=True, ge=1, le=30)
    initial_window_max: int = Field(strict=True, ge=1, le=30)
    cap_step: Literal[1]
    max_cap_advances: Literal[2]

    @model_validator(mode="after")
    def valid_bounds(self) -> Self:
        if not self.window_min <= self.initial_window_max <= self.window_max:
            raise ValueError("initial cap must belong to the registered global domain")
        return self

    @property
    def caps(self):
        return tuple(
            range(self.initial_window_max, min(self.initial_window_max + 2, self.window_max) + 1)
        )


class TrajectoryRegistration(StrictModel):
    kind: Literal["TrajectoryRegistration"]
    schema_version: Literal[1]
    workspace: Path
    source_directory: Path
    worker_attempts: int = Field(strict=True, ge=1)
    registered_worker_seconds: int = Field(strict=True, ge=1)
    max_proposal_stages: int = Field(strict=True, ge=1, le=2)
    max_generations: int = Field(strict=True, ge=1, le=2)

    @model_validator(mode="after")
    def fixed_paths(self) -> Self:
        if not self.workspace.is_absolute() or not self.source_directory.is_absolute():
            raise ValueError("trajectory registration requires absolute fixed paths")
        if (
            self.workspace != self.workspace.resolve()
            or self.source_directory != self.source_directory.resolve()
        ):
            raise ValueError("trajectory registration paths must be canonical")
        require_disjoint_source(self.source_directory, self.workspace / ".quant_rsi" / "trajectory")
        return self


def require_disjoint_source(source, authority):
    source, authority = source.resolve(), authority.resolve()
    if source == authority or source in authority.parents or authority in source.parents:
        raise ValueError("registered root source and trajectory authority must be disjoint")


def domain_ref(plan):
    return digest(canonical(plan.neighbor_domain.model_dump(mode="json")))


def registration_ref(plan):
    return digest(canonical(plan.trajectory_registration.model_dump(mode="json")))


def task_definition(plan, data_ref, template_ref):
    value = dict(data_ref=data_ref, development_template_ref=template_ref)
    if plan.neighbor_domain is not None:
        value.update(
            schema_version=2,
            registered_neighbor_domain_ref=domain_ref(plan),
        )
    return value


def verify_domain(plan, store):
    if plan.neighbor_domain is not None:
        if store.json(domain_ref(plan)) != plan.neighbor_domain.model_dump(mode="json"):
            raise ValueError("registered task domain differs from its frozen definition")
        if store.json(registration_ref(plan)) != plan.trajectory_registration.model_dump(
            mode="json"
        ):
            raise ValueError("registered task authority limits differ from their frozen definition")
