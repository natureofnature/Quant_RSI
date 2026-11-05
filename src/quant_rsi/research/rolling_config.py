"""Instrument-owned window settings for the existing rolling research entry point."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.factors.contracts import InstrumentId


class RollingWindowProfile(StrictModel):
    schema_version: Literal[1]
    kind: Literal["RollingWindowProfile"]
    instrument: InstrumentId
    training_window_sessions: int = Field(ge=61, le=445)
    horizon_sessions: int = Field(ge=1, le=20)

    @field_validator("schema_version", mode="before")
    @classmethod
    def strict_schema_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def enough_mature_samples(self):
        if self.training_window_sessions < 60 + self.horizon_sessions:
            raise ValueError("Training window needs 30 warm-up rows, horizon rows and 30 labels")
        return self


def load_window_profile(path, symbol):
    raw = Path(path).read_bytes()
    profile = RollingWindowProfile.model_validate_json(raw)
    if profile.instrument != symbol:
        raise ValueError("Window profile instrument differs from the requested symbol")
    return profile, raw
