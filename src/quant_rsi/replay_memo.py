"""Bounded pure-result reuse during one locked v2 controller operation.

Authority/CAS/metadata checks and fitting always remain outside this scope.
Private serialized values never cross an operation or process boundary.
"""

import os
import pickle
import threading
import time
import uuid
import warnings
from collections import OrderedDict
from contextvars import ContextVar
from decimal import getcontext
from pathlib import Path
from types import FunctionType

import numpy as np

from quant_rsi.artifacts import canonical, digest
from quant_rsi.schemas import DevelopmentPlan, FactorSpec

_ACTIVE = ContextVar("controller_pure_replay_scope", default=None)
KINDS = ("training_view", "development_predictions", "evaluate_local")
_STANDARD_WARNING_RECEIVER = None


def warnings_allow_reuse():
    """Fail closed for promoted warnings, custom receivers or unknown runtimes."""
    global _STANDARD_WARNING_RECEIVER
    try:
        if warnings.defaultaction == "error" or any(
            rule[0] == "error" for rule in warnings.filters
        ):
            return False
        original_show = getattr(warnings, "_showwarning_orig", None)
        original_format = getattr(warnings, "_formatwarning_orig", None)
        if not callable(original_show) or warnings.showwarning is not original_show:
            return False
        if not callable(original_format) or warnings.formatwarning is not original_format:
            return False
        receiver = warnings._showwarnmsg_impl
        if _STANDARD_WARNING_RECEIVER is None:
            if not (
                type(receiver) is FunctionType
                and receiver.__module__ == "warnings"
                and receiver.__name__ == "_showwarnmsg_impl"
                and receiver.__code__.co_name == "_showwarnmsg_impl"
                and receiver.__globals__ is vars(warnings)
                and Path(receiver.__code__.co_filename).resolve()
                == Path(warnings.__file__).resolve()
            ):
                return False
            _STANDARD_WARNING_RECEIVER = receiver
        return receiver is _STANDARD_WARNING_RECEIVER
    except Exception:
        return False


def numeric_environment():
    if np.geterrcall() is not None:
        raise TypeError("NumPy callback/log configuration is unsupported for reuse")
    context = getcontext()
    return dict(
        decimal={
            name: getattr(context, name)
            for name in ("prec", "rounding", "Emin", "Emax", "capitals", "clamp")
        }
        | dict(
            flags={signal.__name__: value for signal, value in context.flags.items()},
            traps={signal.__name__: value for signal, value in context.traps.items()},
        ),
        numpy_errors=np.geterr(),
    )


class _Pool:
    def __init__(self, max_entries=64, max_bytes=32 * 1024 * 1024):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.values = OrderedDict()
        self.bytes = 0
        self.stats = dict(
            hits=0,
            misses=0,
            computations=0,
            exceptions=0,
            evictions=0,
            oversized=0,
            key_fallbacks=0,
            serialization_skips=0,
            environment_changes=0,
            decode_failures=0,
            peak_entries=0,
            peak_payload_key_bytes=0,
            key_wall=0.0,
            compute_wall=0.0,
            encode_wall=0.0,
            decode_wall=0.0,
        )

    def remove(self, key):
        blob = self.values.pop(key)
        self.bytes -= len(blob) + len(key.encode())

    def clear(self):
        self.values.clear()
        self.bytes = 0

    def put(self, key, blob):
        size = len(blob) + len(key.encode())
        if size > self.max_bytes or self.max_entries <= 0:
            self.stats["oversized"] += 1
            return
        if key in self.values:
            self.remove(key)
        while self.values and (
            len(self.values) >= self.max_entries or self.bytes + size > self.max_bytes
        ):
            self.remove(next(iter(self.values)))
            self.stats["evictions"] += 1
        self.values[key] = blob
        self.bytes += size
        self.stats["peak_entries"] = max(self.stats["peak_entries"], len(self.values))
        self.stats["peak_payload_key_bytes"] = max(self.stats["peak_payload_key_bytes"], self.bytes)
        assert len(self.values) <= self.max_entries and self.bytes <= self.max_bytes

    def report(self):
        return self.stats | dict(
            entries=len(self.values),
            retained_payload_key_bytes=self.bytes,
            max_entries=self.max_entries,
            max_bytes=self.max_bytes,
        )


class CalculationScope:
    def __init__(self, runtime, *, max_entries=64, max_bytes=32 * 1024 * 1024):
        self.runtime = digest(canonical(runtime))
        self.operation = uuid.uuid4().hex
        self.pid = os.getpid()
        self.thread = threading.get_ident()
        self.closed = False
        self.pools = {name: _Pool(max_entries, max_bytes) for name in KINDS}

    def usable(self):
        return (
            not self.closed
            and _ACTIVE.get() is self
            and self.pid == os.getpid()
            and self.thread == threading.get_ident()
        )

    def invalidate(self):
        for pool in self.pools.values():
            pool.clear()

    def key(self, kind, args, *, environment=None):
        from quant_rsi.local_research import LocalResearchData

        if (
            len(args)
            != {"training_view": 3, "development_predictions": 4, "evaluate_local": 3}[kind]
        ):
            raise TypeError("unsupported arity: call original function")
        data, plan, *rest = args
        if type(data) is not LocalResearchData or type(plan) is not DevelopmentPlan:
            raise TypeError("unsupported argument class: use original computation")
        if kind in ("training_view", "development_predictions") and type(rest[0]) is not FactorSpec:
            raise TypeError("unsupported factor class")
        if kind == "development_predictions" and type(rest[1]) is not str:
            raise TypeError("unsupported model text class")
        if kind == "evaluate_local" and type(rest[0]) is not list:
            raise TypeError("unsupported predictions class")
        extra = (
            {"factor": rest[0].model_dump(mode="json")}
            if kind != "evaluate_local"
            else {"predictions": rest[0]}
        )
        if kind == "development_predictions":
            extra["model_text_utf8_hex"] = rest[1].encode("utf-8").hex()
        return digest(
            canonical(
                dict(
                    format=1,
                    kind=kind,
                    runtime=self.runtime,
                    operation=self.operation,
                    numeric_environment=numeric_environment()
                    if environment is None
                    else environment,
                    data=dict(
                        spec=data.spec.model_dump(mode="json"),
                        contract=data.contract.model_dump(mode="json"),
                        snapshot=data.snapshot,
                        dates=data.dates,
                        instruments=data.instruments,
                        value=data.value,
                    ),
                    plan=plan.model_dump(mode="json"),
                    extra=extra,
                )
            )
        )

    def call(self, kind, original, args):
        if not self.usable() or not warnings_allow_reuse():
            return original(*args)
        pool = self.pools[kind]
        start = time.perf_counter()
        try:
            environment = numeric_environment()
            key = self.key(kind, args, environment=environment)
        except Exception:
            pool.stats["key_fallbacks"] += 1
            return original(*args)
        finally:
            pool.stats["key_wall"] += time.perf_counter() - start
        if key in pool.values:
            start = time.perf_counter()
            try:
                result = pickle.loads(pool.values[key])
            except Exception:
                pool.stats["decode_failures"] += 1
                pool.remove(key)
            else:
                pool.values.move_to_end(key)
                pool.stats["hits"] += 1
                return result
            finally:
                pool.stats["decode_wall"] += time.perf_counter() - start
        pool.stats["misses"] += 1
        pool.stats["computations"] += 1
        start = time.perf_counter()
        try:
            result = original(*args)
        except Exception:
            pool.stats["exceptions"] += 1
            raise
        finally:
            pool.stats["compute_wall"] += time.perf_counter() - start
        try:
            unchanged = numeric_environment() == environment
        except Exception:
            unchanged = False
        if not unchanged:
            pool.stats["environment_changes"] += 1
            return result
        start = time.perf_counter()
        try:
            blob = pickle.dumps(result, protocol=5)
        except Exception:
            pool.stats["serialization_skips"] += 1
        else:
            pool.put(key, blob)
        finally:
            pool.stats["encode_wall"] += time.perf_counter() - start
        return result

    def report(self):
        return {name: pool.report() for name, pool in self.pools.items()}

    def __enter__(self):
        if self.closed:
            raise ValueError("scope already closed")
        self.outer = _ACTIVE.get()
        if self.outer:
            raise RuntimeError("only the outer operation creates a scope; nested calls reuse it")
        self.token = _ACTIVE.set(self)
        return self

    def __exit__(self, *args):
        self.invalidate()
        self.closed = True
        _ACTIVE.reset(self.token)


def calculate(kind, original, args):
    """Static shims use the current scope; workers and standalone calls stay cold."""
    scope = _ACTIVE.get()
    if scope is None:
        return original(*args)
    return scope.call(kind, original, args)
