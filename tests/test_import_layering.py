"""Package layering ratchet: no new upward imports; the recorded backlog only shrinks.

Target order (#159 D4), lower layers never importing higher ones:
core < market (data) < factors < evaluation/execution/forecast
< research/intraday/brokers < cli/legacy < __main__.
Frozen pipeline A lives in legacy/ (#159 D2) at the rank of cli: only cli and other
legacy modules may import it; within legacy, ranks are not checked.
Imports within one rank are allowed. Existing upward imports are listed in
import_layering_allowlist.json as "<file relative to src/quant_rsi> -> <package>".
"""

import ast
import json
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "quant_rsi"
ALLOWLIST = Path(__file__).with_name("import_layering_allowlist.json")
RANK = {
    "core": 0,
    "market": 1,
    "factors": 2,
    "evaluation": 3,
    "execution": 3,
    "forecast": 3,
    "research": 4,
    "intraday": 4,
    "brokers": 4,
    "cli": 5,
    "legacy": 5,
    "__main__": 6,
}


def imported_packages(source, relative):
    """Top-level quant_rsi packages imported by one module, whatever the import form."""
    # Relative imports resolve against the containing package: for "core/x.py" and for
    # "core/__init__.py" alike, that is quant_rsi.core.
    package = ["quant_rsi", *Path(relative).parent.parts]
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            bases = [(alias.name, None) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level > len(package):
                    continue  # Beyond the top-level package; not a quant_rsi import.
                anchor = package[: len(package) - (node.level - 1)]
                base = ".".join(anchor + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            bases = [(base, [alias.name for alias in node.names])]
        else:
            continue
        for base, names in bases:
            segments = base.split(".")
            if segments[0] != "quant_rsi":
                continue
            if len(segments) > 1:
                found.add(segments[1])
            elif names:
                # "from quant_rsi import research" or "from .. import research".
                found.update(name for name in names if name != "*")
    return found


def upward_imports():
    found = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE)
        source = relative.parts[0].removesuffix(".py")
        if source not in RANK:
            continue
        for target in imported_packages(path.read_text(encoding="utf-8"), relative):
            if target != source and RANK.get(target, -1) > RANK[source]:
                found.add(f"{relative.as_posix()} -> {target}")
    return found


@pytest.mark.parametrize(
    ("statement", "relative", "expected"),
    [
        ("import quant_rsi.research", "core/probe.py", {"research"}),
        ("from quant_rsi import research", "core/probe.py", {"research"}),
        ("from .. import research", "core/probe.py", {"research"}),
        ("from ..research import study", "core/probe.py", {"research"}),
        ("from ..research.study import run_study", "core/probe.py", {"research"}),
        ("from . import research", "__init__.py", {"research"}),
        ("from .research import study", "__init__.py", {"research"}),
        ("from ...research import study", "forecast/recursive/probe.py", {"research"}),
        ("from . import artifacts", "core/probe.py", {"core"}),
        ("from quant_rsi.core.artifacts import digest", "market/probe.py", {"core"}),
        ("import json", "core/probe.py", set()),
    ],
)
def test_scanner_resolves_absolute_relative_and_package_alias_imports(
    statement, relative, expected
):
    assert imported_packages(statement + "\n", relative) == expected


def test_every_package_has_a_layer():
    packages = {p.name.removesuffix(".py") for p in PACKAGE.iterdir()}
    packages -= {"__init__", "__pycache__"}
    assert packages <= RANK.keys(), f"Assign a layer to: {sorted(packages - RANK.keys())}"


def test_no_new_upward_imports_and_backlog_only_shrinks():
    allowed = set(json.loads(ALLOWLIST.read_text(encoding="utf-8")))
    current = upward_imports()
    new = sorted(current - allowed)
    assert not new, "New upward imports violate the layer order (#159 D4):\n" + "\n".join(new)
    stale = sorted(allowed - current)
    assert not stale, "Remove resolved entries from import_layering_allowlist.json:\n" + "\n".join(
        stale
    )
