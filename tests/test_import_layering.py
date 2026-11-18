"""Package layering ratchet: no new upward imports; the recorded backlog only shrinks.

Target order (#159 D4), lower layers never importing higher ones:
core < market (data) < factors < evaluation/execution/forecast
< research/intraday/brokers < cli/worker < __main__.
Imports within one rank are allowed. Existing upward imports are listed in
import_layering_allowlist.json as "<file relative to src/quant_rsi> -> <package>".
"""

import ast
import json
from pathlib import Path

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
    "worker": 5,
    "__main__": 6,
}


def upward_imports():
    found = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE)
        source = relative.parts[0].removesuffix(".py")
        if source not in RANK:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                parts = module.split(".")
                if parts[0] != "quant_rsi" or len(parts) < 2:
                    continue
                target = parts[1]
                if target != source and RANK.get(target, -1) > RANK[source]:
                    found.add(f"{relative.as_posix()} -> {target}")
    return found


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
