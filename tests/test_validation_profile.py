"""Exercise the real profiling command with tiny offline pytest modules."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/profile_validation.py"


def test_critical_selection_defers_only_named_modules_and_full_keeps_every_module(tmp_path):
    functions = runpy.run_path(str(SCRIPT))
    select = functions["select_tests"]
    select.__globals__["ROOT"] = tmp_path
    tests = tmp_path / "tests"
    tests.mkdir()
    for name in ("test_controller.py", "test_trajectory.py", "new_feature_test.py"):
        (tests / name).write_text("def test_example(): pass\n")
    nested = tests / "nested"
    nested.mkdir()
    (nested / "test_trajectory.py").write_text("def test_new_scenario(): pass\n")
    critical, deferred = select([], "critical")
    assert critical == [
        "tests/nested/test_trajectory.py",
        "tests/new_feature_test.py",
        "tests/test_controller.py",
    ]
    assert set(deferred) == {"tests/test_trajectory.py"}
    full, deferred = select([], "full")
    assert full == critical + ["tests/test_trajectory.py"]
    assert deferred == {}
    explicit = ["tests/test_trajectory.py::test_example"]
    assert select(explicit, "critical") == (explicit, {})


def test_extended_manifest_names_existing_modules_and_retains_primary_safety_coverage():
    functions = runpy.run_path(str(SCRIPT))
    for path in functions["EXTENDED_MODULES"]:
        assert (ROOT / path).is_file()
    critical, _ = functions["select_tests"]([], "critical")
    for name in (
        "test_study_discovery",
        "test_discovery_review_regressions",
        "test_study_resume",
        "test_forecast_features",
        "test_forecast_settlement",
        "test_candidate_factor_runtime",
        "test_provider_process",
        "test_controller",
        "test_research",
        "test_study_ui",
    ):
        assert f"tests/{name}.py" in critical


def test_slow_modules_are_separated_without_splitting_or_losing_selected_tests():
    shard = runpy.run_path(str(SCRIPT))["grouped_shards"]
    heavy = "tests/test_enhanced_forecast_workflow.py"
    other_heavy = "tests/test_forecast_usability.py"
    selectors = [
        f"{heavy}::first",
        "tests/test_factor_discovery.py",
        other_heavy,
        f"{heavy}::second",
    ]
    shards = shard(selectors, 2)
    assert sorted(sum(shards, [])) == sorted(selectors)
    owner = next(group for group in shards if f"{heavy}::first" in group)
    assert f"{heavy}::second" in owner
    assert other_heavy not in owner
    assert shard(selectors, 1) == [[selectors[0], selectors[3], selectors[1], selectors[2]]]
    assert shard([], 2) == []


def run_profile(output, *selectors):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), "--shards", "2", *selectors],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return result, json.loads((output / "summary.json").read_text())


def test_profile_keeps_selected_module_fixture_together_and_records_all_results(tmp_path):
    counter = tmp_path / "setup-count.txt"
    module = tmp_path / "test_shared.py"
    module.write_text(
        "import pytest\nfrom pathlib import Path\n"
        "@pytest.fixture(scope='module', autouse=True)\n"
        "def shared():\n"
        f"    with Path({str(counter)!r}).open('a') as handle: handle.write('setup\\n')\n"
        "def test_first(): pass\ndef test_second(): pass\n"
    )
    other = tmp_path / "test_other.py"
    other.write_text("def test_other(): pass\n")
    result, summary = run_profile(
        tmp_path / "profile", f"{module}::test_first", str(other), f"{module}::test_second"
    )
    assert result.returncode == summary["exit_code"] == 0, result.stdout + result.stderr
    assert len(summary["shards"]) == 2
    assert all(shard["exit_code"] == 0 for shard in summary["shards"])
    assert counter.read_text().splitlines() == ["setup"]
    metadata = json.loads((tmp_path / "profile/environment.json").read_text())
    assert metadata["shard_selectors"][0] == [f"{module}::test_first", f"{module}::test_second"]
    assert metadata["source"]["head"]
    assert metadata["source"]["files"]["examples/profile_validation.py"]
    assert metadata["environment"]["OMP_NUM_THREADS"] == "1"
    assert metadata["loaded_package"] == str(ROOT / "src/quant_rsi/__init__.py")
    assert metadata["suite"] == summary["suite"] == "focused"
    assert metadata["deferred_modules"] == {}
    assert summary["source_changed"] is False
    for index in range(2):
        assert (tmp_path / f"profile/shard-{index}.xml").exists()
        assert (tmp_path / f"profile/shard-{index}.command.json").exists()


@pytest.mark.parametrize("source, code", [("def test_failure(): assert False\n", 1), ("", 5)])
def test_profile_propagates_pytest_failure_or_empty_selection(tmp_path, source, code):
    module = tmp_path / "test_failure.py"
    module.write_text(source)
    result, summary = run_profile(tmp_path / "profile", str(module))
    assert result.returncode == summary["exit_code"] == code
    assert summary["shards"][0]["exit_code"] == code
