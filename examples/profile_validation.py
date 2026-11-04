"""Profile offline pytest runs without splitting a module's fixtures across processes."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = dict(
    PYTHONHASHSEED="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
)
# These integration scenarios remain in the scheduled/manual full suite. All other
# modules, including new modules, stay in the critical tier. See docs/validation_cost.md.
EXTENDED_MODULES = {
    "test_enhanced_forecast_workflow": "Legacy enhanced forecast workflow permutations",
    "test_forecast_usability": "Legacy forecast usability and full campaign scenarios",
    "test_research_workflow": "Legacy forecast-to-strategy workflow scenarios",
    "test_forecast_recursive_preparation": "Recursive forecast preparation scenarios",
    "test_local_comparison": "Protected local C4/C5 end-to-end comparisons",
    "test_trajectory": "Full protected trajectory and saturation scenarios",
    "test_trajectory_recovery_guard": "Protected trajectory recovery permutations",
    "test_packages": "Protected package validation and recovery permutations",
    "test_comparison": "Protected C4/C5 evaluation scenarios",
    "test_local_packages": "Local package staging and recovery scenarios",
    "test_factor_discovery": "Legacy bounded factor-discovery scenarios",
    "test_forecast_campaign": "Multi-candidate forecast campaign permutations",
    "test_observer": "Sealed-forecast observer integration scenarios",
    "test_news_comparison": "Paired legacy news comparison scenarios",
    "test_forecast_comparison": "Paired legacy forecast comparison scenarios",
    "test_intraday_second_review": "Separate intraday execution product scenarios",
    "test_intraday_paper": "Separate intraday paper-account scenarios",
    "test_intraday_review_regressions": "Separate intraday review scenarios",
    "test_intraday_policy": "Separate intraday policy calibration scenarios",
}
# Whole-module scheduling hints from green CI 41346746509 (2026-11-01).
# Modules below 30 seconds averaged 3 seconds. These estimates never select tests.
MODULE_SECONDS = {
    "test_comparison": 91,
    "test_enhanced_forecast_workflow": 618,
    "test_factor_discovery": 36,
    "test_forecast_campaign": 42,
    "test_forecast_recursive_preparation": 47,
    "test_forecast_usability": 494,
    "test_intraday_second_review": 32,
    "test_local_comparison": 250,
    "test_local_packages": 74,
    "test_packages": 98,
    "test_research_workflow": 50,
    "test_trajectory": 235,
    "test_trajectory_recovery_guard": 114,
    # Critical-tier balancing hints from CI 41855398994 (2026-11-04).
    "test_forecast_settlement": 28,
    "test_research": 23,
    "test_candidate_factor_runtime": 19,
    "test_provider_process": 16,
    "test_audit_regressions": 11,
    "test_study_horizon": 11,
    "test_forecast_features": 10,
}


def select_tests(selectors, suite):
    if selectors:
        # Explicit files/nodes always run, even when their module is extended-only.
        return list(selectors), {}
    modules = sorted(
        {
            str(path.relative_to(ROOT))
            for pattern in ("test_*.py", "*_test.py")
            for path in ROOT.glob(f"tests/**/{pattern}")
        }
    )
    deferred = {
        path: EXTENDED_MODULES[Path(path).stem]
        for path in modules
        if suite == "critical" and Path(path).stem in EXTENDED_MODULES
    }
    return [path for path in modules if path not in deferred], deferred


def grouped_shards(selectors, count):
    groups = {}
    for selector in selectors:
        module = str(Path(selector.split("::", 1)[0]).resolve())
        groups.setdefault(module, []).append(selector)
    shards = [[] for _ in range(min(count, len(groups)))]
    if len(shards) < 2:
        return [[selector for group in groups.values() for selector in group]] if groups else []
    estimates = {module: MODULE_SECONDS.get(Path(module).stem, 3) for module in groups}
    loads = [0] * len(shards)
    for module in sorted(groups, key=lambda module: -estimates[module]):
        index = min(range(len(shards)), key=lambda index: loads[index])
        shards[index].extend(groups[module])
        loads[index] += estimates[module]
    return shards


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def source_state():
    paths = [*ROOT.glob("src/**/*.py"), *ROOT.glob("tests/**/*.py"), Path(__file__)]
    return {
        "head": git("rev-parse", "HEAD"),
        "status": git("status", "--porcelain"),
        "diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)
        ).hexdigest(),
        "files": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))
        },
    }


def run_shard(output, index, selectors, env):
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--durations=20",
        "--basetemp",
        str(output / f"tmp-{index}"),
        "--junitxml",
        str(output / f"shard-{index}.xml"),
        *selectors,
    ]
    (output / f"shard-{index}.command.json").write_text(json.dumps(command, indent=2) + "\n")
    started = time.monotonic()
    with (output / f"shard-{index}.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return dict(shard=index, exit_code=result.returncode, seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shards", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--suite",
        choices=("critical", "full"),
        default="critical",
        help="Default: critical PR tier; full includes extended integration scenarios",
    )
    parser.add_argument(
        "--cache-note", default="OS/package caches uncontrolled; fresh pytest basetemp"
    )
    parser.add_argument(
        "selectors", nargs="*", help="Explicit files/node IDs override suite selection"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    selectors, deferred = select_tests(args.selectors, args.suite)
    shards = grouped_shards(selectors, args.shards)
    if not shards:
        parser.error("no test modules found")
    env = dict(os.environ) | THREAD_ENV
    env.pop("PYTHONPATH", None)  # Workers also clear it; use an installed checkout.
    loaded_package = subprocess.check_output(
        [sys.executable, "-c", "import quant_rsi; print(quant_rsi.__file__)"],
        cwd=ROOT,
        env=env,
        text=True,
    ).strip()
    if Path(loaded_package).resolve() != ROOT / "src/quant_rsi/__init__.py":
        parser.error(f"Python imports another checkout: {loaded_package}; install this checkout")
    before = source_state()
    metadata = dict(
        suite="focused" if args.selectors else args.suite,
        deferred_modules=deferred,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source=before,
        loaded_package=loaded_package,
        python=sys.executable,
        python_version=sys.version,
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        load_average=os.getloadavg(),
        resource_limits={
            name: path.read_text().strip() if path.exists() else None
            for name, path in {
                "cpu_max": Path("/sys/fs/cgroup/cpu.max"),
                "memory_max": Path("/sys/fs/cgroup/memory.max"),
            }.items()
        },
        dependencies={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        environment=THREAD_ENV
        | {
            key: env.get(key)
            for key in ("PYTEST_ADDOPTS", "QUANT_RSI_UPSTREAM_CACHE", "QUANT_RSI_WORKSPACE")
        },
        cache_note=args.cache_note,
        shard_selectors=shards,
    )
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(shards)) as pool:
        futures = [pool.submit(run_shard, output, i, tests, env) for i, tests in enumerate(shards)]
        results = [future.result() for future in futures]
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    changed = source_state() != before
    exit_code = next((r["exit_code"] for r in results if r["exit_code"]), 0)
    if exit_code < 0:
        exit_code = 128 - exit_code
    summary = dict(
        suite=metadata["suite"],
        selected_modules=len({selector.split("::", 1)[0] for selector in selectors}),
        deferred_modules=len(deferred),
        elapsed_seconds=time.monotonic() - started,
        child_cpu_seconds=dict(
            user=usage_after.ru_utime - usage_before.ru_utime,
            system=usage_after.ru_stime - usage_before.ru_stime,
        ),
        shards=results,
        source_changed=changed,
        exit_code=exit_code or (1 if changed else 0),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    print(f"Logs, per-test JUnit timings, commands and environment: {output}")
    return summary["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
