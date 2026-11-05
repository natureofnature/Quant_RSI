"""Instrument-window configuration and immutable resume, with offline Ridge fixtures."""

import argparse
import json

import pytest
from test_monthly_rolling import DecisionLLM
from test_monthly_rolling import retained as retained_fixture

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research import rolling
from quant_rsi.research.rolling_config import RollingWindowProfile, load_window_profile

retained = retained_fixture


def specification(**changes):
    return {
        "schema_version": 1,
        "kind": "RollingWindowProfile",
        "instrument": "SOXX",
        "training_window_sessions": 126,
        "horizon_sessions": 3,
    } | changes


def profile_file(path, **changes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(specification(**changes)))
    return path


def arguments(retained, output, *extra, profile=None, month="2026-08"):
    parser = argparse.ArgumentParser()
    rolling.add_arguments(parser.add_subparsers(dest="command"))
    return parser.parse_args(
        [
            "research-rolling",
            "SOXX",
            "--prices-dir",
            str(retained[0]),
            "--first-month",
            month,
            "--last-month",
            month,
            "--model",
            "ridge",
            "--output",
            str(output),
            *(["--profile", str(profile)] if profile else []),
            *extra,
        ]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", 1.0),
        ("schema_version", "1"),
        ("kind", "OtherProfile"),
        ("instrument", "soxx"),
        ("instrument", "SOXX/USD"),
        ("training_window_sessions", 60),
        ("training_window_sessions", 446),
        ("training_window_sessions", 126.0),
        ("training_window_sessions", "126"),
        ("training_window_sessions", True),
        ("horizon_sessions", 0),
        ("horizon_sessions", 21),
        ("horizon_sessions", 3.0),
        ("horizon_sessions", "3"),
        ("horizon_sessions", True),
        ("unexpected", 1),
    ],
)
def test_profile_rejects_invalid_or_coerced_values(field, value):
    with pytest.raises(ValueError):
        RollingWindowProfile.model_validate_json(canonical(specification(**{field: value})))


def test_profile_requires_enough_mature_training_labels_and_matches_symbol(tmp_path):
    with pytest.raises(ValueError, match="30 labels"):
        RollingWindowProfile.model_validate_json(
            canonical(specification(training_window_sessions=62))
        )
    path = profile_file(tmp_path / "rolling.json")
    profile, raw = load_window_profile(path, "SOXX")
    assert profile.training_window_sessions == 126 and profile.horizon_sessions == 3
    assert raw == path.read_bytes()
    with pytest.raises(ValueError, match="differs from the requested symbol"):
        load_window_profile(path, "KO")


def test_profile_drives_exact_126_session_inputs_and_93_real_fit_labels(retained, tmp_path):
    path = profile_file(tmp_path / "SOXX/rolling.json")
    args = arguments(retained, tmp_path / "study", profile=path)
    result = rolling.run_rolling(args)
    assert result["kind"] == "SessionRollingStudy"
    assert result["configuration"]["train_sessions"] == 126
    assert result["configuration"]["horizon_sessions"] == 3
    assert "train_months" not in result["configuration"]
    assert result["configuration"]["input_identity"]["window_profile_sha256"] == digest(
        path.read_bytes()
    )
    assert (args.output / "window_profile.json").read_bytes() == path.read_bytes()
    assert (args.output / "session_plan.json").exists()
    assert result["folds"]
    for fold in result["folds"]:
        assert fold["training_window_sessions"] == 126
        assert fold["horizon_sessions"] == 3
        assert len(fold["training_origins"]) == fold["fits"]["fixed"]["training_samples"] == 93
        assert fold["training_targets"][-1] == fold["origin_session"]
    assert "126-trading-session" in result["notice"]
    report = (args.output / "report.html").read_text()
    assert "126-trading-session" in report and "fixed trading-session window" in report
    assert "<td>93</td>" in report


@pytest.mark.parametrize(
    "flags,length,horizon",
    [
        (["--train-sessions", "90"], 90, 3),
        (["--horizon-sessions", "5"], 126, 5),
        (["--train-sessions", "90", "--horizon-sessions", "5"], 90, 5),
    ],
)
def test_explicit_cli_overrides_are_saved_as_effective_values(
    retained, tmp_path, flags, length, horizon
):
    path = profile_file(tmp_path / "rolling.json")
    result = rolling.run_rolling(arguments(retained, tmp_path / "study", *flags, profile=path))
    assert result["configuration"]["train_sessions"] == length
    assert result["configuration"]["horizon_sessions"] == horizon
    for fold in result["folds"]:
        assert fold["training_window_sessions"] == length and fold["horizon_sessions"] == horizon
        assert fold["fits"]["fixed"]["training_samples"] == length - 30 - horizon


def test_explicit_calendar_month_training_overrides_only_profile_training(retained, tmp_path):
    path = profile_file(tmp_path / "rolling.json")
    result = rolling.run_rolling(
        arguments(retained, tmp_path / "study", "--train-months", "4", profile=path)
    )
    assert result["configuration"]["train_months"] == 4
    assert "train_sessions" not in result["configuration"]
    assert result["configuration"]["horizon_sessions"] == 3
    assert all("training_window_sessions" not in fold for fold in result["folds"])
    assert result["folds"][0]["training_window"] == {"start": "2026-04-01", "end": "2026-07-31"}


@pytest.mark.parametrize("under_instruments", [False, True])
@pytest.mark.parametrize("change", ["edit", "delete"])
def test_resume_uses_retained_profile_after_source_profile_edit(
    retained, tmp_path, monkeypatch, under_instruments, change
):
    if under_instruments:
        # Reproduce an instrument-owned input without editing the checked-in profile.
        monkeypatch.setattr(
            rolling, "__file__", str(tmp_path / "repo/src/quant_rsi/research/rolling.py")
        )
        path = tmp_path / "repo/instruments/US/SOXX/rolling.json"
    else:
        path = tmp_path / "external/rolling.json"
    profile_file(path)
    args = arguments(retained, tmp_path / "study", "--llm", "codex", profile=path)
    llm = DecisionLLM(
        [
            {
                "action": "retain",
                "basis": "price_only",
                "reason": "Keep baseline",
                "candidates": [],
            },
            RuntimeError("offline provider failure"),
        ]
    )
    fits = []
    real_fit = rolling.fit_forecast

    def count_fit(*a, **kw):
        fits.append(1)
        return real_fit(*a, **kw)

    monkeypatch.setattr(rolling, "fit_forecast", count_fit)
    monkeypatch.setattr(rolling, "LLMHypothesizer", lambda *a, **kw: llm)
    with pytest.raises(RuntimeError, match="offline provider failure"):
        rolling.run_rolling(args)
    assert len(fits) == 2
    saved = (args.output / "window_profile.json").read_bytes()
    first = (args.output / "2026-08-05/prediction.json").read_bytes()
    if change == "edit":
        profile_file(path, training_window_sessions=90, horizon_sessions=5)
    else:
        path.unlink()
    args.resume = True
    result = rolling.run_rolling(args)
    assert result["configuration"]["train_sessions"] == 126
    assert result["configuration"]["horizon_sessions"] == 3
    assert len(fits) == len(result["folds"]) == 7
    assert (args.output / "window_profile.json").read_bytes() == saved
    assert (args.output / "2026-08-05/prediction.json").read_bytes() == first


@pytest.mark.parametrize("mutation", ["other_factor", "retained_profile"])
def test_resume_still_rejects_changed_factor_artifacts_or_retained_profile(
    retained, tmp_path, monkeypatch, mutation
):
    monkeypatch.setattr(
        rolling, "__file__", str(tmp_path / "repo/src/quant_rsi/research/rolling.py")
    )
    path = profile_file(tmp_path / "repo/instruments/US/SOXX/rolling.json")
    other = path.parent / "other_factor.json"
    other.write_bytes(canonical({"window": 10}))
    args = arguments(retained, tmp_path / "study", "--llm", "codex", profile=path)
    llm = DecisionLLM([RuntimeError("offline provider failure")])
    monkeypatch.setattr(rolling, "LLMHypothesizer", lambda *a, **kw: llm)
    with pytest.raises(RuntimeError, match="offline provider failure"):
        rolling.run_rolling(args)
    if mutation == "other_factor":
        other.write_bytes(canonical({"window": 20}))
    else:
        profile_file(args.output / "window_profile.json", training_window_sessions=90)
    preserved = {p: p.read_bytes() for p in args.output.rglob("*.json")}
    args.resume = True
    calls = []
    monkeypatch.setattr(rolling, "fit_forecast", lambda *a, **kw: calls.append("fit"))
    with pytest.raises(ValueError, match="configuration changed"):
        rolling.run_rolling(args)
    assert calls == [] and len(llm.contexts) == 1
    assert all(path.read_bytes() == raw for path, raw in preserved.items())


def test_effectively_incompatible_override_fails_before_fits_or_provider(
    retained, tmp_path, monkeypatch
):
    path = profile_file(tmp_path / "rolling.json", training_window_sessions=63)
    args = arguments(
        retained, tmp_path / "study", "--horizon-sessions", "20", "--llm", "codex", profile=path
    )
    calls = []
    monkeypatch.setattr(rolling, "fit_forecast", lambda *a, **kw: calls.append("fit"))
    monkeypatch.setattr(rolling, "LLMHypothesizer", lambda *a, **kw: calls.append("provider"))
    with pytest.raises(ValueError, match="60 \\+ horizon"):
        rolling.run_rolling(args)
    assert calls == [] and not args.output.exists()


def test_resume_override_change_is_rejected_before_work_or_overwriting_evidence(
    retained, tmp_path, monkeypatch
):
    path = profile_file(tmp_path / "rolling.json")
    args = arguments(retained, tmp_path / "study", "--llm", "codex", profile=path)
    llm = DecisionLLM([RuntimeError("offline provider failure")])
    monkeypatch.setattr(rolling, "LLMHypothesizer", lambda *a, **kw: llm)
    with pytest.raises(RuntimeError, match="offline provider failure"):
        rolling.run_rolling(args)
    preserved = {p: p.read_bytes() for p in args.output.rglob("*.json")}
    args.resume = True
    args.train_sessions = 90
    calls = []
    monkeypatch.setattr(rolling, "fit_forecast", lambda *a, **kw: calls.append("fit"))
    with pytest.raises(ValueError, match="configuration changed"):
        rolling.run_rolling(args)
    assert calls == [] and len(llm.contexts) == 1
    assert all(path.read_bytes() == raw for path, raw in preserved.items())


def test_legacy_no_profile_cli_stays_monthly_and_keeps_its_saved_configuration(retained, tmp_path):
    args = arguments(retained, tmp_path / "monthly", month="2026-07")
    result = rolling.run_rolling(args)
    assert result["kind"] == "MonthlyRollingStudy"
    assert len(result["folds"]) == 1 and result["folds"][0]["month"] == "2026-07"
    assert result["configuration"]["train_months"] == 6
    assert not {"profile", "train_sessions", "horizon_sessions"}.intersection(
        result["configuration"]
    )
    assert "window_profile_sha256" not in result["configuration"]["input_identity"]
    assert (args.output / "monthly_plan.json").exists()
    assert not (args.output / "window_profile.json").exists()
    assert not (args.output / "session_plan.json").exists()
    assert "fixed calendar window" in (args.output / "report.html").read_text()
    config = json.loads((args.output / "config.json").read_bytes())
    assert config == result["configuration"]


def test_training_modes_are_mutually_exclusive_at_cli_parse_time(retained, tmp_path):
    with pytest.raises(SystemExit) as exc:
        arguments(retained, tmp_path / "invalid", "--train-sessions", "126", "--train-months", "6")
    assert exc.value.code == 2
