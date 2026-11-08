"""Explicit Codex effort survives isolated CLI calls and native study recovery."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_research_study import command

from quant_rsi.cli import main
from quant_rsi.research import llm, study
from quant_rsi.research.discovery_sources import DiscoverySources
from quant_rsi.research.study_progress import StudyProgress


def test_source_receipts_do_not_reuse_another_reasoning_configuration():
    receipts = {}

    def cached(key, payload, operation):
        if key not in receipts:
            receipts[key] = (payload, operation())
        return receipts[key][1]

    for effort in (None, "high", "low"):
        source = DiscoverySources(
            SimpleNamespace(mode="codex", model="gpt-6.1-sol", reasoning_effort=effort),
            call=cached,
        )
        assert (
            source._operation("analysis", {"question": "Same inputs"}, lambda effort=effort: effort)
            == effort
        )
    assert len(receipts) == 3
    assert "reasoning_effort" not in next(iter(receipts.values()))[0]
    source = DiscoverySources(
        SimpleNamespace(mode="codex", model="gpt-6.1-sol", reasoning_effort="high"), call=cached
    )
    assert (
        source._operation(
            "analysis", {"question": "Same inputs"}, lambda: pytest.fail("Paid call was repeated")
        )
        == "high"
    )


@pytest.mark.parametrize("effort", [None, "high", "ultra"])
@pytest.mark.parametrize("search", [False, True])
def test_codex_effort_is_explicit_even_with_isolated_configuration(
    tmp_path, monkeypatch, effort, search
):
    progress = StudyProgress(tmp_path, "codex", quiet=True)

    def run(arguments, **kwargs):
        assert "--ignore-user-config" in arguments
        assert arguments[arguments.index("--model") + 1] == "gpt-6.1-sol"
        settings = [arguments[i + 1] for i, arg in enumerate(arguments) if arg == "-c"]
        effort_settings = [
            setting for setting in settings if setting.startswith("model_reasoning_effort=")
        ]
        assert effort_settings == ([] if effort is None else [f'model_reasoning_effort="{effort}"'])
        assert arguments[-1] == "-"
        assert ('web_search="live"' in settings) == search
        Path(arguments[arguments.index("--output-last-message") + 1]).write_text("{}")
        if search:
            kwargs["stdout"].write(
                json.dumps(
                    {"type": "item.completed", "item": {"type": "web_search", "query": "fixture"}}
                )
                + "\n"
            )
        kwargs["stdout"].write(
            json.dumps(
                {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}}
            )
            + "\n"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(llm, "run_provider", run)
    provider = llm.LLMHypothesizer(
        "codex", "gpt-6.1-sol", on_usage=progress.usage, reasoning_effort=effort
    )
    if search:
        assert provider.search("offline fixture", {})["text"] == "{}"
    else:
        assert provider.structured("offline fixture", {}) == "{}"
    usage = json.loads((tmp_path / "runtime.json").read_text())["usage"][0]
    if effort is None:
        assert "requested_reasoning_effort" not in usage
    else:
        assert usage["requested_reasoning_effort"] == effort
        assert usage["requested_model"] == "gpt-6.1-sol"


@pytest.mark.parametrize("provider", ["simulated", "api", "deepseek", "cursor"])
def test_effort_rejects_unsupported_study_provider_before_work(tmp_path, capsys, provider):
    root = tmp_path / "not-created"
    assert main(command(root, "--llm", provider, "--llm-reasoning-effort", "high", "--no-ui")) == 1
    assert "--llm-reasoning-effort requires --llm codex" in capsys.readouterr().err
    assert not root.exists()


def test_direct_transport_rejects_ignored_or_invalid_effort():
    with pytest.raises(ValueError, match="requires --llm codex"):
        llm.LLMHypothesizer("api", "test-model", reasoning_effort="high")
    with pytest.raises(ValueError, match="Unsupported Codex reasoning effort"):
        llm.LLMHypothesizer("codex", reasoning_effort="typo")


@pytest.mark.parametrize("mode", ["legacy", "discovery"])
@pytest.mark.parametrize("effort", [None, "high"])
def test_native_configuration_resume_and_identity_keep_requested_effort(
    tmp_path, monkeypatch, capsys, mode, effort
):
    root = tmp_path / "study"
    calls, initialized = [], []
    fail = True
    original = llm.LLMHypothesizer.__post_init__

    def initialize(self):
        original(self)
        initialized.append((self.mode, self.model, self.reasoning_effort))

    def checked(self):
        calls.append((self.model, self.reasoning_effort))
        assert self.model == "gpt-6.1-sol"
        assert self.reasoning_effort == effort
        if fail:
            raise ValueError("Offline provider outage")

    def hypothesis(self, *args, **kwargs):
        checked(self)
        return study.simulated_hypothesis(*args, **kwargs)

    def decision(self, prompt, schema):
        checked(self)
        return json.dumps(
            {
                "action": "stop",
                "reason": "Offline transport test",
                "candidates": [],
                "data_requests": [],
                "query": None,
            }
        )

    monkeypatch.setattr(llm.LLMHypothesizer, "__post_init__", initialize)
    monkeypatch.setattr(llm.LLMHypothesizer, "__call__", hypothesis)
    monkeypatch.setattr(llm.LLMHypothesizer, "_codex", decision)
    options = [] if effort is None else ["--llm-reasoning-effort", effort]
    assert (
        main(
            command(
                root,
                "--llm",
                "codex",
                "--llm-model",
                "gpt-6.1-sol",
                "--research-mode",
                mode,
                "--model",
                "ridge",
                "--no-ui",
                "--no-history",
                *options,
            )
        )
        == 1
    )
    configuration_path = root / "config.json"
    original_config = configuration_path.read_bytes()
    configuration = json.loads(original_config)
    if effort is None:
        assert "llm_reasoning_effort" not in configuration
    else:
        assert configuration["llm_reasoning_effort"] == effort
    identity = json.loads((root / "checkpoints" / "identity.json").read_text())
    assert identity["configuration"].get("llm_reasoning_effort") == effort

    configuration["llm_reasoning_effort"] = "low"
    configuration_path.write_text(json.dumps(configuration))
    assert main(["research-resume", str(root)]) == 1
    assert "configuration changed" in capsys.readouterr().err
    assert len(calls) == 1
    configuration_path.write_bytes(original_config)
    fail = False
    assert main(["research-resume", str(root)]) == 0
    assert configuration_path.read_bytes() == original_config
    assert all(item == ("codex", "gpt-6.1-sol", effort) for item in initialized)
    # Discovery constructs both the ordinary context provider and its tool transport.
    assert len(initialized) == (4 if mode == "discovery" else 2)
    result = json.loads((root / "result.json").read_text())
    assert result["configuration"].get("llm_reasoning_effort") == effort
