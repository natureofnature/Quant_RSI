"""Workspace provider settings and one explicit, tool-free connection check."""

import json
import os
import tempfile
import time
from pathlib import Path

from quant_rsi.core.artifacts import atomic_write, canonical
from quant_rsi.research.llm import LLMHypothesizer

KEYS = {"api": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
PROVIDERS = {"codex", *KEYS}


def _settings(root):
    path = Path(root) / ".provider-settings.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_bytes())
        if not isinstance(value, dict) or set(value) - PROVIDERS:
            raise ValueError
        for provider, settings in value.items():
            validate_settings({"provider": provider, **settings})
        return value
    except (ValueError, TypeError):
        raise ValueError(
            "Invalid workspace provider settings; replace the private settings file"
        ) from None


def validate_settings(options):
    if not isinstance(options, dict) or set(options) - {"provider", "model", "key", "clear_key"}:
        raise ValueError("Unexpected provider settings")
    provider = options.get("provider")
    if not isinstance(provider, str) or provider not in PROVIDERS:
        raise ValueError("Choose Codex, OpenAI API or DeepSeek")
    for field, limit in (("model", 120), ("key", 2048)):
        value = options.get(field, "")
        if not isinstance(value, str) or len(value) > limit or any(ord(c) < 32 for c in value):
            raise ValueError(f"Invalid provider {field}")
    if type(options.get("clear_key", False)) is not bool:
        raise ValueError("Invalid clear key option")
    if options.get("key") and (provider == "codex" or options.get("clear_key")):
        raise ValueError("Codex uses CLI login; clearing a key cannot also set one")
    return provider


def effective_environment(root):
    environment = dict(os.environ)
    for provider, settings in _settings(root).items():
        if provider in KEYS and settings.get("key"):
            environment[KEYS[provider]] = settings["key"]
    return environment


def service_settings(root):
    settings = _settings(root)
    return [
        {
            "provider": provider,
            "model": settings.get(provider, {}).get("model", ""),
            "key_source": (
                "Codex CLI login"
                if provider == "codex"
                else "workspace"
                if settings.get(provider, {}).get("key")
                else "server environment"
                if os.environ.get(KEYS[provider], "").strip()
                else "missing"
            ),
        }
        for provider in ("codex", "api", "deepseek")
    ]


def save_service_settings(root, options):
    provider = validate_settings(options)
    settings = _settings(root)
    entry = settings.setdefault(provider, {})
    entry["model"] = options.get("model", "").strip()
    if options.get("clear_key"):
        entry.pop("key", None)
    elif options.get("key", "").strip():
        entry["key"] = options["key"].strip()
    # mkstemp creates mode 0600 from the outset, including during replacement.
    fd, temporary = tempfile.mkstemp(prefix=".provider-settings-", dir=root)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(settings))
        os.replace(temporary, Path(root) / ".provider-settings.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return service_settings(root)


def check_connection(task, manifest):
    """Runs in the existing queue worker; never retain response text or raw errors."""
    provider = manifest["llm"]
    started = time.time()
    usage = []

    def record_usage(value):
        usage.append(
            {
                k: v
                for k, v in value.items()
                if k in {"input_tokens", "output_tokens", "cached_input_tokens"}
                and type(v) is int
                and v >= 0
            }
        )

    result = {
        "provider": provider,
        "model": manifest["llm_model"] or "provider default",
        "started_at": started,
        "scope": "One tool-free model response; no news search, data collection or model fits. "
        "This checks connectivity now, not future availability or research quality.",
    }
    try:
        client = LLMHypothesizer(
            provider, model=manifest["llm_model"] or None, timeout=60, on_usage=record_usage
        )
        response = client.structured(
            'Connection check. Return exactly {"ok": true}.',
            {
                "type": "object",
                "properties": {"ok": {"type": "boolean", "enum": [True]}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        )
        value = json.loads(response)
        if not isinstance(value, dict) or set(value) != {"ok"} or value["ok"] is not True:
            raise ValueError("Unexpected connection response")
        result.update(status="completed", detail="Received the expected structured response.")
        code = 0
    except Exception:
        result.update(
            status="failed",
            detail="Connection check failed. Check the selected model, credentials/login, "
            "service access and network, then explicitly test again. Raw service errors are "
            "not retained; a failed call may still consume usage.",
        )
        code = 1
    result.update(finished_at=time.time(), usage=usage)
    atomic_write(Path(task) / "connection.json", canonical(result))
    return code
