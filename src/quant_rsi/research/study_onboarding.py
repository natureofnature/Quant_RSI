"""Built-in study starting points and local-only service configuration checks."""

import os
import shutil
from importlib.util import find_spec

from quant_rsi.research.study_queue import study_options


def starter_studies():
    common = {"latest_fit": "selected", "only_new_evidence": False}
    definitions = (
        (
            "offline",
            "Try the complete loop",
            "Synthetic prices, news and financials; simulated researcher; real Ridge fits. "
            "No network or API key. Results only demonstrate the workflow.",
            {
                "llm": "simulated",
                "data": "synthetic",
                "model": "ridge",
                "news_research": False,
                "financials": True,
                "research_mode": "incremental",
                "question": "Can inventory growth relative to revenue help this synthetic example?",
            },
        ),
        (
            "market",
            "Research public market prices",
            "Public daily prices and issuer news with a simulated factor proposer and real "
            "LightGBM fits. Network access required; no model API key. Retrospective results.",
            {"llm": "simulated", "news_research": False},
        ),
        (
            "business",
            "Investigate a business hypothesis",
            "Codex researches news and tests business factors against a fixed baseline. "
            "Requires a working Codex session and public data; service usage may incur costs.",
            {
                "research_mode": "incremental",
                "financials": True,
                "question": "Does inventory growth outpace sales and help explain returns?",
            },
        ),
    )
    return [
        {
            "id": key,
            "name": name,
            "description": description,
            "symbols": ["KO"],
            "options": study_options(common | options),
        }
        for key, name, description, options in definitions
    ]


def local_services(environment=None):
    """Never return credentials, invoke a provider or equate presence with connectivity."""
    environment = os.environ if environment is None else environment
    codex = bool(shutil.which("codex", path=environment.get("PATH")))
    providers = [
        {"id": "simulated", "available": True, "detail": "No credentials or service calls."},
        {
            "id": "codex",
            "available": codex,
            "detail": (
                "Codex executable found. Login, model access and connectivity have not been tested."
                if codex
                else "Codex is not on the server PATH. "
                "Install/configure it or choose the offline example."
            ),
        },
    ]
    for key, variable in (("api", "OPENAI_API_KEY"), ("deepseek", "DEEPSEEK_API_KEY")):
        configured = bool(environment.get(variable, "").strip())
        providers.append(
            {
                "id": key,
                "available": configured,
                "detail": (
                    f"{variable} is configured. "
                    "Credentials, model access and connectivity are unverified."
                    if configured
                    else f"Configure {variable} in Service settings or in the server environment."
                ),
            }
        )
    providers.append(
        {
            "id": "cursor",
            "available": True,
            "detail": "Manual handoff: respond to each request in Cursor; "
            "this is not unattended research.",
        }
    )
    return {
        "scope": "Local configuration only. No network requests, login tests or paid model calls.",
        "providers": providers,
        "models": [
            {"id": name, "available": find_spec(package) is not None}
            for name, package in (("lightgbm", "lightgbm"), ("ridge", "numpy"))
        ],
        "data": "Use Check inputs to inspect data before research; no LLM calls or fits.",
    }
