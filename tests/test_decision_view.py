"""The decision provider sees a bounded view of the frozen context (#159 Phase 2).

Measured on #159: the TSM branch validation run reached 203/203 prepared rows, then its
sixth decision prompt reached 1,234,424 characters and Codex rejected it (limit
1,048,576). Per-row alignment was sent twice, survey articles three times, and every CJK
character as a six-character escape.
"""

import json
from copy import deepcopy

import pytest

from quant_rsi.research import discovery_sources as sources

TEXT = "台積公司2026年第四季的業績展望如下：毛利率預計介於59%到61%之間。" * 40


def article(index):
    return {
        "id": f"source-mops-{index}",
        "provider": "mops",
        "status": "available",
        "url": "https://mops.twse.com.tw/mops/api/t05st01",
        "title": f"Release {index}",
        "published_at": f"2026-0{index + 1}-15T15:00:00+08:00",
        "content_sha256": f"{index:064d}",
        "text": TEXT,
    }


def rows(covered):
    times = [f"2026-01-{day:02d}T16:00:00-05:00" for day in range(1, 29)] * 9
    return [
        {
            "prediction_time": time,
            "observation_id": "obs-a" if covered(i) else None,
            "effective_available_at": "2025-12-01T00:00:00Z",
        }
        for i, time in enumerate(times)
    ]


def bundle(name, covered=lambda i: True):
    input_id = f"acquired.{name}"
    return {
        "identity": f"bundle-{name}",
        "status": "available",
        "covered_rows": 252,
        "request": {"input_id": input_id, "unit": "percent"},
        "observations": [{"observation_id": "obs-a", "value": 59.0}],
        "input_evidence": {input_id: rows(covered)},
        "articles": [article(1)],
        "operations": [{"identity": "op", "input": {"context": "x" * 5000}}] * 4,
        "access_receipts": [{"url": "https://example.com", "status": "available", "x": "y"}],
    }


def context():
    evidence = [article(i) for i in range(4)]
    bundles = [bundle("lower"), bundle("upper")]
    listing = {
        "request": {"year": 2026, "month": 1},
        "status": "available",
        "http_status": 200,
        "raw_path": "/x",
        "items": [
            {"document_ref": f"mops-{i}", "published_at": "2026-01-15", "title": "公告", "x": 1}
            for i in range(30)
        ],
    }
    memory = [
        {"kind": "interpretation", "ref": "decision-1", "decision": {"action": "survey"}},
        {
            "kind": "survey",
            "ref": "survey-1",
            "result": {
                "articles": deepcopy(evidence),
                "evidence": deepcopy(evidence),
                "findings": [{"mechanism": "Guidance", "evidence_ids": ["source-mops-1"]}],
                "search": {"query": "q", "listings": [listing] * 13},
                "operations": [{"input": "x" * 5000}],
                "access_receipts": [{"url": "https://sec.gov", "status": "unavailable"}],
            },
        },
        *(
            {"kind": "acquire", "ref": f"acquire-{i}", "result": deepcopy(b)}
            for i, b in enumerate(bundles)
        ),
    ]
    return {
        "symbol": "TSM",
        "forecast_target": {"horizon_sessions": 3},
        "evidence": evidence,
        "input_bundles": bundles,
        "memory": memory,
        "consumed_memory_refs": [m["ref"] for m in memory],
    }


def size(value):
    return len(json.dumps(value, ensure_ascii=False))


def test_view_sends_each_bundle_and_article_text_once_and_keeps_every_ref():
    frozen = context()
    original = deepcopy(frozen)
    view = sources.decision_view(frozen)
    assert frozen == original  # the frozen context itself is not changed
    assert size(view) < size(frozen) / 3
    assert json.dumps(view, ensure_ascii=False).count(TEXT) == len(frozen["evidence"])
    assert [m["ref"] for m in view["memory"]] == frozen["consumed_memory_refs"]
    acquired = [m for m in view["memory"] if m["kind"] == "acquire"]
    assert [m["result"]["identity"] for m in acquired] == ["bundle-lower", "bundle-upper"]
    assert all(m["result"]["bundle"] == "context.input_bundles" for m in acquired)
    assert [b["identity"] for b in view["input_bundles"]] == ["bundle-lower", "bundle-upper"]
    assert "input_evidence" not in view["input_bundles"][0]
    assert view["input_bundles"][0]["observations"] == frozen["input_bundles"][0]["observations"]
    survey = view["memory"][1]["result"]
    assert survey["findings"] == frozen["memory"][1]["result"]["findings"]
    assert survey["search"]["listings"][0]["items"][0] == {
        "document_ref": "mops-0",
        "published_at": "2026-01-15",
        "title": "公告",
    }


def test_coverage_summary_reports_exactly_the_uncovered_rows():
    full = rows(lambda i: not (i == 0 or 100 <= i < 103 or i == 251))
    summary = sources._coverage_summary(full)
    assert summary["rows"] == 252 and summary["covered_rows"] == 247
    missing = [r["prediction_time"] for r in full if not r["observation_id"]]
    runs = summary["uncovered_runs"]
    assert [run["rows"] for run in runs] == [1, 3, 1]
    assert sum(run["rows"] for run in runs) == len(missing)
    assert [run["first"] for run in runs] == [missing[0], missing[1], missing[4]]
    assert [run["last"] for run in runs] == [missing[0], missing[3], missing[4]]
    assert summary["observations_used"][0]["rows"] == 247


def test_article_not_in_context_evidence_keeps_its_text():
    frozen = context()
    extra = article(9) | {"id": "source-web-extra"}
    frozen["memory"][1]["result"]["articles"].append(extra)
    view = sources.decision_view(frozen)
    assert view["memory"][1]["result"]["articles"][-1] == extra


def decide(frozen, prompts):
    class Transport:
        def structured(self, prompt, schema):
            prompts.append(prompt)
            return json.dumps(
                {"action": "stop", "reason": "Done", "candidates": [], "data_requests": []}
                | {"query": None}
            )

    return sources.discovery_decision(Transport(), frozen)


def test_decision_prompt_is_literal_utf8_and_within_budget():
    prompts = []
    decide(context(), prompts)
    assert "台積公司" in prompts[0] and "\\u53f0" not in prompts[0]
    assert len(prompts[0]) < sources.DECISION_PROMPT_BUDGET


def test_over_budget_prompt_fails_with_its_size_before_any_provider_call(monkeypatch):
    monkeypatch.setattr(sources, "DECISION_PROMPT_BUDGET", 10_000)
    prompts = []
    with pytest.raises(ValueError, match=r"Decision prompt has \d+ characters, over the 10000"):
        decide(context(), prompts)
    assert prompts == []
