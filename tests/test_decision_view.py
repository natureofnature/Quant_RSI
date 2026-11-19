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


def test_article_outside_evidence_keeps_one_full_copy_in_retained_sources():
    frozen = context()
    extra = article(9) | {"id": "source-web-extra", "text": "Only copy of this body."}
    frozen["memory"][1]["result"]["articles"].append(extra)
    view = sources.decision_view(frozen)
    assert view["memory"][1]["result"]["articles"][-1]["full_text"] == ("context.retained_sources")
    assert view["retained_sources"] == [extra]
    assert json.dumps(view, ensure_ascii=False).count("Only copy of this body.") == 1


def test_same_id_with_different_text_is_never_stubbed():
    frozen = context()
    changed = article(1) | {"text": "A different version of the body."}
    frozen["input_bundles"][0]["articles"] = [changed]
    view = sources.decision_view(frozen)
    assert view["input_bundles"][0]["articles"] == [changed]
    assert "retained_sources" not in view


MARKER = "Additional management explanation: inventory normalization is incomplete."


@pytest.mark.parametrize("unit", ["percent", "unsupported unit"])
def test_directly_acquired_article_text_reaches_the_next_decision(tmp_path, monkeypatch, unit):
    # Review on #177: direct acquisition never adds to context.evidence, and a
    # needs_data acquisition is only in memory; the view must still carry its body.
    from test_discovery_loop import cache_at, runner
    from test_mops_usd_guidance import provider_for
    from test_unit_aliases import TIMES, answer

    from quant_rsi.core.artifacts import digest
    from quant_rsi.research.discovery_loop import run_discovery

    request, source, extraction = answer()
    source["text"] += "\n" + MARKER
    source["content_sha256"] = digest(source["text"].encode())
    request["unit"] = unit
    provider = provider_for(source, extraction, [])
    monkeypatch.setattr(sources, "_now", lambda: "2026-10-10T12:01:00Z")
    contexts = []

    def decide(frozen):
        if frozen["iteration"] == 1:
            return {"action": "acquire", "data_requests": [request]}
        return {"action": "stop"}

    run_discovery(
        {"symbol": "TSM", "evidence": []},
        root=tmp_path,
        cache=cache_at(tmp_path),
        decide=decide,
        run_experiment=runner,
        acquire=lambda r: provider.acquire(r, prediction_times=TIMES),
        decision_budget=2,
        on_context=contexts.append,
    )
    frozen = contexts[1]
    assert frozen["evidence"] == [] and MARKER in json.dumps(frozen, ensure_ascii=False)
    view = sources.decision_view(frozen)
    assert json.dumps(view, ensure_ascii=False).count(MARKER) == 1
    assert [a["id"] for a in view["retained_sources"]] == [source["id"]]


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


def experiment_context():
    frozen = context()
    windows = {"partitions": {"train": {"origin_sessions": ["d"] * 200}}}
    frozen["experiment_contract"] = {"windows": windows}
    valuations = [{"session": f"s{i}", "equity": "1.0"} for i in range(24)]
    rows = [{"origin_session": f"o{i}", "predicted_price": 1.0} for i in range(20)]
    frozen["memory"].append(
        {
            "kind": "experiment",
            "ref": "experiment-1",
            "spec": {
                "comparison_key": "group",
                "contract": {"windows": deepcopy(windows), "metric": "mae"},
                "input_bundles": deepcopy(frozen["input_bundles"]),
            },
            "result": {
                "round": 1,
                "metric": 1.5,
                "prediction": {
                    "development": {
                        "rows": rows,
                        "metrics": {"mae_price": 1.5},
                        "baselines": {
                            "unchanged_price": {"metrics": {"mae_price": 2.0}, "rows": rows},
                        },
                    },
                    "strategy": {"strategy": {"valuations": valuations, "cash": "1"}},
                },
            },
        }
    )
    return frozen


def test_fit_records_refer_to_bundles_and_summarize_account_paths():
    # Measured on the TSM run with fits: two experiment records embedded both 150k-character
    # bundles and their daily paths, so decision 6 reached 997,083 characters.
    frozen = experiment_context()
    original = deepcopy(frozen)
    view = sources.decision_view(frozen)
    assert frozen == original
    experiment = view["memory"][-1]
    assert experiment["ref"] == "experiment-1"
    assert experiment["spec"]["input_bundles"] == [
        {"identity": "bundle-lower", "bundle": "context.input_bundles"},
        {"identity": "bundle-upper", "bundle": "context.input_bundles"},
    ]
    assert experiment["spec"]["contract"] == {
        "windows": "context.experiment_contract.windows",
        "metric": "mae",
    }
    prediction = experiment["result"]["prediction"]
    assert len(prediction["development"]["rows"]) == 20  # the fitted model's rows stay
    assert prediction["development"]["baselines"]["unchanged_price"] == {
        "metrics": {"mae_price": 2.0},
        "rows": "retained in the experiment record",
    }
    path = prediction["strategy"]["strategy"]["valuations"]
    assert path["items"] == 24 and path["last"] == {"session": "s23", "equity": "1.0"}
    assert experiment["result"]["metric"] == 1.5
    assert size(experiment) < size(original["memory"][-1]) / 5


def test_bundle_outside_the_view_is_kept_compact_in_the_spec():
    frozen = experiment_context()
    frozen["input_bundles"] = frozen["input_bundles"][:1]
    view = sources.decision_view(frozen)
    kept, compact = view["memory"][-1]["spec"]["input_bundles"]
    assert kept == {"identity": "bundle-lower", "bundle": "context.input_bundles"}
    assert compact["identity"] == "bundle-upper" and "input_coverage" in compact
