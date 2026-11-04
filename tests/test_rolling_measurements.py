"""Offline measured-input discovery reaches real tiny Ridge fits; no quality claim."""

import argparse
import json
from copy import deepcopy

import numpy as np
import pytest
import test_monthly_rolling as monthly
import test_rolling_news_integration as news_tests

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research import discovery_sources, rolling
from quant_rsi.research.discovery_sources import DiscoverySources
from quant_rsi.research.rolling_acquired import AcquiredResolver, load_acquired_history
from quant_rsi.research.rolling_windows import monthly_plan

retained = monthly.retained


def measurement_request(name="inventory"):
    return {
        "input_id": "acquired." + name,
        "symbol": "NVDA",
        "measurement": "Fixture issuer quarterly " + name,
        "unit": "USD million",
        "query": "Fixture issuer disclosed " + name,
        "source_urls": [f"https://example.com/issuer-{i}" for i in range(3)],
        "max_age_days": 185,
    }


class MeasurementLLM(monthly.DecisionLLM):
    """The source text, rather than a catalog mapping, supplies a newly requested value."""

    def __init__(self, decisions=()):
        super().__init__(decisions)
        self.extractions = []
        self.fail_extraction = False

    def structured(self, prompt, schema):
        payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
        if "request" not in payload or "article" not in payload:
            return super().structured(prompt, schema)
        self.extractions.append(deepcopy(payload))
        if self.fail_extraction:
            self.fail_extraction = False
            raise RuntimeError("fixture extraction interruption")
        text = payload["article"]["text"]
        name = payload["request"]["input_id"].split(".", 1)[1]
        # Parse only the deliberately small literal fixture, with no derived measurement.
        value = float(text.split(name + " ", 1)[1].split(" ", 1)[0])
        publication = text.split("Published ", 1)[1].split(".", 1)[0]
        period = text.split("Period ending ", 1)[1].split(":", 1)[0]
        return json.dumps(
            {
                "publication_date": publication,
                "publication_excerpt": "Published " + publication + ".",
                "measurement": payload["request"]["measurement"],
                "observations": [
                    {
                        "period_start": None,
                        "period_end": period,
                        "value": value,
                        "unit": "USD million",
                        "excerpt": text.split("Published " + publication + ". ", 1)[1],
                    }
                ],
                "limitations": ["Synthetic issuer fixture, not market evidence."],
            }
        )


def acquisition_checkpoint(path, monkeypatch, *, late=False, future_inventory=None):
    articles = []
    for index, (published, period, revenue, inventory) in enumerate(
        [
            ("2025-11-16", "2025-10-31", 100, 20),
            ("2026-02-22", "2026-01-31", 120, 35),
            ("2026-05-24", "2026-04-30", 110, 25),
        ]
    ):
        if index == 2 and future_inventory is not None:
            published, period, inventory = "2026-07-24", "2026-06-30", future_inventory
        text = (
            f"Published {published}. Period ending {period}: "
            f"revenue {revenue} USD million; inventory {inventory} USD million."
        )
        receipt = "2026-11-04T12:00:00Z" if late else published + "T21:00:00Z"
        articles.append(
            {
                "id": f"issuer-{index}",
                "url": f"https://example.com/issuer-{index}",
                "status": "available",
                "text": text,
                "content_sha256": digest(text.encode()),
                "published_at": published + "T12:00:00Z",
                "received_at": receipt,
                "version_available_at": receipt,
            }
        )
    monkeypatch.setattr(discovery_sources, "_now", lambda: "2026-11-04T12:00:00+00:00")
    provider = MeasurementLLM()
    sources = DiscoverySources(
        provider,
        retained_articles=articles,
        reader=lambda *_: pytest.fail("fixture attempted an article fetch"),
    )
    request = measurement_request("revenue")
    output = sources.acquire(request, prediction_times=["2026-06-30T20:00:00Z"])
    assert len(output["observations"]) == 3
    path.write_bytes(
        canonical(
            {
                "input_sha256": digest(canonical(request)),
                "output_sha256": digest(canonical(output)),
                "output": output,
            }
        )
    )
    return path


def measured_proposal():
    candidate = monthly.candidate()
    candidate.update(
        name="issuer_inventory",
        description="Measured issuer inventory hypothesis from dated disclosure excerpts",
        inputs=[{"name": "source", "factor_id": "acquired.inventory"}],
        expression={
            "schema_version": 2,
            "nodes": [{"name": "result", "op": "leaf", "leaf": "source"}],
        },
        evidence_ids=["issuer-0", "issuer-1", "issuer-2"],
    )
    return {
        "action": "replace",
        "basis": "news",
        "reason": "Test a new measured business variable appearing in source evidence",
        "candidates": [candidate],
        "news_inputs": [],
        "data_requests": [measurement_request()],
    }


def execute(retained_data, root, history, llm, *, last="2026-08", mode="reconstructed", news=None):
    _, data, info, calendar = retained_data
    root.mkdir()
    cache = monthly.MemoryCache()
    resolver = AcquiredResolver(history, llm, cache, root, max_extractions=6)
    folds = rolling.run_months(
        data,
        info,
        {},
        monthly_plan(data, calendar, "2026-07", last),
        cache,
        root,
        model="ridge",
        llm=llm,
        news_history=news_tests.history() if news is None else news,
        news_required=True,
        news_mode=mode,
        acquired_resolver=resolver,
    )
    return folds, cache, resolver


def test_new_measured_variable_reaches_fit_and_next_month_feedback(retained, tmp_path, monkeypatch):
    source = acquisition_checkpoint(tmp_path / "source.json", monkeypatch)
    original = source.read_bytes()
    history = load_acquired_history([source])
    llm = MeasurementLLM([measured_proposal(), news_tests.retain_news()])
    folds, cache, _ = execute(retained, tmp_path / "study", history, llm)
    assert [fold["candidate_status"] for fold in folds] == ["fitted", "fitted"]
    assert [len(context["mature_feedback"]) for context in llm.contexts] == [0, 1]
    assert len(llm.extractions) == 3
    assert all(p["request"]["input_id"] == "acquired.inventory" for p in llm.extractions)
    for fold in folds:
        fixed = cache.calls[fold["month"] + "_fixed_fit"]
        adaptive = cache.calls[fold["month"] + "_adaptive_fit"]
        assert fixed["y"] == adaptive["y"] and fixed["window"] == adaptive["window"]
        np.testing.assert_array_equal(np.asarray(adaptive["x"])[:, :-1], fixed["x"])
        assert adaptive["factor_ids"] == fixed["factor_ids"] + ["discovery.issuer_inventory"]
        expected = [
            20 if day < "2026-02-23" else 35 if day < "2026-05-25" else 25
            for day in fold["training_origins"]
        ]
        np.testing.assert_array_equal(np.asarray(adaptive["x"])[:, -1], expected)
        assert adaptive["prediction_x"][-1] == 25
        assert fold["fitted_measurement_inputs"] == ["acquired.inventory"]
        coverage = {item["input_id"]: item for item in fold["acquired_inputs"]["coverage"]}[
            "acquired.inventory"
        ]
        assert coverage["training_available"] == coverage["training_samples"] == len(expected)
        assert coverage["distinct_training_values"] == len(set(expected))
        assert coverage["training_disclosures"] == len(set(expected))
    assert "acquired.inventory" not in str(llm.contexts[0]["definitions"])
    assert "acquired.inventory" in str(llm.contexts[1]["acquired_inputs"])
    assert source.read_bytes() == original


def test_late_retained_sources_cannot_extract_or_poison_unrelated_valid_candidate(
    retained, tmp_path, monkeypatch
):
    source = acquisition_checkpoint(tmp_path / "late.json", monkeypatch, late=True)
    decision = measured_proposal()
    news_proposal = news_tests.proposal()
    decision["candidates"].extend(news_proposal["candidates"])
    decision["news_inputs"] = news_proposal["news_inputs"]
    news = news_tests.history()
    for item in news["items"]:
        item["received_at"] = item["version_available_at"] = item["published_at"]
    llm = MeasurementLLM([decision])
    folds, cache, _ = execute(
        retained,
        tmp_path / "retained",
        load_acquired_history([source]),
        llm,
        mode="retained",
        news=news,
        last="2026-07",
    )
    fold = folds[0]
    assert llm.extractions == []
    assert fold["candidate_status"] == "fitted_with_rejections"
    assert cache.calls["2026-07_adaptive_fit"]["factor_ids"][-1] == "discovery.export_pressure"
    assert "discovery.issuer_inventory" not in cache.calls["2026-07_adaptive_fit"]["factor_ids"]
    assert any("issuer_inventory" in str(row) for row in fold["candidate_rejections"])
    assert not fold["acquired_inputs"]["qualified_input_ids"]


def test_cli_retains_acquired_snapshot_and_resumes_without_reextracting(
    retained, tmp_path, monkeypatch
):
    source = acquisition_checkpoint(tmp_path / "source.json", monkeypatch)
    parser = argparse.ArgumentParser()
    rolling.add_arguments(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        [
            "research-rolling",
            "SOXX",
            "--prices-dir",
            str(retained[0]),
            "--first-month",
            "2026-07",
            "--last-month",
            "2026-08",
            "--model",
            "ridge",
            "--llm",
            "codex",
            "--acquired-history",
            str(source),
            "--max-extractions",
            "6",
            "--output",
            str(tmp_path / "cli"),
        ]
    )
    llm = MeasurementLLM([measured_proposal(), RuntimeError("fixture second-month stop")])
    monkeypatch.setattr(rolling, "LLMHypothesizer", lambda *a, **kw: llm)
    with pytest.raises(RuntimeError, match="second-month stop"):
        rolling.run_rolling(args)
    assert len(llm.extractions) == 3
    original = (args.output / "2026-07/prediction.json").read_bytes()
    config = json.loads((args.output / "config.json").read_bytes())
    assert config["max_extractions"] == 6
    assert "acquired" in str(config["input_identity"])
    checkpoint_bytes = {
        file.name: file.read_bytes()
        for file in (args.output / "checkpoints").glob("*.json")
        if "2026-07" in file.name or "extract" in file.name
    }
    source.unlink()
    args.resume = True
    args.max_extractions = 5
    with pytest.raises(ValueError, match="configuration changed"):
        rolling.run_rolling(args)
    args.max_extractions = 6
    llm.decisions = [news_tests.retain_news()]
    result = rolling.run_rolling(args)
    assert result["status"] == "completed"
    assert len(llm.extractions) == 3
    assert (args.output / "2026-07/prediction.json").read_bytes() == original
    for name, raw in checkpoint_bytes.items():
        assert (args.output / "checkpoints" / name).read_bytes() == raw
    assert result["folds"][1]["candidate_status"] == "fitted"
    html = (args.output / "report.html").read_text()
    assert "acquired.inventory" in html
    assert "issuer_inventory" in html
    assert "Fixed baseline" in html and "Sharpe" in html and "<svg" in html
    assert "Source observations in training" in html and "Used by fitted model" in html
    assert result["measurement_usage"]["counts"]["extract"] == 3
    runtime = json.loads((args.output / "runtime.json").read_bytes())
    assert runtime["counts"]["measurement_extraction_attempts_total"] == 3
    assert runtime["counts"]["measurement_extraction_failures_total"] == 0


def test_headline_citation_cannot_substitute_for_measured_source_citation(
    retained, tmp_path, monkeypatch
):
    source = acquisition_checkpoint(tmp_path / "source.json", monkeypatch)
    decision = measured_proposal()
    decision["candidates"][0]["evidence_ids"] = ["first"]
    llm = MeasurementLLM([decision])
    folds, cache, _ = execute(
        retained,
        tmp_path / "wrong-citation",
        load_acquired_history([source]),
        llm,
        last="2026-07",
    )
    assert len(llm.extractions) == 3  # The values exist, but this candidate did not cite them.
    fold = folds[0]
    assert fold["candidate_status"] == "rejected"
    assert "2026-07_adaptive_fit" not in cache.calls
    assert fold["forecasts"]["adaptive"] == fold["forecasts"]["fixed"]
    assert any("cite a source supporting" in str(row) for row in fold["candidate_rejections"])


def test_future_disclosure_values_never_reach_earlier_prompt_extraction_or_matrix(
    retained, tmp_path, monkeypatch
):
    results = []
    for name, future_value in (("first", 999), ("changed", 123456)):
        source = acquisition_checkpoint(
            tmp_path / (name + ".json"), monkeypatch, future_inventory=future_value
        )
        decision = measured_proposal()
        decision["candidates"][0]["evidence_ids"] = ["issuer-0", "issuer-1"]
        decision["data_requests"][0]["source_urls"] = measurement_request()["source_urls"][:2]
        llm = MeasurementLLM([decision])
        folds, cache, _ = execute(
            retained,
            tmp_path / name,
            load_acquired_history([source]),
            llm,
            last="2026-07",
        )
        assert folds[0]["candidate_status"] == "fitted"
        assert len(llm.extractions) == 2
        assert {entry["article"]["id"] for entry in llm.extractions} == {"issuer-0", "issuer-1"}
        assert all("issuer-2" not in str(ctx["acquired_inputs"]) for ctx in llm.contexts)
        results.append((folds[0], cache, llm))
    before, after = results
    assert before[2].contexts == after[2].contexts
    assert before[1].calls["2026-07_fixed_fit"] == after[1].calls["2026-07_fixed_fit"]
    assert before[1].calls["2026-07_adaptive_fit"] == after[1].calls["2026-07_adaptive_fit"]
    assert before[0]["forecasts"] == after[0]["forecasts"]
