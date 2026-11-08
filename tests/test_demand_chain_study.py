"""Native offline cross-company discovery; fixture fits establish no investment result."""

import json
from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pytest
from test_research_study import command

from quant_rsi.cli import main
from quant_rsi.core.artifacts import digest
from quant_rsi.research import study_discovery
from quant_rsi.research.discovery_engine import DiscoveryExperimentRunner
from quant_rsi.research.discovery_factors import DiscoveryFactor
from quant_rsi.research.discovery_sources import DiscoverySources, discovery_decision
from quant_rsi.research.study_estimators import StandardizedRidge


class DemandChainFixture:
    """Only transport responses are scripted; extraction, alignment and fitting stay real."""

    relation = "WMT sells beverages supplied by KO through its retail stores."
    gap = "Supplier-specific beverage purchases are not disclosed."
    input_id = "acquired.retail_demand"

    def __init__(self, outage_iteration):
        self.outage_iteration = outage_iteration
        self.fail = outage_iteration is not None
        self.source_calls = []
        self.contexts = []
        self.articles = []
        self.fits = []
        self.engines = []

    def sources(self, times, *, call, usage_path):
        start = date.fromisoformat(times[0][:10]) - timedelta(days=45)
        end = date.fromisoformat(times[-1][:10])
        self.articles = []
        while start <= end:
            index = len(self.articles)
            value = float(2 + index % 5)
            period_end = (start - timedelta(days=1)).isoformat()
            text = (
                f"OFFLINE FIXTURE. Published {start.isoformat()}. {self.relation} "
                f"WMT reported retail demand {value} percent for period ending {period_end}."
            )
            self.articles.append(
                {
                    "id": f"retailer-{index}",
                    "url": f"https://example.com/wmt/disclosure-{index}",
                    "status": "available",
                    "published_at": start.isoformat() + "T12:00:00Z",
                    "received_at": start.isoformat() + "T13:00:00Z",
                    "version_available_at": start.isoformat() + "T13:00:00Z",
                    "text": text,
                    "content_sha256": digest(text.encode()),
                    "fixture_value": value,
                    "fixture_period_end": period_end,
                }
            )
            start += timedelta(days=30)
        assert len(self.articles) <= 24
        return DiscoverySources(
            self,
            retained_articles=self.articles,
            reader=lambda _: pytest.fail("Retained fixture URLs must not access the network"),
            call=call,
            usage_path=usage_path,
            max_reads=24,
            max_extractions=24,
        )

    def search(self, prompt, schema):
        payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
        self.source_calls.append(("search", payload))
        assert payload["symbol"] == "WMT"
        return {
            "text": json.dumps(
                {
                    "query": payload["query"],
                    "rationale": "Offline retailer disclosure lookup",
                    "results": [
                        {
                            "title": "Retail demand",
                            "url": self.articles[0]["url"],
                            "summary": "Fixture",
                        }
                    ],
                }
            ),
            "search_actions": [{"type": "search", "query": payload["query"]}],
        }

    def structured(self, prompt, schema):
        payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
        self.source_calls.append(("extract" if "request" in payload else "survey", payload))
        if "request" not in payload:
            assert payload["symbol"] == "WMT"
            assert payload["forecast_target"]["symbol"] == "KO"
            assert payload["forecast_target"]["horizon_sessions"] == 5
            return json.dumps(
                {
                    "findings": [
                        {
                            "mechanism": "Retail demand may inform beverage supplier demand.",
                            "evidence_ids": [self.articles[0]["id"]],
                            "measurements": ["Reported retail demand"],
                            "uncertainty": "Cross-company proxy; predictive effect unknown.",
                            "relationships": [self.relationship()],
                        }
                    ],
                    "gaps": [self.gap],
                }
            )
        article = payload["article"]
        missing = payload["request"]["unit"] == "USD"
        return json.dumps(
            {
                "publication_date": article["published_at"][:10],
                "publication_excerpt": f"Published {article['published_at'][:10]}.",
                "measurement": payload["request"]["measurement"],
                "observations": []
                if missing
                else [
                    {
                        "period_start": None,
                        "period_end": article["fixture_period_end"],
                        "value": article["fixture_value"],
                        "unit": "percent",
                        "excerpt": article["text"],
                    }
                ],
                "limitations": [self.gap if missing else "Synthetic engineering observation."],
            }
        )

    def relationship(self):
        return {
            "from_entity": "WMT",
            "to_entity": "KO",
            "relationship": "Retail distribution of supplier beverages",
            "source_id": self.articles[0]["id"],
            "excerpt": self.relation,
        }

    def decide(self, context):
        self.contexts.append(deepcopy(context))
        iteration = context["iteration"]
        if self.fail and iteration == self.outage_iteration:
            raise ValueError("Offline decision outage after completed source work")
        action = {
            "action": "test",
            "reason": "Offline demand-chain trajectory",
            "candidates": [],
            "data_requests": [],
            "query": None,
        }
        if iteration == 1:
            action.update(action="survey", survey_symbol="WMT", query="WMT beverage demand")
        elif iteration in (2, 3):
            if iteration == 3:
                previous = next(m for m in context["memory"] if m["kind"] == "acquire")
                action["reason"] = (
                    "Use disclosed retail demand instead of unavailable supplier purchases: "
                    + "; ".join(previous["result"]["gaps"])
                )
            action.update(
                action="acquire",
                data_requests=[
                    {
                        "input_id": self.input_id,
                        "symbol": "WMT",
                        "measurement": "Supplier purchases" if iteration == 2 else "Retail demand",
                        "unit": "USD" if iteration == 2 else "percent",
                        "query": "WMT retail demand",
                        "source_urls": [a["url"] for a in self.articles[:1]]
                        if iteration == 2
                        else [],
                        "max_age_days": 100,
                    }
                ],
            )
        elif iteration in (4, 5):
            candidate = {
                "name": "retailer_demand_proxy",
                "description": "Retailer measurement as a beverage demand hypothesis",
                "inputs": [{"name": "retail", "factor_id": self.input_id}],
                "expression": {
                    "schema_version": 2,
                    "nodes": [
                        {
                            "name": "retail",
                            "op": "leaf",
                            "leaf": "retail",
                            "inputs": [],
                            "window": None,
                        }
                    ],
                },
                "evidence_ids": [a["id"] for a in self.articles],
                "parent_recipe_refs": [],
            }
            if iteration == 5:
                diagnostic = next(m for m in context["memory"] if m["kind"] == "diagnostic")
                action["reason"] = "Repair " + diagnostic["result"]["reason"]
                candidate["hypothesis"] = {
                    "operating_mechanism": "Retail demand may lead beverage supplier demand.",
                    "measurement_horizon": "Reported operating period; slow-moving proxy",
                    "forecast_channel": "Disclosure surprise may change short-term expectations.",
                    "forecast_horizon_sessions": context["forecast_target"]["horizon_sessions"],
                    "relationships": [self.relationship()],
                }
            action["candidates"] = [candidate]
        else:
            completed = [m for m in context["memory"] if m["kind"] == "experiment"]
            assert len(completed) == 2
            action.update(
                action="retain",
                reason=f"Read {len(completed)} completed development results; stop fixture.",
            )

        class DecisionTransport:
            def structured(self, prompt, schema):
                return json.dumps(action)

        return discovery_decision(DecisionTransport(), context)


@pytest.mark.parametrize("outage_iteration", [None, 5, 6])
def test_native_demand_chain_sources_semantic_repair_and_resume(
    tmp_path, monkeypatch, outage_iteration
):
    fixture = DemandChainFixture(outage_iteration)
    fit = StandardizedRidge.__init__
    initialize = DiscoveryExperimentRunner.__init__

    def record_fit(self, x, y, alpha):
        fixture.fits.append(np.asarray(x).copy())
        fit(self, x, y, alpha)

    def record_engine(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        fixture.engines.append(self)

    monkeypatch.setattr(StandardizedRidge, "__init__", record_fit)
    monkeypatch.setattr(DiscoveryExperimentRunner, "__init__", record_engine)
    monkeypatch.setattr(study_discovery, "simulated_discovery_sources", fixture.sources)
    monkeypatch.setattr(study_discovery, "simulated_discovery_decision", fixture.decide)
    root = tmp_path / "study"
    args = command(
        root,
        "--research-mode",
        "discovery",
        "--llm",
        "simulated",
        "--model",
        "ridge",
        "--rounds",
        "6",
        "--max-fits",
        "2",
        "--horizon-sessions",
        "5",
        "--no-ui",
        "--no-history",
    )
    assert main(args) == (1 if outage_iteration else 0)
    if outage_iteration:
        assert len(fixture.fits) == (0 if outage_iteration == 5 else 2)
        source_calls = deepcopy(fixture.source_calls)
        snapshots = {
            path: path.read_bytes()
            for directory in (root / "checkpoints", root / "discovery" / "memory")
            for path in directory.glob("*.json")
        }
        fixture.fail = False
        assert main(["research-resume", str(root)]) == 0
        assert fixture.source_calls == source_calls
        assert all(path.read_bytes() == data for path, data in snapshots.items())

    result = json.loads((root / "result.json").read_text())
    discovery = result["discovery"]
    records = discovery["memory"]
    assert result["target"] == "KO"
    assert discovery["counts"]["fits"] == len(fixture.fits) == 2
    assert discovery["counts"]["acquisitions"] == 2
    assert discovery["counts"]["surveys"] == 1
    assert discovery["stop_reason"] == "retain"
    contracts = [c["forecast_target"] for c in fixture.contexts]
    assert all(contract == contracts[0] for contract in contracts)
    assert contracts[0]["symbol"] == "KO"
    assert contracts[0]["horizon_sessions"] == 5
    assert contracts[0]["outcome"] == "close at origin plus horizon_sessions"
    contexts = {c["iteration"]: c for c in fixture.contexts}
    survey = next(m for m in contexts[2]["memory"] if m["kind"] == "survey")
    assert survey["request"]["symbol"] == "WMT"
    assert survey["result"]["gaps"] == [fixture.gap]
    missing = next(m for m in contexts[3]["memory"] if m["kind"] == "acquire")
    assert missing["result"]["status"] == "needs_data"
    assert fixture.gap in " ".join(missing["result"]["gaps"])
    decisions = {m["decision"]["action"]: m for m in records if m["kind"] == "interpretation"}
    assert fixture.gap in decisions["acquire"]["decision"]["reason"]
    rejected = next(m for m in contexts[5]["memory"] if m["kind"] == "diagnostic")
    assert rejected["result"]["reason"] == "cross_entity_hypothesis_required"
    assert contexts[5]["counts"]["fits"] == 0
    experiments = [m for m in contexts[6]["memory"] if m["kind"] == "experiment"]
    assert len(experiments) == 2
    candidate = next(m for m in experiments if m["spec"]["kind"] == "candidate")
    failed_recipe = DiscoveryFactor.model_validate_json(
        json.dumps(rejected["candidate"])
    ).recipe_ref
    repaired_recipe = DiscoveryFactor.model_validate_json(
        json.dumps(candidate["spec"]["candidate"])
    ).recipe_ref
    assert failed_recipe == repaired_recipe
    assert candidate["spec"]["candidate"]["parent_recipe_refs"] == []
    acquired = [m["result"] for m in records if m["kind"] == "acquire"][-1]
    assert acquired["request"]["symbol"] == "WMT"
    actual = candidate["result"]["actual_features"]
    assert actual["input_bundle_refs"] == [acquired["identity"]]
    assert actual["definition"]["input_factor_ids"] == [fixture.input_id]
    assert actual["distinct_training_values"] > 1
    engine = fixture.engines[-1]
    aligned = np.array(acquired["columns"][fixture.input_id], dtype=float)[engine.model.train]
    np.testing.assert_array_equal(actual["training_values"], aligned)
    np.testing.assert_array_equal(fixture.fits[-1][:, -1], aligned)
    assert fixture.fits[1].shape[1] == fixture.fits[0].shape[1] + 1
    assert all(a["prediction"]["horizon_sessions"] == 5 for a in result["iterations"]["attempts"])
    assert discovery["source_counts"]["extract"] == len(fixture.articles) + 1
    report = (root / "report.html").read_text()
    assert fixture.relation in report
    assert "retailer_demand_proxy" in report
    assert "short-term expectations" in report
    assert fixture.articles[0]["id"] in report
