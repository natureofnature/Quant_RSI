"""Offline source grounding and provider compatibility; no predictive-quality claims."""

import json

import pytest

from quant_rsi.research.discovery_sources import DiscoverySources, discovery_decision


def source_article():
    return {
        "id": "disclosure-1",
        "url": "https://example.com/customer-release",
        "status": "available",
        "published_at": "2025-01-03T12:00:00Z",
        "received_at": "2025-01-04T12:00:00Z",
        "version_available_at": "2025-01-04T12:00:00Z",
        "text": "Customer ABC purchases chips manufactured by TSM.\nCapital expenditure was USD 8.",
        "content_sha256": "a" * 64,
    }


def finding():
    return {
        "mechanism": "Customer investment may support supplier demand.",
        "evidence_ids": ["disclosure-1"],
        "measurements": ["ABC reported capital expenditure, USD"],
        "uncertainty": "No short-term return relationship established.",
    }


def relationship(**changes):
    return {
        "from_entity": "ABC",
        "to_entity": "TSM",
        "relationship": "customer",
        "source_id": "disclosure-1",
        "excerpt": "Customer ABC purchases chips manufactured by TSM.",
    } | changes


class SurveyTransport:
    def __init__(self, findings):
        self.findings = findings
        self.prompt = None

    def structured(self, prompt, schema):
        self.prompt = prompt
        assert "relationships" in schema["$defs"]["SurveyFinding"]["required"]
        return json.dumps({"findings": self.findings, "gaps": ["No utilization disclosure."]})


def survey(findings, *, article=None):
    transport = SurveyTransport(findings)
    sources = DiscoverySources(
        transport,
        information_mode="retained",
        research_cutoff="2025-02-01T00:00:00Z",
        retained_articles=[article or source_article()],
        reader=lambda _: pytest.fail("Offline survey must not read live pages"),
    )
    return sources.survey({"query": "Investigate supplier demand", "symbol": "TSM"}), transport


def test_survey_retains_literal_relationship_and_original_quote():
    quote = "Customer ABC\n purchases chips manufactured by TSM."
    supplied = finding() | {"relationships": [relationship(excerpt=quote)]}
    result, transport = survey([supplied])
    assert result["findings"] == [supplied]
    assert result["gaps"] == ["No utilization disclosure."]
    assert "Related companies are investigation leads" in transport.prompt
    assert "direct customers from downstream payers" in transport.prompt


@pytest.mark.parametrize(
    "link",
    [
        relationship(excerpt="ABC pays all of TSM's research costs."),
        relationship(excerpt=" " * 12),
        relationship(source_id="unprovided-source"),
    ],
)
def test_unsupported_relationship_is_gap_while_other_findings_survive(link):
    legacy = finding()
    result, _ = survey([finding() | {"relationships": [link]}, legacy])
    assert result["findings"] == [legacy]
    assert result["gaps"][0] == "No utilization disclosure."
    assert len(result["gaps"]) == 2
    assert result["gaps"][1].startswith("Rejected relationship finding:")


def test_relation_source_must_be_cited_by_its_own_finding():
    second = source_article() | {"id": "disclosure-2"}
    transport = SurveyTransport(
        [finding() | {"relationships": [relationship(source_id=second["id"])]}]
    )
    sources = DiscoverySources(
        transport,
        information_mode="retained",
        research_cutoff="2025-02-01T00:00:00Z",
        retained_articles=[source_article(), second],
    )
    result = sources.survey("Customer links")
    assert result["findings"] == []
    assert "relationship_source_not_cited_by_finding" in result["gaps"][-1]


def test_relationship_finding_with_unavailable_evidence_becomes_gap():
    supplied = finding() | {
        "evidence_ids": ["unprovided-source"],
        "relationships": [relationship(source_id="unprovided-source")],
    }
    result, _ = survey([supplied])
    assert result["findings"] == []
    assert "finding_cites_unavailable_source" in result["gaps"][-1]


def test_legacy_survey_without_relationship_field_keeps_serialized_shape():
    result, _ = survey([finding()])
    assert result["findings"] == [finding()]
    assert "relationships" not in result["findings"][0]


@pytest.mark.parametrize("survey_symbol", ["absent", None, "ABC"])
def test_decision_optional_survey_symbol_legacy_decode_and_target_preserved(survey_symbol):
    response = {
        "action": "survey",
        "reason": "Investigate the evidenced customer, without adopting an unverified link.",
        "candidates": [],
        "data_requests": [],
        "query": "ABC reported capital expenditure and operating cash flow",
    }
    if survey_symbol != "absent":
        response["survey_symbol"] = survey_symbol

    class DecisionTransport:
        def structured(self, prompt, schema):
            assert "survey_symbol" in schema["required"]
            assert "forecast_channel" in prompt
            assert "context.forecast_target.horizon_sessions" in prompt
            assert "Never silently convert YTD" in prompt
            payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
            assert payload["context"]["symbol"] == "TSM"
            return json.dumps(response)

    result = discovery_decision(
        DecisionTransport(), {"symbol": "TSM", "forecast_target": {"horizon_sessions": 3}}
    )
    if survey_symbol == "ABC":
        assert result["survey_symbol"] == "ABC"
    else:
        assert "survey_symbol" not in result


def test_decision_rejects_invalid_survey_ticker():
    class DecisionTransport:
        def structured(self, prompt, schema):
            return json.dumps(
                {
                    "action": "survey",
                    "reason": "Look for evidence",
                    "candidates": [],
                    "data_requests": [],
                    "query": "Customer demand",
                    "survey_symbol": "not a ticker",
                }
            )

    with pytest.raises(ValueError, match="survey_symbol"):
        discovery_decision(DecisionTransport(), {"symbol": "TSM"})


@pytest.mark.parametrize("matched_field", ["url", "final_url"])
def test_current_foreign_survey_body_precedes_six_old_target_articles(matched_field):
    requested_url = "https://example.com/wmt-customer-disclosure"
    old = [
        source_article() | {"id": f"target-{i}", "url": f"https://example.com/target-{i}"}
        for i in range(6)
    ]
    foreign = source_article() | {
        "url": "https://example.com/redirected-wmt-disclosure",
        "text": "WMT reported capital expenditure of USD 20 billion.",
        matched_field: requested_url,
    }
    read_calls = []

    class Transport:
        def __init__(self):
            self.search_calls = []
            self.analysis_calls = []

        def search(self, prompt, schema):
            self.search_calls.append(prompt)
            return {
                "text": json.dumps(
                    {
                        "query": "WMT capital expenditure",
                        "rationale": "Inspect the downstream payer's reported investment.",
                        "results": [
                            {"title": "WMT release", "url": requested_url, "summary": "Capex"}
                        ],
                    }
                ),
                "search_actions": [],
            }

        def structured(self, prompt, schema):
            self.analysis_calls.append(json.loads(prompt.split("INPUT_JSON:\n", 1)[1]))
            return json.dumps({"findings": [], "gaps": []})

    def read(url):
        read_calls.append(url)
        return foreign

    transport = Transport()
    sources = DiscoverySources(
        transport, retained_articles=old, reader=read, max_reads=6, max_searches=1
    )
    result = sources.survey({"query": "WMT capital expenditure", "symbol": "WMT"})
    payload = transport.analysis_calls[0]
    assert payload["symbol"] == "WMT"
    assert len(payload["articles"]) == 6
    assert payload["articles"][0]["text"] == foreign["text"]
    assert [article["id"] for article in payload["articles"][1:]] == [
        f"target-{i}" for i in range(5)
    ]
    assert {"id": "target-5", "reason": "context_article_limit"} in result["excluded"]
    assert read_calls == [requested_url]
    assert len(transport.search_calls) == len(transport.analysis_calls) == 1
    assert sources.counts == {"search": 1, "read": 1, "extract": 0, "analysis": 1}
    assert len(sources.articles) == 7
    assert [article["id"] for article in sources.articles[:6]] == [f"target-{i}" for i in range(6)]
