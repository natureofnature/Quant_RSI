"""Saved business hypotheses are visible as proposals, with source subjects and gaps."""

from copy import deepcopy

from quant_rsi.research.discovery_report import render_discovery


def fixture():
    relation = {
        "from_entity": "CUST",
        "to_entity": "TARG",
        "relationship": "Proposed component customer",
        "source_id": "news-customer",
        "excerpt": "CUST purchases components from TARG.",
    }
    candidate = {
        "name": "customer_investment",
        "description": "A proposed business input, not established predictive value.",
        "inputs": [{"name": "capex", "factor_id": "acquired.customer_capex"}],
        "expression": {
            "schema_version": 2,
            "nodes": [{"name": "raw", "op": "leaf", "leaf": "capex"}],
        },
        "evidence_ids": ["news-customer"],
        "hypothesis": {
            "operating_mechanism": "Customer investment may change component demand.",
            "measurement_horizon": "Calendar quarter",
            "forecast_channel": "Unexpected investment disclosures may reprice near-term demand.",
            "forecast_horizon_sessions": 5,
            "relationships": [relation],
        },
    }
    return {
        "symbol": "TARG",
        "configuration": {"question": "Investigate source-backed demand mechanisms"},
        "notice": "Invented report fixture; not predictive evidence.",
        "study": {"conclusion": "Retain missing evidence as gaps"},
        "discovery": {
            "counts": {"surveys": 1, "acquisitions": 1, "fits": 0},
            "groups": {},
            "selected": None,
            "memory": [
                {
                    "kind": "survey",
                    "ref": "survey-reference",
                    "request": {"symbol": "CUST", "query": "Customer investment and financing"},
                    "result": {
                        "search": {
                            "query": "Customer investment and financing",
                            "status": "completed",
                        },
                        "findings": [
                            {
                                "mechanism": "Investment can affect supplier demand.",
                                "evidence_ids": ["news-customer"],
                                "measurements": ["Capital expenditure in USD millions"],
                                "uncertainty": "Orders are not disclosed.",
                                "relationships": [deepcopy(relation)],
                            }
                        ],
                        "gaps": [
                            "Private payer revenue unavailable",
                            "Comparable units not disclosed",
                        ],
                    },
                },
                {
                    "kind": "interpretation",
                    "decision": {"action": "test", "candidates": [candidate]},
                },
                {
                    "kind": "acquire",
                    "request": {
                        "input_id": "acquired.customer_capex",
                        "symbol": "CUST",
                        "measurement": "Capital expenditure",
                        "unit": "USD millions",
                    },
                    "result": {
                        "identity": "bundle-customer",
                        "status": "available",
                        "observations": [
                            {
                                "input_id": "acquired.customer_capex",
                                "period_end": "2026-09-30",
                                "value": 100,
                                "unit": "USD millions",
                                "source_url": "https://example.invalid/customer-quarter",
                                "effective_available_at": "2026-10-20T12:00:00Z",
                                "availability_basis": "retained_source",
                                "excerpt": "Capital expenditure was USD 100 million.",
                            }
                        ],
                    },
                },
            ],
        },
    }


def visible(value):
    markdown, html = render_discovery(value)
    # Complete-evidence JSON is intentionally not sufficient presentation evidence.
    return markdown, html.split("<details>", 1)[0]


def test_report_shows_operating_and_forecast_horizons_relationship_quotes_and_survey_gaps():
    markdown, html = visible(fixture())
    for document in (markdown, html):
        for expected in (
            "Operating mechanism and proposed forecast channel",
            "Calendar quarter",
            "Forecast horizon (sessions)",
            "Unexpected investment disclosures may reprice near-term demand.",
            "Proposed relationships — quoted evidence, not verified links",
            "CUST purchases components from TARG.",
            "news-customer",
            "Customer investment and financing",
            "Capital expenditure in USD millions",
            "Orders are not disclosed.",
            "Private payer revenue unavailable",
            "Comparable units not disclosed",
        ):
            assert expected in document
    assert " | 5" in markdown and "<td>5</td>" in html
    assert "acquired.customer_capex | CUST | Capital expenditure / USD millions" in markdown
    assert "acquired.customer_capex | CUST | 2026-09-30 | 100 USD millions" in markdown


def test_null_forecast_channel_remains_an_explicit_gap_and_html_is_escaped():
    value = fixture()
    hypothesis = value["discovery"]["memory"][1]["decision"]["candidates"][0]["hypothesis"]
    hypothesis["forecast_channel"] = None
    hypothesis["operating_mechanism"] = "<script>Untrusted source text</script>"
    markdown, html = visible(value)
    for document in (markdown, html):
        assert "Unspecified — needs data" in document
    assert "<script>" not in html and "&lt;script&gt;Untrusted source text&lt;/script&gt;" in html


def test_legacy_records_do_not_gain_inferred_horizons_relationships_or_source_symbols():
    value = fixture()
    memory = value["discovery"]["memory"]
    memory.pop(0)
    del memory[0]["decision"]["candidates"][0]["hypothesis"]
    del memory[1]["request"]["symbol"]
    before = deepcopy(value)
    markdown, html = visible(value)
    assert value == before
    for document in (markdown, html):
        assert "Operating mechanism and proposed forecast channel" not in document
        assert "Forecast horizon (sessions)" not in document
        assert "Proposed relationships" not in document
        assert "Calendar quarter" not in document
    assert "acquired.customer_capex | Unavailable | Capital expenditure / USD millions" in markdown
    assert "acquired.customer_capex | Unavailable | 2026-09-30 | 100 USD millions" in markdown
