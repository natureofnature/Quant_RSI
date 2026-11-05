"""Critical offline HTML-to-measurement regression; no real disclosure or research result."""

import hashlib
import json
from datetime import UTC, datetime
from email.message import Message

import pytest

from quant_rsi.research import article_reader, discovery_sources
from quant_rsi.research.discovery_sources import DiscoverySources

URL = "https://issuer.example/quarterly-results"
RECEIVED = "2026-11-05T10:00:00Z"
EXTRACTED = "2026-11-05T10:00:01Z"
BODY = b"""<html><body><form method="post">
<input type="hidden" value="secret-state">
<header>Account navigation</header>
<div><h1>Synthetic quarterly disclosure</h1><p>Published May 24, 2023.</p>
<h2>Data Center</h2><p>Quarter ended April 30, 2023.</p>
<table><tr><th>Revenue (USD billions)</th></tr><tr><td>4.28</td></tr></table></div>
<textarea>Private revenue 99999 USD billions.</textarea><button>Subscribe</button>
<div hidden>Private account</div><footer>Footer navigation</footer>
</form></body></html>"""
REQUEST = {
    "input_id": "acquired.data_center_revenue",
    "symbol": "NVDA",
    "measurement": "Synthetic fixture Data Center quarterly revenue",
    "unit": "USD billions",
    "query": "Synthetic issuer quarterly disclosure",
    "source_urls": [URL],
    "max_age_days": 90,
}
TIMES = [
    "2023-05-24T21:00:00Z",
    "2023-05-25T11:59:59Z",
    "2023-05-25T12:00:00Z",
    "2023-06-01T21:00:00Z",
    "2023-10-01T21:00:00Z",
]


@pytest.fixture
def pipeline(monkeypatch):
    class ReceiptClock(datetime):
        @classmethod
        def now(cls, tz=None):
            # Deterministic present-time receipt, deliberately years after publication.
            return datetime(2026, 11, 5, 10, tzinfo=UTC)

    fetches = []

    def fetch(url):
        fetches.append(url)
        headers = Message()
        headers["Content-Type"] = "text/html; charset=utf-8"
        return 200, headers, BODY

    class LiteralExtractor:
        def __init__(self):
            self.calls = []
            self.value = 4.28
            self.excerpt = "Data Center Quarter ended April 30, 2023. Revenue (USD billions) 4.28"

        def structured(self, prompt, schema):
            payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
            self.calls.append(payload)
            return json.dumps(
                {
                    "publication_date": "2023-05-24",
                    "publication_excerpt": "Published May 24, 2023.",
                    "measurement": REQUEST["measurement"],
                    "observations": [
                        {
                            "period_start": None,
                            "period_end": "2023-04-30",
                            "period_excerpt": "Quarter ended April 30, 2023.",
                            "value": self.value,
                            "unit": "USD billions",
                            "excerpt": self.excerpt,
                        }
                    ],
                    "limitations": ["Synthetic literal extraction, not a real issuer result."],
                }
            )

    monkeypatch.setattr(article_reader, "_fetch", fetch)
    monkeypatch.setattr(article_reader, "datetime", ReceiptClock)
    monkeypatch.setattr(discovery_sources, "_now", lambda: EXTRACTED)
    extractor = LiteralExtractor()
    return DiscoverySources(extractor), extractor, fetches


def test_actual_article_reader_feeds_validated_aligned_measurement(pipeline):
    source, extractor, fetches = pipeline
    bundle = source.acquire(REQUEST, prediction_times=TIMES)
    assert fetches == [URL]
    assert source.counts == {"search": 0, "read": 1, "extract": 1, "analysis": 0}
    assert bundle["columns"][REQUEST["input_id"]] == [None, None, 4.28, 4.28, None]
    assert bundle["gaps"] == []

    article = extractor.calls[0]["article"]
    assert article["text"] == (
        "Synthetic quarterly disclosure Published May 24, 2023. "
        "Data Center Quarter ended April 30, 2023. Revenue (USD billions) 4.28"
    )
    assert article["content_sha256"] == hashlib.sha256(BODY).hexdigest()
    assert article["received_at"] == RECEIVED
    assert article == bundle["articles"][0]
    observation = bundle["observations"][0]
    assert observation["source_sha256"] == article["content_sha256"]
    assert observation["source_id"] == article["id"]
    assert observation["source_url"] == URL
    assert observation["published_at"] == "2023-05-25T12:00:00+00:00"
    assert observation["received_at"] == observation["version_available_at"]
    assert datetime.fromisoformat(observation["received_at"]) == datetime.fromisoformat(RECEIVED)
    assert observation["extracted_at"] == EXTRACTED
    assert observation["availability_basis"] == "reconstructed_public_disclosure_date"
    assert (
        bundle["input_evidence"][REQUEST["input_id"]][2]["observation_id"]
        == observation["observation_id"]
    )
    json.dumps(bundle, allow_nan=False)


@pytest.mark.parametrize("supply_publication_metadata", [False, True])
def test_present_receipt_cannot_be_reused_as_historical_retained_information(
    pipeline, supply_publication_metadata
):
    source, extractor, fetches = pipeline
    bundle = source.acquire(REQUEST, prediction_times=TIMES)
    article = dict(bundle["articles"][0])
    if supply_publication_metadata:
        article.update(
            published_at="2023-05-24T12:00:00Z", version_available_at=article["received_at"]
        )
    historical = DiscoverySources(
        extractor,
        retained_articles=[article],
        information_mode="retained",
        research_cutoff="2023-06-30T21:00:00Z",
    )
    result = historical.acquire(REQUEST, prediction_times=TIMES)
    assert result["columns"][REQUEST["input_id"]] == [None] * len(TIMES)
    assert result["observations"] == result["extractions"] == []
    assert result["excluded"] == [
        {
            "id": article["id"],
            "reason": "source_after_research_cutoff"
            if supply_publication_metadata
            else "missing_retained_timestamp",
        }
    ]
    assert len(extractor.calls) == 1  # No extraction or new read for inadmissible material.
    assert fetches == [URL]
    assert historical.counts == {"search": 0, "read": 0, "extract": 0, "analysis": 0}


def test_form_control_numbers_cannot_support_numerical_extraction(pipeline):
    source, extractor, _ = pipeline
    extractor.value = 99999.0
    extractor.excerpt = "Private revenue 99999 USD billions."
    result = source.acquire(REQUEST, prediction_times=TIMES)
    assert "99999" not in extractor.calls[0]["article"]["text"]
    assert result["observations"] == []
    assert result["columns"][REQUEST["input_id"]] == [None] * len(TIMES)
    assert "unverified_literal_value_or_unit" in result["gaps"]
