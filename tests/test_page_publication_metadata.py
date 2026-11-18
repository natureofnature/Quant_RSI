"""Page-declared publication times bound LLM-quoted dates; the rest is labeled (#159 D3.4)."""

import json
from email.message import Message

import pytest

from quant_rsi.research import article_reader, discovery_sources
from quant_rsi.research.discovery_sources import DiscoverySources, eligible_articles

URL = "https://issuer.example/release"
BODY = "<html><head>{head}</head><body><article>{text}</article></body></html>"
TEXT = (
    "<p>Published May 24, 2023.</p><p>Quarter ended April 30, 2023.</p>"
    "<p>Revenue (USD billions) 4.28</p>"
)
REQUEST = {
    "input_id": "acquired.revenue",
    "symbol": "NVDA",
    "measurement": "Synthetic quarterly revenue",
    "unit": "USD billions",
    "query": "Synthetic issuer quarterly disclosure",
    "source_urls": [URL],
    "max_age_days": 90,
}


def serve(monkeypatch, head="", text=TEXT, content_type="text/html; charset=utf-8"):
    def fetch(url):
        headers = Message()
        headers["Content-Type"] = content_type
        return 200, headers, BODY.format(head=head, text=text).encode()

    monkeypatch.setattr(article_reader, "_fetch", fetch)


def read(monkeypatch, head, **kwargs):
    serve(monkeypatch, head, **kwargs)
    return article_reader.read_article(URL)


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (
            '<meta property="article:published_time" content="2023-05-24T08:00:00-04:00">',
            "2023-05-24T12:00:00Z",
        ),
        ('<meta itemprop="datePublished" content="2023-05-24T12:00:00Z"/>', "2023-05-24T12:00:00Z"),
        (
            '<script type="application/ld+json">'
            '{"@graph": [{"@type": "NewsArticle", "datePublished": "2023-05-24T12:00:00+00:00"}]}'
            "</script>",
            "2023-05-24T12:00:00Z",
        ),
    ],
)
def test_declared_publication_time_is_read_and_normalized(monkeypatch, head, expected):
    result = read(monkeypatch, head)
    assert result["published_at"] == expected
    assert result["publication_metadata"]["status"] == "declared"
    assert "Revenue" in result["text"] and "datePublished" not in result["text"]


@pytest.mark.parametrize(
    ("head", "status"),
    [
        (
            '<meta property="article:published_time" content="2023-05-24T12:00:00Z">'
            '<script type="application/ld+json">{"datePublished": "2023-06-01T12:00:00Z"}</script>',
            "conflicting",
        ),
        (
            '<meta property="article:published_time" content="2023-05-24">',
            "not_zone_aware_or_unparsed",
        ),
        ('<script type="application/ld+json">{not json</script>', None),
    ],
)
def test_ambiguous_or_unusable_metadata_declares_no_time(monkeypatch, head, status):
    result = read(monkeypatch, head)
    assert "published_at" not in result
    assert result.get("publication_metadata", {}).get("status") == status


@pytest.mark.parametrize(
    "head",
    [
        '<meta property="article:modified_time" content="2023-06-01T12:00:00Z">',
        '<script type="application/ld+json">{"dateModified": "2023-06-01T12:00:00Z"}</script>',
        "",
    ],
)
def test_modified_times_and_plain_pages_declare_nothing(monkeypatch, head):
    result = read(monkeypatch, head, text=TEXT + '<time datetime="2023-06-01T12:00:00Z">x</time>')
    assert "published_at" not in result and "publication_metadata" not in result


def test_plain_text_pages_declare_nothing(monkeypatch):
    result = read(monkeypatch, "", content_type="text/plain")
    assert "published_at" not in result


class Extractor:
    def __init__(self, publication_date="2023-05-24", excerpt="Published May 24, 2023."):
        self.publication = (publication_date, excerpt)

    def structured(self, prompt, schema):
        payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
        return json.dumps(
            {
                "publication_date": self.publication[0],
                "publication_excerpt": self.publication[1],
                "measurement": payload["request"]["measurement"],
                "observations": [
                    {
                        "period_start": None,
                        "period_end": "2023-04-30",
                        "period_excerpt": "Quarter ended April 30, 2023.",
                        "value": 4.28,
                        "unit": "USD billions",
                        "excerpt": "Revenue (USD billions) 4.28",
                    }
                ],
                "limitations": ["Synthetic fixture."],
            }
        )


def acquire(monkeypatch, head):
    serve(monkeypatch, head)
    monkeypatch.setattr(discovery_sources, "_now", lambda: "2026-11-05T10:00:01Z")
    return DiscoverySources(Extractor()).acquire(REQUEST, prediction_times=["2023-06-01T20:00:00Z"])


def test_page_metadata_bounds_an_earlier_llm_quoted_date(monkeypatch):
    late = '<meta property="article:published_time" content="2023-06-10T12:00:00Z">'
    bundle = acquire(monkeypatch, late)
    assert bundle["observations"] == []
    assert "publication_before_source_metadata" in " ".join(bundle["gaps"])


@pytest.mark.parametrize(
    ("head", "basis"),
    [
        (
            '<meta property="article:published_time" content="2023-05-24T12:00:00Z">',
            "source_metadata",
        ),
        ("", "llm_quoted_unverified"),
    ],
)
def test_each_observation_records_how_its_publication_date_is_known(monkeypatch, head, basis):
    bundle = acquire(monkeypatch, head)
    assert len(bundle["observations"]) == 1
    observation_id = bundle["observations"][0]["observation_id"]
    assert bundle["publication_basis"] == {observation_id: basis}
    assert bundle["publication_basis_counts"] == {basis: 1}


def test_metadata_does_not_make_a_late_read_eligible_in_retained_mode():
    article = {
        "id": "page",
        "status": "available",
        "published_at": "2023-05-24T12:00:00Z",  # page metadata
        "received_at": "2026-11-05T10:00:00Z",  # read long after the research cutoff
        "version_available_at": "2026-11-05T10:00:00Z",
    }
    admitted, excluded = eligible_articles(
        [article], information_mode="retained", research_cutoff="2023-06-30T20:00:00Z"
    )
    assert admitted == []
    assert excluded == [{"id": "page", "reason": "source_after_research_cutoff"}]
