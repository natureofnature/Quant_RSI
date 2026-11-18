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


ARTICLE_SCOPE = '<div itemscope itemtype="https://schema.org/NewsArticle">{}</div>'


def ld(document):
    return f'<script type="application/ld+json">{json.dumps(document)}</script>'


@pytest.mark.parametrize(
    ("head", "text", "expected"),
    [
        (
            '<meta property="article:published_time" content="2023-05-24T08:00:00-04:00">',
            TEXT,
            "2023-05-24T12:00:00Z",
        ),
        (
            "",
            ARTICLE_SCOPE.format(
                '<meta itemprop="datePublished" content="2023-05-24T12:00:00Z">' + TEXT
            ),
            "2023-05-24T12:00:00Z",
        ),
        (
            ld(
                {
                    "@graph": [
                        {
                            "@type": "NewsArticle",
                            "url": URL,
                            "datePublished": "2023-05-24T12:00:00Z",
                        }
                    ]
                }
            ),
            TEXT,
            "2023-05-24T12:00:00Z",
        ),
        (
            ld(
                {"@type": "NewsArticle", "@id": "#article", "datePublished": "2023-05-24T12:00:00Z"}
            ),
            TEXT,
            "2023-05-24T12:00:00Z",
        ),
        (
            ld(
                {
                    "@graph": [
                        {"@type": "WebPage", "@id": URL + "/", "mainEntity": {"@id": "urn:a"}},
                        {
                            "@type": "Article",
                            "@id": "urn:a",
                            "datePublished": "2023-05-24T12:00:00Z",
                        },
                    ]
                }
            ),
            TEXT,
            "2023-05-24T12:00:00Z",
        ),
    ],
)
def test_declared_publication_time_is_read_and_normalized(monkeypatch, head, text, expected):
    result = read(monkeypatch, head, text=text)
    assert result["published_at"] == expected
    assert result["publication_metadata"]["status"] == "declared"
    assert "Revenue" in result["text"] and "datePublished" not in result["text"]


@pytest.mark.parametrize(
    ("head", "status"),
    [
        (
            '<meta property="article:published_time" content="2023-05-24T12:00:00Z">'
            + ld({"@type": "NewsArticle", "url": URL, "datePublished": "2023-06-01T12:00:00Z"}),
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


OTHER_STORY = "https://issuer.example/other-story"


@pytest.mark.parametrize(
    ("head", "text"),
    [
        # Review on #172: a related story in the page's graph is not this page.
        (
            ld(
                {
                    "@graph": [
                        {"@type": "WebPage", "url": URL},
                        {
                            "@type": "NewsArticle",
                            "url": OTHER_STORY,
                            "@id": OTHER_STORY,
                            "datePublished": "2023-06-10T12:00:00Z",
                        },
                    ]
                }
            ),
            TEXT,
        ),
        # Review on #172: a comment's datePublished is not the article's.
        (
            "",
            TEXT + '<aside itemscope itemtype="https://schema.org/Comment">'
            '<time itemprop="datePublished" datetime="2023-06-10T12:00:00Z">June 10</time></aside>',
        ),
        # An article object that does not identify any page is ambiguous.
        (ld({"@type": "NewsArticle", "datePublished": "2023-06-10T12:00:00Z"}), TEXT),
        # Unscoped microdata, and an article scope in a sidebar.
        ('<meta itemprop="datePublished" content="2023-06-10T12:00:00Z">', TEXT),
        (
            "",
            TEXT
            + "<aside>"
            + ARTICLE_SCOPE.format('<meta itemprop="datePublished" content="2023-06-10T12:00:00Z">')
            + "</aside>",
        ),
        # Two marked-up articles: which one is this page is ambiguous.
        (
            "",
            ARTICLE_SCOPE.format('<meta itemprop="datePublished" content="2023-05-24T12:00:00Z">')
            + ARTICLE_SCOPE.format('<meta itemprop="datePublished" content="2023-05-24T12:00:00Z">')
            + TEXT,
        ),
    ],
)
def test_metadata_not_attached_to_this_page_declares_nothing(monkeypatch, head, text):
    result = read(monkeypatch, head, text=text)
    assert "published_at" not in result


def test_unrelated_metadata_cannot_reject_the_release(monkeypatch):
    comment = (
        '<aside itemscope itemtype="https://schema.org/Comment">'
        '<time itemprop="datePublished" datetime="2023-06-10T12:00:00Z">June 10</time></aside>'
    )
    bundle = acquire(monkeypatch, "", text=TEXT + comment)
    assert len(bundle["observations"]) == 1
    observation_id = bundle["observations"][0]["observation_id"]
    assert bundle["publication_basis"] == {observation_id: "llm_quoted_unverified"}


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


def acquire(monkeypatch, head, text=TEXT):
    serve(monkeypatch, head, text=text)
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
