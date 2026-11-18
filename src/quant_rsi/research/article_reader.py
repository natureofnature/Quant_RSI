"""Bounded public-page text with actual receipt times; historical admission is separate."""

import gzip
import hashlib
import http.client
import io
import ipaddress
import json
import socket
import ssl
import zlib
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit

MAX_BYTES = 1_048_576
MAX_TEXT = 16_000
TIMEOUT = 15
MAX_REDIRECTS = 3


class _Unavailable(ValueError):
    http_requests = 0


def _url(value):
    if not isinstance(value, str) or len(value) > 2048 or any(ord(c) < 33 for c in value):
        raise _Unavailable("unsafe_url")
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or "%" in parsed.hostname
            or parsed.port not in {None, 443 if parsed.scheme == "https" else 80}
        ):
            raise _Unavailable("unsafe_url")
        parsed.hostname.encode("idna")
    except (ValueError, UnicodeError):
        raise _Unavailable("unsafe_url") from None
    return parsed


def _public_target(parsed):
    port = 443 if parsed.scheme == "https" else 80
    addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    if not addresses:
        raise _Unavailable("request_failed")
    for _, _, _, _, address in addresses:
        ip = ipaddress.ip_address(address[0])
        if not ip.is_global or ip.is_multicast:
            raise _Unavailable("non_public_address")
    # Prefer IPv4 when both are available; connect to the validated address, not the hostname.
    return min(addresses, key=lambda item: item[0] != socket.AF_INET)


def _fetch(url, *, json_body=None):
    payload = None
    if json_body is not None:
        if not isinstance(json_body, dict):
            raise _Unavailable("invalid_json_request")
        payload = json.dumps(json_body, allow_nan=False, separators=(",", ":")).encode()
        if len(payload) > 16_384:
            raise _Unavailable("request_too_large")
    parsed = _url(url)
    family, kind, protocol, _, address = _public_target(parsed)
    connection = http.client.HTTPConnection(parsed.hostname, address[1], timeout=TIMEOUT)
    sock = socket.socket(family, kind, protocol)
    http_requests = 0
    try:
        sock.settimeout(TIMEOUT)
        sock.connect(address)
        if parsed.scheme == "https":
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=parsed.hostname)
        connection.sock = sock
        path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        http_requests = 1
        headers = {
            "User-Agent": "QuantRSI/0.1 (public research article reader)",
            "Accept": "application/json"
            if payload is not None
            else "text/html,text/plain,application/xhtml+xml",
            "Accept-Encoding": "gzip, identity",
        }
        kwargs = {"headers": headers}
        if payload is not None:
            headers["Content-Type"] = "application/json"
            kwargs["body"] = payload
        connection.request("POST" if payload is not None else "GET", path, **kwargs)
        response = connection.getresponse()
        if response.status in {301, 302, 303, 307, 308}:
            if payload is not None:
                raise _Unavailable("redirect_not_allowed")
            return response.status, response.headers, b""
        if response.status != 200:
            raise _Unavailable(f"http_{response.status}")
        content_types = (
            {"application/json"}
            if payload is not None
            else {
                "text/html",
                "text/plain",
                "application/xhtml+xml",
            }
        )
        if response.headers.get_content_type() not in content_types:
            raise _Unavailable("unsupported_content_type")
        if response.headers.get("Content-Encoding", "identity").lower() not in {"identity", "gzip"}:
            raise _Unavailable("unsupported_content_encoding")
        body = response.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise _Unavailable("response_too_large")
        return response.status, response.headers, body
    except (OSError, http.client.HTTPException, ValueError) as exc:
        failure = exc if isinstance(exc, _Unavailable) else _Unavailable("request_failed")
        failure.http_requests = http_requests
        raise failure from None
    finally:
        connection.close()
        sock.close()


# Publication metadata must describe the page being read, not a related story, a comment or
# a sidebar item that happens to carry its own datePublished (#159 D3.4, review on #172).
ARTICLE_TYPES = {
    "Article",
    "NewsArticle",
    "AnalysisNewsArticle",
    "BackgroundNewsArticle",
    "OpinionNewsArticle",
    "ReportageNewsArticle",
    "BlogPosting",
    "Report",
    "ScholarlyArticle",
    "TechArticle",
}
_SECONDARY = {"aside", "nav", "footer", "header", "template"}


def _schema_types(value):
    values = value.split() if isinstance(value, str) else value if isinstance(value, list) else []
    return {str(v).rstrip("/").rsplit("/", 1)[-1] for v in values if isinstance(v, str)}


def _canonical_page(url):
    parts = urlsplit(urldefrag(url)[0])
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/") or "/", parts.query, "")
    )


def _same_page(reference, page_url):
    if isinstance(reference, dict):
        reference = reference.get("@id") or reference.get("url")
    if not isinstance(reference, str) or not reference.strip():
        return False
    return _canonical_page(urljoin(page_url, reference.strip())) == _canonical_page(page_url)


def _linked_data_objects(value, found, depth=0):
    if depth > 6 or len(found) > 200:
        return
    if isinstance(value, list):
        for item in value[:50]:
            _linked_data_objects(item, found, depth + 1)
    elif isinstance(value, dict):
        found.append(value)
        for key in ("@graph", "mainEntity"):
            if key in value:
                _linked_data_objects(value[key], found, depth + 1)


def _linked_data_publication(documents, page_url):
    """datePublished of article objects that identify this page, or that an object
    identifying this page names as its mainEntity; anything else is not this page's."""
    objects = []
    for document in documents:
        _linked_data_objects(document, objects)
    main_entities = []
    for item in objects:
        if _same_page(item.get("url"), page_url) or _same_page(item.get("@id"), page_url):
            entity = item.get("mainEntity")
            main_entities.extend(entity if isinstance(entity, list) else [entity])
    found = []
    for item in objects:
        if not _schema_types(item.get("@type")) & ARTICLE_TYPES:
            continue
        own = any(_same_page(item.get(key), page_url) for key in ("url", "@id", "mainEntityOfPage"))
        named = any(
            entity is item
            or (
                isinstance(entity, dict)
                and isinstance(item.get("@id"), str)
                and entity.get("@id") == item["@id"]
            )
            for entity in main_entities
        )
        if (own or named) and isinstance(item.get("datePublished"), str):
            found.append(("json-ld:datePublished", item["datePublished"]))
    return found


def _publication_metadata(candidates):
    """The page's own declared publication time, only when unambiguous and zone-aware (D3.4)."""
    if not candidates:
        return {}
    parsed, unparsed = set(), []
    for _, raw in candidates:
        try:
            value = datetime.fromisoformat(raw.strip())
        except ValueError:
            value = None
        if value is None or value.tzinfo is None:
            unparsed.append(raw)
        else:
            parsed.add(value.astimezone(UTC).isoformat().replace("+00:00", "Z"))
    sources = sorted({source for source, _ in candidates})
    if len(parsed) == 1 and not unparsed:
        return {
            "published_at": parsed.pop(),
            "publication_metadata": {"status": "declared", "sources": sources},
        }
    status = "conflicting" if len(parsed) > 1 else "not_zone_aware_or_unparsed"
    return {"publication_metadata": {"status": status, "sources": sources}}


class _ArticleText(HTMLParser):
    _ignored = {
        "head",
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "noscript",
        "svg",
        "textarea",
        "select",
        "option",
        "optgroup",
        "button",
        "label",
        "datalist",
        "output",
        "aside",
        "template",
    }
    _void = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.hidden_depths = []
        self.parts = {"article": [], "main": [], "body": []}
        # Declarations the page makes about itself; never dateModified or a bare <time>.
        self.page_meta = []
        self.linked_documents, self.linked_data = [], None
        self.scopes = []  # (stack depth, schema types, inside a secondary container)
        self.article_scopes = 0
        self.microdata = []

    def _metadata(self, tag, attributes):
        name = (attributes.get("property") or attributes.get("name") or "").strip().lower()
        if tag == "meta" and name == "article:published_time" and attributes.get("content"):
            self.page_meta.append(("meta:article:published_time", attributes["content"]))
        if (attributes.get("itemprop") or "").strip() == "datePublished" and self.scopes:
            # Microdata belongs to its innermost itemscope; only a primary article counts.
            _, types, secondary = self.scopes[-1]
            value = attributes.get("datetime") or attributes.get("content")
            if value and types & ARTICLE_TYPES and not secondary:
                if not _SECONDARY.intersection(self.stack):
                    self.microdata.append(("itemprop:datePublished", value))
        kind = (attributes.get("type") or "").strip().lower()
        if tag == "script" and kind == "application/ld+json":
            self.linked_data = []

    def publication_candidates(self, page_url):
        # Microdata is unambiguous only when the page marks up exactly one primary article.
        microdata = self.microdata if self.article_scopes == 1 else []
        linked = _linked_data_publication(self.linked_documents, page_url)
        return self.page_meta + microdata + linked

    def handle_starttag(self, tag, attrs):
        self._metadata(tag, dict(attrs))
        if tag not in self._void:
            self.stack.append(tag)
            attributes = dict(attrs)
            if "itemscope" in attributes:
                types = _schema_types(attributes.get("itemtype") or "")
                secondary = bool(_SECONDARY.intersection(self.stack))
                self.scopes.append((len(self.stack), types, secondary))
                if types & ARTICLE_TYPES and not secondary:
                    self.article_scopes += 1
            styles = {}
            for declaration in (attributes.get("style") or "").lower().split(";"):
                name, separator, value = declaration.partition(":")
                if separator:
                    styles[name.strip()] = value.strip().removesuffix("!important").strip()
            if (
                "hidden" in attributes
                or (attributes.get("aria-hidden") or "").strip().lower() == "true"
                or styles.get("display") == "none"
                or styles.get("visibility") == "hidden"
            ):
                self.hidden_depths.append(len(self.stack))

    def handle_startendtag(self, tag, attrs):
        self._metadata(tag, dict(attrs))

    def handle_endtag(self, tag):
        if tag == "script" and self.linked_data is not None:
            try:
                self.linked_documents.append(json.loads("".join(self.linked_data)))
            except (ValueError, RecursionError):
                pass  # Malformed linked data declares nothing.
            self.linked_data = None
        if tag in self.stack:
            index = len(self.stack) - 1 - self.stack[::-1].index(tag)
            del self.stack[index:]
            self.hidden_depths = [depth for depth in self.hidden_depths if depth <= len(self.stack)]
            self.scopes = [scope for scope in self.scopes if scope[0] <= len(self.stack)]

    def handle_data(self, data):
        if self.linked_data is not None:
            self.linked_data.append(data)
            return
        if self.hidden_depths or self._ignored.intersection(self.stack):
            return
        self.parts["body"].append(data)
        for tag in ("article", "main"):
            if tag in self.stack:
                self.parts[tag].append(data)

    def text(self):
        for tag in ("article", "main", "body"):
            value = " ".join(" ".join(self.parts[tag]).split())
            if value:
                return value
        return ""


def read_article(url, *, avoid_hosts=None):
    """Read at most one bounded public page, following at most three checked redirects.

    An available page is extracted text, not proof of completeness, publication
    time, source accuracy, or a verified relationship. No browser/JS or paywall bypass.
    ``avoid_hosts`` blocks every hop before resolution/connection. ``http_requests``
    counts dispatched HTTP requests (including failed requests), not DNS lookups.
    """
    result = {
        "url": None,
        "final_url": None,
        "status": "unavailable",
        "text": "",
        "text_truncated": False,
        "content_sha256": None,
        "content_extent": "bounded_public_page_text_not_verified_full_article",
        "fetch_attempted": False,
        "http_requests": 0,
    }
    try:
        _url(url)
        result["url"] = url
        current = url
        for redirects in range(MAX_REDIRECTS + 1):
            parsed = _url(current)
            result["final_url"] = current
            host = parsed.hostname.rstrip(".").lower().encode("idna").decode()
            if host in (avoid_hosts or {}):
                result["host_failure"] = avoid_hosts[host]
                result["content_extent"] = "not_read"
                raise _Unavailable("skipped_denied_host")
            result["fetch_attempted"] = True
            status, headers, body = _fetch(current)
            result["http_requests"] += 1
            if status in {301, 302, 303, 307, 308}:
                if redirects == MAX_REDIRECTS:
                    raise _Unavailable("redirect_limit")
                location = headers.get("Location")
                if not location:
                    raise _Unavailable("redirect_without_location")
                current = urljoin(current, location)
                continue
            content_hash = hashlib.sha256(body).hexdigest()
            if headers.get("Content-Encoding", "identity").lower() == "gzip":
                try:
                    with gzip.GzipFile(fileobj=io.BytesIO(body)) as compressed:
                        body = compressed.read(MAX_BYTES + 1)
                except (OSError, EOFError, zlib.error):
                    raise _Unavailable("invalid_compressed_body") from None
                if len(body) > MAX_BYTES:
                    raise _Unavailable("decompressed_response_too_large")
            charset = headers.get_content_charset() or "utf-8"
            try:
                decoded = body.decode(charset, errors="replace")
            except LookupError:
                raise _Unavailable("unsupported_charset") from None
            publication = []
            if headers.get_content_type() == "text/plain":
                text = " ".join(decoded.split())
            else:
                parser = _ArticleText()
                parser.feed(decoded)
                parser.close()
                text = parser.text()
                publication = parser.publication_candidates(current)
            if not text:
                raise _Unavailable("no_readable_text")
            result.update(
                final_url=current,
                status="available",
                text=text[:MAX_TEXT],
                text_truncated=len(text) > MAX_TEXT,
                content_sha256=content_hash,
            )
            result |= _publication_metadata(publication)
            break
    except _Unavailable as exc:
        result["reason"] = str(exc)
        result["http_requests"] += exc.http_requests
    except (OSError, http.client.HTTPException, ValueError):
        result["reason"] = "request_failed"
    result["received_at"] = (
        None
        if result.get("reason") == "skipped_denied_host" and not result["http_requests"]
        else datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    return result
