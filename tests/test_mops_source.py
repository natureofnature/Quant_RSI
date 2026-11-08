"""Offline official-format fixtures test acquisition/provenance, not research efficacy."""

import gzip
import json
import socket
from copy import deepcopy
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path

import pytest
from test_article_reader import Response
from test_article_reader import transport as transport

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research import article_reader
from quant_rsi.research import mops_source as mops

TITLE = "Fixture company discloses a demand measurement"
BODY = "Original fixture text.\nReported demand: 12.0 percent.\n"
SELECTOR = {"marketKind": "sii", "enterDate": "1151008", "serialNumber": "1", "companyId": "2330"}


def payload(detail=False):
    names = (
        ["序號", "發言日期", "發言時間", "主旨", "說明"]
        if detail
        else ["公司代號", "公司名稱", "發言日期", "發言時間", "主旨", "詳細資料"]
    )
    row = (
        ["1", "115/10/08", "13:54:04", TITLE + "   ", BODY]
        if detail
        else [
            "2330",
            "Fixture issuer",
            "115/10/08",
            "13:54:04",
            TITLE,
            {"parameters": deepcopy(SELECTOR), "apiName": "t05st01_detail"},
        ]
    )
    return {
        "code": 200,
        "datetime": "115/11/08 17:36:50",
        "result": {
            "companyId": "2330",
            "titles": [{"main": name, "sub": []} for name in names],
            "data": [row],
        },
    }


def fetcher(data, *, status=200, content_type="application/json", encoding="identity"):
    raw = data if isinstance(data, bytes) else canonical(data)
    calls = []

    def fetch(url, *, json_body):
        calls.append((url, deepcopy(json_body)))
        headers = Message()
        headers["Content-Type"] = content_type
        headers["Content-Encoding"] = encoding
        return status, headers, raw

    fetch.calls, fetch.raw = calls, raw
    return fetch


def listed(tmp_path):
    return mops.list_disclosures("2330", 2026, 10, root=tmp_path, fetch=fetcher(payload()))[
        "items"
    ][0]


def test_list_and_detail_keep_exact_raw_bytes_request_identity_and_actual_receipt(tmp_path):
    fetch = fetcher(payload())
    before = datetime.now(UTC)
    listing = mops.list_disclosures("2330", 2026, 10, root=tmp_path, fetch=fetch)
    assert listing["status"] == "available"
    assert fetch.calls == [
        (
            mops.LIST_URL,
            {
                "companyId": "2330",
                "year": "115",
                "month": "10",
                "firstDay": "1",
                "lastDay": "31",
            },
        )
    ]
    assert listing["request_sha256"] == digest(
        canonical(
            {
                "method": "POST",
                "url": mops.LIST_URL,
                "body": listing["request"],
            }
        )
    )
    assert Path(listing["raw_path"]).read_bytes() == fetch.raw
    assert listing["raw_sha256"] == digest(fetch.raw)
    item = listing["items"][0]
    assert item["published_at"] == "2026-10-08T13:54:04+08:00"
    detail_fetch = fetcher(payload(True))
    article = mops.read_disclosure(item, root=tmp_path, fetch=detail_fetch)
    assert article["status"] == "available" and article["provider"] == "mops"
    assert detail_fetch.calls == [(mops.DETAIL_URL, SELECTOR)]
    assert article["source_request"] == SELECTOR
    assert item["document_ref"] == "mops-" + article["request_sha256"]
    assert article["id"] == "source-mops-" + digest(
        canonical(
            [
                article["request_sha256"],
                article["raw_sha256"],
            ]
        )
    )
    assert article["content_sha256"] == digest(detail_fetch.raw)
    assert article["title_raw"] == TITLE + "   " and article["text"].endswith(BODY)
    assert "115/10/08" in article["text"] and "13:54:04" in article["text"]
    assert article["publication_date_raw"] == "115/10/08"
    assert article["publication_time_raw"] == "13:54:04"
    assert article["published_at"] != article["received_at"] == article["version_available_at"]
    for receipt in (listing, article):
        assert before <= datetime.fromisoformat(receipt["received_at"]) <= datetime.now(UTC)
        assert receipt["http_requests"] == 1 and receipt["fetch_attempted"]


@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda p: p["result"].update(companyId="9999"), "response_company_mismatch"),
        (lambda p: p["result"]["data"][0].__setitem__(0, "9999"), "row_company_mismatch"),
        (lambda p: p["result"]["data"][0].__setitem__(2, "115/09/08"), "row_month_mismatch"),
        (lambda p: p["result"]["data"][0].__setitem__(3, ""), "invalid_publication_time"),
        (
            lambda p: p["result"]["data"][0][5].update(apiName="untrusted_endpoint"),
            "unsupported_detail_endpoint",
        ),
        (
            lambda p: p["result"]["data"][0][5]["parameters"].update(companyId="9999"),
            "detail_selector_binding_mismatch",
        ),
        (
            lambda p: p["result"]["data"][0][5]["parameters"].update(enterDate="1151009"),
            "detail_selector_binding_mismatch",
        ),
        (
            lambda p: p["result"]["data"][0][5]["parameters"].update(url="https://private.test"),
            "invalid_detail_selector",
        ),
        (
            lambda p: p["result"]["data"].append(deepcopy(p["result"]["data"][0])),
            "duplicate_document_identity",
        ),
    ],
)
def test_list_rejects_unbound_or_missing_source_metadata(tmp_path, change, reason):
    data = payload()
    change(data)
    fetch = fetcher(data)
    result = mops.list_disclosures("2330", 2026, 10, root=tmp_path, fetch=fetch)
    assert result["status"] == "unavailable" and result["reason"] == reason
    assert result["items"] == [] and len(fetch.calls) == 1
    assert Path(result["raw_path"]).read_bytes() == fetch.raw


@pytest.mark.parametrize(
    "column,value,reason",
    [
        (0, "2", "detail_serial_mismatch"),
        (1, "115/10/09", "detail_publication_mismatch"),
        (2, "13:54:05", "detail_publication_mismatch"),
        (3, "Different story", "detail_title_mismatch"),
        (4, "   ", "missing_disclosure_body"),
    ],
)
def test_detail_must_match_selected_listing_before_body_is_evidence(
    tmp_path, column, value, reason
):
    item = listed(tmp_path)
    data = payload(True)
    data["result"]["data"][0][column] = value
    result = mops.read_disclosure(item, root=tmp_path, fetch=fetcher(data))
    assert result["status"] == "unavailable" and result["reason"] == reason
    assert result["text"] == "" and Path(result["raw_path"]).is_file()


def test_selectors_cannot_be_rebound_and_each_post_identity_is_distinct(tmp_path):
    first = listed(tmp_path)
    altered = deepcopy(first)
    altered["selector"]["serialNumber"] = "2"
    with pytest.raises(ValueError, match="document_identity_mismatch"):
        mops.read_disclosure(
            altered, root=tmp_path, fetch=lambda *a, **kw: pytest.fail("No request")
        )
    data = payload()
    data["result"]["data"][0][5]["parameters"]["serialNumber"] = "2"
    second = mops.list_disclosures("2330", 2026, 10, root=tmp_path, fetch=fetcher(data))["items"][0]
    assert first["document_ref"] != second["document_ref"]


def test_immutable_artifacts_reuse_bytes_and_fail_if_existing_contents_change(tmp_path):
    fetch = fetcher(payload())
    first = mops.list_disclosures("2330", 2026, 10, root=tmp_path, fetch=fetch)
    second = mops.list_disclosures("2330", 2026, 10, root=tmp_path, fetch=fetch)
    assert first["raw_path"] == second["raw_path"]
    Path(first["raw_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="immutable content identity"):
        mops.list_disclosures("2330", 2026, 10, root=tmp_path, fetch=fetch)


def test_denied_host_skips_both_operations_without_receipt_or_network(tmp_path):
    item = listed(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Blocked host must not reach transport")

    options = {"root": tmp_path, "avoid_hosts": {mops.HOST: "http_403"}, "fetch": forbidden}
    for result in (
        mops.list_disclosures("2330", 2026, 10, **options),
        mops.read_disclosure(item, **options),
    ):
        assert result["reason"] == "skipped_denied_host"
        assert result["http_requests"] == 0 and not result["fetch_attempted"]
        assert result["received_at"] is None and result["raw_path"] is None


@pytest.mark.parametrize(
    "data,status,content_type,reason",
    [
        (b"{}", 403, "application/json", "http_403"),
        (b"", 302, "application/json", "redirect_not_allowed"),
        (b"<html>not JSON</html>", 200, "text/html", "unsupported_content_type"),
        (b"{bad", 200, "application/json", "invalid_json_response"),
        ({"code": 500, "result": {}}, 200, "application/json", "application_error"),
        (b"x" * (article_reader.MAX_BYTES + 1), 200, "application/json", "response_too_large"),
    ],
)
def test_errors_are_explicit_bounded_and_never_retried(
    tmp_path, data, status, content_type, reason
):
    fetch = fetcher(data, status=status, content_type=content_type)
    result = mops.list_disclosures("2330", 2026, 10, root=tmp_path, fetch=fetch)
    assert result["status"] == "unavailable" and result["reason"] == reason
    assert result["items"] == [] and len(fetch.calls) == 1


def test_compressed_response_bound_and_text_truncation_preserve_original_payload(tmp_path):
    huge = gzip.compress(b"x" * (article_reader.MAX_BYTES + 1))
    result = mops.list_disclosures(
        "2330", 2026, 10, root=tmp_path, fetch=fetcher(huge, encoding="gzip")
    )
    assert result["reason"] == "decompressed_response_too_large"
    assert Path(result["raw_path"]).read_bytes() == huge
    data = payload(True)
    data["result"]["data"][0][-1] = BODY * 500
    fetch = fetcher(data)
    result = mops.read_disclosure(listed(tmp_path), root=tmp_path, fetch=fetch)
    assert result["status"] == "available" and result["text_truncated"]
    assert len(result["text"]) == article_reader.MAX_TEXT
    assert Path(result["raw_path"]).read_bytes() == fetch.raw


@pytest.fixture
def post_transport(transport, monkeypatch):
    base = article_reader.http.client.HTTPConnection

    class Connection(base):
        def request(self, method, path, headers, body=None):
            assert self.sock is not None
            transport["request"].append((method, path, headers, body))

    monkeypatch.setattr(article_reader.http.client, "HTTPConnection", Connection)
    return transport


def test_default_json_post_reuses_pinned_dns_and_tls_transport(tmp_path, post_transport):
    post_transport["responses"].append(
        Response(canonical(payload()), **{"Content-Type": "application/json"})
    )
    result = mops.list_disclosures("2330", 2026, 10, root=tmp_path)
    assert result["status"] == "available"
    assert post_transport["dns"] == [(mops.HOST, 443)]
    assert post_transport["connect"] == [("93.184.216.34", 443)]
    assert post_transport["tls"] == [mops.HOST]
    method, path, headers, body = post_transport["request"][0]
    assert method == "POST" and path == "/mops/api/t05st01"
    assert headers["Content-Type"] == headers["Accept"] == "application/json"
    assert json.loads(body) == result["request"]


@pytest.mark.parametrize("status,reason", [(302, "redirect_not_allowed"), (403, "http_403")])
def test_default_post_never_follows_redirect_or_retries_denial(
    tmp_path, post_transport, status, reason
):
    post_transport["responses"].append(Response(status=status, Location="https://private.test"))
    result = mops.list_disclosures("2330", 2026, 10, root=tmp_path)
    assert result["reason"] == reason and result["http_requests"] == 1
    assert post_transport["dns"] == [(mops.HOST, 443)]
    assert len(post_transport["request"]) == 1


def test_default_post_rejects_non_public_dns_before_connecting(
    tmp_path, post_transport, monkeypatch
):
    monkeypatch.setattr(
        article_reader.socket,
        "getaddrinfo",
        lambda *a, **kw: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 443)),
        ],
    )
    result = mops.list_disclosures("2330", 2026, 10, root=tmp_path)
    assert result["reason"] == "non_public_address" and result["http_requests"] == 0
    assert post_transport["connect"] == post_transport["request"] == []


@pytest.mark.parametrize(
    "company,year,month", [("../../2330", 2026, 10), ("2330", 1911, 10), ("2330", 2026, 13)]
)
def test_bad_arguments_fail_before_transport(tmp_path, company, year, month):
    with pytest.raises(ValueError):
        mops.list_disclosures(
            company, year, month, root=tmp_path, fetch=lambda *a, **k: pytest.fail("No transport")
        )


def test_empty_month_and_leap_year_use_calendar_bounds_without_fabricating_items(tmp_path):
    data = payload()
    data["result"]["data"] = []
    fetch = fetcher(data)
    result = mops.list_disclosures("2330", 2024, 2, root=tmp_path, fetch=fetch)
    assert result["status"] == "available" and result["items"] == []
    assert result["request"]["year"] == "113" and result["request"]["lastDay"] == "29"


def test_detail_company_mismatch_and_changed_body_have_explicit_identities(tmp_path):
    item = listed(tmp_path)
    first = mops.read_disclosure(item, root=tmp_path, fetch=fetcher(payload(True)))
    revision = payload(True)
    revision["result"]["data"][0][-1] += "A later revised statement."
    second = mops.read_disclosure(item, root=tmp_path, fetch=fetcher(revision))
    assert first["document_ref"] == second["document_ref"]
    assert first["id"] != second["id"] and first["raw_path"] != second["raw_path"]
    revision["result"]["companyId"] = "9999"
    rejected = mops.read_disclosure(item, root=tmp_path, fetch=fetcher(revision))
    assert rejected["reason"] == "response_company_mismatch" and rejected["text"] == ""


def test_transport_errors_do_not_leak_exception_details_or_retry(tmp_path):
    calls = []

    def fetch(*args, **kwargs):
        calls.append(1)
        raise OSError("private endpoint or credential must not be disclosed")

    result = mops.list_disclosures("2330", 2026, 10, root=tmp_path, fetch=fetch)
    assert calls == [1] and result["reason"] == "request_failed"
    assert "credential" not in str(result) and "private endpoint" not in str(result)


def test_invalid_gzip_keeps_original_bytes_and_reports_parse_failure(tmp_path):
    fetch = fetcher(b"not gzip", encoding="gzip")
    result = mops.list_disclosures("2330", 2026, 10, root=tmp_path, fetch=fetch)
    assert result["reason"] == "invalid_json_response"
    assert Path(result["raw_path"]).read_bytes() == b"not gzip"
