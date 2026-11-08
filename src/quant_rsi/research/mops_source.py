"""Bounded official MOPS disclosures; collection does not select or create factors."""

import calendar
import gzip
import io
import json
import re
import zlib
from copy import deepcopy
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research import article_reader

HOST = "mops.twse.com.tw"
LIST_URL = f"https://{HOST}/mops/api/t05st01"
DETAIL_URL = f"https://{HOST}/mops/api/t05st01_detail"
MAX_RECORDS = 256


class _Invalid(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _Invalid(reason)


def _company(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9]{4,6}", value), "invalid_company_id")
    return value


def _publication(day, clock):
    _require(
        isinstance(day, str) and re.fullmatch(r"[0-9]{1,4}/[0-9]{2}/[0-9]{2}", day),
        "invalid_publication_date",
    )
    _require(
        isinstance(clock, str) and re.fullmatch(r"[0-9]{2}:[0-9]{2}:[0-9]{2}", clock),
        "invalid_publication_time",
    )
    year, month, day_number = map(int, day.split("/"))
    _require(year > 0, "invalid_publication_date")
    try:
        published = datetime.fromisoformat(f"{date(year + 1911, month, day_number)}T{clock}")
    except ValueError:
        raise _Invalid("invalid_publication_datetime") from None
    return published.replace(tzinfo=ZoneInfo("Asia/Taipei"))


def _request_hash(url, body):
    return digest(canonical({"method": "POST", "url": url, "body": body}))


def _selector(value, company, day):
    _require(
        isinstance(value, dict)
        and set(value) == {"marketKind", "enterDate", "serialNumber", "companyId"},
        "invalid_detail_selector",
    )
    _require(value["marketKind"] == "sii", "unsupported_market")
    _require(
        value["companyId"] == company and value["enterDate"] == day.replace("/", ""),
        "detail_selector_binding_mismatch",
    )
    _require(
        isinstance(value["serialNumber"], str)
        and re.fullmatch(r"[1-9][0-9]{0,8}", value["serialNumber"]),
        "invalid_serial_number",
    )
    return deepcopy(value)


def _retain(root, body):
    sha = digest(body)
    path = Path(root).resolve() / "raw" / (sha + ".body")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(body)
    except FileExistsError:
        if path.is_symlink() or path.stat().st_size != len(body) or path.read_bytes() != body:
            raise ValueError(
                "Retained MOPS response differs from its immutable content identity"
            ) from None
    return sha, str(path)


def _post(url, request, *, root, avoid_hosts, fetch):
    receipt = {
        "provider": "mops",
        "status": "unavailable",
        "url": url,
        "method": "POST",
        "request": deepcopy(request),
        "request_sha256": _request_hash(url, request),
        "received_at": None,
        "raw_sha256": None,
        "raw_path": None,
        "fetch_attempted": False,
        "http_requests": 0,
    }
    if HOST in (avoid_hosts or {}):
        return receipt | {"reason": "skipped_denied_host", "host_failure": avoid_hosts[HOST]}, None
    receipt["fetch_attempted"] = True
    try:
        status, headers, body = (fetch or article_reader._fetch)(url, json_body=request)
        receipt["http_requests"] = 1
    except article_reader._Unavailable as exc:
        return receipt | {
            "reason": str(exc),
            "http_requests": exc.http_requests,
            "received_at": datetime.now(UTC).isoformat(),
        }, None
    except (OSError, ValueError):
        return receipt | {
            "reason": "request_failed",
            "received_at": datetime.now(UTC).isoformat(),
        }, None
    receipt["received_at"] = datetime.now(UTC).isoformat()
    receipt["http_status"] = status
    if status != 200:
        reason = "redirect_not_allowed" if status in {301, 302, 303, 307, 308} else f"http_{status}"
        return receipt | {"reason": reason}, None
    if not isinstance(body, bytes) or len(body) > article_reader.MAX_BYTES:
        return receipt | {"reason": "response_too_large"}, None
    receipt["raw_sha256"], receipt["raw_path"] = _retain(root, body)
    if headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
        return receipt | {"reason": "unsupported_content_type"}, None
    encoding = headers.get("Content-Encoding", "identity").lower()
    receipt["content_encoding"] = encoding
    try:
        decoded = body
        if encoding == "gzip":
            with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:
                decoded = stream.read(article_reader.MAX_BYTES + 1)
        _require(encoding in {"identity", "gzip"}, "unsupported_content_encoding")
        _require(len(decoded) <= article_reader.MAX_BYTES, "decompressed_response_too_large")
        payload = json.loads(
            decoded.decode("utf-8"),
            parse_constant=lambda _: _require(False, "invalid_json_response"),
        )
        _require(
            isinstance(payload, dict)
            and type(payload.get("code")) is int
            and payload["code"] == 200,
            "application_error",
        )
        _require(isinstance(payload.get("result"), dict), "invalid_response_shape")
    except _Invalid as exc:
        return receipt | {"reason": str(exc)}, None
    except (ValueError, OSError, EOFError, zlib.error):
        return receipt | {"reason": "invalid_json_response"}, None
    return receipt, payload["result"]


def _table(result, company, required):
    _require(result.get("companyId") == company, "response_company_mismatch")
    titles, data = result.get("titles"), result.get("data")
    _require(isinstance(titles, list) and 1 <= len(titles) <= 32, "invalid_table_headers")
    names = [item.get("main") if isinstance(item, dict) else None for item in titles]
    _require(all(isinstance(name, str) for name in names), "invalid_table_headers")
    _require(len(set(names)) == len(names) and required <= set(names), "invalid_table_headers")
    _require(isinstance(data, list) and len(data) <= MAX_RECORDS, "invalid_table_rows")
    _require(
        all(isinstance(row, list) and len(row) == len(names) for row in data), "invalid_table_rows"
    )
    return [dict(zip(names, row, strict=True)) for row in data]


def list_disclosures(company_id, year, month, *, root, avoid_hosts=None, fetch=None):
    """List one Gregorian calendar month; no selection, redirects or implicit retries."""
    company_id = _company(company_id)
    _require(type(year) is int and 1912 <= year <= 9999, "invalid_gregorian_year")
    _require(type(month) is int and 1 <= month <= 12, "invalid_month")
    request = {
        "companyId": company_id,
        "year": str(year - 1911),
        "month": f"{month:02d}",
        "firstDay": "1",
        "lastDay": str(calendar.monthrange(year, month)[1]),
    }
    receipt, response = _post(LIST_URL, request, root=root, avoid_hosts=avoid_hosts, fetch=fetch)
    receipt["items"] = []
    if response is None:
        return receipt
    try:
        rows = _table(
            response, company_id, {"公司代號", "發言日期", "發言時間", "主旨", "詳細資料"}
        )
        seen, items = set(), []
        for row in rows:
            _require(row["公司代號"] == company_id, "row_company_mismatch")
            publication = _publication(row["發言日期"], row["發言時間"])
            _require((publication.year, publication.month) == (year, month), "row_month_mismatch")
            _require(
                isinstance(row["主旨"], str) and 0 < len(row["主旨"].rstrip()) <= 1000,
                "invalid_disclosure_title",
            )
            detail = row["詳細資料"]
            _require(
                isinstance(detail, dict) and detail.get("apiName") == "t05st01_detail",
                "unsupported_detail_endpoint",
            )
            selector = _selector(detail.get("parameters"), company_id, row["發言日期"])
            ref = "mops-" + _request_hash(DETAIL_URL, selector)
            _require(ref not in seen, "duplicate_document_identity")
            seen.add(ref)
            items.append(
                {
                    "document_ref": ref,
                    "title": row["主旨"],
                    "company_id": company_id,
                    "published_at": publication.isoformat(),
                    "publication_date_raw": row["發言日期"],
                    "publication_time_raw": row["發言時間"],
                    "selector": selector,
                }
            )
    except _Invalid as exc:
        return receipt | {"reason": str(exc)}
    return receipt | {"status": "available", "items": items}


def read_disclosure(item, *, root, avoid_hosts=None, fetch=None):
    """Read exactly the listed POST selector and retain the original disclosure payload."""
    _require(isinstance(item, dict), "invalid_disclosure_item")
    company = _company(item.get("company_id"))
    publication = _publication(item.get("publication_date_raw"), item.get("publication_time_raw"))
    selector = _selector(item.get("selector"), company, item["publication_date_raw"])
    _require(
        item.get("document_ref") == "mops-" + _request_hash(DETAIL_URL, selector),
        "document_identity_mismatch",
    )
    _require(item.get("published_at") == publication.isoformat(), "publication_binding_mismatch")
    _require(
        isinstance(item.get("title"), str) and 0 < len(item["title"].rstrip()) <= 1000,
        "invalid_disclosure_title",
    )
    receipt, response = _post(DETAIL_URL, selector, root=root, avoid_hosts=avoid_hosts, fetch=fetch)
    receipt.update(
        {
            "document_ref": item["document_ref"],
            "source_request": deepcopy(selector),
            "title": item["title"],
            "text": "",
            "text_truncated": False,
            "content_extent": "official_disclosure_fields_not_verified_complete_article",
            "content_sha256": receipt["raw_sha256"],
            "company_id": company,
            "published_at": publication.isoformat(),
            "version_available_at": receipt["received_at"],
            "publication_date_raw": item["publication_date_raw"],
            "publication_time_raw": item["publication_time_raw"],
            "id": "source-mops-"
            + digest(canonical([receipt["request_sha256"], receipt["raw_sha256"]])),
        }
    )
    if response is None:
        return receipt
    try:
        rows = _table(response, company, {"序號", "發言日期", "發言時間", "主旨", "說明"})
        _require(len(rows) == 1, "detail_row_count_mismatch")
        row = rows[0]
        _require(row["序號"] == selector["serialNumber"], "detail_serial_mismatch")
        _require(
            row["發言日期"] == item["publication_date_raw"]
            and row["發言時間"] == item["publication_time_raw"],
            "detail_publication_mismatch",
        )
        _require(
            isinstance(row["主旨"], str) and row["主旨"].rstrip() == item["title"].rstrip(),
            "detail_title_mismatch",
        )
        _require(
            isinstance(row["說明"], str) and bool(row["說明"].strip()), "missing_disclosure_body"
        )
        text = (
            f"公司代號: {company}\n序號: {row['序號']}\n發言日期: {row['發言日期']}\n"
            f"發言時間: {row['發言時間']}\n主旨: {row['主旨']}\n說明:\n{row['說明']}"
        )
    except _Invalid as exc:
        return receipt | {"reason": str(exc)}
    return receipt | {
        "status": "available",
        "title_raw": row["主旨"],
        "text": text[: article_reader.MAX_TEXT],
        "text_truncated": len(text) > article_reader.MAX_TEXT,
    }
