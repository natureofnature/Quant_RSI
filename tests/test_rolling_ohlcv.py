"""Offline optional OHLCV admission; no network, fits, or mutation of retained inputs."""

import json
from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest
from test_daily_data import chart, install

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.core.data import MarketData
from quant_rsi.market.daily_data import collect_daily
from quant_rsi.research.demo_prices import read_daily_history
from quant_rsi.research.rolling_ohlcv import load_ohlcv_history


def retained_ohlcv(monkeypatch, root, mutate=None):
    payload = chart("2026-11-03")
    item = payload["chart"]["result"][0]
    item["meta"]["symbol"] = "SOXX"
    item["events"] = {}
    quotes = item["indicators"]["quote"][0]
    quotes["high"] = [max(o, c) + 2 for o, c in zip(quotes["open"], quotes["close"], strict=True)]
    quotes["low"] = [min(o, c) - 2 for o, c in zip(quotes["open"], quotes["close"], strict=True)]
    quotes["volume"] = [1000 + i for i in range(len(item["timestamp"]))]
    if mutate:
        mutate(item)
    install(monkeypatch, payload, now="2026-11-04T12:00:00+00:00")
    collect_daily(
        "SOXX",
        root,
        historical_sessions=209,
        future_sessions=20,
        as_of=datetime.fromisoformat("2026-10-31T23:59:59-04:00"),
    )
    data, info = read_daily_history("SOXX", root)
    return root, data, info


def load(retained):
    root, data, info = retained
    return load_ohlcv_history("SOXX", root, data, info)


def rebind_raw(root, mutate):
    """Produce internally rebound fixture evidence to exercise semantic admission checks."""
    raw_path = root / "yahoo_chart_raw.json"
    payload = json.loads(raw_path.read_bytes())
    mutate(payload["chart"]["result"][0])
    body = json.dumps(payload).encode()
    raw_path.write_bytes(body)
    receipt_path = root / "collection_receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    record = next(
        row for row in receipt["downloads"] if row["path"].endswith("yahoo_chart_raw.json")
    )
    record.update(bytes=len(body), sha256=digest(body))
    identity = {
        "response_ref": digest(body),
        "calendar_ref": receipt["result"]["calendar_evidence_ref"],
        "price_cutoff": receipt["price_cutoff"],
    }
    contract_path = root / "data_contract.json"
    contract = json.loads(contract_path.read_bytes())
    contract["source_id"] = "yahoo_SOXX_2026-01-02_2026-10-30_" + digest(canonical(identity))
    contract_body = json.dumps(contract).encode()
    contract_path.write_bytes(contract_body)
    receipt["normalized_inputs"]["contract_file_sha256"] = digest(contract_body)
    receipt["result"]["contract"] = contract
    receipt_path.write_text(json.dumps(receipt))
    data, info = read_daily_history("SOXX", root)
    return root, data, info


def test_retained_columns_align_and_leave_all_original_bytes_and_arrays_unchanged(
    tmp_path, monkeypatch
):
    retained = retained_ohlcv(monkeypatch, tmp_path / "prices")
    root, data, info = retained
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    opens, closes = data.opens.copy(), data.closes.copy()
    bundle = load(retained)
    assert bundle["dates"] == list(data.dates)
    assert len(bundle["dates"]) == 209
    assert set(bundle["columns"]) == {"price.high", "price.low", "price.volume"}
    np.testing.assert_allclose(bundle["columns"]["price.high"], data.closes[:, 0] + 2)
    np.testing.assert_allclose(bundle["columns"]["price.low"], data.opens[:, 0] - 2)
    np.testing.assert_array_equal(bundle["columns"]["price.volume"], np.arange(1000, 1209))
    assert all(row["available"] == 209 and row["missing"] == 0 for row in bundle["coverage"])
    provenance = bundle["provenance"]
    assert provenance["raw_sha256"] == digest(before["yahoo_chart_raw.json"])
    assert provenance["received_at"] == "2026-11-04T12:00:00+00:00"
    assert provenance["source_id"] == info["source_id"]
    json.dumps({k: v for k, v in bundle.items() if k != "columns"}, allow_nan=False)
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    np.testing.assert_array_equal(data.opens, opens)
    np.testing.assert_array_equal(data.closes, closes)


def test_null_values_and_observed_zero_volume_remain_distinct(tmp_path, monkeypatch):
    def mutate(item):
        quote = item["indicators"]["quote"][0]
        quote["high"][40] = None
        quote["low"][41] = None
        quote["volume"][42], quote["volume"][43] = None, 0

    bundle = load(retained_ohlcv(monkeypatch, tmp_path / "nulls", mutate))
    assert np.isnan(bundle["columns"]["price.high"][40])
    assert np.isnan(bundle["columns"]["price.low"][41])
    assert np.isnan(bundle["columns"]["price.volume"][42])
    assert bundle["columns"]["price.volume"][43] == 0
    coverage = {r["factor_id"]: r for r in bundle["coverage"]}
    assert coverage["price.volume"]["available"] == 208
    assert coverage["price.volume"]["observed_zero"] == 1


def test_absent_optional_columns_are_unavailable_not_zero(tmp_path, monkeypatch):
    def mutate(item):
        for field in ("high", "low", "volume"):
            del item["indicators"]["quote"][0][field]

    bundle = load(retained_ohlcv(monkeypatch, tmp_path / "missing", mutate))
    assert all(np.isnan(values).all() for values in bundle["columns"].values())
    assert all(not row["field_present"] and row["missing"] == 209 for row in bundle["coverage"])
    assert all(not row["runnable"] for row in bundle["definitions"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("volume", -1),
        ("volume", 1.5),
        ("volume", True),
        ("volume", float("inf")),
        ("high", 0),
        ("high", 1),
        ("low", 999),
        ("low", "20"),
    ],
)
def test_invalid_nonmissing_values_fail_without_filling(tmp_path, monkeypatch, field, value):
    def mutate(item):
        item["indicators"]["quote"][0][field][40] = value

    retained = retained_ohlcv(monkeypatch, tmp_path / "invalid", mutate)
    with pytest.raises(ValueError, match="OHLCV"):
        load(retained)


def test_raw_tamper_and_receipt_only_rebinding_fail(tmp_path, monkeypatch):
    retained = retained_ohlcv(monkeypatch, tmp_path / "tamper")
    root = retained[0]
    raw = root / "yahoo_chart_raw.json"
    raw.write_bytes(raw.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="bytes/hash"):
        load(retained)
    path = root / "collection_receipt.json"
    receipt = json.loads(path.read_bytes())
    receipt["downloads"][0].update(bytes=raw.stat().st_size, sha256=digest(raw.read_bytes()))
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="normalized source identity"):
        load(retained)


@pytest.mark.parametrize(
    "field,value",
    [
        ("http_status", 404),
        ("response_url", "https://example.com/fake"),
        ("truncated", True),
        ("received_at", "2026-11-05T12:00:00+00:00"),
    ],
)
def test_source_receipt_requires_registered_success_and_actual_times(
    tmp_path, monkeypatch, field, value
):
    retained = retained_ohlcv(monkeypatch, tmp_path / "receipt")
    path = retained[0] / "collection_receipt.json"
    receipt = json.loads(path.read_bytes())
    receipt["downloads"][0][field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="OHLCV"):
        load(retained)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_date",
        "shifted_opening",
        "duplicate_timestamp",
        "duplicate_session",
        "open_mismatch",
        "wrong_symbol",
    ],
)
def test_rebound_source_still_requires_exact_dates_and_matching_quotes(
    tmp_path, monkeypatch, problem
):
    root, _, _ = retained_ohlcv(monkeypatch, tmp_path / "rebound")

    def mutate(item):
        if problem == "missing_date":
            item["timestamp"].pop(40)
            for values in item["indicators"]["quote"][0].values():
                values.pop(40)
        elif problem == "shifted_opening":
            item["timestamp"][40] += 60
        elif problem == "duplicate_timestamp":
            item["timestamp"][40] = item["timestamp"][39]
        elif problem == "duplicate_session":
            item["timestamp"].insert(41, item["timestamp"][40] + 60)
            for values in item["indicators"]["quote"][0].values():
                values.insert(41, values[40])
        elif problem == "open_mismatch":
            item["indicators"]["quote"][0]["open"][40] += 0.5
        else:
            item["meta"]["symbol"] = "KO"

    rebound = rebind_raw(root, mutate)
    with pytest.raises(ValueError):
        load(rebound)


def test_future_optional_values_cannot_change_selected_columns(tmp_path, monkeypatch):
    first = load(retained_ohlcv(monkeypatch, tmp_path / "original"))

    def mutate(item):
        quotes = item["indicators"]["quote"][0]
        for i, t in enumerate(item["timestamp"]):
            if t > datetime.fromisoformat("2026-10-31T23:59:59-04:00").timestamp():
                quotes["high"][i], quotes["low"][i], quotes["volume"][i] = 0, -1, None

    second = load(retained_ohlcv(monkeypatch, tmp_path / "future", mutate))
    assert first["dates"] == second["dates"]
    for key, values in first["columns"].items():
        np.testing.assert_array_equal(values, second["columns"][key])


def test_caller_cannot_mix_normalized_collection_or_metadata(tmp_path, monkeypatch):
    root, data, info = retained_ohlcv(monkeypatch, tmp_path / "caller")
    other = MarketData(data.dates, data.instruments, data.opens.copy(), data.closes + 1)
    with pytest.raises(ValueError, match="target differs"):
        load_ohlcv_history("SOXX", root, other, info)
    changed = deepcopy(info)
    changed["source_id"] += "-different"
    with pytest.raises(ValueError, match="target differs"):
        load_ohlcv_history("SOXX", root, data, changed)


def legacy_source_id(root):
    """Older collectors retained price_cutoff but bound only raw/calendar hashes."""
    receipt_path, contract_path = root / "collection_receipt.json", root / "data_contract.json"
    receipt = json.loads(receipt_path.read_bytes())
    identity = {
        "response_ref": digest((root / "yahoo_chart_raw.json").read_bytes()),
        "calendar_ref": receipt["result"]["calendar_evidence_ref"],
    }
    contract = json.loads(contract_path.read_bytes())
    contract["source_id"] = "yahoo_SOXX_2026-01-02_2026-10-30_" + digest(canonical(identity))
    contract_body = json.dumps(contract).encode()
    contract_path.write_bytes(contract_body)
    receipt["normalized_inputs"]["contract_file_sha256"] = digest(contract_body)
    receipt["result"]["contract"] = contract
    receipt_path.write_text(json.dumps(receipt))
    data, info = read_daily_history("SOXX", root)
    return root, data, info


def test_known_legacy_source_identity_with_cutoff_loads_without_rewriting(tmp_path, monkeypatch):
    current = retained_ohlcv(monkeypatch, tmp_path / "legacy")
    modern = load(current)
    assert modern["provenance"]["source_id_schema"] == "raw_calendar_cutoff"
    legacy = legacy_source_id(current[0])
    before = {p.name: p.read_bytes() for p in legacy[0].iterdir()}
    bundle = load(legacy)
    assert bundle["provenance"]["source_id_schema"] == "raw_calendar"
    assert bundle["provenance"]["price_cutoff"] == modern["provenance"]["price_cutoff"]
    for key in modern["columns"]:
        np.testing.assert_array_equal(bundle["columns"][key], modern["columns"][key])
    assert {p.name: p.read_bytes() for p in legacy[0].iterdir()} == before


def test_legacy_identity_does_not_bypass_explicit_price_cutoff(tmp_path, monkeypatch):
    root, _, _ = retained_ohlcv(monkeypatch, tmp_path / "legacy")
    legacy_source_id(root)
    path = root / "collection_receipt.json"
    receipt = json.loads(path.read_bytes())
    receipt["price_cutoff"] = "2026-10-30T15:59:59-04:00"
    path.write_text(json.dumps(receipt))
    data, info = read_daily_history("SOXX", root)
    with pytest.raises(ValueError, match="selected close exceeds retained cutoff"):
        load_ohlcv_history("SOXX", root, data, info)


def test_legacy_identity_still_rejects_rebound_raw_bytes(tmp_path, monkeypatch):
    root, _, _ = retained_ohlcv(monkeypatch, tmp_path / "legacy")
    legacy_source_id(root)
    raw_path = root / "yahoo_chart_raw.json"
    raw_path.write_bytes(raw_path.read_bytes() + b"\n")
    receipt_path = root / "collection_receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["downloads"][0].update(
        bytes=raw_path.stat().st_size, sha256=digest(raw_path.read_bytes())
    )
    receipt_path.write_text(json.dumps(receipt))
    data, info = read_daily_history("SOXX", root)
    with pytest.raises(ValueError, match="normalized source identity"):
        load_ohlcv_history("SOXX", root, data, info)
