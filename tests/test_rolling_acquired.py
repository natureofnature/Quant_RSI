"""Bounded source replay and extraction tests use literal synthetic issuer excerpts only."""

import json
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_rolling_measurements import MeasurementLLM, acquisition_checkpoint, measurement_request

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research import discovery_sources
from quant_rsi.research.discovery_sources import (
    DataRequest,
    NumericalExtraction,
    align_input_bundle,
    validate_extraction,
)
from quant_rsi.research.rolling_acquired import (
    AcquiredResolver,
    align_acquired_history,
    load_acquired_history,
)
from quant_rsi.research.study_resume import StudyCache

TIMES = ["2026-01-05T21:00:00Z", "2026-03-02T21:00:00Z", "2026-06-30T20:00:00Z"]
CUTOFF = TIMES[-1]


@pytest.fixture
def archive(tmp_path, monkeypatch):
    return acquisition_checkpoint(tmp_path / "archive.json", monkeypatch)


def save(path, body, *, rebuild=False):
    if rebuild:
        request = DataRequest.model_validate_json(canonical(body["request"]))
        articles = {row["id"]: row for row in body["articles"]}
        observations = []
        for extraction in body["extractions"]:
            rows, _ = validate_extraction(
                request,
                articles[extraction["article_id"]],
                NumericalExtraction.model_validate_json(canonical(extraction["result"])),
                extracted_at=extraction["extracted_at"],
                information_mode="retrospective",
                research_cutoff=None,
            )
            observations.extend(rows)
        old_times = [row["prediction_time"] for row in body["input_evidence"][request.input_id]]
        body |= align_input_bundle(request, observations, old_times)
    path.write_bytes(
        canonical(
            {"input_sha256": "a" * 64, "output": body, "output_sha256": digest(canonical(body))}
        )
    )
    return path


def resolver(history, provider, directory, *, cache=None, maximum=6):
    directory.mkdir(exist_ok=True)
    cache = cache or StudyCache(
        directory, {"factor_root": str(Path(__file__).resolve().parents[1] / "instruments")}
    )
    return AcquiredResolver(history, provider, cache, directory, max_extractions=maximum), cache


def test_archive_is_path_independent_and_origin_is_not_invented(archive, tmp_path):
    history = load_acquired_history([archive])
    copied = tmp_path / "retained.json"
    copied.write_bytes(archive.read_bytes())
    assert history == load_acquired_history([copied])
    assert history["provenance"][0]["acquisition_origin"] == "not_attested"
    assert history["provenance"][0]["original_http_bytes_verified"] is False
    output = json.loads(archive.read_bytes())["output"]
    output["acquisition_origin"] = "manual_reviewed"
    save(copied, output)
    assert (
        load_acquired_history([copied])["provenance"][0]["acquisition_origin"] == "manual_reviewed"
    )
    with pytest.raises(ValueError, match="Duplicate acquired input"):
        load_acquired_history([archive, archive])


@pytest.mark.parametrize(
    "damage", ["checksum", "column", "quote", "future_publication", "duplicate_source"]
)
def test_archive_hashes_do_not_replace_source_revalidation(archive, damage):
    receipt = json.loads(archive.read_bytes())
    body = receipt["output"]
    if damage in {"checksum", "column"}:
        body["columns"]["acquired.revenue"][0] += 999
    elif damage == "quote":
        body["extractions"][0]["result"]["observations"][0]["excerpt"] = "invented revenue 100"
    elif damage == "future_publication":
        body["extractions"][0]["result"]["publication_date"] = "2030-01-01"
    else:
        body["articles"].append(deepcopy(body["articles"][0]))
    if damage == "checksum":
        archive.write_bytes(canonical(receipt))
    else:
        save(archive, body)
    with pytest.raises(ValueError, match="checksum|replayed|Duplicate"):
        load_acquired_history([archive])


def test_availability_staleness_conflicts_and_empty_rows(archive):
    history = load_acquired_history([archive])
    view = align_acquired_history(
        history,
        ["2025-11-17T11:59:59Z", "2025-11-17T12:00:00Z"],
        "2025-11-18T00:00:00Z",
        "reconstructed",
    )
    np.testing.assert_allclose(view["columns"]["acquired.revenue"], [np.nan, 100], equal_nan=True)
    assert view["coverage"][0]["factor_id"] == "acquired.revenue"
    assert view["coverage"][0]["symbol"] == "NVDA"
    assert view["input_evidence"]["acquired.revenue"][0]["observation_id"] is None
    retained = align_acquired_history(history, TIMES, CUTOFF, "retained")
    assert np.isnan(retained["columns"]["acquired.revenue"]).all() and not retained["evidence"]
    stale = align_acquired_history(
        history, ["2027-01-04T21:00:00Z"], "2027-01-04T21:00:00Z", "reconstructed"
    )
    assert not stale["qualified_input_ids"]
    assert (
        len(
            align_acquired_history(history, [], CUTOFF, "reconstructed")["columns"][
                "acquired.revenue"
            ]
        )
        == 0
    )
    body = json.loads(archive.read_bytes())["output"]
    article = deepcopy(body["articles"][2])
    extraction = deepcopy(body["extractions"][2])
    article.update(
        id="conflicting",
        url="https://example.com/conflict",
        text=article["text"].replace("revenue 110", "revenue 111"),
    )
    article["content_sha256"] = digest(article["text"].encode())
    extraction["article_id"] = "conflicting"
    observation = extraction["result"]["observations"][0]
    observation["value"] = 111.0
    observation["excerpt"] = observation["excerpt"].replace("revenue 110", "revenue 111")
    body["articles"].append(article)
    body["extractions"].append(extraction)
    save(archive, body, rebuild=True)
    conflict = align_acquired_history(
        load_acquired_history([archive]), TIMES, CUTOFF, "reconstructed"
    )
    assert np.isnan(conflict["columns"]["acquired.revenue"][-1])
    assert conflict["coverage"][0]["conflicts"]


def test_new_literal_measurement_uses_only_verified_spans_and_success_replays(archive, tmp_path):
    body = json.loads(archive.read_bytes())["output"]
    for article in body["articles"]:
        article["text"] += " Future 2028 navigation: inventory 999999 USD million."
        article["content_sha256"] = digest(article["text"].encode())
    save(archive, body, rebuild=True)
    history = load_acquired_history([archive])
    provider = MeasurementLLM()
    source, cache = resolver(history, provider, tmp_path / "study")
    request = measurement_request()
    request["source_urls"] = []
    result = source.acquire([request], [request["input_id"]], TIMES, CUTOFF, "reconstructed")
    np.testing.assert_array_equal(result["view"]["columns"]["acquired.inventory"], [20, 35, 25])
    assert len(provider.extractions) == 3 and not result["blocked_input_ids"]
    originals = {row["id"]: row for row in body["articles"]}
    for call in provider.extractions:
        article = call["article"]
        original = originals[article["id"]]
        assert "999999" not in article["text"] and "Future 2028" not in article["text"]
        assert article["content_sha256"] == original["content_sha256"]
        assert article["projection_sha256"] == digest(article["text"].encode())
        for span in article["projection_spans"]:
            assert original["text"][span["start"] : span["end"]] == span["text"]
    before = len(provider.extractions)
    restarted, _ = resolver(history, provider, tmp_path / "study", cache=cache)
    replay = restarted.acquire([request], [request["input_id"]], TIMES, CUTOFF, "reconstructed")
    assert len(provider.extractions) == before
    assert replay["acquisitions"] == result["acquisitions"]
    assert not restarted.acquire([request], [request["input_id"]], TIMES, CUTOFF, "reconstructed")[
        "acquisitions"
    ]
    for evidence in restarted.view(TIMES, CUTOFF, "reconstructed")["evidence"]:
        for span in evidence["projection_spans"]:
            assert originals[evidence["id"]]["text"][span["start"] : span["end"]] == span["text"]


def test_invalid_conflicting_and_unused_requests_do_not_discard_independent_variable(
    archive, tmp_path
):
    history = load_acquired_history([archive])
    provider = MeasurementLLM()
    source, _ = resolver(history, provider, tmp_path / "study")
    bad = measurement_request("revenue")
    bad["unit"] = "different unit"
    good = measurement_request()
    result = source.acquire(
        [bad, good], [bad["input_id"], good["input_id"]], TIMES, CUTOFF, "reconstructed"
    )
    assert result["blocked_input_ids"] == ["acquired.revenue"]
    assert "acquired.inventory" in result["view"]["qualified_input_ids"]
    conflicting = good | {"max_age_days": 7}
    duplicated = source.acquire(
        [good, conflicting], [good["input_id"]], TIMES, CUTOFF, "reconstructed"
    )
    assert duplicated["blocked_input_ids"] == [good["input_id"]]
    assert len(provider.extractions) == 3
    assert not source.acquire([measurement_request("ignored")], [], TIMES, CUTOFF, "reconstructed")[
        "acquisitions"
    ]
    invalid = good | {"input_id": "acquired.other", "unit": 42}
    malformed = source.acquire([invalid], ["acquired.other"], TIMES, CUTOFF, "reconstructed")
    assert malformed["blocked_input_ids"] == ["acquired.other"]


def test_company_and_url_restrictions_do_not_fetch(archive, tmp_path):
    source, _ = resolver(load_acquired_history([archive]), MeasurementLLM(), tmp_path / "study")
    request = measurement_request()
    for changed in (
        request | {"symbol": "KO"},
        request | {"source_urls": ["https://unrelated.example/release"]},
    ):
        response = source.acquire([changed], [changed["input_id"]], TIMES, CUTOFF, "reconstructed")
        assert response["gaps"] and not response["acquisitions"]
    assert not source.llm.extractions


def test_paid_failure_counts_survive_restart_and_never_become_candidate_rejection(
    archive, tmp_path
):
    provider = MeasurementLLM()
    provider.fail_extraction = True
    history = load_acquired_history([archive])
    source, cache = resolver(history, provider, tmp_path / "study", maximum=1)
    request = measurement_request()
    with pytest.raises(RuntimeError, match="acquisition/provider replay"):
        source.acquire([request], [request["input_id"]], TIMES, CUTOFF, "reconstructed")
    usage = json.loads((tmp_path / "study/acquired_source_usage.json").read_bytes())["usage"]
    assert usage["counts"]["extract"] == 1 and usage["operations"][0]["status"] == "failed"
    restarted, _ = resolver(history, provider, tmp_path / "study", cache=cache, maximum=1)
    result = restarted.acquire([request], [request["input_id"]], TIMES, CUTOFF, "reconstructed")
    assert len(provider.extractions) == 1
    assert "acquired.inventory" not in result["view"]["qualified_input_ids"]
    assert any("allowance" in row["reason"].lower() for row in result["gaps"])


def test_new_retained_extraction_cannot_be_backdated_and_empty_retained_bundle_replays(
    archive, tmp_path, monkeypatch
):
    body = json.loads(archive.read_bytes())["output"]
    for article, extraction in zip(body["articles"], body["extractions"], strict=True):
        extraction["extracted_at"] = article["received_at"]
    save(archive, body, rebuild=True)
    history = load_acquired_history([archive])
    provider = MeasurementLLM()
    source, _ = resolver(history, provider, tmp_path / "study")
    monkeypatch.setattr(discovery_sources, "_now", lambda: "2026-11-04T12:00:00Z")
    request = measurement_request()
    result = source.acquire([request], [request["input_id"]], TIMES, CUTOFF, "retained")
    assert (
        not provider.extractions and not result["view"]["columns"]["acquired.inventory"].size == 0
    )
    assert np.isnan(result["view"]["columns"]["acquired.inventory"]).all()
    saved = save(tmp_path / "empty_retained.json", result["acquisitions"][0])
    assert load_acquired_history([saved])["acquisitions"][0]["status"] == "needs_data"


def test_unknown_unquoted_measurement_remains_needs_data(archive, tmp_path):
    class MissingMeasurement:
        def structured(self, prompt, schema):
            payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
            publication = re.search(r"Published ([0-9-]+)\.", payload["article"]["text"])[1]
            return json.dumps(
                {
                    "publication_date": publication,
                    "publication_excerpt": "Published " + publication + ".",
                    "measurement": payload["request"]["measurement"],
                    "observations": [],
                    "limitations": ["No backlog measurement in excerpts"],
                }
            )

    source, _ = resolver(load_acquired_history([archive]), MissingMeasurement(), tmp_path / "study")
    request = measurement_request("backlog")
    result = source.acquire([request], [request["input_id"]], TIMES, CUTOFF, "reconstructed")
    assert result["acquisitions"][0]["status"] == "needs_data"
    assert request["input_id"] not in result["view"]["qualified_input_ids"]
    assert np.isnan(result["view"]["columns"][request["input_id"]]).all()


def test_new_extraction_cannot_quote_across_artificial_excerpt_boundaries(archive, tmp_path):
    body = json.loads(archive.read_bytes())["output"]
    for article in body["articles"]:
        article["text"] = article["text"].replace(
            ". Period", ". Unrelated navigation omitted. Period"
        )
        article["content_sha256"] = digest(article["text"].encode())
    save(archive, body, rebuild=True)

    class CrossingQuote:
        def structured(self, prompt, schema):
            payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
            text = payload["article"]["text"]
            publication = re.search(r"Published ([0-9-]+)\.", text)[1]
            period = re.search(r"Period ending ([0-9-]+):", text)[1]
            value = float(re.search(r"inventory ([0-9]+)", text)[1])
            assert "[retained excerpt boundary]" in text
            return json.dumps(
                {
                    "publication_date": publication,
                    "publication_excerpt": "Published " + publication + ".",
                    "measurement": payload["request"]["measurement"],
                    "observations": [
                        {
                            "period_start": None,
                            "period_end": period,
                            "value": value,
                            "unit": payload["request"]["unit"],
                            "excerpt": text,
                        }
                    ],
                    "limitations": [],
                }
            )

    source, _ = resolver(load_acquired_history([archive]), CrossingQuote(), tmp_path / "study")
    request = measurement_request()
    with pytest.raises(RuntimeError, match="acquisition/provider replay") as failure:
        source.acquire([request], [request["input_id"]], TIMES, CUTOFF, "reconstructed")
    assert "source spans" in str(failure.value.__cause__)
    assert (
        request["input_id"]
        not in source.view(TIMES, CUTOFF, "reconstructed")["qualified_input_ids"]
    )


def test_empty_retained_archive_is_not_reopened_by_reconstructed_lookup(archive, tmp_path):
    body = json.loads(archive.read_bytes())["output"]
    body |= align_input_bundle(body["request"], [], TIMES)
    body.update(information_mode="retained", research_cutoff=CUTOFF)
    retained = save(tmp_path / "empty_retained.json", body)
    history = load_acquired_history([retained])
    assert not history["acquisitions"][0]["observations"]
    for mode in ("reconstructed", "retained"):
        view = align_acquired_history(history, TIMES, CUTOFF, mode)
        assert np.isnan(view["columns"]["acquired.revenue"]).all()
        assert not view["evidence"] and not view["qualified_input_ids"]


@pytest.mark.parametrize("damage", [None, "content_sha256", "offset", "url"])
def test_original_and_projected_acquisitions_share_verified_source_identity(
    archive, tmp_path, damage
):
    source, _ = resolver(load_acquired_history([archive]), MeasurementLLM(), tmp_path / "study")
    request = measurement_request()
    output = source.acquire([request], [request["input_id"]], TIMES, CUTOFF, "reconstructed")[
        "acquisitions"
    ][0]
    if damage == "content_sha256":
        output["articles"][0][damage] = "f" * 64
    elif damage == "url":
        output["articles"][0]["url"] = "https://different.example/issuer"
    elif damage == "offset":
        for segment in output["articles"][0]["projection_segments"]:
            segment["start"] += 1
            segment["end"] += 1
    derived = save(tmp_path / "derived.json", output, rebuild=damage in {"content_sha256", "url"})
    # Both individually replay; combining must additionally verify their shared origin.
    assert load_acquired_history([derived])["acquisitions"]
    for paths in ([archive, derived], [derived, archive]):
        if damage:
            with pytest.raises(ValueError, match="Conflicting acquired source|exceeds original"):
                load_acquired_history(paths)
        else:
            view = align_acquired_history(
                load_acquired_history(paths), TIMES, CUTOFF, "reconstructed"
            )
            np.testing.assert_array_equal(view["columns"]["acquired.revenue"], [100, 120, 110])
            np.testing.assert_array_equal(view["columns"]["acquired.inventory"], [20, 35, 25])
            assert len(view["evidence"]) == 3


@pytest.mark.parametrize("derived_first", [False, True])
def test_shared_source_projection_unites_wider_and_disjoint_original_spans(
    archive, tmp_path, derived_first
):
    source, _ = resolver(load_acquired_history([archive]), MeasurementLLM(), tmp_path / "study")
    request = measurement_request()
    output = source.acquire([request], [request["input_id"]], TIMES, CUTOFF, "reconstructed")[
        "acquisitions"
    ][0]
    originals = {a["id"]: a for a in json.loads(archive.read_bytes())["output"]["articles"]}
    for article, extraction in zip(output["articles"], output["extractions"], strict=True):
        observation = extraction["result"]["observations"][0]
        observation["period_excerpt"] = "Period ending " + observation["period_end"]
        observation["excerpt"] = re.search(r"inventory [0-9]+ USD million\.", article["text"])[0]
        original = originals[article["id"]]["text"]
        quotes = [
            extraction["result"]["publication_excerpt"],
            observation["period_excerpt"],
            observation["excerpt"],
        ]
        segments = [
            {"start": original.index(q), "end": original.index(q) + len(q), "text": q}
            for q in quotes
        ]
        article["projection_segments"] = segments
        article["projection_spans"] = deepcopy(segments)
        article["text"] = "\n\n[retained excerpt boundary]\n\n".join(quotes)
        article["projection_sha256"] = digest(article["text"].encode())
        assert "revenue" not in article["text"]
    derived = save(tmp_path / "narrow_inventory.json", output, rebuild=True)
    paths = [derived, archive] if derived_first else [archive, derived]
    view = align_acquired_history(load_acquired_history(paths), TIMES, CUTOFF, "reconstructed")
    np.testing.assert_array_equal(view["columns"]["acquired.inventory"], [20, 35, 25])
    np.testing.assert_array_equal(view["columns"]["acquired.revenue"], [100, 120, 110])
    for evidence in view["evidence"]:
        assert "revenue" in evidence["excerpt"] and "inventory" in evidence["excerpt"]
        for span in evidence["projection_spans"]:
            assert originals[evidence["id"]]["text"][span["start"] : span["end"]] == span["text"]
