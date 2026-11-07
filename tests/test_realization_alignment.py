"""Offline actual/guidance joins exercise source admission and production consumers."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research.discovery_engine import DiscoveryExperimentRunner
from quant_rsi.research.discovery_factors import DiscoveryFactor, compute_factor
from quant_rsi.research.discovery_sources import (
    DataRequest,
    NumericalExtraction,
    align_input_bundle,
    validate_extraction,
)
from quant_rsi.research.realization_alignment import prepare_realization_inputs
from quant_rsi.research.rolling_acquired import _history, _replay, align_acquired_history
from quant_rsi.research.rolling_candidates import screen_candidates
from quant_rsi.research.rolling_research import verify_news_dependency

TIMES = ["2026-01-20T21:00:00Z", "2026-04-20T20:00:00Z", "2026-04-21T20:00:00Z"]
RECEIPT = "2026-11-05T10:00:00Z"
PERIODS = {
    "2025-Q4": ("2025-10-01", "2025-12-31"),
    "2026-Q1": ("2026-01-01", "2026-03-31"),
    "2026-Q2": ("2026-04-01", "2026-06-30"),
}


def source(identity, day, quarter, values):
    start, end = PERIODS[quarter]
    period = f"{quarter}: {start} to {end}"
    quotes = {role: f"{role} gross margin {value} percent." for role, value in values.items()}
    text = f"Published {day}. Results for {period}. " + " ".join(quotes.values())
    article = {
        "id": identity,
        "url": f"https://example.com/{identity}",
        "status": "available",
        "text": text,
        "published_at": day + "T12:00:00Z",
        "received_at": RECEIPT,
        "version_available_at": RECEIPT,
        "content_sha256": digest(text.encode()),
    }
    return article, {
        role: {
            "publication_date": day,
            "publication_excerpt": f"Published {day}.",
            "measurement": f"TSMC {role} gross margin",
            "observations": [
                {
                    "period_start": start,
                    "period_end": end,
                    "value": value,
                    "unit": "percent",
                    "excerpt": quotes[role],
                    "period_excerpt": period,
                    **({"target_quarter": quarter} if role != "actual" else {}),
                }
            ],
            "limitations": ["Synthetic source; no predictive efficacy evidence."],
        }
        for role, value in values.items()
    }


def bundles():
    actuals = [
        source("actual-q4", "2026-01-16", "2025-Q4", {"actual": 61.0}),
        source("actual-q1", "2026-04-16", "2026-Q1", {"actual": 66.2}),
    ]
    guides = [
        source("guide-q4", "2025-10-15", "2025-Q4", {"lower": 60.0, "upper": 62.0}),
        source("guide-q1", "2026-01-15", "2026-Q1", {"lower": 63.0, "upper": 65.0}),
        source("guide-q2", "2026-04-16", "2026-Q2", {"lower": 65.5, "upper": 67.5}),
    ]
    result = []
    for role in ("actual", "lower", "upper"):
        records = actuals if role == "actual" else guides
        request = {
            "input_id": "acquired." + role,
            "symbol": "TSM",
            "measurement": f"TSMC {role} gross margin",
            "unit": "percent",
            "query": "TSMC original quarterly margin release",
            "source_urls": [article["url"] for article, _ in records],
            "max_age_days": 365,
        }
        if role == "actual":
            request["realized_measure"] = {
                "product": "TSMC consolidated gross margin",
                "basis": "calendar quarter, percent",
            }
            request["original_actual_releases"] = [
                {
                    "period_start": extraction["actual"]["observations"][0]["period_start"],
                    "period_end": extraction["actual"]["observations"][0]["period_end"],
                    "source_url": article["url"],
                    "publication_date": extraction["actual"]["publication_date"],
                    "release_excerpt": extraction["actual"]["observations"][0]["period_excerpt"],
                }
                for article, extraction in records
            ]
        else:
            request["forecast"] = {
                "product": "TSMC consolidated gross margin",
                "statistic": role,
                "basis": "calendar quarter, percent",
                "pairing_group": "margin",
            }
        observed = []
        for article, extraction in records:
            rows, reasons = validate_extraction(
                DataRequest.model_validate_json(canonical(request)),
                article,
                NumericalExtraction.model_validate_json(canonical(extraction[role])),
                extracted_at=RECEIPT,
                information_mode="retrospective",
                research_cutoff=RECEIPT,
            )
            assert not reasons and len(rows) == 1
            observed.extend(rows)
        result.append(
            align_input_bundle(request, observed, TIMES)
            | {
                "articles": [article for article, _ in records],
                "extractions": [
                    {
                        "article_id": article["id"],
                        "result": extraction[role],
                        "extracted_at": RECEIPT,
                    }
                    for article, extraction in records
                ],
            }
        )
    return result


def proposal():
    return DiscoveryFactor.model_validate_json(
        canonical(
            {
                "name": "guidance_position",
                "description": "Actual relative to same-quarter management guidance range",
                "inputs": [
                    {"name": role, "factor_id": "acquired." + role}
                    for role in ("actual", "lower", "upper")
                ],
                "expression": {
                    "schema_version": 2,
                    "nodes": [
                        {"name": role, "op": "leaf", "leaf": role}
                        for role in ("actual", "lower", "upper")
                    ]
                    + [
                        {"name": "gap", "op": "subtract", "inputs": ["actual", "lower"]},
                        {"name": "width", "op": "subtract", "inputs": ["upper", "lower"]},
                        {"name": "position", "op": "divide", "inputs": ["gap", "width"]},
                    ],
                },
                "realization_alignment": {
                    "actual": "actual",
                    "lower": "lower",
                    "upper": "upper",
                    "policy": "latest_pre_actual",
                },
                "evidence_ids": ["actual-q4", "actual-q1", "guide-q4", "guide-q1"],
            }
        )
    )


def prepared_values(history, candidate=None):
    candidate = candidate or proposal()
    view = prepare_realization_inputs(candidate, history, TIMES)
    definitions = [item for bundle in history for item in bundle["definitions"]]
    values = compute_factor(
        candidate.definition(definitions),
        view["columns"],
        definitions,
        input_evidence=view["input_evidence"],
    )
    return values, view, definitions


def test_same_quarter_recovers_prior_guidance_without_changing_default_alignment():
    history = bundles()
    assert history[1]["columns"]["acquired.lower"][-1] == 65.5
    values, view, _ = prepared_values(history)
    np.testing.assert_allclose(values, [0.5, 1.6, 1.6])
    assert view["input_evidence_ids"]["acquired.lower"] == ["guide-q1", "guide-q4"]
    assert history[1]["columns"]["acquired.lower"][-1] == 65.5
    for bundle in history:
        assert _replay(bundle) == bundle


@pytest.mark.parametrize(
    ("publication", "expected"),
    [
        ("2026-03-02T12:00:00+00:00", 2.1),
        ("2026-04-17T12:00:00+00:00", 1.6),
        ("2026-04-18T12:00:00+00:00", 1.6),
    ],
)
def test_latest_prior_update_excludes_same_day_and_post_actual(publication, expected):
    history = bundles()
    for bundle, value in zip(history[1:], [62.0, 64.0], strict=True):
        update = deepcopy(bundle["observations"][1])
        update.update(
            value=value,
            published_at=publication,
            effective_available_at=publication,
            source_id="guide-update",
            observation_id=digest(canonical([publication, value])),
        )
        bundle["observations"].append(update)
    values, _, _ = prepared_values(history)
    np.testing.assert_allclose(values, [0.5, expected, expected])


@pytest.mark.parametrize("problem", ["missing", "conflict", "zero_width", "negative_width"])
def test_incomplete_or_conflicting_ranges_are_missing(problem):
    history = bundles()
    rows = history[2]["observations"]
    q1 = next(row for row in rows if row["target_quarter"] == "2026-Q1")
    if problem == "missing":
        rows.remove(q1)
    elif problem == "conflict":
        rows.append(q1 | {"value": 64.0, "observation_id": "conflicting"})
    else:
        q1["value"] = 63.0 if problem == "zero_width" else 62.0
    values, _, _ = prepared_values(history)
    np.testing.assert_allclose(values, [0.5, np.nan, np.nan], equal_nan=True)


def test_newest_anchor_or_update_missing_never_falls_back_to_older_comparison():
    history = bundles()
    history[0]["observations"] = history[0]["observations"][:1]
    values, _, _ = prepared_values(history)
    np.testing.assert_allclose(values, [0.5, np.nan, np.nan], equal_nan=True)

    history = bundles()
    previous = history[1]["observations"][1]
    history[1]["observations"].append(
        previous
        | {
            "value": 62.0,
            "published_at": "2026-03-02T12:00:00+00:00",
            "effective_available_at": "2026-03-02T12:00:00+00:00",
            "source_id": "unmatched-lower-update",
            "observation_id": "unmatched-lower-update",
        }
    )
    values, _, _ = prepared_values(history)
    np.testing.assert_allclose(values, [0.5, np.nan, np.nan], equal_nan=True)


def test_future_receipt_and_measure_basis_mismatch_cannot_enter_comparison():
    history = bundles()
    history[1]["observations"][1]["effective_available_at"] = RECEIPT
    values, _, _ = prepared_values(history)
    np.testing.assert_allclose(values, [0.5, np.nan, np.nan], equal_nan=True)

    history = bundles()
    history[0]["request"]["realized_measure"]["basis"] = "incompatible non-GAAP adjusted margin"
    with pytest.raises(ValueError, match="compatible range bounds"):
        prepared_values(history)


def test_original_release_anchor_rejects_later_comparative_source_and_replay_tampering():
    actual = bundles()[0]
    request = DataRequest.model_validate_json(canonical(actual["request"]))
    article = deepcopy(actual["articles"][1])
    result = deepcopy(actual["extractions"][1]["result"])
    article["url"] = "https://example.com/later-comparative-release"
    rows, reasons = validate_extraction(
        request,
        article,
        NumericalExtraction.model_validate_json(canonical(result)),
        extracted_at=RECEIPT,
        information_mode="retrospective",
        research_cutoff=RECEIPT,
    )
    assert rows == [] and reasons == ["unverified_original_actual_release"]
    actual["observations"][0]["original_actual_release"]["publication_date"] = "2026-02-01"
    with pytest.raises(ValueError, match="replayed observations"):
        _replay(actual)


@pytest.mark.parametrize("mismatch", ["publication", "period_start", "release_excerpt"])
def test_actual_release_anchor_requires_exact_date_period_and_literal_context(mismatch):
    actual = bundles()[0]
    request = deepcopy(actual["request"])
    anchor = request["original_actual_releases"][1]
    if mismatch == "publication":
        anchor["publication_date"] = "2026-04-17"
    elif mismatch == "period_start":
        # Keep a valid declared quarter but attempt a different extraction start.
        actual["extractions"][1]["result"]["observations"][0]["period_start"] = None
    else:
        anchor["release_excerpt"] = "Results for 2026-Q1"
    rows, reasons = validate_extraction(
        DataRequest.model_validate_json(canonical(request)),
        actual["articles"][1],
        NumericalExtraction.model_validate_json(canonical(actual["extractions"][1]["result"])),
        extracted_at=RECEIPT,
        information_mode="retrospective",
        research_cutoff=RECEIPT,
    )
    assert rows == [] and reasons == ["unverified_original_actual_release"]


@pytest.mark.parametrize(
    ("header", "prefix", "accepted"),
    [
        ("first quarter ended March 31, 2026", "", True),
        ("month ended March 31, 2026 (2026-01-01 to 2026-03-31)", "", False),
        ("year to date 2026-01-01 to 2026-03-31", "", False),
        ("first quarter ended March 31, 2026", "fiscal ", False),
        ("2026-Q1: 2026-01-01 to 2026-03-31", "FY2026 ", False),
    ],
)
def test_original_release_quarter_header_supports_literal_ended_date_only_when_opted_in(
    header, prefix, accepted
):
    actual = bundles()[0]
    request = deepcopy(actual["request"])
    article = deepcopy(actual["articles"][1])
    extracted = deepcopy(actual["extractions"][1]["result"])
    previous = extracted["observations"][0]["period_excerpt"]
    article["text"] = article["text"].replace(previous, prefix + header)
    article["content_sha256"] = digest(article["text"].encode())
    extracted["observations"][0]["period_excerpt"] = header
    request["original_actual_releases"][1]["release_excerpt"] = header

    def check(value):
        return validate_extraction(
            DataRequest.model_validate_json(canonical(value)),
            article,
            NumericalExtraction.model_validate_json(canonical(extracted)),
            extracted_at=RECEIPT,
            information_mode="retrospective",
            research_cutoff=RECEIPT,
        )

    rows, reasons = check(request)
    if accepted:
        assert len(rows) == 1 and reasons == []
        assert rows[0]["period_start"] == "2026-01-01"
        request.pop("original_actual_releases")
        legacy_rows, legacy_reasons = check(request)
        assert legacy_rows == [] and legacy_reasons == ["period_start_not_in_excerpt"]
    else:
        assert rows == [] and reasons == ["unverified_original_actual_release"]


def test_quarter_token_does_not_turn_a_cumulative_half_year_into_quarter_actual():
    actual = bundles()[0]
    request, article, extracted = (
        actual["request"],
        actual["articles"][1],
        actual["extractions"][1]["result"],
    )
    header = "2026-Q2 year-to-date 2026-01-01 to 2026-06-30"
    request["original_actual_releases"][1].update(
        period_start="2026-04-01",
        period_end="2026-06-30",
        publication_date="2026-07-16",
        release_excerpt=header,
    )
    extracted.update(publication_date="2026-07-16", publication_excerpt="Published 2026-07-16.")
    extracted["observations"][0].update(
        period_start="2026-04-01",
        period_end="2026-06-30",
        period_excerpt=header,
    )
    article["text"] = (
        "Published 2026-07-16. " + header + ". " + extracted["observations"][0]["excerpt"]
    )
    article["content_sha256"] = digest(article["text"].encode())
    rows, reasons = validate_extraction(
        DataRequest.model_validate_json(canonical(request)),
        article,
        NumericalExtraction.model_validate_json(canonical(extracted)),
        extracted_at=RECEIPT,
        information_mode="retrospective",
        research_cutoff=RECEIPT,
    )
    assert rows == [] and reasons == ["unverified_original_actual_release"]


@pytest.mark.parametrize(
    ("period", "value_quote", "accepted"),
    [
        ("Six months ended June 30, 2026", "Year-to-date gross margin was 58.0 percent.", False),
        ("Six months ended June 30, 2026", "Gross margin was 58.0 percent.", False),
        ("2026-Q2: 2026-04-01 to 2026-06-30", "Year-to-date gross margin was 58.0 percent.", False),
        ("Month ended June 30, 2026", "Gross margin was 58.0 percent.", False),
        ("2026-04-01 to 2026-06-30", "Gross margin was 58.0 percent.", True),
    ],
)
def test_valid_release_anchor_does_not_admit_cumulative_or_monthly_observation(
    period, value_quote, accepted
):
    request = bundles()[0]["request"]
    article, results = source("actual-q2", "2026-07-16", "2026-Q2", {"actual": 58.0})
    header = "Results for the second quarter ended June 30, 2026."
    request["original_actual_releases"] = [
        {
            "period_start": "2026-04-01",
            "period_end": "2026-06-30",
            "source_url": article["url"],
            "publication_date": "2026-07-16",
            "release_excerpt": header,
        }
    ]
    article["text"] = f"Published 2026-07-16. {header} {period}. {value_quote}"
    article["content_sha256"] = digest(article["text"].encode())
    results["actual"]["observations"][0].update(period_excerpt=period, excerpt=value_quote)
    rows, reasons = validate_extraction(
        DataRequest.model_validate_json(canonical(request)),
        article,
        NumericalExtraction.model_validate_json(canonical(results["actual"])),
        extracted_at=RECEIPT,
        information_mode="retrospective",
        research_cutoff=RECEIPT,
    )
    assert bool(rows) == accepted
    assert reasons == ([] if accepted else ["unverified_original_actual_observation_period"])


@pytest.mark.parametrize("role_index", [1, 2])
def test_original_guidance_spelled_out_calendar_quarter_is_admitted(role_index):
    guide = bundles()[role_index]
    article, extracted = guide["articles"][1], guide["extractions"][1]["result"]
    previous = extracted["observations"][0]["period_excerpt"]
    article["text"] = article["text"].replace(previous, "first quarter 2026")
    article["content_sha256"] = digest(article["text"].encode())
    extracted["observations"][0]["period_excerpt"] = "first quarter 2026"
    rows, reasons = validate_extraction(
        DataRequest.model_validate_json(canonical(guide["request"])),
        article,
        NumericalExtraction.model_validate_json(canonical(extracted)),
        extracted_at=RECEIPT,
        information_mode="retrospective",
        research_cutoff=RECEIPT,
    )
    assert len(rows) == 1 and reasons == []
    assert rows[0]["target_quarter"] == "2026-Q1"
    assert rows[0]["period_start"] == "2026-01-01"
    assert rows[0]["period_end"] == "2026-03-31"


def test_alignment_is_opt_in_identity_bound_and_requires_prepared_evidence():
    candidate = proposal()
    record = candidate.model_dump(mode="json")
    record.pop("realization_alignment")
    legacy = DiscoveryFactor.model_validate_json(canonical(record))
    assert "realization_alignment" not in legacy.model_dump(mode="json")
    assert legacy.recipe_ref == digest(
        canonical(legacy.expression.normalized({i.name: i.factor_id for i in legacy.inputs}))
    )
    assert legacy.recipe_ref != candidate.recipe_ref
    history = bundles()
    columns = {key: values for bundle in history for key, values in bundle["columns"].items()}
    evidence = {key: rows for bundle in history for key, rows in bundle["input_evidence"].items()}
    definitions = [item for bundle in history for item in bundle["definitions"]]
    with pytest.raises(ValueError, match="prepared source evidence"):
        compute_factor(
            candidate.definition(definitions), columns, definitions, input_evidence=evidence
        )
    with pytest.raises(ValueError, match="raw histories"):
        prepare_realization_inputs(candidate, history[:1], TIMES)


@pytest.mark.parametrize("generic_conflicts", [False, True])
def test_study_and_rolling_use_same_matched_sources_and_preserve_dependency_probes(
    generic_conflicts,
):
    history, candidate = bundles(), proposal()
    if generic_conflicts:
        # Unrelated Q2 publications conflict at every generic row, while the
        # pre-actual Q4/Q1 ranges remain complete and unambiguous.
        for record in history[1:]:
            role = record["request"]["forecast"]["statistic"]
            for day in ("2026-01-18", "2026-04-16"):
                for offset in (0, 1):
                    article, extracted = source(
                        f"conflict-{day}-{offset}", day, "2026-Q2", {role: 70.0 + offset}
                    )
                    rows, reasons = validate_extraction(
                        DataRequest.model_validate_json(canonical(record["request"])),
                        article,
                        NumericalExtraction.model_validate_json(canonical(extracted[role])),
                        extracted_at=RECEIPT,
                        information_mode="retrospective",
                        research_cutoff=RECEIPT,
                    )
                    assert not reasons
                    record["observations"].extend(rows)
                    record["articles"].append(article)
                    record["extractions"].append(
                        {
                            "article_id": article["id"],
                            "result": extracted[role],
                            "extracted_at": RECEIPT,
                        }
                    )
            record.update(align_input_bundle(record["request"], record["observations"], TIMES))
            assert not record["definitions"][0]["runnable"]
    runner = DiscoveryExperimentRunner.__new__(DiscoveryExperimentRunner)
    runner.model = SimpleNamespace(
        train=np.array([0, 1]),
        origins=np.array([2]),
        data=SimpleNamespace(dates=[instant[:10] for instant in TIMES]),
    )
    runner.cutoffs, runner.definitions, runner.columns = TIMES, [], {}
    _, study_values, _ = runner.prepare(candidate.model_dump(mode="json"), history)
    np.testing.assert_allclose(study_values, [0.5, 1.6, 1.6])
    view = align_acquired_history(_history(history, []), TIMES, TIMES[-1], "reconstructed")
    args = {
        "columns": view["columns"] | {"base": np.array([0.2, 0.3, 0.4])},
        "definitions": view["definitions"] + [{"factor_id": "base", "unit": "dimensionless"}],
        "baseline": ["base"],
        "news": {"items": []},
        "times": TIMES,
        "train": np.array([0, 1]),
        "news_required": True,
        "require_evidence": True,
        "acquired_inputs": view,
    }
    result = screen_candidates([candidate.model_dump(mode="json")], [], **args)
    assert result["rejections"] == []
    np.testing.assert_allclose(result["matrix"][:, -1], study_values)
    blocked = view | {"qualified_input_ids": [], "raw_qualified_input_ids": []}
    assert not screen_candidates(
        [candidate.model_dump(mode="json")], [], **(args | {"acquired_inputs": blocked})
    )["candidates"]
    uncited = candidate.model_dump(mode="json")
    uncited["evidence_ids"] = ["actual-q1", "actual-q4", "guide-q2"]
    rejected = screen_candidates([uncited], [], **args)
    assert not rejected["candidates"]
    assert "supporting acquired." in rejected["rejections"][0]["reason"]

    prepared = prepare_realization_inputs(candidate, history, TIMES)
    definitions = [
        item | {"runnable": True} for bundle in history for item in bundle["definitions"]
    ]
    verify_news_dependency(
        [candidate.model_dump(mode="json")],
        prepared["columns"],
        definitions,
        np.array([0, 1, 2]),
        acquired_inputs=tuple(prepared["columns"]),
        input_evidence=prepared["input_evidence"],
    )
    cancelled = candidate.model_dump(mode="json")
    cancelled["expression"]["nodes"].append(
        {"name": "cancelled", "op": "subtract", "inputs": ["position", "position"]}
    )
    with pytest.raises(ValueError, match="does not numerically depend"):
        verify_news_dependency(
            [cancelled],
            prepared["columns"],
            definitions,
            np.array([0, 1, 2]),
            acquired_inputs=tuple(prepared["columns"]),
            input_evidence=prepared["input_evidence"],
        )
