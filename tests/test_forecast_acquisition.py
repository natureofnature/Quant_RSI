"""Published expectations remain distinct from actuals; synthetic fixtures show wiring only."""

import calendar
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from test_discovery_sources import article, extraction, request

from quant_rsi.core.artifacts import canonical
from quant_rsi.research.discovery_engine import DiscoveryExperimentRunner
from quant_rsi.research.discovery_factors import DiscoveryFactor, compute_factor
from quant_rsi.research.discovery_sources import (
    DataRequest,
    DiscoverySources,
    NumericalExtraction,
    _date_in_excerpt,
    align_input_bundle,
    validate_extraction,
)
from quant_rsi.research.rolling_acquired import _history, _replay, align_acquired_history
from quant_rsi.research.rolling_candidates import screen_candidates

TIMES = ["2026-01-06T21:00:00Z", "2026-02-03T21:00:00Z", "2026-04-02T20:00:00Z"]


def forecast_request(product="dram"):
    return request() | {
        "input_id": "acquired." + product,
        "symbol": "SOXX",
        "measurement": product + " contract-price growth forecast lower endpoint",
        "unit": "% QoQ",
        "forecast": {
            "product": product,
            "statistic": "lower",
            "basis": "QoQ",
            "pairing_group": "memory_prices",
        },
        "source_urls": [f"https://example.com/release-{n}" for n in range(1, 4)],
        "max_age_days": 45,
    }


def forecast_extraction(number=1, product="dram"):
    day, quote, quarter, start, end = [
        ("2026-01-05", "5 January 2026", "2026-Q1", "2026-01-01", "2026-03-31"),
        ("2026-02-02", "2 February 2026", "2026-Q1", "2026-01-01", "2026-03-31"),
        ("2026-03-31", "31 March 2026", "2026-Q2", "2026-04-01", "2026-06-30"),
    ][number - 1]
    value = ([55.0, 90.0, 58.0] if product == "dram" else [33.0, 55.0, 70.0])[number - 1]
    period = "1Q26" if number < 3 else "Q2 2026"
    return {
        "publication_date": day,
        "publication_excerpt": quote,
        "measurement": forecast_request(product)["measurement"],
        "observations": [
            {
                "period_start": start,
                "period_end": end,
                "target_quarter": quarter,
                "value": value,
                "unit": "% QoQ",
                "period_excerpt": period,
                "excerpt": f"{product} forecast {value} % QoQ for {period}",
            }
        ],
        "limitations": ["Synthetic article; not a predictive result."],
    }


def forecast_article(number=1):
    left, right = forecast_extraction(number), forecast_extraction(number, "nand")
    return article(number) | {
        "text": " ".join(
            [
                left["publication_excerpt"],
                left["observations"][0]["excerpt"],
                right["observations"][0]["excerpt"],
            ]
        ),
    }


def checked(req, source, result, mode="retrospective", cutoff=None):
    return validate_extraction(
        DataRequest.model_validate_json(canonical(req)),
        source,
        NumericalExtraction.model_validate_json(canonical(result)),
        extracted_at="2026-11-05T10:00:00Z",
        information_mode=mode,
        research_cutoff=cutoff,
    )


def bundle(product="dram", times=TIMES):
    class Provider:
        def structured(self, prompt, schema):
            import json

            payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
            number = int(payload["article"]["url"].rsplit("-", 1)[1])
            return canonical(forecast_extraction(number, product)).decode()

    sources = DiscoverySources(Provider(), reader=lambda url: forecast_article(int(url[-1])))
    return sources.acquire(forecast_request(product), prediction_times=times)


def proposal(evidence=()):
    return {
        "name": "memory_forecast_spread",
        "description": "Same-release DRAM minus NAND expectation",
        "inputs": [{"name": p, "factor_id": "acquired." + p} for p in ("dram", "nand")],
        "expression": {
            "schema_version": 2,
            "nodes": [
                {"name": "d", "op": "leaf", "leaf": "dram"},
                {"name": "n", "op": "leaf", "leaf": "nand"},
                {"name": "spread", "op": "subtract", "inputs": ["d", "n"]},
            ],
        },
        "evidence_ids": list(evidence),
    }


def test_legacy_realized_identities_from_main_11471dd_are_unchanged():
    req = DataRequest.model_validate_json(canonical(request()))
    assert req.model_dump(mode="json") == request()
    rows, reasons = checked(request(), article(), extraction())
    assert reasons == []
    assert (
        rows[0]["observation_id"]
        == "ab426a63d20e6aab57eb5b45641f4bfeaccd1ece6e65c7bb0f482184e4a49772"
    )
    result = align_input_bundle(req, rows, ["2025-01-03T21:00:00Z", "2025-01-06T21:00:00Z"])
    assert result["identity"] == "f18e384be3442648e4ed6d5ed37aba0f533c6c0825cf620d7cd8e36cc18b61e8"


@pytest.mark.parametrize("quote", ["5 January 2026", "05 January 2026", "5 Jan 2026"])
def test_day_first_literal_dates(quote):
    assert _date_in_excerpt("2026-01-05", quote)
    assert not _date_in_excerpt("2026-01-06", quote)
    assert not _date_in_excerpt("2026-01-05", "05/01/2026")


@pytest.mark.parametrize(
    "quarter,ordinal", [(1, "first"), (2, "second"), (3, "third"), (4, "fourth")]
)
@pytest.mark.parametrize("connector", ["", "of "])
@pytest.mark.parametrize("year", [2025, 2026])
def test_spelled_out_guidance_quarter_preserves_literal_source_quote(
    quarter, ordinal, connector, year
):
    source, result = forecast_article(), forecast_extraction()
    quote = f"{ordinal} quarter {connector}{year}"
    source["text"] = source["text"].replace("1Q26", quote).replace("2026", str(year))
    result["publication_date"] = f"{year}-01-05"
    result["publication_excerpt"] = result["publication_excerpt"].replace("2026", str(year))
    observation = result["observations"][0]
    observation.update(
        period_start=f"{year}-{3 * quarter - 2:02}-01",
        period_end=f"{year}-{3 * quarter:02}-{calendar.monthrange(year, 3 * quarter)[1]}",
        target_quarter=f"{year}-Q{quarter}",
        period_excerpt=quote,
        excerpt=observation["excerpt"].replace("1Q26", quote),
    )
    original_text = source["text"]
    rows, reasons = checked(forecast_request(), source, result)
    assert reasons == [] and len(rows) == 1
    assert rows[0]["period_excerpt"] == quote
    assert rows[0]["excerpt"] == observation["excerpt"]
    assert rows[0]["target_quarter"] == f"{year}-Q{quarter}"
    assert source["text"] == original_text


@pytest.mark.parametrize(
    "quote,prefix",
    [
        ("first quarter of 2025", ""),
        ("second quarter of 2026", ""),
        ("first quarter", ""),
        ("first month of 2026", ""),
        ("January 2026", ""),
        ("first quarter of 2026", "fiscal "),
        ("first quarter of 2026", "FY2026 "),
    ],
)
def test_optional_of_does_not_relax_forecast_year_quarter_or_fiscal_guards(quote, prefix):
    source, result = forecast_article(), forecast_extraction()
    source["text"] = source["text"].replace("1Q26", prefix + quote)
    observation = result["observations"][0]
    observation["period_excerpt"] = quote
    observation["excerpt"] = observation["excerpt"].replace("1Q26", prefix + quote)
    rows, reasons = checked(forecast_request(), source, result)
    assert rows == [] and reasons == ["unverified_forecast_target_quarter"]


@pytest.mark.parametrize("change", [{"period_start": "2026-02-01"}, {"period_end": "2026-03-30"}])
def test_spelled_out_guidance_still_requires_exact_calendar_bounds(change):
    source, result = forecast_article(), forecast_extraction()
    quote = "first quarter of 2026"
    source["text"] = source["text"].replace("1Q26", quote)
    observation = result["observations"][0]
    observation.update(change)
    observation["period_excerpt"] = quote
    observation["excerpt"] = observation["excerpt"].replace("1Q26", quote)
    rows, reasons = checked(forecast_request(), source, result)
    assert rows == [] and reasons == ["unverified_forecast_target_quarter"]


def test_forecast_future_target_keeps_actual_publication_availability():
    rows, reasons = checked(forecast_request(), forecast_article(), forecast_extraction())
    assert reasons == []
    assert rows[0]["target_quarter"] == "2026-Q1"
    assert rows[0]["period_end"] == "2026-03-31"
    assert rows[0]["effective_available_at"] == "2026-01-06T12:00:00+00:00"
    times = ["2026-01-05T21:00:00Z", *TIMES[:1], "2026-03-01T21:00:00Z"]
    result = align_input_bundle(forecast_request(), rows, times)
    assert result["columns"]["acquired.dram"] == [None, 55, None]
    retained, reasons = checked(
        forecast_request(),
        forecast_article(),
        forecast_extraction(),
        "retained",
        "2026-07-14T20:00:00Z",
    )
    assert retained == [] and reasons == ["extraction_after_research_cutoff"]


@pytest.mark.parametrize(
    "change",
    [
        {"target_quarter": None},
        {"target_quarter": "2026-Q2"},
        {"period_end": "2026-01-05"},
        {"period_excerpt": "2Q26"},
        {"period_excerpt": "fiscal Q1 2026"},
    ],
)
def test_forecast_requires_literal_correct_quarter_and_bounds(change):
    result = forecast_extraction()
    result["observations"][0].update(change)
    rows, reasons = checked(forecast_request(), forecast_article(), result)
    assert rows == [] and reasons == ["unverified_forecast_target_quarter"]


def test_realized_does_not_accept_forecast_or_future_period():
    req = forecast_request()
    req.pop("forecast")
    for quarter in (None, "2026-Q1"):
        result = forecast_extraction()
        result["observations"][0]["target_quarter"] = quarter
        rows, reasons = checked(req, forecast_article(), result)
        assert rows == [] and reasons == ["invalid_period"]


def test_acquire_replay_and_alignment_preserve_vintages_and_missing_rows():
    times = [*TIMES, "2026-07-14T20:00:00Z"]
    result = bundle(times=times)
    assert result["columns"]["acquired.dram"] == [55, 90, 58, None]
    assert (
        result["input_evidence"]["acquired.dram"][0]["forecast_vintage"]["target_quarter"]
        == "2026-Q1"
    )
    assert (
        result["input_evidence"]["acquired.dram"][2]["forecast_vintage"]["target_quarter"]
        == "2026-Q2"
    )
    assert _replay(result) == result
    aligned = align_acquired_history(_history([result], []), times, times[-1], "reconstructed")
    np.testing.assert_allclose(
        aligned["columns"]["acquired.dram"], [55, 90, 58, np.nan], equal_nan=True
    )
    tampered = deepcopy(result)
    tampered["input_evidence"]["acquired.dram"][0]["forecast_vintage"]["target_quarter"] = "2026-Q2"
    with pytest.raises(ValueError, match="replayed input_evidence"):
        _replay(tampered)


def test_production_candidate_paths_require_matching_forecast_vintages():
    bundles = [bundle(), bundle("nand")]
    view = align_acquired_history(_history(bundles, []), TIMES, TIMES[-1], "reconstructed")
    candidate = proposal([item["id"] for item in view["evidence"]])
    definitions = view["definitions"] + [
        {"factor_id": "base", "unit": "dimensionless", "runnable": True}
    ]
    columns = view["columns"] | {"base": np.array([1.0, 2.0, 3.0])}
    factor = DiscoveryFactor.model_validate_json(canonical(candidate))
    values = compute_factor(
        factor.definition(definitions), columns, definitions, input_evidence=view["input_evidence"]
    )
    np.testing.assert_allclose(values, [22, 35, -12])
    with pytest.raises(ValueError, match="vintage evidence"):
        compute_factor(factor.definition(definitions), columns, definitions)
    for key, other in (
        ("source_id", "different"),
        ("published_at", TIMES[0]),
        ("target_quarter", "2026-Q2"),
    ):
        evidence = deepcopy(view["input_evidence"])
        evidence["acquired.nand"][1]["forecast_vintage"][key] = other
        values = compute_factor(
            factor.definition(definitions), columns, definitions, input_evidence=evidence
        )
        np.testing.assert_allclose(values, [22, np.nan, -12], equal_nan=True)
    args = dict(
        columns=columns,
        definitions=definitions,
        baseline=["base"],
        news={"items": []},
        times=TIMES,
        train=np.array([0, 1]),
        news_required=True,
        require_evidence=True,
        acquired_inputs=view,
    )
    result = screen_candidates([candidate], [], **args)
    assert len(result["candidates"]) == 1 and result["matrix"].shape == (3, 2)
    bad_view = deepcopy(view)
    bad_view["input_evidence"]["acquired.nand"][1]["forecast_vintage"]["target_quarter"] = "2026-Q2"
    result = screen_candidates([candidate], [], **(args | {"acquired_inputs": bad_view}))
    assert result["candidates"] == []
    assert "unavailable on the fixed" in result["rejections"][0]["reason"]

    # Exercise the actual study prepare/validate boundary on the same fixed population.
    runner = DiscoveryExperimentRunner.__new__(DiscoveryExperimentRunner)
    runner.model = SimpleNamespace(
        train=np.array([0, 1]),
        origins=np.array([2]),
        data=SimpleNamespace(dates=[t[:10] for t in TIMES]),
    )
    runner.cutoffs = TIMES
    runner.definitions = []
    runner.columns = {}
    runner.evidence_ids = set(candidate["evidence_ids"])
    assert runner.validate(candidate, bundles)["status"] == "ready"
    # Remove NAND's February vintage; independent carry keeps January but pairing must mask it.
    late = deepcopy(bundles[1])
    late["observations"] = [row for row in late["observations"] if row["value"] != 55]
    late |= align_input_bundle(late["request"], late["observations"], TIMES)
    result = runner.validate(candidate, [bundles[0], late])
    assert result["status"] == "needs_data"
    assert result["missing_sessions"] == ["2026-02-03"]


def test_pairing_is_explicit_and_temporal_transforms_do_not_cross_vintages():
    view = align_acquired_history(
        _history([bundle(), bundle("nand")], []), TIMES, TIMES[-1], "reconstructed"
    )
    candidate = proposal()
    factor = DiscoveryFactor.model_validate_json(canonical(candidate))
    definition = factor.definition(view["definitions"])
    # The generated definition retains constraints even when callers omit the optional catalog.
    with pytest.raises(ValueError, match="vintage evidence"):
        compute_factor(definition, view["columns"])
    independent = deepcopy(view["definitions"])
    for item in independent:
        item["forecast"]["pairing_group"] = None
    plain = factor.definition(independent)
    assert "forecast_pairing" not in plain
    np.testing.assert_allclose(compute_factor(plain, view["columns"], independent), [22, 35, -12])

    candidate["expression"]["nodes"].append(
        {"name": "old_spread", "op": "lag", "inputs": ["spread"], "window": 1}
    )
    factor = DiscoveryFactor.model_validate_json(canonical(candidate))
    values = compute_factor(
        factor.definition(view["definitions"]),
        view["columns"],
        view["definitions"],
        input_evidence=view["input_evidence"],
    )
    np.testing.assert_allclose(values, [np.nan, 22, 35], equal_nan=True)
    # lag(DRAM) - NAND must not turn a valid Jan pair into a Feb cross-vintage comparison.
    candidate = proposal()
    candidate["expression"]["nodes"].insert(
        2, {"name": "old_d", "op": "lag", "inputs": ["d"], "window": 1}
    )
    candidate["expression"]["nodes"][-1]["inputs"] = ["old_d", "n"]
    factor = DiscoveryFactor.model_validate_json(canonical(candidate))
    with pytest.raises(ValueError, match="before temporal"):
        compute_factor(
            factor.definition(view["definitions"]),
            view["columns"],
            view["definitions"],
            input_evidence=view["input_evidence"],
        )


def test_forecast_does_not_admit_publication_after_receipt():
    source = forecast_article()
    source["received_at"] = "2026-01-04T20:00:00Z"
    rows, reasons = checked(forecast_request(), source, forecast_extraction())
    assert rows == [] and reasons == ["publication_after_receipt"]


def test_late_receipt_does_not_refresh_an_old_forecast():
    rows, reasons = checked(
        forecast_request(),
        forecast_article(),
        forecast_extraction(),
        "retained",
        "2026-11-06T20:00:00Z",
    )
    assert reasons == [] and len(rows) == 1
    result = align_input_bundle(forecast_request(), rows, ["2026-11-06T20:00:00Z"])
    assert result["columns"]["acquired.dram"] == [None]


@pytest.mark.parametrize(
    "vintage",
    [
        {"source_id": "same_source"},
        {"source_id": None, "published_at": None, "target_quarter": None},
        {"source_id": "x", "published_at": "invalid", "target_quarter": "2026-Q1"},
        {"source_id": "x", "published_at": TIMES[0], "target_quarter": "2026-Q5"},
    ],
)
def test_malformed_pairing_metadata_is_rejected_even_when_both_sides_match(vintage):
    view = align_acquired_history(
        _history([bundle(), bundle("nand")], []), TIMES, TIMES[-1], "reconstructed"
    )
    for rows in view["input_evidence"].values():
        for row in rows:
            row["forecast_vintage"] = vintage
    factor = DiscoveryFactor.model_validate_json(canonical(proposal()))
    with pytest.raises(ValueError):
        compute_factor(
            factor.definition(view["definitions"]),
            view["columns"],
            input_evidence=view["input_evidence"],
        )


def test_short_quarter_quote_cannot_hide_fiscal_source_context():
    result = forecast_extraction()
    result["observations"][0]["period_excerpt"] = "Q1 2026"
    result["observations"][0]["excerpt"] = "dram forecast 55.0 % QoQ"
    for text in (
        "dram forecast 55.0 % QoQ for fiscal Q1 2026 (December 1 2025 to February 28 2026)",
        "dram forecast 55.0 % QoQ for Q1 2026 (FY2026)",
    ):
        source = forecast_article() | {"text": "5 January 2026 " + text}
        rows, reasons = checked(forecast_request(), source, result)
        assert rows == [] and reasons == ["unverified_forecast_target_quarter"]
