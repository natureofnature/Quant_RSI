"""'%' names the requested 'percent' unit under recorded rules (#159 Phase 2).

Measured on #159: in the TSM run after #176, 4 of 8 gross-margin guidance answers
returned unit '%' for a 'percent' request and were rejected, so no quarter had both
bounds. Synthetic figures here are not market evidence.
"""

import json

import pytest
from test_mops_extraction_boundaries import records
from test_mops_usd_guidance import provider_for
from test_rolling_acquired import save

from quant_rsi.research import discovery_sources as sources
from quant_rsi.research.rolling_acquired import align_acquired_history, load_acquired_history

QUOTE = "年增12.3%"
EARLIER = sources.EXTRACTION_RULE_SETS[0]  # recorded by #176 bundles
TIMES = ["2026-10-09T06:30:00Z", "2026-10-09T06:30:01Z"]


def answer(requested="percent", returned="%", quote=QUOTE, value=12.3):
    request, article, extraction = records(
        "2026年9月 " + quote + "。", quote=quote, value=value, unit=requested
    )
    extraction["observations"][0]["unit"] = returned
    return request, article, extraction


def check(request, article, extraction, rules=sources.CURRENT_EXTRACTION_RULES):
    return sources.validate_extraction(
        sources.DataRequest.model_validate_json(json.dumps(request)),
        article,
        sources.NumericalExtraction.model_validate_json(json.dumps(extraction)),
        extracted_at="2026-10-10T12:01:00Z",
        information_mode="retrospective",
        research_cutoff=None,
        rules=rules,
    )


@pytest.mark.parametrize(("requested", "returned"), [("percent", "%"), ("%", "percent")])
def test_percent_names_are_one_unit_and_rows_carry_the_requested_name(requested, returned):
    rows, reasons = check(*answer(requested, returned))
    assert reasons == []
    assert rows[0]["value"] == 12.3
    assert rows[0]["unit"] == requested


@pytest.mark.parametrize("rules", [EARLIER, None])
def test_earlier_rules_still_require_the_exact_unit_name(rules):
    assert check(*answer(), rules=rules) == ([], ["unverified_literal_value_or_unit"])


@pytest.mark.parametrize(
    ("requested", "returned", "quote", "value"),
    [
        ("percent", "pct", QUOTE, 12.3),
        ("percent", "percentage points", QUOTE, 12.3),
        ("percent", "TWD", QUOTE, 12.3),
        ("USD hundred million", "億美元", "營收322億美元", 322),
        ("TWD million", "TWD", "新台幣51百萬元", 51),
    ],
)
def test_other_unit_names_still_match_exactly(requested, returned, quote, value):
    rows, reasons = check(*answer(requested, returned, quote, value))
    assert (rows, reasons) == ([], ["unverified_literal_value_or_unit"])


def test_extraction_prompt_asks_for_the_requested_unit_name(monkeypatch):
    monkeypatch.setattr(sources, "_now", lambda: "2026-10-10T12:01:00Z")
    request, article, extraction = answer()
    prompts = []
    provider_for(article, extraction, prompts).acquire(request, prediction_times=TIMES)
    assert "set each observation's unit to exactly request.unit" in prompts[0]


def acquired(monkeypatch):
    monkeypatch.setattr(sources, "_now", lambda: "2026-10-10T12:01:00Z")
    request, article, extraction = answer()
    return request, provider_for(article, extraction, []).acquire(request, prediction_times=TIMES)


def column(history):
    view = align_acquired_history(history, TIMES, TIMES[-1], "reconstructed")
    values = view["columns"]["acquired.monthly_measurement"]
    return [None if value != value else value for value in values]


def test_current_bundle_records_alias_rules_and_replays(tmp_path, monkeypatch):
    _, bundle = acquired(monkeypatch)
    assert bundle["extraction_rules"] == sources.CURRENT_EXTRACTION_RULES
    assert bundle["covered_rows"] == 1
    assert column(load_acquired_history([save(tmp_path / "current.json", bundle)])) == [
        None,
        12.3,
    ]


def test_bundle_recorded_under_earlier_rules_keeps_its_rejection(tmp_path, monkeypatch):
    request, bundle = acquired(monkeypatch)
    # A #176 bundle: the same '%' answer, rejected, with the earlier rules recorded.
    earlier = bundle | sources.align_input_bundle(request, [], TIMES)
    earlier |= {"extraction_rules": dict(EARLIER)}
    history = load_acquired_history([save(tmp_path / "earlier.json", earlier)])
    assert history["acquisitions"][0]["observations"] == []
    assert column(history) == [None, None]

    # The recorded rules decide: the admitted claim under the earlier rules does not verify.
    claimed = bundle | {"extraction_rules": dict(EARLIER)}
    with pytest.raises(ValueError, match="differs from replayed observations"):
        load_acquired_history([save(tmp_path / "claimed.json", claimed)])
