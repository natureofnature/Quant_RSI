"""Executable, evidence-linked headline baskets for monthly research.

These inputs measure matches within a retrieved corpus, not actual event frequency,
verified business measurements or semantic sentiment. Recipes selected at a fold's
cutoff may be recomputed over its history; that is retrospective feature research.
"""

import re
from datetime import date, timedelta
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.market.market_contracts import timestamp

Term = Annotated[str, Field(min_length=2, max_length=100)]


def _normalize(value):
    return " ".join(value.casefold().split())


class NewsInput(StrictModel):
    input_id: str = Field(pattern=r"^news\.[a-z][a-z0-9_]{0,63}$")
    hypothesis: str = Field(min_length=1, max_length=1500)
    operation: Literal["count", "balance"]
    terms: tuple[Term, ...] = Field(min_length=1, max_length=16)
    negative_terms: tuple[Term, ...] = Field(default=(), max_length=16)
    window_days: int = Field(ge=1, le=30)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def coherent_basket(self):
        terms = [_normalize(term) for term in self.terms + self.negative_terms]
        if any(len(term) < 2 for term in terms) or len(terms) != len(set(terms)):
            raise ValueError("Headline terms must be nonempty and distinct across both baskets")
        if self.operation == "count" and self.negative_terms:
            raise ValueError("A count basket cannot have negative terms")
        if self.operation == "balance" and not self.negative_terms:
            raise ValueError("A balance basket needs explicit negative terms")
        if len(self.evidence_ids) != len(set(self.evidence_ids)) or any(
            not value.strip() for value in self.evidence_ids
        ):
            raise ValueError("News evidence IDs must be nonempty and distinct")
        return self


def _matches(text, terms):
    return any(
        re.search(r"(?<!\w)" + re.escape(_normalize(term)) + r"(?!\w)", text) for term in terms
    )


def compile_news_inputs(
    recipes,
    articles,
    prediction_times,
    *,
    mode="reconstructed",
    history_start=None,
    coverage=None,
):
    """Return numerical columns, discovery definitions and auditable per-row diagnostics.

    Each article counts once per basket regardless of term count. Balance is positive
    article count minus negative article count; a headline matching both contributes
    zero. Retained mode requires original receipt and version times at each row.
    Reconstructed mode keeps those actual times but aligns today's headline version
    by its reported publication time. It cannot establish contemporaneous availability.
    An incomplete trailing window before history_start or across an unqueried/failed
    coverage date produces NaN, not a zero count. Diagnostics serialize that as None.
    Coverage intervals use inclusive UTC dates; successful empty searches establish
    only an empty retrieved corpus, never the absence of real-world events.
    """
    if mode not in {"retained", "reconstructed"}:
        raise ValueError("Unknown news information mode")
    prediction_times = list(prediction_times)
    times = [timestamp(value) for value in prediction_times]
    if not times or any(a >= b for a, b in zip(times, times[1:], strict=False)):
        raise ValueError("News prediction times must be nonempty and strictly increasing")
    boundary = timestamp(history_start) if history_start else None
    covered_days = None
    if coverage is not None:
        covered_days = set()
        for interval in coverage:
            first, last = date.fromisoformat(interval["start"]), date.fromisoformat(interval["end"])
            if first > last or (last - first).days > 1100:
                raise ValueError("Invalid bounded news coverage interval")
            if interval["status"] in {"partial_results", "empty_search_results"}:
                covered_days.update(
                    first + timedelta(days=i) for i in range((last - first).days + 1)
                )
    parsed = [NewsInput.model_validate_json(canonical(recipe)) for recipe in recipes]
    if len({recipe.input_id for recipe in parsed}) != len(parsed):
        raise ValueError("Duplicate news input ID")
    records = {}
    for article in articles:
        identifier = article["id"]
        if identifier in records:
            if records[identifier][0] != article:
                raise ValueError("Conflicting duplicate news evidence ID")
            continue
        published = timestamp(article["published_at"])
        received = timestamp(article["received_at"])
        version = timestamp(article.get("version_available_at") or article["received_at"])
        available = max(published, received, version) if mode == "retained" else published
        if available <= times[-1]:
            records[identifier] = (article, published, available)
    columns, definitions, diagnostics = {}, [], []
    for recipe in parsed:
        absent = set(recipe.evidence_ids) - records.keys()
        if absent:
            raise ValueError("News recipe cites unknown or cutoff-ineligible evidence")
        matches = {}
        for identifier, (article, published, available) in records.items():
            text = _normalize(article["title"])
            positive = bool(_matches(text, recipe.terms))
            negative = bool(_matches(text, recipe.negative_terms))
            if positive or negative:
                matches[identifier] = (published, available, int(positive) - int(negative))
        if not set(recipe.evidence_ids) <= matches.keys():
            raise ValueError("Each cited headline must match its declared news basket")
        values, row_ids, row_status = [], [], []
        for instant in times:
            start = instant - timedelta(days=recipe.window_days)
            missing_history = boundary is not None and start < boundary
            missing_coverage = covered_days is not None and any(
                start.date() + timedelta(days=i) not in covered_days
                for i in range((instant.date() - start.date()).days + 1)
            )
            if missing_history or missing_coverage:
                row_ids.append([])
                values.append(None)
                row_status.append(
                    "before_history_start" if missing_history else "missing_query_coverage"
                )
                continue
            identifiers = sorted(
                key
                for key, (published, available, _) in matches.items()
                if start < published <= instant and available <= instant
            )
            row_ids.append(identifiers)
            values.append(sum(matches[key][2] for key in identifiers))
            row_status.append("observed_corpus")
        array = np.asarray(values, dtype=float)
        columns[recipe.input_id] = array
        dumped = recipe.model_dump(mode="json")
        definitions.append(
            {
                "factor_id": recipe.input_id,
                "description": recipe.hypothesis,
                "kind": "numeric",
                "unit": "count",
                "runnable": True,
                "recipe": {"op": "headline_basket", **dumped},
                "recipe_ref": digest(canonical(dumped)),
                "evidence_ids": list(recipe.evidence_ids),
            }
        )
        diagnostics.append(
            {
                "input_id": recipe.input_id,
                "recipe": dumped,
                "information_mode": mode,
                "measurement": "retrieved_headline_match_count"
                if recipe.operation == "count"
                else "retrieved_positive_minus_negative_headline_match_count",
                "coverage": "partial_retrieved_corpus; zero is not evidence of no event",
                "availability_basis": "publication_receipt_and_version"
                if mode == "retained"
                else "reported_publication_of_currently_retrieved_headline_version",
                "prediction_times": list(prediction_times),
                "values": values,
                "row_evidence_ids": row_ids,
                "row_status": row_status,
                "history_start": history_start,
                "query_coverage_checked": coverage is not None,
                "distinct_values": len(np.unique(array[np.isfinite(array)])),
                "nonzero_rows": int(np.count_nonzero(array[np.isfinite(array)])),
                "unavailable_rows": int(np.count_nonzero(~np.isfinite(array))),
                "matched_articles": [
                    {
                        key: article.get(key)
                        for key in (
                            "id",
                            "title",
                            "url",
                            "published_at",
                            "received_at",
                            "version_available_at",
                        )
                    }
                    for identifier, (article, _, _) in sorted(records.items())
                    if identifier in matches
                ],
            }
        )
    return columns, definitions, diagnostics
