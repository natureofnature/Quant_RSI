"""Evidence and forecast meaning for business candidates, independent of their recipe."""

from pydantic import Field

from quant_rsi.core.contracts_base import StrictModel


class EvidenceRelationship(StrictModel):
    # Tickers for listed entities; names for unsupported/private entities. These are
    # source-grounded proposed links, not a verified company master or knowledge graph.
    from_entity: str = Field(min_length=1, max_length=200)
    to_entity: str = Field(min_length=1, max_length=200)
    relationship: str = Field(min_length=1, max_length=1000)
    source_id: str = Field(min_length=1, max_length=200)
    excerpt: str = Field(min_length=12, max_length=2000)

    def evidence_error(self, articles):
        article = articles.get(self.source_id)
        if not article:
            return "relationship_source_not_admitted"
        quote = " ".join(self.excerpt.split())
        if len(quote) < 12:
            return "relationship_quote_too_short"
        if quote not in " ".join(article.get("text", "").split()):
            return "relationship_quote_not_in_source"
        return None


class ResearchHypothesis(StrictModel):
    operating_mechanism: str = Field(min_length=1, max_length=2000)
    measurement_horizon: str = Field(min_length=1, max_length=300)
    forecast_channel: str | None = Field(default=None, min_length=1, max_length=2000)
    forecast_horizon_sessions: int = Field(ge=1, le=252)
    relationships: tuple[EvidenceRelationship, ...] = Field(default=(), max_length=12)


def hypothesis_gap(candidate, bundles, *, target_symbol, horizon, articles):
    """Return a concrete research gap before fitting; never retarget the experiment.

    Quote checks establish provenance, not truth or the usefulness of a causal claim.
    Existing target-only recipes remain readable without semantic metadata.
    """
    used = {item.factor_id for item in candidate.inputs}
    used_bundles = [b for b in bundles if b.get("request", {}).get("input_id") in used]
    foreign = {
        b["request"]["symbol"] for b in used_bundles if b["request"]["symbol"] != target_symbol
    }
    hypothesis = candidate.hypothesis
    if foreign and hypothesis is None:
        return "cross_entity_hypothesis_required"
    if hypothesis is None:
        return None
    if hypothesis.forecast_horizon_sessions != horizon:
        return "hypothesis_forecast_horizon_mismatch"
    if not hypothesis.forecast_channel or not hypothesis.forecast_channel.strip():
        return "forecast_channel_unspecified"
    for link in hypothesis.relationships:
        if link.source_id not in candidate.evidence_ids:
            return "relationship_source_not_cited_by_candidate"
        if error := link.evidence_error(articles):
            return error
    # Only the small candidate-local list is traversed. Related-profile seeds never
    # establish a company relationship by themselves.
    reached = {target_symbol}
    for _ in hypothesis.relationships:
        for link in hypothesis.relationships:
            if {link.from_entity, link.to_entity} & reached:
                reached.update((link.from_entity, link.to_entity))
    if foreign - reached:
        return "cross_entity_relationship_missing"
    for bundle in used_bundles:
        sources = {row["source_id"] for row in bundle.get("observations", [])}
        if not sources.intersection(candidate.evidence_ids):
            return "acquired_measurement_source_not_cited"
    return None
