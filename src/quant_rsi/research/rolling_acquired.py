"""Replay measured disclosures and extract new literals from retained excerpts only.

Checkpoint hashes establish retained-artifact integrity, not authenticity of HTTP
bytes or financial interpretation. Reconstructed disclosures are not point-in-time.
"""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.market.market_contracts import timestamp
from quant_rsi.research import discovery_sources
from quant_rsi.research.discovery_sources import (
    DataRequest,
    DiscoverySources,
    NumericalExtraction,
    align_input_bundle,
    validate_extraction,
)

MAX_FILES, MAX_BYTES = 32, 8 * 1024 * 1024
EXCERPT_BOUNDARY = "\n\n[retained excerpt boundary]\n\n"
NOTICE = (
    "Literal extracted measurements require interpretation review. Repeated daily values are "
    "not independent disclosures. Retrospective publication dates and excerpt projections are "
    "not point-in-time proof. Original HTTP content hashes are retained attestations; original "
    "HTTP bytes are not revalidated."
)


def _model(model, value):
    return model.model_validate_json(canonical(value))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _mode(mode):
    _require(mode in {"reconstructed", "retained"}, "Unknown acquired information mode")
    return "retrospective" if mode == "reconstructed" else "retained"


def _segments(article):
    segments = article.get("projection_segments")
    if segments is None:
        return [{"start": 0, "end": len(article["text"]), "text": article["text"]}]
    _require(len(segments) <= 96, "Too many retained projection segments")
    previous = -1
    for segment in segments:
        _require(
            type(segment["start"]) is int
            and type(segment["end"]) is int
            and previous <= segment["start"] < segment["end"]
            and segment["end"] - segment["start"] == len(segment["text"]),
            "Invalid original-text projection offsets",
        )
        previous = segment["end"]
    _require(
        article["text"] == EXCERPT_BOUNDARY.join(s["text"] for s in segments)
        and article["projection_sha256"] == digest(article["text"].encode()),
        "Projection text differs from retained original spans",
    )
    return segments


def _quote_span(article, quote):
    for segment in _segments(article):
        offset = segment["text"].find(quote)
        if offset >= 0:
            return segment["start"] + offset
    raise ValueError("Extraction quotation crosses or exceeds retained source spans")


def _replay(bundle):
    """Rebuild original aligned values from retained source text and extraction receipts."""
    request = _model(DataRequest, bundle["request"])
    articles, extractions, observations = (
        bundle[key] for key in ("articles", "extractions", "observations")
    )
    _require(
        len(articles) <= 64 and len(extractions) <= 64 and len(observations) <= 2048,
        "Acquired archive exceeds source/observation bounds",
    )
    by_id = {}
    for article in articles:
        _require(
            isinstance(article.get("id"), str) and article["id"] not in by_id,
            "Duplicate or missing acquired article ID",
        )
        _require(
            isinstance(article.get("text"), str) and len(canonical(article)) <= 128_000,
            "Invalid or oversized retained article",
        )
        _require(article.get("status") == "available", "Unavailable source in acquired archive")
        _segments(article)
        timestamp(article["received_at"])
        if article.get("version_available_at"):
            timestamp(article["version_available_at"])
        by_id[article["id"]] = article
    basis = {row["availability_basis"] for row in observations}
    _require(
        len(basis) <= 1 and basis <= {"observed_receipts", "reconstructed_public_disclosure_date"},
        "Unknown or mixed acquired availability basis",
    )
    inferred = "retained" if basis == {"observed_receipts"} else "retrospective"
    mode = bundle.get("information_mode", inferred)
    _require(
        mode in {"retrospective", "retained"} and (not basis or mode == inferred),
        "Archive availability mode conflicts",
    )
    cutoff = max((timestamp(row["effective_available_at"]) for row in observations), default=None)
    if bundle.get("research_cutoff"):
        cutoff = timestamp(bundle["research_cutoff"])
    _require(mode != "retained" or cutoff is not None, "Retained archive lacks original cutoff")
    checked, seen = [], set()
    for extraction in extractions:
        identity = extraction["article_id"]
        _require(
            identity in by_id and identity not in seen, "Unknown or duplicate extraction source"
        )
        seen.add(identity)
        instant = timestamp(extraction["extracted_at"])
        _require(
            instant >= timestamp(by_id[identity]["received_at"]),
            "Extraction precedes source receipt",
        )
        rows, _ = validate_extraction(
            request,
            by_id[identity],
            _model(NumericalExtraction, extraction["result"]),
            extracted_at=extraction["extracted_at"],
            information_mode=mode,
            research_cutoff=cutoff.isoformat() if cutoff else extraction["extracted_at"],
        )
        for row in rows:
            for quote in (
                row["publication_excerpt"],
                row["excerpt"],
                row.get("period_excerpt"),
                row.get("original_actual_release", {}).get("release_excerpt"),
            ):
                if quote:
                    _quote_span(by_id[identity], quote)
        checked.extend(rows)
    evidence = bundle["input_evidence"][request.input_id]
    _require(len(evidence) <= 10000, "Acquired alignment exceeds row bound")
    times = [row["prediction_time"] for row in evidence]
    expected = align_input_bundle(request, checked, times)
    for key in ("observations", "identity", "columns", "input_evidence", "definitions"):
        _require(
            canonical(bundle[key]) == canonical(expected[key]),
            f"Acquired archive differs from replayed {key}",
        )
    return deepcopy(bundle)


def _history(acquisitions, provenance):
    body = {"acquisitions": deepcopy(acquisitions), "provenance": deepcopy(provenance)}
    return body | {"identity": digest(canonical(body))}


def _compatible_source(left, right):
    """Match immutable source identity and overlapping original-text spans."""
    derived = {
        "text",
        "projection_sha256",
        "projection_spans",
        "projection_segments",
        "original_text_sha256",
        "original_url",
        "symbol",
        "measurement",
        "published_at",
    }

    def identity(article):
        return {k: v for k, v in article.items() if k not in derived} | {
            "original_text_sha256": article.get("original_text_sha256")
            or digest(article["text"].encode()),
            "original_url": article.get("original_url") or article["url"],
        }

    _require(identity(left) == identity(right), "Conflicting acquired source ID")
    if "projection_segments" not in left and "projection_segments" not in right:
        _require(
            left.get("published_at") == right.get("published_at"),
            "Conflicting acquired source publication",
        )
    for a in _segments(left):
        for b in _segments(right):
            start, end = max(a["start"], b["start"]), min(a["end"], b["end"])
            if start < end:
                _require(
                    a["text"][start - a["start"] : end - a["start"]]
                    == b["text"][start - b["start"] : end - b["start"]],
                    "Conflicting acquired source spans",
                )
    # A supplied complete original must cover every claimed projection offset.
    for original, projected in ((left, right), (right, left)):
        if "projection_segments" not in original:
            _require(
                all(s["end"] <= len(original["text"]) for s in _segments(projected)),
                "Projection exceeds original source",
            )


def load_acquired_history(paths):
    """Import explicit acquisition checkpoints; filenames never affect their identity."""
    paths = list(paths)
    _require(len(paths) <= MAX_FILES, "Too many acquired history files")
    acquisitions, provenance, ids, sources = [], [], set(), {}
    for path in paths:
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        _require(len(raw) <= MAX_BYTES, "Acquired checkpoint exceeds size bound")
        receipt = json.loads(raw)
        output = receipt["output"]
        _require(
            receipt["output_sha256"] == digest(canonical(output)),
            "Acquired checkpoint checksum differs",
        )
        bundle = _replay(output)
        identity = bundle["request"]["input_id"]
        _require(identity not in ids, "Duplicate acquired input ID")
        ids.add(identity)
        for article in bundle["articles"]:
            for previous in sources.setdefault(article["id"], []):
                _compatible_source(previous, article)
            sources[article["id"]].append(article)
        acquisitions.append(bundle)
        provenance.append(
            {
                "checkpoint_sha256": digest(raw),
                "output_sha256": receipt["output_sha256"],
                "input_sha256_opaque": receipt.get("input_sha256"),
                "acquisition_origin": bundle.get("acquisition_origin", "not_attested"),
                "original_http_bytes_verified": False,
            }
        )
    return _history(acquisitions, provenance)


def _rows(bundle, cutoff, mode):
    request = _model(DataRequest, bundle["request"])
    articles = {a["id"]: a for a in bundle["articles"]}
    observations = []
    observed = bundle.get("information_mode") == "retained" or any(
        o["availability_basis"] == "observed_receipts" for o in bundle["observations"]
    )
    for extraction in bundle["extractions"]:
        rows, _ = validate_extraction(
            request,
            articles[extraction["article_id"]],
            _model(NumericalExtraction, extraction["result"]),
            extracted_at=extraction["extracted_at"],
            information_mode="retained" if observed else _mode(mode),
            research_cutoff=cutoff,
        )
        observations.extend(
            row for row in rows if timestamp(row["effective_available_at"]) <= timestamp(cutoff)
        )
    return observations


def _projections(acquisitions, cutoff, mode):
    """Keep exact contiguous spans, never the unfiltered current article/page body."""
    documents = {}
    for bundle in acquisitions:
        request = bundle["request"]
        articles = {a["id"]: a for a in bundle["articles"]}
        for row in _rows(bundle, cutoff, mode):
            article = articles[row["source_id"]]
            key = (request["symbol"], article["id"])
            document = documents.setdefault(
                key,
                {
                    "article": article,
                    "fragments": [],
                    "spans": [],
                    "measurements": set(),
                    "available_at": row["effective_available_at"],
                },
            )
            for fragment in _segments(article):
                if fragment not in document["fragments"]:
                    document["fragments"].append(fragment)
            document["measurements"].add(request["measurement"])
            document["available_at"] = min(document["available_at"], row["effective_available_at"])
            for role, excerpt in (
                ("publication", row["publication_excerpt"]),
                ("period", row.get("period_excerpt") or row["excerpt"]),
                ("value", row["excerpt"]),
                (
                    "original_actual_release",
                    row.get("original_actual_release", {}).get("release_excerpt"),
                ),
            ):
                if not excerpt:
                    continue
                start = _quote_span(article, excerpt)
                span = {
                    "start": start,
                    "end": start + len(excerpt),
                    "text": excerpt,
                    "role": role,
                    "input_id": request["input_id"],
                    "period_end": row["period_end"],
                }
                if span not in document["spans"]:
                    document["spans"].append(span)
    result = []
    for (symbol, _), document in documents.items():
        article = document["article"]
        fragments = sorted(document["fragments"], key=lambda s: (s["start"], s["end"]))

        def original_text(start, end, fragments=fragments):
            # Only concatenate contiguous original offsets; never bridge an unknown gap.
            if start > end:
                return None
            cursor, parts = start, []
            for fragment in fragments:
                if fragment["start"] <= cursor <= fragment["end"]:
                    stop = min(end, fragment["end"])
                    parts.append(
                        fragment["text"][cursor - fragment["start"] : stop - fragment["start"]]
                    )
                    cursor = stop
                    if cursor == end:
                        return "".join(parts)
            return None

        intervals = []
        for span in sorted(document["spans"], key=lambda s: (s["start"], s["end"])):
            gap = original_text(intervals[-1][1], span["start"]) if intervals else None
            if intervals and (
                span["start"] <= intervals[-1][1] or (gap is not None and not gap.strip())
            ):
                intervals[-1][1] = max(intervals[-1][1], span["end"])
            else:
                intervals.append([span["start"], span["end"]])
        segments = [{"start": a, "end": b, "text": original_text(a, b)} for a, b in intervals]
        _require(
            all(segment["text"] is not None for segment in segments),
            "Projection crosses original spans",
        )
        text = EXCERPT_BOUNDARY.join(segment["text"] for segment in segments)
        projected = deepcopy(article) | {
            "text": text,
            "symbol": symbol,
            "measurement": "; ".join(sorted(document["measurements"])),
            "projection_sha256": digest(text.encode()),
            "projection_spans": document["spans"],
            "projection_segments": segments,
            "original_text_sha256": article.get("original_text_sha256")
            or digest(article["text"].encode()),
            "original_url": article.get("original_url") or article["url"],
            "published_at": document["available_at"],
        }
        result.append(projected)
    return result


def align_acquired_history(history, row_times, cutoff, mode):
    _mode(mode)
    _require(
        history["identity"] == _history(history["acquisitions"], history["provenance"])["identity"],
        "Acquired history identity changed",
    )
    _require(
        all(timestamp(t) <= timestamp(cutoff) for t in row_times),
        "Acquired rows extend beyond cutoff",
    )
    columns, definitions, coverage, mapping, input_evidence = {}, [], [], {}, {}
    raw_bundles = []
    for bundle in history["acquisitions"]:
        rows = _rows(bundle, cutoff, mode)
        aligned = align_input_bundle(bundle["request"], rows, row_times)
        raw_bundles.append({"request": aligned["request"], "observations": rows})
        identity = bundle["request"]["input_id"]
        values = np.asarray(aligned["columns"][identity], dtype=float)
        columns[identity] = values
        input_evidence[identity] = aligned["input_evidence"][identity]
        definitions.extend(aligned["definitions"])
        used = {
            row["observation_id"]
            for row in aligned["input_evidence"][identity]
            if row["observation_id"]
        }
        mapping[identity] = sorted(
            {row["source_id"] for row in rows if row["observation_id"] in used}
        )
        coverage.append(
            {
                "input_id": identity,
                "factor_id": identity,
                "symbol": bundle["request"]["symbol"],
                "unit": bundle["request"]["unit"],
                "measurement": bundle["request"]["measurement"],
                "available_rows": int(np.isfinite(values).sum()),
                "total_rows": len(values),
                "distinct_observations": len(used),
                "distinct_values": aligned["distinct_values"],
                "input_evidence": aligned["input_evidence"][identity],
                "conflicts": aligned["conflicts"],
                "max_age_days": bundle["request"]["max_age_days"],
            }
        )
    projections = _projections(history["acquisitions"], cutoff, mode)
    evidence = [
        {
            "id": a["id"],
            "symbol": a["symbol"],
            "measurement": a["measurement"],
            "headline": a["measurement"],
            "source_url": a.get("final_url") or a["url"],
            "published_at": a["published_at"],
            "excerpt": a["text"],
            "projection_sha256": a["projection_sha256"],
            "projection_spans": a["projection_spans"],
            "original_text_sha256": a["original_text_sha256"],
            "content_sha256": a.get("content_sha256"),
        }
        for a in projections
    ]
    return {
        "columns": columns,
        "definitions": definitions,
        "evidence": evidence,
        "coverage": coverage,
        # Raw admitted history may support a candidate-specific join even when the
        # generic latest view conflicts. Every candidate still checks its fixed rows.
        "qualified_input_ids": sorted(
            bundle["request"]["input_id"] for bundle in raw_bundles if bundle["observations"]
        ),
        "input_evidence_ids": mapping,
        "input_evidence": input_evidence,
        "raw_bundles": raw_bundles,
        "notice": NOTICE,
    }


class AcquiredResolver:
    """Small source-only bridge over existing acquisition and operation checkpoints."""

    def __init__(self, history, llm, cache, root, max_extractions=6):
        _require(
            type(max_extractions) is int and 0 <= max_extractions <= 6,
            "Extraction limit must be in 0..6",
        )
        self.history = deepcopy(history)
        self.llm, self.cache, self.root, self.max_extractions = (
            llm,
            cache,
            Path(root),
            max_extractions,
        )

    def view(self, row_times, cutoff, mode):
        return align_acquired_history(self.history, row_times, cutoff, mode)

    def acquire(self, requests, referenced_ids, row_times, cutoff, mode):
        _require(len(requests) <= 2, "At most two measured data requests per decision")
        _mode(mode)
        results, gaps, blocked, parsed = [], [], set(), {}
        existing = {b["request"]["input_id"]: b for b in self.history["acquisitions"]}
        for raw in requests:
            identity = raw.get("input_id") if isinstance(raw, dict) else None
            if identity not in referenced_ids:
                continue
            try:
                request = _model(DataRequest, raw)
                if identity in parsed and parsed[identity] != request:
                    raise ValueError("Conflicting same-decision acquired requests")
                if identity in existing and canonical(existing[identity]["request"]) != canonical(
                    request.model_dump(mode="json")
                ):
                    raise ValueError("Acquired input ID cannot be overwritten")
                parsed[identity] = request
            except ValueError as exc:
                blocked.add(identity)
                gaps.append({"input_id": identity, "reason": str(exc)})
        for identity, request in parsed.items():
            if identity in blocked or identity in existing:
                continue
            documents = [
                a
                for a in _projections(self.history["acquisitions"], cutoff, mode)
                if a["symbol"] == request.symbol
            ]
            aliases = {url for a in documents for url in (a["url"], a.get("final_url")) if url}
            if set(request.source_urls) - aliases:
                gaps.append(
                    {
                        "input_id": request.input_id,
                        "reason": "URL is outside cutoff-eligible source archive",
                    }
                )
                continue
            if request.source_urls:
                documents = [
                    a
                    for a in documents
                    if a["url"] in request.source_urls or a.get("final_url") in request.source_urls
                ]
            documents = documents[:6]
            if not documents:
                gaps.append(
                    {
                        "input_id": request.input_id,
                        "reason": "No cutoff-eligible archived excerpts for company",
                    }
                )
                continue
            # Filling source_urls prevents the existing provider from attempting a search.
            source_request = request.model_copy(
                update={"source_urls": tuple(a["url"] for a in documents)}
            )
            inputs = {
                "request": request.model_dump(mode="json"),
                "documents": documents,
                "cutoff": cutoff,
                "mode": mode,
                "row_times": list(row_times),
                "history_identity": self.history["identity"],
            }

            def execute(
                source_request=source_request, documents=documents, request=request, inputs=inputs
            ):
                def no_network(*args, **kwargs):
                    raise ValueError("Acquired monthly inputs cannot read or search live sources")

                if mode == "retained" and timestamp(discovery_sources._now()) > timestamp(cutoff):
                    return align_input_bundle(request, [], row_times) | {
                        "articles": [],
                        "extractions": [],
                        "gaps": ["New extraction is after retained cutoff"],
                        "information_mode": "retained",
                        "research_cutoff": cutoff,
                        "acquisition_origin": "not_extracted_retained_cutoff",
                    }
                sources = DiscoverySources(
                    self.llm,
                    information_mode=_mode(mode),
                    research_cutoff=cutoff,
                    retained_articles=documents,
                    reader=no_network,
                    call=self.cache.call,
                    max_searches=0,
                    max_reads=6,
                    max_extractions=self.max_extractions,
                    policy={"monthly_source_request": digest(canonical(inputs))},
                    usage_path=self.root / "acquired_source_usage.json",
                )
                output = sources.acquire(
                    source_request.model_dump(mode="json"), prediction_times=row_times
                )
                aligned = align_input_bundle(request, output["observations"], row_times)
                return (
                    output
                    | aligned
                    | {
                        "acquisition_origin": "llm_extraction_from_retained_excerpts",
                        "information_mode": _mode(mode),
                        "research_cutoff": cutoff,
                    }
                )

            try:
                output = self.cache.call(
                    "rolling_acquire_" + digest(canonical(inputs))[:20], inputs, execute
                )
                output = _replay(output)
            except Exception as exc:
                raise RuntimeError("Measured acquisition/provider replay failed") from exc
            self.history = _history(
                self.history["acquisitions"] + [output],
                self.history["provenance"]
                + [
                    {
                        "output_sha256": digest(canonical(output)),
                        "acquisition_origin": output["acquisition_origin"],
                        "original_http_bytes_verified": False,
                    }
                ],
            )
            existing[request.input_id] = output
            results.append(deepcopy(output))
            gaps.extend(
                {"input_id": request.input_id, "reason": reason}
                for reason in output.get("gaps", [])
            )
        return {
            "view": self.view(row_times, cutoff, mode),
            "acquisitions": results,
            "gaps": gaps,
            "blocked_input_ids": sorted(blocked),
        }
