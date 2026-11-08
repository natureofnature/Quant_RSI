"""Readable discovery lineage and separately grouped experimental results."""

import json
from html import escape

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research.discovery_factors import DiscoveryFactor
from quant_rsi.research.discovery_report_comparisons import append_comparisons


def render_discovery(result):
    discovery = result["discovery"]
    preparation_only = discovery.get("mode") == "preparation_only"
    markdown = [f"# {result['symbol']} — news-driven factor discovery", result["notice"]]
    parts = [
        f"<h1>{escape(result['symbol'])} · Factor discovery</h1>",
        f"<p>{escape(result['notice'])}</p>",
    ]

    def table(title, headers, rows):
        markdown.extend(
            ["", "## " + title, "", " | ".join(headers), " | ".join(["---"] * len(headers))]
        )
        parts.append(
            "<section><h2>"
            + escape(title)
            + "</h2><div class='scroll'><table><thead><tr>"
            + "".join("<th>" + escape(h) + "</th>" for h in headers)
            + "</tr></thead><tbody>"
        )
        for row in rows:
            texts = [str(v) if v is not None else "Unavailable" for v in row]
            markdown.append(" | ".join(t.replace("|", "\\|").replace("\n", " ") for t in texts))
            parts.append("<tr>" + "".join("<td>" + escape(t) + "</td>" for t in texts) + "</tr>")
        parts.append("</tbody></table></div></section>")

    table(
        "Research question",
        ["Question", "Outcome"],
        [[result["configuration"]["question"], result["study"]["conclusion"]]],
    )
    table("Operations", ["Operation", "Count"], list(discovery["counts"].items()))
    if preparation_only:
        table(
            "Preparation only — no model fitting",
            ["Scope", "Evidence boundary"],
            [
                [
                    "Acquire source inputs and compute candidate columns on the frozen population.",
                    "No baseline or candidate model was fitted. No forecast, accuracy, strategy "
                    "return or predictive improvement is evaluated. Prior experiment feedback "
                    "is not loaded; only this run's preparation feedback informs later actions.",
                ]
            ],
        )
    else:
        append_comparisons(discovery, table, parts)
    records = discovery["memory"]
    decisions = [m for m in records if m.get("kind") == "interpretation"]
    table(
        "Research iterations",
        ["Action", "Reason", "Feedback references"],
        [
            [
                m["decision"].get("action"),
                m["decision"].get("reason"),
                ", ".join(ref[:12] for ref in m.get("consumed_memory_refs", [])),
            ]
            for m in decisions
        ],
    )
    candidates = []
    revisions = []
    hypotheses = []
    relationships = []
    for m in decisions:
        for index, c in enumerate(m["decision"].get("candidates", [])):
            try:
                proposal = DiscoveryFactor.model_validate_json(canonical(c))
                view = proposal.model_dump(mode="json")
            except (ValueError, TypeError):
                view = {
                    "name": c.get("name", "Invalid candidate"),
                    "description": "Invalid definition; see the recorded lineage diagnostic.",
                    "evidence_ids": [],
                    "inputs": [],
                    "expression": {"nodes": []},
                }
            candidates.append(
                [
                    view["name"],
                    view["description"],
                    ", ".join(view["evidence_ids"]),
                    ", ".join(i["factor_id"] for i in view["inputs"]),
                    " → ".join(
                        n["op"] + (f"({n['window']})" if n.get("window") else "")
                        for n in view["expression"]["nodes"]
                    ),
                ]
            )
            hypothesis = view.get("hypothesis")
            if hypothesis:
                hypotheses.append(
                    [
                        view["name"],
                        hypothesis.get("operating_mechanism"),
                        hypothesis.get("measurement_horizon"),
                        hypothesis.get("forecast_channel") or "Unspecified — needs data",
                        hypothesis.get("forecast_horizon_sessions"),
                    ]
                )
                relationships.extend(
                    [
                        view["name"],
                        *[
                            relation.get(key)
                            for key in (
                                "from_entity",
                                "to_entity",
                                "relationship",
                                "source_id",
                                "excerpt",
                            )
                        ],
                    ]
                    for relation in hypothesis.get("relationships", [])
                )
            lineage = m.get("candidate_lineage", [])
            row = lineage[index] if index < len(lineage) else {}
            recipe_ref = row.get("recipe_ref")
            if not recipe_ref:
                try:
                    recipe_ref = DiscoveryFactor.model_validate_json(canonical(c)).recipe_ref
                except (ValueError, TypeError):
                    recipe_ref = "Unavailable (invalid definition)"
            parents = c.get("parent_recipe_refs", [])
            parent_text = (
                ", ".join(str(ref) for ref in parents)
                if isinstance(parents, (tuple, list))
                else str(parents)
            )
            revisions.append(
                [
                    c.get("name", "Invalid candidate"),
                    recipe_ref,
                    parent_text
                    or ("None declared" if "parent_recipe_refs" in c else "Unspecified (legacy)"),
                    row.get("status", "legacy_unspecified"),
                    row.get("reason", "Historical records are not assigned inferred parents."),
                ]
            )
    table(
        "Evidence → candidate → inputs → expression",
        ["Candidate", "Mechanism", "Evidence", "Inputs", "Expression"],
        candidates,
    )
    table(
        "Candidate revisions",
        ["Candidate", "Recipe reference", "Parent recipe references", "Lineage status", "Reason"],
        revisions,
    )
    if hypotheses:
        table(
            "Operating mechanism and proposed forecast channel — hypotheses, not proven effects",
            [
                "Candidate",
                "Operating mechanism",
                "Measurement horizon",
                "Proposed forecast channel",
                "Forecast horizon (sessions)",
            ],
            hypotheses,
        )
    surveys = [m for m in records if m.get("kind") == "survey"]
    if surveys:
        table(
            "Survey requests and source coverage",
            ["Source entity / symbol", "Search question", "Search status", "Search limitation"],
            [
                [
                    m.get("request", {}).get("symbol"),
                    m.get("request", {}).get("query")
                    or m.get("result", {}).get("search", {}).get("query"),
                    m.get("result", {}).get("search", {}).get("status", "Not recorded"),
                    m.get("result", {}).get("search", {}).get("reason", "None recorded"),
                ]
                for m in surveys
            ],
        )
        findings = []
        for m in surveys:
            for finding in m.get("result", {}).get("findings", []):
                findings.append(
                    [
                        m.get("request", {}).get("symbol"),
                        finding.get("mechanism"),
                        "; ".join(finding.get("measurements", [])),
                        finding.get("uncertainty"),
                        ", ".join(finding.get("evidence_ids", [])),
                    ]
                )
                relationships.extend(
                    [
                        "Survey finding",
                        *[
                            relation.get(key)
                            for key in (
                                "from_entity",
                                "to_entity",
                                "relationship",
                                "source_id",
                                "excerpt",
                            )
                        ],
                    ]
                    for relation in finding.get("relationships", [])
                )
        table(
            "Survey findings — proposed mechanisms and measurements",
            [
                "Source entity / symbol",
                "Mechanism",
                "Suggested measurements",
                "Uncertainty",
                "Evidence",
            ],
            findings,
        )
    if relationships:
        table(
            "Proposed relationships — quoted evidence, not verified links",
            [
                "Origin",
                "From entity",
                "To entity",
                "Proposed relationship",
                "Source ID",
                "Retained supporting quote",
            ],
            relationships,
        )
    acquisitions = [m for m in records if m.get("kind") == "acquire"]
    bundles = {
        m["result"]["identity"]: m["result"]
        for m in acquisitions
        if m.get("result", {}).get("identity")
    }
    table(
        "Acquired input coverage",
        [
            "Input",
            "Source entity / symbol",
            "Measurement / unit",
            "Status",
            "Distinct observations / values",
            "Covered rows",
            "Bundle reference",
        ],
        [
            [
                m.get("request", {}).get("input_id"),
                m.get("request", {}).get("symbol"),
                str(m.get("request", {}).get("measurement", ""))
                + " / "
                + str(m.get("request", {}).get("unit", "")),
                m.get("result", {}).get("status"),
                str(m.get("result", {}).get("distinct_observations", 0))
                + " / "
                + str(m.get("result", {}).get("distinct_values", 0)),
                m.get("result", {}).get("covered_rows"),
                m.get("result", {}).get("identity"),
            ]
            for m in acquisitions
        ],
    )
    observations = []
    requests = {
        m["result"]["identity"]: m.get("request", {})
        for m in acquisitions
        if m.get("result", {}).get("identity")
    }
    for bundle in bundles.values():
        for observation in bundle.get("observations", [])[:20]:
            observations.append(
                [
                    observation.get("input_id"),
                    requests[bundle["identity"]].get("symbol"),
                    observation.get("period_end"),
                    str(observation.get("value")) + " " + str(observation.get("unit", "")),
                    observation.get("source_url"),
                    observation.get("effective_available_at"),
                    observation.get("availability_basis"),
                    observation.get("excerpt"),
                ]
            )
    table(
        "Source observations — first 20 per acquired input",
        [
            "Input",
            "Source entity / symbol",
            "Measurement period",
            "Value / unit",
            "Source",
            "Available from",
            "Availability basis",
            "Retained excerpt",
        ],
        observations,
    )
    if preparation_only:
        _append_preparations(records, bundles, table)
    lineage = []
    for m in records:
        measured = m.get("result", {})
        actual = measured.get("actual_features")
        if not actual:
            continue
        definition = actual["definition"]
        for input_id in definition.get("input_factor_ids", []):
            matching = [
                b
                for key, b in bundles.items()
                if key in actual.get("input_bundle_refs", []) and input_id in b.get("columns", {})
            ]
            lineage.append(
                [
                    definition["factor_id"],
                    ", ".join(definition.get("evidence_ids", [])),
                    input_id,
                    ", ".join(b["identity"] for b in matching) or "Previously admitted input",
                    measured.get("round"),
                    measured.get("fit_identity"),
                    measured.get("prediction", {})
                    .get("development", {})
                    .get("metrics", {})
                    .get("mae_price"),
                ]
            )
    if not preparation_only:
        table(
            "Candidate → acquired input → actual fit",
            [
                "Candidate",
                "News evidence",
                "Used input",
                "Acquisition reference",
                "Fit",
                "Fit identity",
                "Development MAE",
            ],
            lineage,
        )
    features = []
    for m in records:
        actual = m.get("result", {}).get("actual_features")
        if actual:
            features.append(
                [
                    actual["definition"]["factor_id"],
                    actual["distinct_training_values"],
                    ", ".join(actual["definition"]["input_factor_ids"]),
                    ", ".join(str(v) for v in actual["training_values"][:12]),
                ]
            )
    if not preparation_only:
        table(
            "Executed feature values",
            ["Factor", "Distinct training values", "Inputs", "First training values"],
            features,
        )
    gaps = [
        [m.get("kind"), m.get("reason") or m.get("result", {}).get("reason"), m.get("ref", "")[:12]]
        for m in records
        if m.get("kind") in {"diagnostic", "failure"}
    ]
    gaps.extend(
        ["survey", gap, m.get("ref", "")[:12]]
        for m in surveys
        for gap in m.get("result", {}).get("gaps", [])
    )
    if preparation_only:
        gaps.extend(
            [
                "preparation",
                m["result"].get("reason") or m["result"].get("status"),
                m.get("ref", "")[:12],
            ]
            for m in records
            if m.get("kind") == "preparation" and m["result"].get("status") != "ready"
        )
    table("Gaps and failures", ["Kind", "Reason", "Reference"], gaps)
    selected = discovery.get("selected")
    if not preparation_only:
        table(
            "Publication selection",
            ["Selected fit", "Rule / reason"],
            [
                [
                    selected.get("result", {}).get("round") if selected else None,
                    discovery.get("selection_note", "No selection recorded"),
                ]
            ],
        )
    if not preparation_only and selected and result.get("terminal_evaluation"):
        table(
            "Reserved terminal evaluation",
            ["Status", "Samples"],
            [
                [
                    result["terminal_evaluation"]["status"],
                    result["terminal_evaluation"].get("sample_count"),
                ]
            ],
        )
    parts.append(
        "<details><summary>Complete evidence JSON</summary><pre>"
        + escape(json.dumps(result, indent=2, allow_nan=False))
        + "</pre></details>"
    )
    html = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>News-driven factor discovery</title><style>
body{margin:0;background:#0c1322;color:#e8eef8;font:16px system-ui}
main{max-width:1250px;margin:auto;padding:32px}h1{color:#8ae8d2}
section{margin:24px 0;padding:20px;background:#152137;
border:1px solid #2a3e5a;border-radius:14px}
h2{font-size:20px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}
th,td{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #30425b}
th{color:#8ae8d2}td{overflow-wrap:anywhere}p{line-height:1.6;color:#c0cee2}
pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer;padding:16px}
.comparison-group{border-color:#497d88}.comparison-group>section{padding:12px;margin:14px 0}
.comparison-chart{margin:22px 0;padding:14px;background:#0c192b;border-radius:10px}
.comparison-chart figcaption{font-weight:600;color:#8ae8d2;margin-bottom:8px}
.comparison-chart svg{width:100%;height:auto}.comparison-chart text{fill:#b7c9dd;font-size:13px}
.comparison-legend{display:flex;flex-wrap:wrap;gap:10px 24px;padding-left:20px;font-size:13px}
@media(max-width:680px){main{padding:14px}section{padding:12px}.comparison-chart{padding:4px}}
</style><main>"""
        + "".join(parts)
        + "</main></html>"
    )
    return "\n".join(markdown), html


def _append_preparations(records, bundles, table):
    """Keep acquisition, partial computation and population readiness visibly distinct."""
    candidates, coverage, provenance = [], [], []
    for record in records:
        if record.get("kind") != "preparation":
            continue
        candidate, result = record["candidate"], record["result"]
        prepared = result.get("preparation") or {}
        name = candidate.get("name", "Invalid candidate")
        candidates.append(
            [
                name,
                prepared.get("candidate_ref") or digest(canonical(candidate)),
                result.get("status"),
                result.get("reason") or "None recorded",
            ]
        )
        counts = prepared.get("coverage", {})
        bundle_refs = prepared.get("input_bundle_refs", [])
        if not prepared:
            acquisition = "Not established for this candidate"
        elif not bundle_refs:
            acquisition = "No acquired inputs used"
        elif any(ref not in bundles for ref in bundle_refs):
            acquisition = "Referenced acquisition receipt unavailable"
        else:
            acquisition = f"Acquired: {len(bundle_refs)} source-bound bundle(s)"
        computed, required = counts.get("computed_rows", 0), counts.get("required_rows", 0)
        ready = (
            result.get("status") == "ready"
            and required > 0
            and computed == required
            and counts.get("missing_rows") == 0
        )
        coverage.append(
            [
                name,
                acquisition,
                f"{computed} / {required} rows computable" if prepared else "Not computed",
                f"{counts.get('computed_training_rows', 0)} / {counts.get('training_rows', 0)}"
                if prepared
                else None,
                f"{counts.get('computed_development_rows', 0)} / "
                f"{counts.get('development_rows', 0)}"
                if prepared
                else None,
                counts.get("missing_rows") if prepared else None,
                "Full frozen population ready" if ready else "Not ready",
            ]
        )
        if prepared:
            provenance.append(
                [
                    name,
                    ", ".join(
                        f"{item['name']} = {item['factor_id']}"
                        for item in prepared.get("input_bindings", [])
                    ),
                    ", ".join(bundle_refs) or "None used",
                    ", ".join(prepared.get("source_ids", [])) or "None recorded",
                    prepared.get("column_ref"),
                    prepared.get("recipe_ref"),
                    prepared.get("definition_ref"),
                ]
            )
    table(
        "Prepared candidate decisions",
        ["Candidate", "Candidate reference", "Validation", "Reason"],
        candidates,
    )
    table(
        "Preparation coverage — acquired inputs, computable rows and readiness",
        [
            "Candidate",
            "Input acquisition",
            "Feature computation",
            "Training rows",
            "Development rows",
            "Missing rows",
            "Frozen population",
        ],
        coverage,
    )
    table(
        "Preparation provenance — values are retained in the evidence JSON",
        [
            "Candidate",
            "Input bindings",
            "Acquisition references",
            "Numerical source IDs",
            "Column reference",
            "Recipe reference",
            "Definition reference",
        ],
        provenance,
    )
