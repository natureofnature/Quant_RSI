"""Readable discovery lineage and separately grouped experimental results."""

import json
from html import escape


def render_discovery(result):
    discovery = result["discovery"]
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
    for m in decisions:
        for c in m["decision"].get("candidates", []):
            candidates.append(
                [
                    c["name"],
                    c["description"],
                    ", ".join(c.get("evidence_ids", [])),
                    ", ".join(i["factor_id"] for i in c["inputs"]),
                    " → ".join(
                        n["op"] + (f"({n['window']})" if n.get("window") else "")
                        for n in c["expression"]["nodes"]
                    ),
                ]
            )
    table(
        "Evidence → candidate → inputs → expression",
        ["Candidate", "Mechanism", "Evidence", "Inputs", "Expression"],
        candidates,
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
            "Measurement / unit",
            "Status",
            "Distinct observations / values",
            "Covered rows",
            "Bundle reference",
        ],
        [
            [
                m.get("request", {}).get("input_id"),
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
    for bundle in bundles.values():
        for observation in bundle.get("observations", [])[:20]:
            observations.append(
                [
                    observation.get("input_id"),
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
            "Measurement period",
            "Value / unit",
            "Source",
            "Available from",
            "Availability basis",
            "Retained excerpt",
        ],
        observations,
    )
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
    for key, group in discovery["groups"].items():
        rows = []
        for record in [group["baseline"], *group["candidates"]]:
            measured = record["result"]
            prediction = measured.get("prediction", {})
            metrics = prediction.get("development", {}).get("metrics", {})
            strategy = prediction.get("strategy", {}).get("strategy", {}).get("metrics", {})
            rows.append(
                [
                    measured.get("round"),
                    record["spec"]["kind"],
                    metrics.get("mae_price"),
                    strategy.get("net_return"),
                    strategy.get("sharpe_ratio"),
                    record["ref"] == group["incumbent"]["ref"],
                ]
            )
        table(
            "Comparable experiment " + key[:12],
            ["Fit", "Kind", "Price MAE", "Net return", "Sharpe", "Retained"],
            rows,
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
    table("Gaps and failures", ["Kind", "Reason", "Reference"], gaps)
    selected = discovery.get("selected")
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
    if selected and result.get("terminal_evaluation"):
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
</style><main>"""
        + "".join(parts)
        + "</main></html>"
    )
    return "\n".join(markdown), html
