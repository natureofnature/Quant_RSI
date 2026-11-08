"""Persist bounded news/feedback paired diagnostics before invoking real research decisions.

This writes a separate diagnostic workspace, never a study's measured memory.
Outputs await an explicit evidence-based assessment; unequal LLM outputs do not pass.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from quant_rsi.core.artifacts import atomic_write, canonical, digest
from quant_rsi.research.discovery_contracts import now

# Rebuild prompts from explicit scientific inputs. Derived summaries, cache, prior
# candidates and memory are dropped from BOTH news arms instead of guessing provenance.
CONTEXT_FIELDS = {
    "symbol",
    "question",
    "research_question",
    "information_mode",
    "research_cutoff",
    "prediction_times",
    "requested_data_period",
    "source_limits",
    "iteration",
    "definitions",
    "catalog",
    "survey_candidates",
    "policy",
    "experiment_contract",
    "forecast_target",
    "preparation_only",
    "counts",
    "remaining_fits",
    "evidence",
}


def paired_contexts(plan):
    if plan.get("kind") not in {"news_ablation", "counterfactual_feedback"}:
        raise ValueError("Unknown paired diagnostic")
    if type(plan.get("max_pairs")) is not int or not 1 <= plan["max_pairs"] <= 5:
        raise ValueError("Declare one to five pairs before execution")
    if not isinstance(plan.get("criteria"), str) or not plan["criteria"].strip():
        raise ValueError("Declare observable evidence/action criteria before execution")
    if not isinstance(plan.get("provider"), dict):
        raise ValueError("Persist provider settings before execution")
    context = deepcopy(plan["context"])
    if plan["kind"] == "news_ablation":
        left = {key: value for key, value in context.items() if key in CONTEXT_FIELDS}
        # Catalog/definitions must be declared pre-evidence references, not a prior
        # generation output that would leak the removed story through its description.
        if plan.get("catalog_provenance") != "pre_evidence":
            raise ValueError("News ablation requires a declared pre-evidence catalog")
        ids = {item["id"] for item in left.get("evidence", [])}
        removed = set(plan.get("removed_evidence_ids", []))
        if not removed or not removed <= ids:
            raise ValueError("Removed evidence IDs must exist in the original inputs")
        left.update(memory=[], prior_memory=[], consumed_memory_refs=[], visible_candidates=[])
        right = deepcopy(left)
        right["evidence"] = [item for item in right["evidence"] if item["id"] not in removed]
        cleanup = {
            "dropped_context_fields": sorted(set(context) - CONTEXT_FIELDS),
            "memory": "empty in both arms",
            "live_search": "disabled; decision only",
            "catalog_provenance": "pre_evidence",
            "removed_evidence_ids": sorted(removed),
        }
    else:
        branches = plan.get("feedback_branches", [])
        if len(branches) != 2 or any(not b.get("synthetic") for b in branches):
            raise ValueError("Supply exactly two explicitly synthetic feedback branches")
        if not plan.get("feedback_consistency_note"):
            raise ValueError("Explain the internally consistent feedback intervention")
        left, right = deepcopy(context), deepcopy(context)
        for target, feedback in zip((left, right), branches, strict=True):
            target.setdefault("memory", []).append(
                {
                    "kind": "counterfactual_feedback",
                    "partition": "development",
                    "synthetic": True,
                    "result": deepcopy(feedback),
                    "ref": digest(canonical(feedback)),
                }
            )
            target["consumed_memory_refs"] = [item.get("ref") for item in target["memory"]]
            target["diagnostic_notice"] = (
                "Synthetic counterfactual branch, not a measured result. "
                "Do not store in authentic study memory."
            )
        cleanup = {"live_search": "disabled; decision only", "synthetic_feedback": True}
    return left, right, cleanup


def _write(path, value):
    atomic_write(path, canonical(value))


def run_pairs(plan, output, decide):
    output = Path(output)
    # New directory prevents cherry-picked appends/reusing altered input plans.
    output.mkdir(parents=True, exist_ok=False)
    left, right, cleanup = paired_contexts(plan)
    envelope = {
        "plan": deepcopy(plan),
        "effective_contexts": [left, right],
        "cleanup": cleanup,
        "created_at": now(),
    }
    envelope["identity"] = digest(canonical(envelope))
    _write(output / "plan.json", envelope)
    outcomes = []
    for pair in range(plan["max_pairs"]):
        row = {"pair": pair, "plan_identity": envelope["identity"], "arms": {}}
        # Alternate order; each transport call is otherwise stateless.
        for arm in ("original", "intervention") if pair % 2 == 0 else ("intervention", "original"):
            context = left if arm == "original" else right
            path = output / f"pair-{pair}-{arm}.json"
            request = {
                "context": context,
                "input_identity": digest(canonical(context)),
                "plan_identity": envelope["identity"],
                "pair": pair,
                "arm": arm,
            }
            _write(output / f"pair-{pair}-{arm}-input.json", request)
            try:
                result = decide(deepcopy(context))
                record = request | {"status": "completed", "output": result}
            except Exception as exc:
                record = request | {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            _write(path, record)
            row["arms"][arm] = {"artifact": path.name, "status": record["status"]}
        row["assessment"] = "inconclusive_pending_evidence_review"
        outcomes.append(row)
        _write(
            output / "results.json",
            {
                "plan_identity": envelope["identity"],
                "pairs": outcomes,
                "status": "inconclusive",
                "notice": "Output inequality is not evidence of influence.",
            },
        )
    return outcomes


def assess_pairs(output, assessment):
    """Record an explicit reviewer disposition bound to planned criteria and real artifacts."""
    output = Path(output)
    envelope = json.loads((output / "plan.json").read_text())
    if assessment.get("plan_identity") != envelope["identity"]:
        raise ValueError("Assessment must refer to the exact frozen plan")
    if not assessment.get("reviewer") or not assessment.get("rationale"):
        raise ValueError("Assessment requires an identified reviewer and reasoning")
    if assessment.get("status") not in {"pass", "fail", "inconclusive"}:
        raise ValueError("Assessment status must be pass/fail/inconclusive")
    refs = assessment.get("artifact_refs", [])
    if not refs or any(Path(ref).name != ref or not (output / ref).is_file() for ref in refs):
        raise ValueError("Assessment must cite existing paired artifacts")
    if assessment["status"] == "pass":
        records = [json.loads((output / ref).read_text()) for ref in refs]
        completed = [r for r in records if r.get("status") == "completed" and "output" in r]
        arms = {(r.get("pair"), r.get("arm")) for r in completed}
        paired = [
            i
            for i in range(envelope["plan"]["max_pairs"])
            if (i, "original") in arms and (i, "intervention") in arms
        ]
        if not paired:
            raise ValueError("Pass requires a completed original/intervention pair")
        if not assessment.get("observed_actions") or not assessment.get("evidence_links"):
            raise ValueError("Pass requires observed actions and supporting evidence links")
        if not assessment.get("alternative_explanations"):
            raise ValueError("Pass requires consideration of randomness and redundant evidence")
    record = deepcopy(assessment) | {
        "created_at": now(),
        "assessment_kind": "explicit_review",
        "statistical_claim": "none",
    }
    _write(output / "assessment.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--assessment", type=Path)
    args = parser.parse_args(argv)
    if args.assessment:
        result = assess_pairs(args.output, json.loads(args.assessment.read_text()))
        print(result["status"])
        return
    if args.plan is None:
        parser.error("--plan is required for paired runs")
    plan = json.loads(args.plan.read_text())
    # Validate before initializing an external service.
    paired_contexts(plan)
    from quant_rsi.research.discovery_sources import discovery_decision
    from quant_rsi.research.llm import LLMHypothesizer

    provider = plan["provider"]
    llm = LLMHypothesizer(provider["mode"], provider.get("model"), provider.get("timeout", 180))
    run_pairs(plan, args.output, lambda context: discovery_decision(llm, context))
    print(f"Paired observations: {args.output}; evidence review pending")


if __name__ == "__main__":
    main()
