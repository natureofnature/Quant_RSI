"""Outer research decisions over existing operation receipts and numerical callbacks.

Runner callbacks perform one fit or return zero-fit diagnostics. They never choose
another investigation. Results are persisted before the next decision callback.
"""

import json
import math
from copy import deepcopy
from pathlib import Path

from quant_rsi.core.artifacts import atomic_write, canonical, digest
from quant_rsi.research.discovery_contracts import guard_records, now, research_policy


def run_discovery(
    context,
    *,
    root,
    cache,
    decide,
    run_experiment,
    acquire=None,
    survey=None,
    policy=None,
    fit_budget=3,
    decision_budget=3,
    acquisition_budget=2,
    survey_budget=2,
    publication_experiment=None,
    validate_candidate=None,
    on_context=None,
):
    for limit in (fit_budget, decision_budget, acquisition_budget, survey_budget):
        if type(limit) is not int or limit < 0:
            raise ValueError("Discovery budgets must be nonnegative integers")
    if publication_experiment not in {None, "first", "latest"}:
        raise ValueError(
            "Declare first/latest publication experiment or leave incomparable results separate"
        )
    root = Path(root)
    policy = deepcopy(policy or research_policy())
    if (
        research_policy(policy["settings"], parent=policy.get("parent"))["identity"]
        != policy["identity"]
    ):
        raise ValueError("Research policy identity mismatch")
    failures_root = root / "discovery" / "fit_failures"
    previous_failures = [
        json.loads(path.read_text()) for path in sorted(failures_root.glob("*.json"))
    ]
    failed_charges = sum(record["fit_count"] for record in previous_failures)
    counts = dict(
        decisions=0,
        acquisitions=0,
        surveys=0,
        fits=failed_charges,
        failed_fits=failed_charges,
        successful_fits=0,
        proposed=0,
        distinct_candidates=0,
        proposal_duplicates=0,
        duplicates=0,
        screened=0,
        executable=0,
        trained=0,
    )
    memory, bundles, groups, seen = [], [], {}, set()
    discovered, screened, executable, trained = set(), set(), set(), set()
    evidence, report = guard_records(
        context.get("evidence", []),
        mode=context.get("information_mode", "retrospective"),
        cutoff=context.get("research_cutoff"),
    )
    prior, prior_guard = guard_records(
        context.get("prior_memory", []),
        gate="G3",
        mode=context.get("information_mode", "retrospective"),
        cutoff=context.get("research_cutoff"),
    )
    base = deepcopy(context) | {
        "evidence": evidence,
        "context_guard": report,
        "prior_memory": prior,
        "prior_memory_guard": prior_guard,
    }
    # Terminal artifacts belong only in finalization, never in a decision's input.
    for key in ("terminal", "terminal_evaluation", "terminal_results"):
        base.pop(key, None)
    baseline_contract = deepcopy(context.get("experiment_contract", {}))
    stop_reason = "decision_budget_exhausted"

    def action_budgets():
        return {
            key: {"limit": limit, "used": counts[key], "remaining": max(0, limit - counts[key])}
            for key, limit in (
                ("decisions", decision_budget),
                ("surveys", survey_budget),
                ("acquisitions", acquisition_budget),
                ("fits", fit_budget),
            )
        }

    def persist(name, value):
        def create():
            record = deepcopy(value) | {"created_at": now(), "partition": "development"}
            return record | {"ref": digest(canonical(record))}

        record = cache.call("discovery_record_" + name, value, create)
        atomic_write(root / "discovery" / "memory" / (name + ".json"), canonical(record))
        memory.append(record)
        return record

    def operation(name, inputs, callback):
        try:
            return cache.call("discovery_" + name, inputs, callback)
        except Exception as exc:
            # Preserve the failed call independently; explicit resume may retry it.
            atomic_write(
                root / "discovery" / "failures" / (name + ".json"),
                canonical(
                    {
                        "operation": name,
                        "input_ref": digest(canonical(inputs)),
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "created_at": now(),
                        "policy_identity": policy["identity"],
                    }
                ),
            )
            raise

    def fit(name, spec):
        nonlocal counts
        # Rebuild in-memory numerical state on resume; the runner's existing fit cache
        # owns physical fit reuse. Persisted facts below retain their original identity.
        try:
            result = run_experiment(deepcopy(spec))
        except Exception as exc:
            charge = getattr(exc, "fit_count", 1)
            if type(charge) is not int or charge not in (0, 1):
                charge = 1
            failure = {
                "kind": "experiment_failure",
                "spec": spec,
                "operation": name,
                "reason": str(exc),
                "exception": type(exc).__name__,
                "fit_count": charge,
                "created_at": now(),
                "partition": "development",
                "charge_note": getattr(exc, "fit_accounting", "conservative_attempt_charge"),
            }
            failure["ref"] = digest(canonical(failure))
            atomic_write(failures_root / (failure["ref"] + ".json"), canonical(failure))
            atomic_write(
                root / "discovery" / "memory" / (name + "_failure.json"), canonical(failure)
            )
            raise
        charged = result.get("fit_count", 0)
        if type(charged) is not int or charged not in (0, 1):
            raise ValueError("Experiment runner must account for exactly zero or one fit")
        if result.get("partition", "development") != "development":
            raise ValueError("Research runner returned prohibited terminal feedback")
        if _has_terminal(result):
            raise ValueError("Research runner mixed terminal data into development output")
        counts["fits"] += charged
        counts["successful_fits"] += charged
        return persist(
            name,
            {
                "kind": "experiment",
                "spec": spec,
                "result": result,
                "policy_identity": policy["identity"],
            },
        )

    for iteration in range(1, decision_budget + 1):
        # Same-run facts are available immediately; historical retrieval was guarded upstream.
        eligible, memory_guard = guard_records(memory, gate="G3")
        current = deepcopy(base) | {
            "iteration": iteration,
            "memory": prior + previous_failures + eligible,
            "consumed_memory_refs": [
                row.get("ref", row.get("id")) for row in prior + previous_failures + eligible
            ],
            "memory_guard": memory_guard,
            "input_bundles": deepcopy(bundles),
            "policy": deepcopy(policy),
            "counts": deepcopy(counts),
            "action_budgets": action_budgets(),
            "remaining_fits": fit_budget - counts["fits"],
        }
        current["visible_candidates"] = _visible_candidates(current["memory"])
        # Freeze each decision input once. On resume the same decision sees its original
        # context; newly reached decisions include all durable failed-fit charges.
        current = cache.call(
            f"discovery_context_{iteration}",
            {"iteration": iteration, "base": base, "policy": policy},
            lambda current=current: deepcopy(current),
        )
        if on_context is not None:
            on_context(deepcopy(current))
        action = operation(
            f"decision_{iteration}", current, lambda current=current: decide(deepcopy(current))
        )
        counts["decisions"] += 1
        lineage = _candidate_lineage(action, current["memory"])
        persist(
            f"decision_{iteration}",
            {
                "kind": "interpretation",
                "decision": action,
                "candidate_lineage": lineage,
                "consumed_memory_refs": current["consumed_memory_refs"],
                "policy_identity": policy["identity"],
            },
        )
        # Discovery happens before a test request: acquisition and stop decisions may
        # carry hypotheses that could not yet be executed. Preserve those in the funnel.
        proposals = action.get("candidates", [])
        if isinstance(proposals, list):
            for candidate in proposals:
                identity = _candidate_identity(candidate)
                counts["proposed"] += 1
                if identity in discovered:
                    counts["proposal_duplicates"] += 1
                discovered.add(identity)
            counts["distinct_candidates"] = len(discovered)
        if any(item["status"] == "rejected" for item in lineage):
            persist(
                f"lineage_{iteration}",
                {
                    "kind": "diagnostic",
                    "reason": "invalid_candidate_lineage",
                    "result": {"status": "rejected", "candidate_lineage": lineage},
                },
            )
            continue
        kind = action.get("action")
        if kind in {"stop", "retain", "withdraw"}:
            if kind == "withdraw" and groups:
                # Withdrawal concerns the latest investigated experiment, not unrelated groups.
                group = next(reversed(groups.values()))
                group["incumbent"] = group["baseline"]
            stop_reason = kind
            break
        if kind in {"acquire", "survey"}:
            requests = (
                action.get("data_requests", [])
                if kind == "acquire"
                else [
                    {
                        "query": action.get("query"),
                        "reason": action.get("reason", ""),
                        **(
                            {"symbol": action["survey_symbol"]}
                            if action.get("survey_symbol")
                            else {}
                        ),
                    }
                ]
            )
            counter, limit, provider = (
                ("acquisitions", acquisition_budget, acquire)
                if kind == "acquire"
                else ("surveys", survey_budget, survey)
            )
            if not requests or provider is None:
                persist(
                    f"diagnostic_{iteration}",
                    {"kind": "diagnostic", "reason": "missing_request_or_provider"},
                )
                continue
            for index, request in enumerate(requests):
                if counts[counter] >= limit:
                    stop_reason = counter + "_budget_exhausted"
                    persist(
                        f"budget_{iteration}_{index}",
                        {
                            "kind": "diagnostic",
                            "status": "not_executed",
                            "reason": stop_reason,
                            "operation": kind,
                            "requests": deepcopy(requests[index:]),
                            "action_budgets": action_budgets(),
                        },
                    )
                    break
                inputs = {"request": request, "policy_identity": policy["identity"]}
                result = operation(
                    f"{kind}_{iteration}_{index}",
                    inputs,
                    lambda request=request, provider=provider: provider(deepcopy(request)),
                )
                counts[counter] += 1
                if kind == "survey":
                    records, guard = guard_records(
                        result.get("evidence", []),
                        mode=context.get("information_mode", "retrospective"),
                        cutoff=context.get("research_cutoff"),
                    )
                    base["evidence"].extend(records)
                    admitted = {r.get("id") for r in records}
                    result = {
                        "evidence": records,
                        "articles": records,
                        "guard": guard,
                        "counts": result.get("counts", {}),
                        "operations": result.get("operations", []),
                        "access_receipts": result.get("access_receipts", []),
                        "gaps": result.get("gaps", []),
                        "search": result.get("search", {}),
                        "excluded": result.get("excluded", []),
                        "findings": [
                            f
                            for f in result.get("findings", [])
                            if f.get("evidence_ids") and set(f["evidence_ids"]) <= admitted
                        ],
                    }
                elif result.get("status") in {"completed", "available", "ready"}:
                    input_id = result.get("request", {}).get("input_id")
                    if input_id:
                        bundles[:] = [
                            b for b in bundles if b.get("request", {}).get("input_id") != input_id
                        ]
                    bundles.append(deepcopy(result))
                persist(
                    f"{kind}_{iteration}_{index}",
                    {
                        "kind": kind,
                        "request": request,
                        "result": result,
                        "policy_identity": policy["identity"],
                    },
                )
            continue
        if kind not in {"test", "revise"}:
            persist(f"diagnostic_{iteration}", {"kind": "diagnostic", "reason": "unknown_action"})
            continue
        candidates = action.get("candidates", [])
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 8:
            persist(
                f"diagnostic_{iteration}", {"kind": "diagnostic", "reason": "candidate_batch_limit"}
            )
            continue
        for index, candidate in enumerate(candidates):
            discovery_identity = _candidate_identity(candidate)
            screened.add(discovery_identity)
            counts["screened"] = len(screened)
            validation = {}
            if validate_candidate is not None:
                validation = validate_candidate(deepcopy(candidate), deepcopy(bundles))
            if fit_budget == 0:
                # Preparation is an explicit product: keep executable column evidence
                # and gaps even though no baseline or candidate fit may be started.
                if validate_candidate is None:
                    validation = {
                        "status": "needs_data",
                        "reason": "preparation_validator_unavailable",
                    }
                persist(
                    f"preparation_{iteration}_{index}",
                    {"kind": "preparation", "candidate": candidate, "result": validation},
                )
                if validation.get("status") == "ready":
                    executable.add(discovery_identity)
                    counts["executable"] = len(executable)
                continue
            if validate_candidate is not None:
                if validation.get("status") not in {"ready", "completed", "pass"}:
                    persist(
                        f"invalid_{iteration}_{index}",
                        {
                            "kind": "diagnostic",
                            "candidate": candidate,
                            "result": validation,
                        },
                    )
                    continue
            executable.add(discovery_identity)
            counts["executable"] = len(executable)
            # Validator/runtime supplies canonical recipe identities; raw identity only prevents
            # exact repeated submissions before validation, not symbolic equivalence claims.
            identity = validation.get("recipe_ref") or discovery_identity
            bundle_refs = [digest(canonical(bundle)) for bundle in bundles]
            comparison_key = digest(
                canonical({"contract": baseline_contract, "inputs": bundle_refs})
            )
            duplicate_key = (comparison_key, identity)
            if duplicate_key in seen:
                counts["duplicates"] += 1
                persist(
                    f"duplicate_{iteration}_{index}",
                    {"kind": "diagnostic", "reason": "duplicate", "candidate_ref": identity},
                )
                continue
            seen.add(duplicate_key)
            new_group = comparison_key not in groups
            required = 2 if new_group else 1
            if fit_budget - counts["fits"] < required:
                stop_reason = "fit_budget_exhausted"
                persist(
                    f"budget_{iteration}_{index}",
                    {"kind": "diagnostic", "reason": stop_reason, "required_fits": required},
                )
                break
            spec = {
                "comparison_key": comparison_key,
                "contract": baseline_contract,
                "input_bundles": deepcopy(bundles),
                "policy_identity": policy["identity"],
            }
            if new_group:
                baseline = fit(
                    f"baseline_{iteration}_{index}", spec | {"kind": "baseline", "candidate": None}
                )
                groups[comparison_key] = {
                    "baseline": baseline,
                    "incumbent": baseline,
                    "candidates": [],
                }
            group = groups[comparison_key]
            if group["baseline"]["result"].get("status") != "completed":
                continue
            result = fit(
                f"candidate_{iteration}_{index}",
                spec | {"kind": "candidate", "candidate": candidate},
            )
            group["candidates"].append(result)
            if result["result"].get("status") == "completed" and result["result"].get("fit_count"):
                trained.add(discovery_identity)
                counts["trained"] = len(trained)
            value, incumbent = result["result"], group["incumbent"]["result"]
            metrics = (value.get("metric"), incumbent.get("metric"))
            comparable = all(
                type(metric) in (int, float) and math.isfinite(metric) for metric in metrics
            )
            if value.get("status") == "completed" and comparable and metrics[0] < metrics[1]:
                group["incumbent"] = result
    if not groups and counts["fits"] < fit_budget:
        # A failed discovery still produces a useful baseline report on original inputs.
        comparison_key = digest(canonical({"contract": baseline_contract, "inputs": []}))
        baseline = fit(
            "baseline_only",
            {
                "kind": "baseline",
                "candidate": None,
                "comparison_key": comparison_key,
                "contract": baseline_contract,
                "input_bundles": [],
                "policy_identity": policy["identity"],
            },
        )
        groups[comparison_key] = {"baseline": baseline, "incumbent": baseline, "candidates": []}
    incumbents = [group["incumbent"] for group in groups.values()]
    selected = incumbents[0] if len(incumbents) == 1 else None
    if incumbents and publication_experiment:
        selected = incumbents[0 if publication_experiment == "first" else -1]
    output = {
        "status": "completed",
        "stop_reason": stop_reason,
        "counts": counts,
        "groups": groups,
        "selected": selected,
        "memory": memory,
        "fit_failures": previous_failures,
        "policy": policy,
        "publication_rule": publication_experiment,
        "selection_note": (
            "Development selection within each experiment; incompatible groups are not ranked."
        ),
    }
    if fit_budget == 0:
        output["mode"] = "preparation_only"
        output["selection_note"] = (
            "Preparation only: no fitting, predictions or outcome evaluation."
        )
    atomic_write(root / "discovery" / "result.json", canonical(output))
    return output


def _has_terminal(value):
    if isinstance(value, dict):
        if value.get("partition") == "terminal" or any(
            key in value for key in ("terminal", "terminal_results", "terminal_evaluation")
        ):
            return True
        return any(_has_terminal(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_terminal(item) for item in value)
    return False


def _candidate_identity(candidate):
    """Discovery dedup never prevents a later acquire→test transition."""
    from quant_rsi.research.discovery_factors import DiscoveryFactor

    try:
        return DiscoveryFactor.model_validate_json(canonical(candidate)).recipe_ref
    except (ValueError, TypeError):
        # Keep malformed proposals visible too; execution validation rejects them.
        return digest(canonical(candidate))


def _visible_candidates(records):
    """Index only definitions in the decision's admitted Memory, not hidden run history."""
    from quant_rsi.research.discovery_factors import DiscoveryFactor

    found = {}
    for record in records:
        candidates = record.get("decision", {}).get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        lineages = record.get("candidate_lineage", [])
        candidates = [
            candidate
            for index, candidate in enumerate(candidates)
            if not lineages or (index < len(lineages) and lineages[index]["status"] != "rejected")
        ]
        candidate = record.get("spec", record.get("experiment_context", {})).get("candidate")
        if candidate:
            candidates.append(candidate)
        for candidate in candidates:
            try:
                proposal = DiscoveryFactor.model_validate_json(canonical(candidate))
            except (ValueError, TypeError):
                continue
            entry = found.setdefault(
                proposal.recipe_ref,
                {"recipe_ref": proposal.recipe_ref, "name": proposal.name, "memory_refs": []},
            )
            if record.get("ref") and record["ref"] not in entry["memory_refs"]:
                entry["memory_refs"].append(record["ref"])
    return list(found.values())


def _candidate_lineage(action, records):
    """Keep each proposed revision even when its numerical recipe is later deduplicated."""
    from quant_rsi.research.discovery_factors import DiscoveryFactor

    visible = {item["recipe_ref"] for item in _visible_candidates(records)}
    result = []
    candidates = action.get("candidates", [])
    if not isinstance(candidates, list):
        return result
    for candidate in candidates:
        if not isinstance(candidate, dict):
            result.append(
                {
                    "recipe_ref": _candidate_identity(candidate),
                    "parent_recipe_refs": [],
                    "status": "rejected",
                    "reason": "Candidate must be an object",
                }
            )
            continue
        row = {
            "recipe_ref": _candidate_identity(candidate),
            "parent_recipe_refs": candidate.get("parent_recipe_refs", []),
            "status": "legacy_unspecified",
            "reason": "Legacy candidate has no declared parent metadata; no parent is inferred.",
        }
        if "parent_recipe_refs" in candidate:
            try:
                proposal = DiscoveryFactor.model_validate_json(canonical(candidate))
                if action.get("action") == "revise" and not proposal.parent_recipe_refs:
                    raise ValueError("A revision requires parent_recipe_refs")
                if set(proposal.parent_recipe_refs) - visible:
                    raise ValueError("Parent candidate is not visible in this decision's Memory")
                row.update(
                    status="accepted", reason="Declared parents are visible earlier candidates."
                )
            except (ValueError, TypeError) as exc:
                row.update(status="rejected", reason=str(exc))
        result.append(row)
    return result
