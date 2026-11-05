"""Screen a proposed factor pool independently on one unchanged monthly sample.

These gates establish executable inputs, not predictive usefulness. A bad candidate
does not discard unrelated valid candidates, and no missing rows are removed.
"""

from copy import deepcopy

import numpy as np

from quant_rsi.core.artifacts import canonical
from quant_rsi.research.discovery_factors import DiscoveryFactor
from quant_rsi.research.rolling_news_features import NewsInput, compile_news_inputs
from quant_rsi.research.rolling_research import candidate_matrix, verify_news_dependency


def screen_candidates(
    candidates,
    news_inputs,
    columns,
    definitions,
    baseline,
    news,
    times,
    train,
    *,
    news_required=False,
    mode="reconstructed",
    coverage=None,
    allowed_parent_refs=(),
    accepted_incumbent_records=(),
    require_evidence=False,
    acquired_inputs=None,
):
    """Keep the first valid, distinct columns and explain every rejected proposal.

    Shared news input IDs may reuse identical declarations. Conflicting declarations
    make only their dependent candidates unavailable. Returned recipes cover accepted
    columns. Diagnostics retain all compiled inputs and mark model use; rejected/unused
    declarations remain in ``rejections``.
    """
    train = np.asarray(train, dtype=int)
    required = np.append(train, len(times) - 1)
    matrix = [columns[name] for name in baseline]
    if not np.isfinite(np.column_stack(matrix)[required]).all():
        raise ValueError("Baseline unavailable on the fixed training/prediction sample")
    recipes, conflicts, rejections = {}, set(), []
    for index, value in enumerate(news_inputs):
        identity = value.get("input_id", f"news_input_{index + 1}")
        try:
            value = NewsInput.model_validate_json(canonical(value)).model_dump(mode="json")
        except ValueError as exc:
            rejections.append({"kind": "news_input", "name": identity, "reason": str(exc)})
            continue
        if identity in recipes and canonical(recipes[identity]) != canonical(value):
            conflicts.add(identity)
        recipes.setdefault(identity, value)
    accepted, ids, selected_inputs, news_columns, diagnostics = (
        [],
        list(baseline),
        {},
        {},
        {},
    )
    refs = set()
    incumbents = {
        canonical(DiscoveryFactor.model_validate_json(canonical(value)).model_dump(mode="json"))
        for value in accepted_incumbent_records
    }
    evidence = {item["id"] for item in (news or {}).get("items", [])}
    acquired = acquired_inputs or {}
    qualified = set(acquired.get("qualified_input_ids", ()))
    evidence.update(item["id"] for item in acquired.get("evidence", ()))
    for index, value in enumerate(candidates):
        name = value.get("name", f"candidate_{index + 1}")
        try:
            candidate = DiscoveryFactor.model_validate_json(canonical(value))
            record = candidate.model_dump(mode="json")
            used = {
                item.factor_id for item in candidate.inputs if item.factor_id.startswith("news.")
            }
            measurements = {
                item.factor_id
                for item in candidate.inputs
                if item.factor_id.startswith("acquired.")
            }
            if not measurements <= qualified:
                raise ValueError("Candidate requires an unavailable or unverified measurement")
            for identity in measurements:
                sources = acquired.get("input_evidence_ids", {}).get(identity, ())
                if not set(candidate.evidence_ids).intersection(sources):
                    raise ValueError(f"Candidate must cite a source supporting {identity}")
                values = columns.get(identity)
                if values is None or not np.isfinite(values[required]).all():
                    raise ValueError(f"Measurement unavailable on fixed sample: {identity}")
                if len(np.unique(values[train])) < 2:
                    raise ValueError(f"Measurement has no training variation: {identity}")
            if not set(candidate.evidence_ids) <= evidence:
                raise ValueError("Candidate cites unknown or cutoff-ineligible evidence")
            if require_evidence and not candidate.evidence_ids:
                raise ValueError("Candidate requires cutoff-eligible evidence")
            # Exact admitted records retain their validated ancestry, including in
            # partial replacements. Modified/new records may cite only current parents.
            if canonical(record) not in incumbents and not set(candidate.parent_recipe_refs) <= set(
                allowed_parent_refs
            ):
                raise ValueError("Candidate cites unknown parent recipe")
            if news_required and (not (used or measurements) or not candidate.evidence_ids):
                raise ValueError(
                    "News research requires an actual news input or verified measurement "
                    "with cited evidence"
                )
            if used & conflicts:
                raise ValueError("Candidate depends on conflicting news input declarations")
            if not used <= recipes.keys():
                raise ValueError("Candidate needs an undeclared news input")
            selected = [recipes[identity] for identity in sorted(used)]
            nc, nd, audit = (
                compile_news_inputs(
                    selected,
                    (news or {}).get("items", []),
                    times,
                    mode=mode,
                    history_start=times[0],
                    coverage=coverage,
                )
                if selected
                else ({}, [], [])
            )
            diagnostics.update({item["input_id"]: item for item in audit})
            for identity, values in nc.items():
                if not np.isfinite(values[required]).all():
                    raise ValueError(f"News input unavailable on fixed sample: {identity}")
                if len(np.unique(values[train])) < 2:
                    raise ValueError(f"News input has no training variation: {identity}")
            active_columns, active_definitions = columns | nc, definitions + nd
            proposed_matrix, proposed_ids = candidate_matrix(
                [record],
                active_columns,
                active_definitions,
                baseline,
                required,
                input_evidence=acquired.get("input_evidence"),
            )
            values = proposed_matrix[:, -1]
            if news_required or measurements:
                verify_news_dependency(
                    [record],
                    active_columns,
                    active_definitions,
                    required,
                    acquired_inputs=qualified,
                    input_evidence=acquired.get("input_evidence"),
                )
            if len(np.unique(values[train])) < 2:
                raise ValueError("Candidate expression has no training variation")
            if proposed_ids[-1] in ids:
                raise ValueError("Duplicate candidate name")
            if candidate.recipe_ref in refs:
                raise ValueError("Duplicate candidate recipe")
            if any(np.array_equal(values[required], prior[required]) for prior in matrix):
                raise ValueError("Candidate duplicates an existing model column")
            accepted.append(record)
            refs.add(candidate.recipe_ref)
            ids.append(proposed_ids[-1])
            matrix.append(values)
            for recipe in selected:
                selected_inputs[recipe["input_id"]] = deepcopy(recipe)
            news_columns.update(nc)
            diagnostics.update({item["input_id"]: item for item in audit})
        except ValueError as exc:
            rejections.append({"kind": "candidate", "name": name, "reason": str(exc)})
    for identity in sorted(recipes.keys() - selected_inputs.keys()):
        rejections.append(
            {
                "kind": "news_input",
                "name": identity,
                "reason": "Conflicting news input declarations"
                if identity in conflicts
                else "News input is not used by an accepted candidate",
            }
        )
    return {
        "candidates": accepted,
        "news_inputs": list(selected_inputs.values()),
        "matrix": np.column_stack(matrix),
        "ids": ids,
        "news_columns": news_columns,
        "diagnostics": [
            item | {"used_by_model": identity in selected_inputs}
            for identity, item in diagnostics.items()
        ],
        "rejections": rejections,
    }
