"""Offline evaluation for evidence-ID selection and host-rendered citations."""
from collections import Counter

from evaluation.agent01_metrics import review_finding
from evaluation.agent02_metrics import candidate_group as _candidate_group
from src.agent.tools import REFERENCE_MAX_CHARS


def candidate_group(task, candidate_reviews, candidate_ids=None):
    if not task.get("scopes"):
        return "PREFLIGHT_CONTROL"
    return _candidate_group(task.get("oracle"), candidate_reviews,
                            candidate_ids=candidate_ids)


def _bounds_valid(reference, registry_row, session):
    excerpt = reference.get("excerpt")
    span = reference.get("span")
    if not isinstance(excerpt, str) or not isinstance(span, dict):
        return False
    if len(excerpt) > REFERENCE_MAX_CHARS or span.get("length") != len(excerpt):
        return False
    ranges = span.get("ranges")
    if ranges is None:
        return (span.get("coordinate") == "child"
                and excerpt == registry_row.get("text")
                and len(excerpt) <= REFERENCE_MAX_CHARS)
    if span.get("coordinate") != "parent" or not isinstance(ranges, list) or not ranges:
        return False
    parent = session._parents[tuple(registry_row[k] for k in
                                    ("source", "page", "block_type", "block_index"))]["text"]
    pieces = []
    for row in ranges:
        start, end = row.get("start"), row.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(parent):
            return False
        pieces.append(parent[start:end])
    return "".join(pieces) == excerpt


def score_reference(state, outcomes, task, candidate_reviews, session=None):
    """Score attempted model selection separately from accepted host outcomes."""
    oracle = task.get("oracle") or {}
    observed_ids = {row["evidence_id"] for history in state.get("search_history", [])
                    for row in history.get("results", [])
                    if isinstance(row, dict) and "evidence_id" in row}
    explicit_ids = any("expected_evidence_ids" in item for item in oracle.values())
    group = candidate_group(task, candidate_reviews, observed_ids)
    host_finish_accepted = state.get("status") == "finished"
    outcome_by_scope = {row.get("scope"): row for row in outcomes}
    rendered_by_scope = {row.get("scope"): row for row in state.get("findings", [])}
    refs, reviews, authenticity, provenance, bounds = [], [], [], [], []
    context_chars, excerpt_sizes, whole_parent = [], [], 0
    model_quote_fields = 0
    for outcome in outcomes:
        model_quote_fields += int("quote" in outcome)
        if outcome.get("status") != "supported":
            continue
        rendered = rendered_by_scope.get(outcome["scope"], {})
        rendered_evidence = {r.get("evidence_id"): r for r in rendered.get("evidence", [])}
        for evidence_id in outcome.get("evidence_ids", []):
            model_quote_fields += int("quote" in rendered_evidence.get(evidence_id, {}))
            reference = rendered_evidence.get(evidence_id, {})
            registry_row = getattr(session, "registry", {}).get(evidence_id) if session is not None else None
            expected_reference = None
            if session is not None:
                try:
                    expected_reference = session.render_evidence_reference(evidence_id)
                except (AttributeError, KeyError, ValueError):
                    pass
                authentic = registry_row is not None and expected_reference is not None and reference == expected_reference
                authenticity.append(authentic)
                aligned_provenance = bool(registry_row) and all(
                    reference.get(key) == registry_row.get(key)
                    for key in ("evidence_id", "source", "page", "block_type", "block_index", "chunk_id"))
                provenance.append(aligned_provenance)
                bounds.append(bool(registry_row) and _bounds_valid(reference, registry_row, session))
            if registry_row:
                context_chars.append(reference.get("context_chars", 0))
                excerpt_sizes.append(len(reference.get("excerpt", "")))
                parent = session._parents[tuple(registry_row[k] for k in
                                                ("source", "page", "block_type", "block_index"))]["text"]
                whole_parent += int(reference.get("excerpt") == parent)
            review = candidate_reviews.get(evidence_id, {})
            label = review.get("label", "UNCERTAIN")
            expected_ids = oracle.get(outcome["scope"], {}).get("expected_evidence_ids")
            expected_id_match = (evidence_id in expected_ids
                                 if isinstance(expected_ids, list) else None)
            expected_id_correct = (expected_id_match is True if explicit_ids
                                   else expected_id_match is not False)
            evidence = state.get("looked_up_evidence", {}).get(evidence_id)
            alignment = review_finding({"scope": outcome["scope"], "quote": reference.get("excerpt", "")},
                                       evidence, oracle.get(outcome["scope"]))
            reviews.append(dict(scope=outcome["scope"], evidence_id=evidence_id, label=label,
                                expected_id_match=expected_id_match,
                                correct_expected_id=bool(label == "RELEVANT" and expected_id_correct),
                                field_alignment=alignment["field_alignment"],
                                unit_alignment=alignment["unit_alignment"],
                                condition_alignment=alignment["condition_alignment"],
                                scope_alignment=alignment["scope_alignment"]))
            refs.append(dict(scope=outcome["scope"], evidence_id=evidence_id, label=label,
                             expected_id_match=expected_id_match,
                             authentic=authentic if session is not None else None,
                             provenance=aligned_provenance if session is not None else None,
                             bounded=bounds[-1] if session is not None else None,
                             context_chars=reference.get("context_chars", 0),
                             excerpt_chars=len(reference.get("excerpt", ""))))

    correct_scope_ids = []
    for scope, review in oracle.items():
        if review.get("expected_available") is not True:
            continue
        expected_ids = review.get("expected_evidence_ids")
        if isinstance(expected_ids, list):
            required = {eid for eid in expected_ids if eid in observed_ids
                        and candidate_reviews.get(eid, {}).get("scope") == scope
                        and candidate_reviews.get(eid, {}).get("label") == "RELEVANT"}
        else:
            required = {eid for eid, row in candidate_reviews.items()
                        if row.get("scope") == scope and row.get("label") == "RELEVANT"}
        selected = set(outcome_by_scope.get(scope, {}).get("evidence_ids", []))
        correct_scope_ids.append(bool(required and selected & required))
    id_success = None if group != "CANDIDATE_AVAILABLE" else bool(correct_scope_ids and all(correct_scope_ids))

    unsupported_checks = []
    for scope, review in oracle.items():
        if review.get("expected_available") is not False:
            continue
        has_candidates = any(h.get("scopes") == [scope] and h.get("status") == "ok"
                             and bool(h.get("results")) for h in state.get("search_history", []))
        expected_status = "insufficient_evidence" if has_candidates else "no_candidates"
        actual = outcome_by_scope.get(scope, {})
        unsupported_checks.append(actual.get("status") == expected_status
                                  and not actual.get("evidence_ids"))

    fully_relevant = None
    if group == "CANDIDATE_AVAILABLE":
        all_refs_relevant = all(row["correct_expected_id"] for row in reviews)
        all_supported = all(outcome_by_scope.get(scope, {}).get("status") == "supported"
                            and any(row["scope"] == scope and row["label"] == "RELEVANT" for row in reviews)
                            for scope, item in oracle.items() if item.get("expected_available") is True)
        fully_relevant = bool(host_finish_accepted and id_success and all_refs_relevant and all_supported
                              and all(unsupported_checks))

    def alignment_count(key):
        assessed = [row[key] for row in reviews if row[key] is not None]
        return {"passed": sum(value is True for value in assessed), "assessed": len(assessed)}

    return dict(candidate_group=group,
                expected_id_mode=explicit_ids,
                evidence_id_selection_success=id_success,
                host_finish_accepted=host_finish_accepted,
                fully_relevant_task_success=fully_relevant,
                unsupported_side_correct={"passed": sum(unsupported_checks),
                                          "assessed": len(unsupported_checks)},
                accepted_unsupported_side_correct={
                    "passed": sum(host_finish_accepted and passed for passed in unsupported_checks),
                    "assessed": len(unsupported_checks)},
                scope_statuses=dict(Counter(row.get("status", "invalid") for row in outcomes)),
                selected_ids=len(refs), selected_relevant_ids=sum(r["label"] == "RELEVANT" for r in refs),
                correct_evidence_ids=sum(r["correct_expected_id"] for r in reviews),
                attempted_evidence_ids=len(refs),
                relevance_labels=dict(Counter(r["label"] for r in reviews)),
                citation_authenticity=({"passed": sum(authenticity), "assessed": len(authenticity)}
                                       if session is not None else
                                       {"passed": None, "assessed": 0, "status": "not_recomputed"}),
                citation_provenance=({"passed": sum(provenance), "assessed": len(provenance)}
                                     if session is not None else
                                     {"passed": None, "assessed": 0, "status": "not_recomputed"}),
                span_bounds=({"passed": sum(bounds), "assessed": len(bounds)}
                             if session is not None else
                             {"passed": None, "assessed": 0, "status": "not_recomputed"}),
                field_alignment=alignment_count("field_alignment"),
                unit_alignment=alignment_count("unit_alignment"),
                condition_alignment=alignment_count("condition_alignment"),
                scope_alignment=alignment_count("scope_alignment"),
                model_quote_fields=model_quote_fields,
                span_context=dict(references=len(refs), added_chars=sum(context_chars),
                                  mean_added_chars=(sum(context_chars) / len(context_chars) if context_chars else None),
                                  max_excerpt_chars=max(excerpt_sizes, default=0),
                                  whole_parent_excerpts=whole_parent if session is not None else None,
                                  citation_integrity_recomputed=session is not None,
                                  ambiguous_fallbacks=sum(r.get("span", {}).get("coordinate") == "child"
                                                          for row in rendered_by_scope.values()
                                                          for r in row.get("evidence", []))))
