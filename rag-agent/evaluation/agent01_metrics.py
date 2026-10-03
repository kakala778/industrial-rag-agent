"""Offline observable metrics and frozen, PDF-reviewed alignment checks.

The lexical review is deliberately conservative, not an LLM judge or an
engineering semantic-equivalence evaluator. Its labels require offline review.
"""
from collections import Counter
import json
import hashlib
import re
import unicodedata

from src.agent.progress import snapshot


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace(r"\times", "x").replace("×", "x").replace("²", "2")
    text = re.sub(r"\\[a-z]+", "", text)
    return re.sub(r"[\s{}^_\\]", "", text)


def _aligned(quote, groups, *, unit_tokens=False):
    if not groups:
        return None
    text = normalize(quote)
    def matches(term):
        token = normalize(term)
        if re.fullmatch(r"\d+(?:\.\d+)?(?:x\d+(?:\.\d+)?)?", token):
            return re.search(r"(?<![\d.])" + re.escape(token) + r"(?![\d.])", text) is not None
        if unit_tokens and re.fullmatch(r"[a-z]+\d*", token):
            return re.search(r"(?<![a-z])" + re.escape(token) + r"(?![a-z\d])", text) is not None
        return token in text
    return all(any(matches(term) for term in alternatives) for alternatives in groups)


def review_finding(finding, evidence, oracle):
    if oracle is None or not oracle.get("reviewed") or oracle.get("expected_available") is None:
        return dict(label="UNCERTAIN", field_alignment=None, unit_alignment=None,
                    condition_alignment=None, scope_alignment=None)
    scope = evidence is not None and evidence.get("source") == finding["scope"]
    scope = scope and evidence.get("page") in oracle.get("pages", [])
    field = _aligned(finding["quote"], oracle.get("field_groups"))
    unit = _aligned(finding["quote"], oracle.get("unit_groups"), unit_tokens=True)
    condition = _aligned(finding["quote"], oracle.get("condition_groups"))
    if not oracle["expected_available"] or not scope or field is False:
        label = "IRRELEVANT"
    elif field is None:
        label = "UNCERTAIN"
    elif unit is False or condition is False:
        label = "PARTIAL"
    else:
        label = "RELEVANT"
    return dict(label=label, field_alignment=field, unit_alignment=unit,
                condition_alignment=condition, scope_alignment=bool(scope))


def finding_fingerprint(finding):
    return hashlib.sha256(json.dumps(finding, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def score_state(state, oracle=None, *, max_steps=12, max_search=4, max_lookup=6, finding_reviews=None):
    # Saved states use only public dataclass fields; never hidden reasoning.
    if isinstance(state, dict):
        from types import SimpleNamespace
        state = SimpleNamespace(**state)
    before = Counter()
    repeated = Counter()
    rejected = 0
    for t in state.trace:
        kind = t["action"]
        if t["tool_status"] == "no_progress":
            rejected += 1
        if kind in ("SEARCH", "LOOKUP") and t["tool_status"] in ("ok", "no_evidence"):
            signature = json.dumps([kind, t["tool_input"]], sort_keys=True)
            repeated[kind] += int(before[signature] > 0)
            before[signature] += 1
    progress = snapshot(state)
    findings = state.findings
    grounded = all(f["evidence_id"] in state.looked_up_evidence
                   and f["scope"] == state.looked_up_evidence[f["evidence_id"]]["source"]
                   and f["quote"] in state.looked_up_evidence[f["evidence_id"]]["text"] for f in findings)
    reviews = [dict(scope=f["scope"], evidence_id=f["evidence_id"],
                    **review_finding(f, state.looked_up_evidence.get(f["evidence_id"]),
                                     (oracle or {}).get(f["scope"]))) for f in findings]
    if finding_reviews is not None:
        if len(finding_reviews) != len(findings):
            raise ValueError("review must cover every final finding")
        for finding, review, row in zip(findings, finding_reviews, reviews):
            if (review.get("finding_sha256") != finding_fingerprint(finding)
                    or review.get("label") not in ("RELEVANT", "PARTIAL", "IRRELEVANT", "UNCERTAIN")):
                raise ValueError("stale or invalid finding review")
            for key in ("field_alignment", "unit_alignment", "condition_alignment", "scope_alignment"):
                value = review.get(key)
                if value is not None and type(value) is not bool:
                    raise ValueError("invalid alignment review")
                row[key] = value
            row["label"] = review["label"]
    mechanical = state.status in ("finished", "incomplete") and not progress["remaining_scopes"]
    success = None
    if oracle and all(o.get("reviewed") and o.get("expected_available") is not None for o in oracle.values()):
        required = {s for s, o in oracle.items() if o["expected_available"]}
        relevant = {r["scope"] for r in reviews if r["label"] == "RELEVANT"}
        success = bool(mechanical and grounded and required <= relevant
                       and all(r["label"] == "RELEVANT" for r in reviews))
    invalid_executed = sum(h["scopes"][0] not in state.resolved_scopes
                           or h["query"] != state.original_query for h in state.search_history)
    observed = {r["evidence_id"] for h in state.search_history for r in h["results"]}
    invalid_executed += sum(h["evidence_id"] not in observed for h in state.lookup_history)
    return dict(status=state.status, mechanical_completion=mechanical,
                review_method="offline_pdf_review" if finding_reviews is not None else "frozen_lexical_screen",
                relevant_task_success=success, coverage=progress["coverage"],
                distinct_successful_lookup_scopes=len(progress["successful_lookup_scopes"]),
                lookup_scope_coverage=(len(progress["successful_lookup_scopes"]) / len(state.resolved_scopes)
                                       if state.resolved_scopes else None),
                repeated_search_executed=repeated["SEARCH"], repeated_lookup_executed=repeated["LOOKUP"],
                guard_rejections=rejected, no_progress_actions=repeated["SEARCH"] + repeated["LOOKUP"] + rejected,
                grounding=grounded if findings else None,
                citation_grounding=(grounded and all(f["evidence_id"] in state.answer for f in findings)) if findings else None,
                relevance_review=reviews, invalid_executed_calls=invalid_executed,
                budget_violation=state.step_count > max_steps or state.search_calls > max_search or state.lookup_calls > max_lookup,
                steps=state.step_count, search_calls=state.search_calls, lookup_calls=state.lookup_calls)


def aggregate(records):
    rows = list(records.values())
    reviews = [r for row in rows for r in row["relevance_review"]]
    def fraction(key):
        eligible = [r[key] for r in rows if r[key] is not None]
        return {"passed": sum(bool(x) for x in eligible), "assessed": len(eligible)}
    coverage = [r["lookup_scope_coverage"] for r in rows if r["lookup_scope_coverage"] is not None]
    return dict(tasks=len(rows), mechanical_completion=sum(r["mechanical_completion"] for r in rows),
                review_methods=dict(Counter(r["review_method"] for r in rows)),
                relevant_task_success=fraction("relevant_task_success"),
                mean_lookup_scope_coverage=sum(coverage) / len(coverage) if coverage else None,
                grounding=fraction("grounding"), citation_grounding=fraction("citation_grounding"),
                relevance_labels=dict(Counter(r["label"] for r in reviews)),
                **{k: sum(r[k] for r in rows) for k in ("repeated_search_executed", "repeated_lookup_executed",
                    "guard_rejections", "no_progress_actions", "invalid_executed_calls", "budget_violation", "steps")})
