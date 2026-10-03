"""Observable task progress and mechanical coverage, never semantic relevance."""


TERMINAL_COVERAGE = {"LOOKED_UP_EVIDENCE", "NO_EVIDENCE", "INVALID_SCOPE"}


def snapshot(state):
    searched = set()
    empty = set()
    candidates = set()
    for h in state.search_history:
        scope = h["scopes"][0]
        status = h.get("status", "ok" if h["results"] else "no_evidence")
        if status in ("ok", "no_evidence"):
            searched.add(scope)
            if h["results"]:
                candidates.add(scope)
            elif status == "no_evidence":
                empty.add(scope)
    looked = {r["source"] for r in state.looked_up_evidence.values()}
    coverage = {}
    for scope in state.requested_scopes:
        if not isinstance(scope, str):
            continue
        if state.status == "invalid_scope" and scope not in state.resolved_scopes:
            coverage[scope] = "INVALID_SCOPE"
        elif scope in looked:
            coverage[scope] = "LOOKED_UP_EVIDENCE"
        elif scope in candidates:
            coverage[scope] = "CANDIDATES_OBSERVED"
        elif scope in empty:
            coverage[scope] = "NO_EVIDENCE"
        else:
            coverage[scope] = "UNSEARCHED"
    return dict(searched_scopes=sorted(searched), successful_lookup_scopes=sorted(looked),
                observed_evidence_ids=list(state.evidence_ids),
                looked_up_evidence_ids=list(state.looked_up_evidence), coverage=coverage,
                covered_scopes=[s for s, c in coverage.items() if c in TERMINAL_COVERAGE],
                remaining_scopes=[s for s, c in coverage.items() if c not in TERMINAL_COVERAGE])


def repeated_action(state, action):
    if action["action"] == "SEARCH":
        return any(h["query"] == action["query"] and h["scopes"] == action["scopes"]
                   and h.get("status") in ("ok", "no_evidence") for h in state.search_history)
    return action["action"] == "LOOKUP" and action["evidence_id"] in state.looked_up_evidence


def eligible_search_scopes(state):
    return [s for s in state.resolved_scopes if not repeated_action(
        state, dict(action="SEARCH", query=state.original_query, scopes=[s]))]


def eligible_lookup_ids(state):
    pending = [eid for eid in state.evidence_ids if eid not in state.looked_up_evidence]
    outstanding = {s for s, c in snapshot(state)["coverage"].items() if c == "CANDIDATES_OBSERVED"}
    if not outstanding:
        return pending
    sources = {r["evidence_id"]: r["source"] for h in state.search_history for r in h["results"]}
    return [eid for eid in pending if sources.get(eid) in outstanding]


def diagnostics(state, before, status):
    after = snapshot(state)
    new_ids = len(set(after["observed_evidence_ids"]) - set(before.get("observed_evidence_ids", [])))
    new_lookups = len(set(after["looked_up_evidence_ids"]) - set(before.get("looked_up_evidence_ids", [])))
    progress = (new_ids > 0 or new_lookups > 0
                or set(after["searched_scopes"]) != set(before.get("searched_scopes", []))
                or status == "ok" and state.status in ("finished", "incomplete", "clarify"))
    after.update(progress=bool(progress), new_evidence_ids=new_ids,
                 new_lookup_ids=new_lookups,
                 no_progress_actions=sum(t["tool_status"] == "no_progress" for t in state.trace)
                                     + int(status == "no_progress"))
    return after
