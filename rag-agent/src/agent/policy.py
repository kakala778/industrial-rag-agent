"""Deterministic harness validation policy; no model needed."""


class DeterministicPolicy:
    def __call__(self, state):
        attempted = {h["scopes"][0] for h in state.search_history}
        for scope in state.resolved_scopes:
            if scope not in attempted:
                return {"action": "SEARCH", "query": state.original_query, "scopes": [scope]}
        first_ids = []
        child_text = {}
        for scope in state.resolved_scopes:
            candidates = [r for h in state.search_history for r in h["results"]
                          if r["source"] == scope]
            if candidates:
                first_ids.append(candidates[0]["evidence_id"])
                child_text[candidates[0]["evidence_id"]] = candidates[0]["text"]
        for evidence_id in first_ids:
            if evidence_id not in state.looked_up_evidence:
                return {"action": "LOOKUP", "evidence_id": evidence_id}
        return {"action": "FINISH", "findings": [
            {"scope": state.looked_up_evidence[eid]["source"], "evidence_id": eid,
             "quote": (state.looked_up_evidence[eid]["text"]
                       if len(state.looked_up_evidence[eid]["text"]) <= 1000
                       else child_text[eid])}
            for eid in first_ids]}


class DeterministicReferencePolicy:
    """Offline reference-contract policy; records looked-up source IDs only."""

    action_contract = "evidence_reference"

    def __call__(self, state):
        searched = {h["scopes"][0] for h in state.search_history
                    if h.get("status") in ("ok", "no_evidence")}
        for scope in state.resolved_scopes:
            if scope not in searched:
                return {"action": "SEARCH", "query": state.original_query, "scopes": [scope]}

        candidates_by_scope = {
            scope: [row["evidence_id"] for history in state.search_history
                    if history["scopes"][0] == scope and history.get("status") == "ok"
                    for row in history["results"]]
            for scope in state.resolved_scopes
        }
        for scope in state.resolved_scopes:
            candidate_ids = candidates_by_scope[scope]
            if not candidate_ids:
                continue
            looked_candidates = [eid for eid in candidate_ids
                                 if eid in state.looked_up_evidence
                                 and state.looked_up_evidence[eid].get("source") == scope]
            if not looked_candidates:
                for evidence_id in candidate_ids:
                    if evidence_id not in state.looked_up_evidence:
                        return {"action": "LOOKUP", "evidence_id": evidence_id}

        outcomes = []
        for scope in state.resolved_scopes:
            candidate_ids = candidates_by_scope[scope]
            if not candidate_ids:
                outcomes.append({"scope": scope, "status": "no_evidence_found"})
                continue
            looked_candidates = [eid for eid in candidate_ids
                                 if eid in state.looked_up_evidence
                                 and state.looked_up_evidence[eid].get("source") == scope]
            if looked_candidates:
                outcomes.append({"scope": scope, "status": "evidence_found",
                                 "evidence_ids": [looked_candidates[0]]})
            else:
                outcomes.append({"scope": scope, "status": "insufficient_scope"})
        return {"action": "FINISH", "outcomes": outcomes}
