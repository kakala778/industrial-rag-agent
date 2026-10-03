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
