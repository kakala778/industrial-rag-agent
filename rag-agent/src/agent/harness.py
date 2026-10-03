"""Choose → validate → execute → observe → update → terminate."""

from copy import deepcopy
import json

from .actions import InvalidAction, finish_ready, validate_action
from .state import AgentState
from .tools import ToolResult
from .progress import diagnostics, eligible_lookup_ids, eligible_search_scopes, repeated_action, snapshot


class AgentHarness:
    def __init__(self, session, selector, *, max_steps=12, max_search_calls=4,
                 max_lookup_calls=6):
        for value in (max_steps, max_search_calls, max_lookup_calls):
            if type(value) is not int or value < 0:
                raise ValueError("budgets must be nonnegative integers")
        self.session, self.selector = session, selector
        self.action_contract = getattr(selector, "action_contract", "copied_quote")
        if self.action_contract not in ("copied_quote", "evidence_reference"):
            raise ValueError("unknown action contract")
        self.max_steps, self.max_search_calls = max_steps, max_search_calls
        self.max_lookup_calls = max_lookup_calls

    @staticmethod
    def _trace(state, action, inputs, status, summary):
        state.progress = diagnostics(state, state.progress, status)
        state.trace.append(dict(step=state.step_count, action=action,
                                tool_input=deepcopy(inputs), tool_status=status,
                                observation_summary=summary,
                                state_transition=f"running->{state.status}",
                                terminal_status=state.status if state.status != "running" else None,
                                progress=deepcopy(state.progress)))

    def run(self, query, scopes=None, *, clarification_required=""):
        if not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise ValueError("task query must be a nonempty bounded string")
        requested = list(scopes) if isinstance(scopes, (list, tuple)) else []
        state = AgentState(query, requested)
        if not isinstance(clarification_required, str) or len(clarification_required) > 500:
            raise ValueError("clarification requirement must be a bounded string")
        state.clarification_required = clarification_required.strip()
        if (len(requested) != 2 or any(not isinstance(s, str) for s in requested)
                or len(set(requested)) != 2):
            state.pending_clarification = "请明确指定两份不同的文档别名。"
            state.status, state.answer = "clarify", state.pending_clarification
            self._trace(state, "CLARIFY", {"question": state.answer}, "ok", "scope required")
            return state
        resolved = self.session.resolve_scopes(requested)
        if resolved is None:
            state.resolved_scopes = [s for s in requested if self.session.resolve_scopes([s]) is not None]
            state.status, state.answer = "invalid_scope", "指定的文档范围不存在。"
            self._trace(state, "PREFLIGHT", {"scopes": requested}, "invalid_scope", "scope rejected")
            return state
        state.resolved_scopes = requested.copy()
        state.progress = snapshot(state)
        if state.clarification_required:
            state.pending_clarification = state.clarification_required
            state.status, state.answer = "clarify", state.pending_clarification
            self._trace(state, "CLARIFY", {"question": state.answer}, "ok", "missing task constraint")
            return state
        while state.status == "running":
            if state.step_count >= self.max_steps:
                self._budget(state, "step limit")
                break
            state.step_count += 1
            state.remaining_budget = dict(steps=self.max_steps - state.step_count,
                                          search=self.max_search_calls - state.search_calls,
                                          lookup=self.max_lookup_calls - state.lookup_calls)
            if (not finish_ready(state)
                    and not (state.remaining_budget["search"] > 0 and eligible_search_scopes(state))
                    and not (state.remaining_budget["lookup"] > 0 and eligible_lookup_ids(state))):
                self._budget(state, "no eligible tool within remaining budgets")
                break
            action = None
            try:
                # Selectors receive a snapshot: they cannot bypass validation by mutation.
                action = validate_action(self.selector(deepcopy(state)),
                                         contract=self.action_contract)
                self._validate_state_action(state, action)
            except InvalidAction as exc:
                state.status, state.answer = "invalid_action", "动作无效，已停止，未执行该动作的工具。"
                state.errors.append(f"action:{exc}")
                kind = action["action"] if action else "INVALID"
                status = "invalid_evidence_id" if kind == "LOOKUP" and action["evidence_id"] not in state.evidence_ids else "invalid_action"
                inputs = {"evidence_id": action["evidence_id"]} if kind == "LOOKUP" else {}
                self._trace(state, kind, inputs, status, str(exc))
                break
            except TimeoutError:
                state.status, state.answer = "timeout", "动作选择服务超时。"
                state.errors.append("selector:timeout")
                self._trace(state, "SELECT", {}, "timeout", "selector timeout")
                break
            except Exception:
                state.status, state.answer = "tool_error", "动作选择服务失败。"
                state.errors.append("selector:error")
                self._trace(state, "SELECT", {}, "error", "selector failure")
                break
            kind = action["action"]
            inputs = {key: value for key, value in action.items() if key != "action"}
            if kind == "CLARIFY":
                state.pending_clarification = action["question"]
                state.status, state.answer = "clarify", action["question"]
                self._trace(state, kind, inputs, "ok", "clarification requested")
            elif kind == "FINISH":
                if self.action_contract == "copied_quote":
                    self._finish(state, action["findings"])
                    inputs = {"evidence_ids": [f["evidence_id"] for f in state.findings]}
                    summary = f"{len(state.findings)} grounded excerpts; {state.comparison}"
                else:
                    inputs = {"scope_statuses": [{"scope": row["scope"], "status": row["status"]}
                                                 for row in action["outcomes"]],
                              "evidence_ids": [eid for row in action["outcomes"]
                                               for eid in row.get("evidence_ids", [])]}
                    try:
                        self._finish_references(state, action["outcomes"])
                    except InvalidAction:
                        state.status, state.answer = "invalid_action", "宿主无法生成有效引用；未返回 findings。"
                        state.errors.append("reference:render_failed")
                        self._trace(state, kind, inputs, "invalid_action",
                                    {"terminal_status": state.status,
                                     "scope_statuses": inputs["scope_statuses"],
                                     "evidence_ids": inputs["evidence_ids"]})
                        break
                    summary = self._reference_trace_summary(state)
                self._trace(state, kind, inputs, "ok", summary)
            else:
                if repeated_action(state, action):
                    self._trace(state, kind, inputs, "no_progress", "unchanged successful action; tool not executed")
                else:
                    self._execute(state, kind, inputs)
        return state

    def _validate_state_action(self, state, action):
        kind = action["action"]
        if kind == "SEARCH":
            if action["scopes"][0] not in state.resolved_scopes or action["query"] != state.original_query:
                raise InvalidAction("search scope/query differs from task")
        elif kind == "LOOKUP":
            if action["evidence_id"] not in state.evidence_ids:
                raise InvalidAction("unobserved evidence")
            if not repeated_action(state, action) and action["evidence_id"] not in eligible_lookup_ids(state):
                raise InvalidAction("uncovered candidate scope must be looked up first")
        elif kind == "CLARIFY":
            if not state.clarification_required:
                raise InvalidAction("no missing user constraint; continue investigation or finish")
        elif kind == "FINISH":
            if not finish_ready(state):
                raise InvalidAction("unsearched or unlooked-up comparison scope")
            if self.action_contract == "copied_quote":
                seen = set()
                for finding in action["findings"]:
                    evidence = state.looked_up_evidence.get(finding["evidence_id"])
                    if (evidence is None or finding["scope"] not in state.resolved_scopes
                            or finding["scope"] != evidence["source"]
                            or finding["quote"] not in evidence["text"]
                            or finding["evidence_id"] in seen):
                        raise InvalidAction("unsupported finding")
                    seen.add(finding["evidence_id"])
            else:
                if ({row["scope"] for row in action["outcomes"]} != set(state.resolved_scopes)
                        or len(action["outcomes"]) != len(state.resolved_scopes)):
                    raise InvalidAction("scope outcomes must cover each resolved scope once")
                seen = set()
                candidates = {scope: [r for h in state.search_history
                                      if h["scopes"][0] == scope and h.get("status") == "ok"
                                      for r in h["results"]] for scope in state.resolved_scopes}
                for outcome in action["outcomes"]:
                    scope, status = outcome["scope"], outcome["status"]
                    if status == "no_candidates" and candidates[scope]:
                        raise InvalidAction("no_candidates contradicts observed candidates")
                    if status == "insufficient_evidence" and not candidates[scope]:
                        raise InvalidAction("insufficient_evidence requires observed candidates")
                    if status == "supported":
                        for evidence_id in outcome["evidence_ids"]:
                            evidence = state.looked_up_evidence.get(evidence_id)
                            if (evidence_id in seen or evidence_id not in state.evidence_ids
                                    or evidence is None or evidence.get("source") != scope
                                    or not self._owns_session_evidence(evidence_id)):
                                raise InvalidAction("evidence reference is not active, observed, looked up and in scope")
                            seen.add(evidence_id)

    def _budget(self, state, reason, kind="BUDGET", inputs=None):
        state.status, state.answer = "budget_exceeded", "执行预算已耗尽；调查未完成。"
        self._trace(state, kind, inputs or {}, "budget_exceeded", reason)

    def _execute(self, state, kind, inputs):
        if kind == "SEARCH":
            if state.search_calls >= self.max_search_calls:
                self._budget(state, "search limit", kind, inputs)
                return
            state.search_calls += 1
            tool = self.session.search_knowledge
        else:
            if state.lookup_calls >= self.max_lookup_calls:
                self._budget(state, "lookup limit", kind, inputs)
                return
            state.lookup_calls += 1
            tool = self.session.lookup_evidence
        try:
            result = tool(**inputs)
            if not isinstance(result, ToolResult):
                result = ToolResult("error", message="Malformed tool result")
        except TimeoutError:
            result = ToolResult("timeout")
        except Exception:
            result = ToolResult("error")
        history = dict(inputs, status=result.status, results=deepcopy(result.results))
        if kind == "SEARCH":
            state.search_history.append(history)
            if result.status == "ok":
                for row in result.results:
                    if row["evidence_id"] not in state.evidence_ids:
                        state.evidence_ids.append(row["evidence_id"])
        else:
            state.lookup_history.append(history)
            if result.status == "ok":
                for row in result.results:
                    state.looked_up_evidence[row["evidence_id"]] = deepcopy(row)
        if result.status not in ("ok", "no_evidence"):
            state.errors.append(f"{kind}:{result.status}")
            state.status = "timeout" if result.status == "timeout" else "tool_error"
            state.answer = f"工具执行失败（{result.status}）；不能把工具失败解释为没有答案。"
        self._trace(state, kind, inputs, result.status,
                    {"result_count": len(result.results),
                     "evidence_ids": [r.get("evidence_id") for r in result.results]})

    @staticmethod
    def _finish(state, findings):
        state.findings = deepcopy(findings)
        covered = {row["scope"] for row in findings}
        missing = [scope for scope in state.resolved_scopes if scope not in covered]
        state.status = "incomplete" if missing else "finished"
        if not missing:
            quote_sets = [{r["quote"] for r in findings if r["scope"] == scope}
                          for scope in state.resolved_scopes]
            state.comparison = "same_text" if quote_sets[0] == quote_sets[1] else "different_text"
        lines = []
        for row in findings:
            evidence = state.looked_up_evidence[row["evidence_id"]]
            lines.append(f'{row["scope"]}: {row["quote"]}\n'
                         f'[{row["evidence_id"]}; source={evidence["source"]}; '
                         f'page={evidence["page"]}; block={evidence["block_type"]}:'
                         f'{evidence["block_index"]}; chunk={evidence["chunk_id"]}]')
        if missing:
            lines.append("缺少可引用证据：" + ", ".join(missing) + "；无法完成比较。")
        else:
            lines.append("摘录文本相同。" if state.comparison == "same_text" else "摘录文本不同。")
            lines.append("这只是证据文本核对；参数含义、适用条件及语义一致性仍需复核。")
        state.answer = "\n\n".join(lines)

    def _session(self):
        session = self.session
        seen = set()
        while not hasattr(session, "registry") and hasattr(session, "session") and id(session) not in seen:
            seen.add(id(session))
            session = session.session
        return session

    def _owns_session_evidence(self, evidence_id):
        registry = getattr(self._session(), "registry", None)
        return isinstance(registry, dict) and evidence_id in registry

    def _render_reference(self, evidence_id):
        renderer = getattr(self._session(), "render_evidence_reference", None)
        if not callable(renderer):
            raise InvalidAction("session cannot render evidence references")
        try:
            return renderer(evidence_id)
        except (KeyError, ValueError) as exc:
            raise InvalidAction("host reference rendering failed") from exc

    def _finish_references(self, state, outcomes):
        findings = []
        for outcome in outcomes:
            row = {"scope": outcome["scope"], "status": outcome["status"]}
            if outcome["status"] == "supported":
                row["claim"] = outcome["claim"]
                row["evidence"] = [self._render_reference(evidence_id)
                                   for evidence_id in outcome["evidence_ids"]]
            findings.append(row)
        state.findings = findings
        state.status = "finished"
        state.comparison = "not_evaluated"
        state.answer = json.dumps({"comparison": state.comparison, "findings": findings},
                                  ensure_ascii=False, indent=2)

    @staticmethod
    def _reference_trace_summary(state):
        host_spans = []
        provenance = []
        for row in state.findings:
            for evidence in row.get("evidence", []):
                host_spans.append({"scope": row["scope"], "evidence_id": evidence["evidence_id"],
                                   "span": deepcopy(evidence["span"])})
                provenance.append({"scope": row["scope"], "evidence_id": evidence["evidence_id"],
                                   **{key: evidence[key] for key in
                                      ("source", "page", "block_type", "block_index", "chunk_id",
                                       "excerpt_sha256")}})
        return {"scope_statuses": [{"scope": row["scope"], "status": row["status"]}
                                   for row in state.findings],
                "host_spans": host_spans, "citation_provenance": provenance,
                "terminal_status": state.status}
