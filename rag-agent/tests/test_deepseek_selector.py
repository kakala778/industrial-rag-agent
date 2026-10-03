import io
import json
import os
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from src.agent.state import AgentState
from src.agent.actions import InvalidAction


class DeepSeekSelectorTests(unittest.TestCase):
    def state(self):
        return AgentState('pressure', ['A','B'], ['A','B'])

    def test_missing_key_fails_before_http(self):
        from src.agent.deepseek_selector import DeepSeekActionSelector
        with patch.dict(os.environ, {}, clear=True), patch('src.agent.deepseek_selector.urlopen') as http:
            with self.assertRaisesRegex(RuntimeError, 'missing_key'):
                DeepSeekActionSelector()(self.state())
            http.assert_not_called()

    def test_common_messages_nonthinking_short_json_and_sanitized_usage(self):
        from src.agent.deepseek_selector import DeepSeekActionSelector
        from src.agent.selector import OllamaActionSelector
        received = []
        action = dict(action='SEARCH', query='pressure', scopes=['A'])
        def transport(request, timeout):
            received.append(json.loads(request.data))
            return io.BytesIO(json.dumps(dict(choices=[dict(message=dict(content=json.dumps(action), reasoning_content='discard'))],
                                               usage=dict(prompt_tokens=100, completion_tokens=20, total_tokens=120,
                                                          prompt_cache_hit_tokens=80, prompt_cache_miss_tokens=20))).encode())
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY':'test-only-token'}), patch('src.agent.deepseek_selector.urlopen', side_effect=transport):
            selector = DeepSeekActionSelector()
            self.assertEqual(selector(self.state()), action)
        def ollama_transport(request, timeout):
            received.append(json.loads(request.data))
            return io.BytesIO(json.dumps(dict(message=dict(content=json.dumps(action)))).encode())
        with patch('src.agent.selector.urlopen', side_effect=ollama_transport):
            OllamaActionSelector(max_tokens=512)(self.state())
        self.assertEqual(received[0]['messages'], received[1]['messages'])
        self.assertEqual(received[0]['thinking'], {'type':'disabled'})
        self.assertEqual(received[0]['max_tokens'],512)
        self.assertNotIn('tools',received[0])
        self.assertEqual(selector.events[0]['usage']['total_tokens'],120)
        self.assertNotIn('test-only-token',json.dumps(selector.events))
        self.assertNotIn('discard',json.dumps(selector.events))

    def test_auth_error_has_no_retry_or_server_body_leak(self):
        from src.agent.deepseek_selector import DeepSeekActionSelector
        error = HTTPError('https://api.deepseek.com/chat/completions',401,'private details',{},io.BytesIO(b'secret body'))
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY':'test-only-token'}), patch('src.agent.deepseek_selector.urlopen',side_effect=error) as http:
            selector = DeepSeekActionSelector()
            with self.assertRaisesRegex(RuntimeError, 'http_401') as caught:
                selector(self.state())
            self.assertEqual(http.call_count,1)
            self.assertNotIn('private',str(caught.exception))
            self.assertEqual(selector.events[0]['error'],'http_401')

    def test_transient_retry_is_bounded(self):
        from src.agent.deepseek_selector import DeepSeekActionSelector
        error = HTTPError('https://api.deepseek.com/chat/completions',429,'details',{},None)
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY':'test-only-token'}), patch('src.agent.deepseek_selector.urlopen',side_effect=error) as http:
            selector=DeepSeekActionSelector()
            with self.assertRaisesRegex(RuntimeError,'http_429'): selector(self.state())
            self.assertEqual(http.call_count,2)
            self.assertEqual(len(selector.events),2)
            self.assertTrue(selector.events[1]['retry'])

    def test_json_mode_cannot_bypass_dynamic_eligibility(self):
        from src.agent.deepseek_selector import DeepSeekActionSelector
        response=dict(choices=[dict(message=dict(content='{"action":"FINISH","findings":[]}'))],usage={})
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY':'test-only-token'}), patch('src.agent.deepseek_selector.urlopen',return_value=io.BytesIO(json.dumps(response).encode())):
            with self.assertRaises(InvalidAction): DeepSeekActionSelector()(self.state())

    def test_timeout_and_malformed_json_are_explicit_and_bounded(self):
        from src.agent.deepseek_selector import DeepSeekActionSelector
        for outcome, expected in ((TimeoutError(),TimeoutError),(io.BytesIO(b'not json'),InvalidAction)):
            with patch.dict(os.environ, {'DEEPSEEK_API_KEY':'test-only-token'}), patch('src.agent.deepseek_selector.urlopen') as http:
                if isinstance(outcome,Exception): http.side_effect=outcome
                else: http.return_value=outcome
                selector=DeepSeekActionSelector(retries=0)
                with self.assertRaises(expected): selector(self.state())
                self.assertEqual(http.call_count,1)

    def test_cost_reservation_stops_before_request(self):
        from src.agent.deepseek_selector import DeepSeekActionSelector, ApiCostBudget
        budget=ApiCostBudget(input_rmb_per_million=2, output_rmb_per_million=8, stop_rmb=0.00001)
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY':'test-only-token'}), patch('src.agent.deepseek_selector.urlopen') as http:
            with self.assertRaisesRegex(RuntimeError,'cost_limit'): DeepSeekActionSelector(budget=budget)(self.state())
            http.assert_not_called()

    def test_evidence_reference_contract_uses_id_status_schema_and_no_quotes(self):
        from src.agent.deepseek_selector import DeepSeekActionSelector
        state = self.state()
        state.search_history = [
            {"scopes": ["A"], "query": "pressure", "status": "ok",
             "results": [{"evidence_id": "ev_a", "source": "A"}]},
            {"scopes": ["B"], "query": "pressure", "status": "ok",
             "results": [{"evidence_id": "ev_b", "source": "B"}]},
        ]
        state.evidence_ids = ["ev_a", "ev_b"]
        state.looked_up_evidence = {"ev_a": {"source": "A"}, "ev_b": {"source": "B"}}
        state.remaining_budget = {"steps": 1, "search": 2, "lookup": 4}
        action = {"action": "FINISH", "outcomes": [
            {"scope": "A", "status": "evidence_found",
             "evidence_ids": ["ev_a"]},
            {"scope": "B", "status": "insufficient_scope"},
        ]}
        payloads = []
        def transport(request, timeout):
            payloads.append(json.loads(request.data))
            body = {"choices": [{"message": {"content": json.dumps(action)}}], "usage": {}}
            return io.BytesIO(json.dumps(body).encode())
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-token"}), \
             patch("src.agent.deepseek_selector.urlopen", side_effect=transport):
            selector = DeepSeekActionSelector(action_contract="evidence_reference")
            self.assertEqual(selector(state), action)
        prompt = payloads[0]["messages"][0]["content"]
        normalized_prompt = " ".join(prompt.split())
        schema = json.loads(payloads[0]["messages"][1]["content"])["action_schema"]
        self.assertIn("no_evidence_found", prompt)
        self.assertIn("insufficient_scope", prompt)
        self.assertIn("evidence_found", prompt)
        self.assertIn("citation", prompt.lower())
        self.assertIn("does not establish relevance", normalized_prompt)
        self.assertIn("terminal CLARIFY", normalized_prompt)
        self.assertIn("engineering verdict", prompt)
        self.assertNotIn("claim", action["outcomes"][0])
        finish = next(row for row in schema["oneOf"]
                      if row["properties"]["action"]["const"] == "FINISH")
        self.assertIn("outcomes", finish["properties"])
        self.assertNotIn("findings", finish["properties"])
        outcomes = finish["properties"]["outcomes"]["items"]["oneOf"]
        self.assertEqual({row["properties"]["scope"]["const"] for row in outcomes}, {"A", "B"})

    def test_reference_schema_is_dynamic_and_statuses_follow_host_coverage(self):
        from src.agent.selector import action_schema_for_state
        from src.agent.state import AgentState

        for scopes in (["A", "B"], ["A", "B", "C"], ["A", "B", "C", "D"]):
            state = AgentState("pressure", list(scopes), list(scopes))
            state.search_history = [
                {"scopes": ["A"], "query": "pressure", "status": "ok",
                 "results": [{"evidence_id": "ev_a", "source": "A"}]},
                {"scopes": ["B"], "query": "pressure", "status": "no_evidence", "results": []},
                *([{"scopes": [scope], "query": "pressure", "status": "ok",
                    "results": [{"evidence_id": f"ev_{scope.lower()}", "source": scope}]} for scope in scopes[2:]])
            ]
            state.evidence_ids = ["ev_a", *(f"ev_{scope.lower()}" for scope in scopes[2:])]
            state.looked_up_evidence = {eid: {"source": source}
                                        for source, eid in [("A", "ev_a"),
                                                            *((scope, f"ev_{scope.lower()}") for scope in scopes[2:])]}
            schema = action_schema_for_state(state, contract="evidence_reference")
            finish = next(row for row in schema["oneOf"]
                          if row["properties"]["action"]["const"] == "FINISH")
            self.assertEqual(finish["properties"]["outcomes"]["minItems"], len(scopes))
            self.assertEqual(finish["properties"]["outcomes"]["maxItems"], len(scopes))
            variants = finish["properties"]["outcomes"]["items"]["oneOf"]
            aliases = {row["properties"]["scope"]["const"] for row in variants}
            self.assertEqual(aliases, set(scopes))
            statuses = {scope: set() for scope in scopes}
            for row in variants:
                statuses[row["properties"]["scope"]["const"]].add(
                    row["properties"]["status"]["const"])
            self.assertEqual(statuses["A"], {"evidence_found", "insufficient_scope"})
            self.assertEqual(statuses["B"], {"no_evidence_found"})
            for scope in scopes[2:]:
                self.assertEqual(statuses[scope], {"evidence_found", "insufficient_scope"})
            evidence_variant = next(row for row in variants
                                    if row["properties"]["scope"]["const"] == "A"
                                    and row["properties"]["status"]["const"] == "evidence_found")
            self.assertEqual(evidence_variant["properties"]["evidence_ids"]["items"]["enum"], ["ev_a"])

    def test_reference_finish_stays_hidden_until_candidate_scope_is_looked_up(self):
        from src.agent.selector import action_schema_for_state
        state = self.state()
        state.search_history = [
            {"scopes": ["A"], "query": "pressure", "status": "ok",
             "results": [{"evidence_id": "ev_a", "source": "A"}]},
            {"scopes": ["B"], "query": "pressure", "status": "no_evidence", "results": []},
        ]
        state.evidence_ids = ["ev_a"]
        self.assertNotIn("FINISH", [row["properties"]["action"]["const"] for row in
                                   action_schema_for_state(state, contract="evidence_reference")["oneOf"]])
        state.looked_up_evidence["ev_a"] = {"source": "A"}
        schema = action_schema_for_state(state, contract="evidence_reference")
        finish = next(row for row in schema["oneOf"] if row["properties"]["action"]["const"] == "FINISH")
        variants = finish["properties"]["outcomes"]["items"]["oneOf"]
        status_by_scope = {}
        for row in variants:
            status_by_scope.setdefault(row["properties"]["scope"]["const"], set()).add(
                row["properties"]["status"]["const"])
        self.assertEqual(status_by_scope, {"A": {"evidence_found", "insufficient_scope"},
                                           "B": {"no_evidence_found"}})

    def test_reference_selector_rejects_status_contradicting_scope_candidates(self):
        from src.agent.selector import parse_selector_action
        state = self.state()
        state.search_history = [
            {"scopes": ["A"], "query": "pressure", "status": "ok",
             "results": [{"evidence_id": "ev_a", "source": "A"}]},
            {"scopes": ["B"], "query": "pressure", "status": "no_evidence", "results": []},
        ]
        state.evidence_ids = ["ev_a"]
        state.looked_up_evidence = {"ev_a": {"source": "A"}}
        invalid = {"action": "FINISH", "outcomes": [
            {"scope": "A", "status": "no_evidence_found"},
            {"scope": "B", "status": "no_evidence_found"},
        ]}
        with self.assertRaises(InvalidAction):
            parse_selector_action(invalid, state, contract="evidence_reference")

    def test_reference_clarify_is_selectable_but_copied_quote_gate_is_unchanged(self):
        from src.agent.selector import action_schema_for_state
        state = self.state()
        copied = action_schema_for_state(state)
        reference = action_schema_for_state(state, contract="evidence_reference")
        self.assertNotIn("CLARIFY", [row["properties"]["action"]["const"] for row in copied["oneOf"]])
        self.assertIn("CLARIFY", [row["properties"]["action"]["const"] for row in reference["oneOf"]])

    def test_deterministic_reference_policy_searches_and_looks_up_each_candidate_scope(self):
        from src.agent.policy import DeterministicReferencePolicy
        state = self.state()
        policy = DeterministicReferencePolicy()
        self.assertEqual(policy(state), {"action": "SEARCH", "query": "pressure", "scopes": ["A"]})
        state.search_history.append({"scopes": ["A"], "query": "pressure", "status": "ok",
                                     "results": [{"evidence_id": "ev_a", "source": "A"}]})
        state.evidence_ids = ["ev_a"]
        self.assertEqual(policy(state), {"action": "SEARCH", "query": "pressure", "scopes": ["B"]})
        state.search_history.append({"scopes": ["B"], "query": "pressure", "status": "no_evidence", "results": []})
        self.assertEqual(policy(state), {"action": "LOOKUP", "evidence_id": "ev_a"})
        state.looked_up_evidence["ev_a"] = {"source": "A", "text": "pressure value"}
        self.assertEqual(policy(state), {"action": "FINISH", "outcomes": [
            {"scope": "A", "status": "evidence_found", "evidence_ids": ["ev_a"]},
            {"scope": "B", "status": "no_evidence_found"},
        ]})
