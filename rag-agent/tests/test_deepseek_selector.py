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
