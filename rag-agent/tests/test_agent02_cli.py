import io
import os
import unittest
from unittest.mock import patch, Mock


class Agent02CliTests(unittest.TestCase):
    def test_optional_api_policy_missing_key_stops_before_ingestion(self):
        from src.agent_demo import main
        with patch.dict(os.environ,{},clear=True), patch('src.agent_demo.load_corpus') as load, patch('sys.stdout',new_callable=io.StringIO) as out:
            self.assertEqual(main(['--query','q','--document','A=a.md','--document','B=b.md','--scopes','A','B','--policy','deepseek']),1)
            load.assert_not_called()
            self.assertIn('missing_key',out.getvalue())

    def test_optional_api_policy_uses_existing_harness(self):
        from src.agent_demo import main
        from src.agent.state import AgentState
        policy=Mock()
        harness=Mock()
        harness.run.return_value=AgentState('q',['A','B'],status='incomplete')
        with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'test-only-token'}), patch('src.agent_demo.DeepSeekActionSelector',return_value=policy) as factory, patch('src.agent_demo.load_corpus',return_value={'A':[],'B':[]}), patch('src.agent_demo.AgentHarness',return_value=harness) as host, patch('sys.stdout',new_callable=io.StringIO):
            self.assertEqual(main(['--query','q','--document','A=a.md','--document','B=b.md','--scopes','A','B','--policy','deepseek']),2)
            factory.assert_called_once_with()
            self.assertIs(host.call_args.args[1],policy)
            harness.run.assert_called_once_with('q',['A','B'])

    def test_evidence_reference_policy_is_opt_in_on_same_harness(self):
        from src.agent_demo import main
        from src.agent.state import AgentState
        policy = Mock()
        harness = Mock()
        harness.run.return_value = AgentState("q", ["A", "B"], status="incomplete")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only-token"}), \
             patch("src.agent_demo.DeepSeekActionSelector", return_value=policy) as factory, \
             patch("src.agent_demo.load_corpus", return_value={"A": [], "B": []}), \
             patch("src.agent_demo.AgentHarness", return_value=harness) as host, \
             patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(main(["--query", "q", "--document", "A=a.md", "--document", "B=b.md",
                                   "--scopes", "A", "B", "--policy", "deepseek-reference"]), 2)
        factory.assert_called_once_with(action_contract="evidence_reference")
        self.assertIs(host.call_args.args[1], policy)
