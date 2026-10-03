import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class Agent02RunnerTests(unittest.TestCase):
    def test_missing_key_never_opens_inputs_or_calls_api(self):
        from evaluation.evaluate_agent02 import run_comparison
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'missing_key'):
                run_comparison('absent','absent','absent','absent')

    def test_recording_wrapper_preserves_rejected_finish_without_changing_host(self):
        from evaluation.evaluate_agent02 import RecordingSelector
        from src.agent.harness import AgentHarness
        from src.agent.tools import ToolResult
        class Session:
            def resolve_scopes(self,scopes): return tuple(scopes)
            def search_knowledge(self,query,scopes):
                s=scopes[0]
                return ToolResult('ok',[dict(evidence_id='ev_'+s,source=s,text='original')])
            def lookup_evidence(self,evidence_id):
                return ToolResult('ok',[dict(evidence_id=evidence_id,source=evidence_id[-1],text='original')])
        actions=iter([dict(action='SEARCH',query='q',scopes=[s]) for s in ['A','B']]+
                     [dict(action='LOOKUP',evidence_id='ev_'+s) for s in ['A','B']]+
                     [dict(action='FINISH',findings=[dict(scope='A',evidence_id='ev_A',quote='rewritten')])])
        selector=RecordingSelector(lambda _:next(actions))
        state=AgentHarness(Session(),selector).run('q',['A','B'])
        self.assertEqual(state.status,'invalid_action')
        self.assertEqual(state.findings,[])
        self.assertEqual(selector.actions[-1]['findings'][0]['quote'],'rewritten')

    def test_input_guard_detects_added_files_and_changed_observation(self):
        from evaluation.evaluate_agent02 import FrozenInputs
        from evaluation.evaluate_agent01 import inventory
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); protected=root/'input'; protected.mkdir()
            p=protected/'a'; p.write_text('a')
            task=root/'tasks.json'; task.write_text(json.dumps(dict(protected_hashes=inventory([p]),protected_roots=[str(protected)])))
            obs=root/'obs.json'; obs.write_text('{}')
            guard=FrozenInputs(task,obs)
            guard.verify()
            (protected/'b').write_text('b')
            with self.assertRaises(RuntimeError): guard.verify()

    def test_rmb_cost_uses_cache_rates_and_unknown_usage_is_not_zero(self):
        from evaluation.evaluate_agent02 import usage_summary
        price=dict(input_hit_rmb=0.02,input_miss_rmb=1.0,output_rmb=4.0)
        events=[dict(retry=False,error=None,usage=dict(prompt_tokens=100,completion_tokens=10,total_tokens=110,
                    prompt_cache_hit_tokens=80,prompt_cache_miss_tokens=20)),dict(retry=True,error='timeout',usage={})]
        summary=usage_summary(events,price)
        self.assertEqual(summary['requests'],2)
        self.assertEqual(summary['retries'],1)
        self.assertAlmostEqual(summary['metered_rmb'],(80*.02+20+10*4)/1e6)
        self.assertEqual(summary['unmetered_requests'],1)
