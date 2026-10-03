"""Controlled model/provider comparison over existing frozen real observations.

Only parsed actions and public states are saved privately; no keys or reasoning.
No model installation, retrieval recalculation or permissive quote correction.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
from urllib.request import urlopen
from uuid import uuid4

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_agent01 import inventory, SearchReplaySession
from evaluation.agent01_metrics import score_state
from evaluation.agent02_metrics import score_selection
from src.agent.harness import AgentHarness
from src.agent.io import APP_ROOT, load_corpus
from src.agent.selector import OllamaActionSelector
from src.agent.deepseek_selector import DeepSeekActionSelector, ApiCostBudget
from src.agent.tools import KnowledgeBaseSession

PRIVATE_ROOT=APP_ROOT/'outputs'/'agent0_2'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class FrozenInputs:
    def __init__(self,manifest,observations,*other_paths):
        self.paths=[Path(p).resolve() for p in (manifest,observations,*other_paths)]
        self.hashes={str(p):digest(p.read_bytes()) for p in self.paths}
        self.manifest=json.loads(self.paths[0].read_bytes())

    def verify(self):
        expected=self.manifest['protected_hashes']
        current={str(p.resolve()) for r in self.manifest['protected_roots'] for p in Path(r).rglob('*') if p.is_file()}
        if current!=set(expected) or inventory(current)!=expected or any(digest(p.read_bytes())!=self.hashes[str(p)] for p in self.paths):
            raise RuntimeError('frozen inputs/observations/review/pricing changed')


class RecordingSelector:
    """Evaluation-only parsed action capture, before host quote validation."""
    def __init__(self,selector):
        self.selector=selector
        self.actions=[]
        self.failures=[]

    def __call__(self,state):
        try:
            action=self.selector(state)
        except Exception as exc:
            # Client exceptions contain fixed safe codes; do not serialize repr.
            self.failures.append(str(exc) if str(exc).startswith('deepseek:') else type(exc).__name__)
            raise
        self.actions.append(deepcopy(action))
        return action


def usage_summary(events,pricing):
    result=dict(requests=len(events),retries=sum(e['retry'] for e in events),
                prompt_tokens=0,completion_tokens=0,total_tokens=0,metered_rmb=0.0,
                unmetered_requests=0,errors={})
    for event in events:
        usage=event['usage']
        if event['error']:
            result['errors'][event['error']]=result['errors'].get(event['error'],0)+1
        if not all(k in usage for k in ('prompt_tokens','completion_tokens','total_tokens')):
            result['unmetered_requests']+=1
            continue
        for key in ('prompt_tokens','completion_tokens','total_tokens'):
            result[key]+=usage[key]
        hit=usage.get('prompt_cache_hit_tokens',0)
        miss=usage.get('prompt_cache_miss_tokens',usage['prompt_tokens']-hit)
        if hit+miss!=usage['prompt_tokens']:
            hit,miss=0,usage['prompt_tokens']
        result['metered_rmb']+=(hit*pricing['input_hit_rmb']+miss*pricing['input_miss_rmb']+
                                usage['completion_tokens']*pricing['output_rmb'])/1e6
    return result


def run_comparison(manifest_path,observations_path,review_path,pricing_path):
    if not os.environ.get('DEEPSEEK_API_KEY','').strip():
        raise RuntimeError('deepseek:missing_key')
    guard=FrozenInputs(manifest_path,observations_path,review_path,pricing_path)
    guard.verify()
    try:
        with urlopen('http://localhost:11434/api/tags',timeout=10) as response:
            local_models=json.loads(response.read(64000))
        if 'qwen3:4b' not in {r['name'] for r in local_models['models']}:
            raise RuntimeError('local_model_missing')
    except Exception:
        raise RuntimeError('Ollama preflight failed; no API experiment started') from None
    manifest=guard.manifest
    observations=json.loads(Path(observations_path).read_bytes())
    review=json.loads(Path(review_path).read_bytes())
    pricing=json.loads(Path(pricing_path).read_bytes())
    if review['manifest_sha256']!=guard.hashes[str(Path(manifest_path).resolve())] or review['observations_sha256']!=guard.hashes[str(Path(observations_path).resolve())]:
        raise ValueError('candidate review belongs to different frozen inputs')
    if observations['manifest_sha256']!=review['manifest_sha256']:
        raise ValueError('observations belong to a different manifest')
    if len(manifest['tasks'])!=8 or set(review['tasks'])!={t['id'] for t in manifest['tasks']}:
        raise ValueError('matching frozen eight-task set required')
    if pricing.get('model')!='deepseek-flash' or pricing.get('currency')!='CNY' or pricing.get('source')!='https://api-docs.deepseek.com/zh-cn/quick_start/pricing':
        raise ValueError('verified official Flash RMB pricing required')
    budget=ApiCostBudget(input_rmb_per_million=pricing['peak_input_miss_rmb'],
                         output_rmb_per_million=pricing['peak_output_rmb'])
    api=DeepSeekActionSelector(budget=budget)
    output=dict(baseline='7f513a5',started_utc=datetime.now(timezone.utc).isoformat(),
                frozen_hashes=guard.hashes,pricing=pricing,arms={},api_events=[],
                protected_inputs_unchanged=False,forbidden_operations={},
                review_authority=review['authority'],max_tokens=512,
                contract='model_copied_quote',measurement='frozen_search_real_cache_lookup')
    sources=list((APP_ROOT/'src'/'agent').glob('*.py'))+[Path(__file__),APP_ROOT/'evaluation'/'agent02_metrics.py']
    output['inference_code_sha256']={str(p.relative_to(APP_ROOT)):digest(p.read_bytes()) for p in sources}
    directory=PRIVATE_ROOT/('run_'+uuid4().hex)
    if not directory.resolve().is_relative_to((APP_ROOT/'outputs').resolve()):
        raise ValueError('output escapes ignored directory')
    directory.mkdir(parents=True)
    target=directory/'results.json'
    def save():
        output['api_events']=deepcopy(api.events)
        output['api_usage']=usage_summary(api.events,pricing)
        output['api_usage']['conservative_upper_rmb']=budget.reserved_rmb
        target.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    def forbidden(name):
        def reject(*args,**kwargs):
            output['forbidden_operations'][name]=output['forbidden_operations'].get(name,0)+1
            raise RuntimeError('forbidden evaluation operation')
        return reject
    sessions={}
    try:
        with ExitStack() as stack:
            for name in ('src.mineru_loader._run_mineru','src.agent.tools.embed_chunks',
                         'src.agent.tools.load_model','src.agent.tools.load_reranker',
                         'src.agent.tools.retrieve','src.agent.tools.retrieve_hybrid',
                         'src.agent.tools.rerank_candidates','src.agent.tools.BM25Index'):
                stack.enter_context(patch(name,side_effect=forbidden(name)))
            for arm in ('A','B'):
                output['arms'][arm]={}
                for task in manifest['tasks']:
                    specs=tuple(task['documents'])
                    if specs not in sessions:
                        sessions[specs]=KnowledgeBaseSession(load_corpus(specs,parser='mineru',cache_root=manifest['cache_root']))
                    history=observations['arms']['A']['states'][task['id']]['search_history']
                    if (any(h['query']!=task['query'] or h['status'] not in ('ok','no_evidence') or len(h['scopes'])!=1 for h in history)
                            or {h['scopes'][0] for h in history}!=set(task['scopes'])):
                        raise ValueError('incomplete/mismatched successful observations')
                    session=SearchReplaySession(sessions[specs],history)
                    candidates={r['evidence_id'] for h in history for r in h['results']}
                    task_reviews=review['tasks'][task['id']]
                    if set(task_reviews)!=candidates:
                        raise ValueError('candidate review must cover every frozen ID')
                    for eid,row in task_reviews.items():
                        evidence=sessions[specs].lookup_evidence(eid).results[0]
                        if row['lookup_sha256']!=digest(json.dumps(evidence,sort_keys=True,ensure_ascii=False).encode()):
                            raise ValueError('review does not match original lookup evidence')
                    selector=RecordingSelector(OllamaActionSelector(max_tokens=512) if arm=='A' else api)
                    state=AgentHarness(session,selector).run(task['query'],task['scopes'])
                    saved=asdict(state)
                    output['arms'][arm][task['id']]=dict(state=saved,actions=selector.actions,selector_failures=selector.failures,
                        mechanical=score_state(saved,task['oracle']),
                        selection=score_selection(saved,selector.actions,task['oracle'] if task['scopes'] else None,task_reviews))
                    guard.verify()
                    save()
                    print(json.dumps(dict(arm=arm,task=task['id'],status=state.status,steps=state.step_count,
                         group=output['arms'][arm][task['id']]['selection']['group'],
                         id_success=output['arms'][arm][task['id']]['selection']['id_selection_success'])),flush=True)
                    if arm=='B' and any(f in ('deepseek:http_401','deepseek:http_403','deepseek:cost_limit','deepseek:missing_key') for f in selector.failures):
                        output['stop_reason']=selector.failures[-1]
                        break
    finally:
        try:
            guard.verify()
            output['protected_inputs_unchanged']=True
        finally:
            save()
            print('Private results: '+str(target),flush=True)
    if output['forbidden_operations']:
        raise RuntimeError('frozen retrieval boundary violated')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','observations','review','pricing'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    run_comparison(args.manifest,args.observations,args.review,args.pricing)


if __name__=='__main__':
    main()
