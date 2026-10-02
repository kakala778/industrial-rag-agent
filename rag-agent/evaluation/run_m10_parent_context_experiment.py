"""Frozen candidate replay; parent context changes BGE input only. No embedding."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from evaluation import run_m9_hybrid_retrieval_experiment as m9
from evaluation.evaluate_pdf_retrieval import evaluate_questions, load_questions, _evidence_rank_in_results, _summarize_cases
from src.lexical_retrieval import chunk_identity
from src.parent_context import ParentContextIndex, pair_tokens
from src.retrieval import split_documents
from src.reranker import load_reranker, rerank

OUTPUT_ROOT = PROJECT_ROOT / 'outputs/m10_parent_context'
PAIR_TOKEN_CAP = 768
PROTECTED = (*m9.CORE_FILES, 'src/lexical_retrieval.py', 'src/hybrid_retrieval.py',
             'src/pdf_rag_demo.py', 'evaluation/evaluate_pdf_retrieval.py')


def safe_output_root(path):
    path = Path(path).resolve()
    if not path.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError('Outputs must stay under outputs/m10_parent_context')
    if subprocess.run(['git','check-ignore','-q',str(path/'summary.json')],cwd=PROJECT_ROOT).returncode:
        raise ValueError('Output is not ignored')
    return path


def replay_pool(rows, chunks, aliases):
    pool=[]
    for row in rows:
        i=row['corpus_index']
        if isinstance(i,bool) or not isinstance(i,int) or not 0 <= i < len(chunks):
            raise ValueError('Invalid corpus ordinal')
        child=dict(chunks[i],corpus_index=i)
        metadata=child['metadata']
        expected={'source_id':aliases[metadata['source']], **{k:metadata.get(k) for k in ('page','block_type','block_index')}}
        if row['metadata']!=expected or row['chunk_id']!=child['chunk_id']:
            raise ValueError('Stored candidate identity mismatch')
        for key in ('score','rrf_score','dense_rank','bm25_rank','dense_score','bm25_score'):
            if key in row: child[key]=row[key]
        pool.append(child)
    if len(pool)>20 or len({chunk_identity(c) for c in pool})!=len(pool):
        raise ValueError('Frozen pool must contain at most 20 unique children')
    return pool


def rerank_original_children(query, children, expansions, model):
    if len(children)!=len(expansions):
        raise ValueError('Expansion count must equal candidate count')
    temporary=[dict(c,text=e['text']) for c,e in zip(children,expansions)]
    originals={chunk_identity(c):c for c in children}
    return [dict(originals[chunk_identity(c)],reranker_score=c['reranker_score'])
            for c in rerank(query,temporary,model,top_k=len(children))]


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def mechanism_passed(row):
    target=row['candidates'][13]
    return (target['mapping_status']=='EXPANDED' and target['context_type']=='table_boundary_rows'
            and bool(target['row_indices']) and target['parent_rank']<target['baseline_rank'])


def frozen_inputs(args):
    questions,_,documents,inventory=m9.load_frozen_inputs(m9.build_argument_parser().parse_args([]))
    old=_json(args.original_summary)
    independent=_json(args.independent_summary)
    manifest=_json(args.independent_manifest)
    if inventory!=old['configuration']['input_inventory'] or inventory!=manifest['original_input_inventory']:
        raise ValueError('Frozen PDF/QA/cache inventory changed')
    if manifest!=independent['frozen_manifest'] or m9._sha256_file(args.independent_dataset)!=manifest['qa_sha256']:
        raise ValueError('Independent frozen manifest/QA changed')
    # M9.3 changed citation formatting only; rag_demo is unused by this runner.
    for hashes in (old['configuration']['core_sha256'], manifest['core_sha256']):
        for name,digest in hashes.items():
            if name!='src/rag_demo.py' and m9._sha256_file(PROJECT_ROOT/name)!=digest:
                raise ValueError('Historical protected retrieval source changed: '+name)
    chunks=split_documents(documents)
    if len(chunks)!=old['configuration']['chunks'] or len(chunks)!=independent['corpus']['chunks']:
        raise ValueError('Corpus ordinal count changed')
    if len({chunk_identity(c) for c in chunks})!=len(chunks):
        raise ValueError('Corpus identities are not unique')
    cohorts={}
    for name,qs,summary in (('Q',questions,old),('I',load_questions(args.independent_dataset),independent)):
        aliases={source:f'S{i:03d}' for i,source in enumerate(sorted({m9._expected_source(q) for q in qs if m9._expected_source(q) is not None}),1)}
        cases=[]
        for q,row in zip(qs,summary['cases']):
            pool=replay_pool(row['modes']['hybrid']['candidates'],chunks,aliases)
            if _evidence_rank_in_results(q,pool)!=row['modes']['hybrid']['evidence_rank']:
                raise ValueError('Frozen evidence rank does not reproduce')
            cases.append(pool)
        if len(cases)!=len(qs): raise ValueError('QA/candidate count mismatch')
        cohorts[name]=(qs,cases,summary)
    return documents,chunks,cohorts,inventory


def run(args):
    documents,chunks,cohorts,inventory=frozen_inputs(args)
    before={f:m9._sha256_file(PROJECT_ROOT/f) for f in PROTECTED}
    index=ParentContextIndex(documents)
    q25=cohorts['Q'][1][24][13]
    parents=index.parents.get(tuple(q25['metadata'][k] for k in ('source','page','block_type','block_index')),[])
    if len(parents)!=1 or parents[0]['text'].count(q25['text'])!=1:
        raise ValueError('PARENT_MAPPING_INSUFFICIENT: Q25')
    root=safe_output_root(args.output_dir)
    run_dir=root/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ')
    run_dir.mkdir(parents=True,exist_ok=False)
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    print('Frozen candidates verified. Loading cached BGE only; no embedding/index rebuild.',flush=True)
    started=time.perf_counter(); model=load_reranker(); load_seconds=time.perf_counter()-started
    limits=[PAIR_TOKEN_CAP]
    for value in (getattr(model,'max_length',None),getattr(model.tokenizer,'model_max_length',None)):
        if isinstance(value,int) and 0<value<1000000: limits.append(value)
    cap=min(limits)
    records={}; rankings={'A':{},'B':{}}; private=[]

    def pair(cohort,qid):
        qs,pools,prior=cohorts[cohort]; query=qs[qid-1]['question']; pool=pools[qid-1]
        started=time.perf_counter()
        expansions=[index.expand(c,query,model.tokenizer,cap) for c in pool]
        expansion_seconds=time.perf_counter()-started
        if any(e['status']=='CHILD_OVER_BUDGET' for e in expansions):
            raise ValueError('Frozen child exceeds conservative budget; refusing altered baseline')
        started=time.perf_counter(); a=rerank(query,pool,model,top_k=len(pool)); at=time.perf_counter()-started
        started=time.perf_counter(); b=rerank_original_children(query,pool,expansions,model); bt=time.perf_counter()-started
        if [c['corpus_index'] for c in a]!=[c['corpus_index'] for c in sorted(
            prior['cases'][qid-1]['modes']['hybrid_reranker']['candidates'],key=lambda c:c['rank'])]:
            raise ValueError('Baseline BGE order did not reproduce; stop controlled inference')
        ar={chunk_identity(c):i for i,c in enumerate(a,1)}; br={chunk_identity(c):i for i,c in enumerate(b,1)}
        row={'case_id':f'{cohort}{qid:02d}','baseline_evidence_rank':_evidence_rank_in_results(qs[qid-1],a),
             'parent_evidence_rank':_evidence_rank_in_results(qs[qid-1],b),
             'expansion_seconds':expansion_seconds,'baseline_bge_seconds':at,'parent_bge_seconds':bt,
             'baseline_pair_tokens':[pair_tokens(model.tokenizer,query,c['text']) for c in pool],
             'parent_pair_tokens':[e['pair_tokens'] for e in expansions],
             'candidates':[{'corpus_index':c['corpus_index'],'hybrid_rank':i,'baseline_rank':ar[chunk_identity(c)],
                            'parent_rank':br[chunk_identity(c)],'context_type':e['context_type'],
                            'mapping_status':e['status'],'row_indices':e['row_indices']}
                           for i,(c,e) in enumerate(zip(pool,expansions),1)]}
        records[row['case_id']]=row
        rankings['A'][(cohort,qid)]=a; rankings['B'][(cohort,qid)]=b
        private.append({'case_id':row['case_id'],'children':pool,'expansions':expansions})
        print(row['case_id']+': A/B complete; evidence rank '+str(row['baseline_evidence_rank'])+' -> '+str(row['parent_evidence_rank']),flush=True)
        return row

    mechanism=pair('Q',25)
    target=mechanism['candidates'][13]
    improved=mechanism_passed(mechanism)
    summary={'experiment':'M10.1','status':'MECHANISM_PASSED' if improved else 'STOPPED_NO_Q25_RANK_GAIN',
             'pair_token_cap':cap,'reranker_load_seconds':load_seconds,'device':str(model.model.device),
             'mineru_calls':0,'embedding_calls':0,'candidate_replay':True,'q25':mechanism,'cohort_results':{},
             'protected_core_before':before,'original_inventory':inventory}
    if improved:
        for cohort,(qs,pools,prior) in cohorts.items():
            for qid in range(1,len(qs)+1):
                if (cohort,qid) not in rankings['A']: pair(cohort,qid)
            reports={arm:evaluate_questions(qs,None,chunks,[],documents=documents,diagnostic_k=20,
                candidate_provider=lambda query,limit,a=arm,c=cohort:rankings[a][(c,next(i for i,q in enumerate(qs,1) if q['question']==query))][:limit])
                for arm in ('A','B')}
            metrics={arm:{name:{'hits':report[h],'total':report[t]} for name,(h,t) in m9.METRICS.items()} for arm,report in reports.items()}
            for name,values in metrics['A'].items():
                if values!=prior['metrics']['hybrid_reranker'][name]: raise ValueError('Frozen baseline metrics changed')
            transitions={name:m9._transition(reports,lambda c,f=field:c[f] is True,'A','B')
                         for name,field in (('page','top3_page_hit'),('evidence','normalized_keyword_hit'),('strict','keyword_hit'))}
            summary['cohort_results'][cohort]={'metrics':metrics,'transitions':transitions}
            if cohort=='Q':
                summary['q17_sensitivity']={arm:{name:{'hits':r[h],'total':r[t]} for name,(h,t) in m9.METRICS.items()}
                    for arm,report in reports.items() for r in [_summarize_cases([c for i,c in enumerate(report['cases'],1) if i!=17])]}
    _,_,_,after_inventory=frozen_inputs(args)
    if inventory!=after_inventory or before!={f:m9._sha256_file(PROJECT_ROOT/f) for f in PROTECTED}:
        raise ValueError('Protected inputs changed during run')
    summary['protected_inputs_unchanged']=True; summary['cases']=list(records.values())
    m9._write_json(run_dir/'summary.json',summary)
    m9._write_json(run_dir/'private_contexts.local.json',private)
    print('Decision gate: '+summary['status'],flush=True)
    print('Output: '+str(run_dir.relative_to(PROJECT_ROOT)),flush=True)
    return summary


def build_argument_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-summary',type=Path,default=PROJECT_ROOT/'outputs/m9_hybrid_retrieval/run_20261002T093708Z/summary.json')
    parser.add_argument('--independent-summary',type=Path,default=PROJECT_ROOT/'outputs/m9_independent_validation/run_20261002T110427Z/summary.json')
    parser.add_argument('--independent-manifest',type=Path,default=PROJECT_ROOT/'outputs/m9_independent_validation/frozen_manifest.json')
    parser.add_argument('--independent-dataset',type=Path,default=PROJECT_ROOT/'outputs/m9_independent_validation/m9_independent_qa.local.json')
    parser.add_argument('--output-dir',type=Path,default=OUTPUT_ROOT)
    return parser


if __name__=='__main__':
    run(build_argument_parser().parse_args())
