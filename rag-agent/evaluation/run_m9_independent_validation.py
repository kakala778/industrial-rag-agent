"""Frozen M9.2 independent five-arm validation and cache-only ranking audit."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from evaluation import run_m9_hybrid_retrieval_experiment as m9
from evaluation.evaluate_pdf_retrieval import (evaluate_questions, load_questions,
    _expected_source, _evidence_rank_in_results, _matches_case_scope)
from src.lexical_retrieval import BM25Index, rrf_fuse, chunk_identity
from src.retrieval import split_documents, load_model, embed_chunks, retrieve
from src.reranker import load_reranker, rerank

MODES = m9.MODES
ROOT = PROJECT_ROOT / "outputs" / "m9_independent_validation"
AUDIT_ROOT = PROJECT_ROOT / "outputs" / "m9_ranking_audit"
PROTECTED = m9.CORE_FILES + ("src/lexical_retrieval.py", "evaluation/evaluate_pdf_retrieval.py",
                           "evaluation/run_m9_hybrid_retrieval_experiment.py")


def frozen_configuration():
    config = {"dense_limit": m9.CANDIDATE_K, "bm25_limit": m9.CANDIDATE_K,
              "hybrid_limit": m9.CANDIDATE_K, "top_k": m9.TOP_K,
              "bm25_k1": m9.BM25_K1, "bm25_b": m9.BM25_B, "rrf_k": m9.RRF_K,
              "max_chars": 500, "overlap": 80, "fusion_weights": [1,1],
              "representation": "structured", "embedding_model": m9.MODEL_NAME,
              "reranker_model": m9.DEFAULT_RERANKER_MODEL}
    if (m9.CANDIDATE_K,m9.TOP_K,m9.BM25_K1,m9.BM25_B,m9.RRF_K) != (20,3,1.2,.75,60):
        raise ValueError("M9.1 retrieval parameters changed")
    return config


def safe_root(path):
    path = Path(path).resolve()
    if not any(path.is_relative_to(root.resolve()) for root in (ROOT,AUDIT_ROOT)):
        raise ValueError("Output must remain in an ignored M9.2 directory")
    result = subprocess.run(["git","check-ignore","-q",str(path / "probe.json")],
                            cwd=PROJECT_ROOT,capture_output=True)
    if result.returncode:
        raise ValueError("Private output is not Git ignored")
    return path


def _norm(text):
    return "".join(unicodedata.normalize("NFKC",text).casefold().split())


def build_manifest(questions,original,qa_hash,inventory,core_hashes):
    old_queries={_norm(q["question"]) for q in original}
    old_pages={(_expected_source(q),q.get("expected_page")) for q in original}
    if len({_norm(q["question"]) for q in questions}) != len(questions):
        raise ValueError("Independent queries must be unique")
    for q in questions:
        if (_norm(q["question"]) in old_queries or
            (_expected_source(q),q.get("expected_page")) in old_pages):
            raise ValueError("Independent questions must use new facts/pages, not M6 copies")
        if (q.get("original_pdf_page_visual_reviewed") is not True or q.get("review_required") or
            not q.get("answer") or not q.get("expected_keywords") or not q.get("expected_page") or
            not _expected_source(q)):
            raise ValueError("Every independent question needs reliable PDF-reviewed ground truth")
    aliases={source:f"S{i:03d}" for i,source in enumerate(sorted({_expected_source(q) for q in questions}),1)}
    return {"question_count":len(questions),"qa_sha256":qa_hash,
            "category_distribution":dict(sorted(Counter(q["category"] for q in questions).items())),
            "source_distribution":dict(sorted(Counter(aliases[_expected_source(q)] for q in questions).items())),
            "configuration":frozen_configuration(),"original_input_inventory":inventory,"core_sha256":core_hashes}


def require_manifest(expected,observed):
    if expected != observed:
        raise ValueError("Frozen QA, corpus or M9.1 configuration changed; refusing evaluation")


def load_inputs(args):
    old_args=m9.build_argument_parser().parse_args([])
    original,old_analysis,documents,inventory=m9.load_frozen_inputs(old_args)
    questions=load_questions(args.dataset)
    hashes={name:m9._sha256_file(PROJECT_ROOT/name) for name in PROTECTED}
    known_sources={d["metadata"]["source"] for d in documents}
    if any(_expected_source(q) not in known_sources for q in questions):
        raise ValueError("Independent source is outside the frozen eight PDFs")
    manifest=build_manifest(questions,original,m9._sha256_file(args.dataset),inventory,hashes)
    return questions,original,documents,manifest


def freeze(args):
    safe_root(args.dataset.parent); safe_root(args.manifest.parent)
    _,_,_,manifest=load_inputs(args)
    if args.manifest.exists():
        raise ValueError("Manifest already exists; refusing overwrite")
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    m9._write_json(args.manifest,manifest)
    print(json.dumps(manifest,indent=2),flush=True)
    return manifest


def build_summary(questions,reports,documents,chunks):
    if any(len(r["cases"])!=len(questions) for r in reports.values()):
        raise ValueError("All arms must preserve independent QA order")
    sources={s:f"S{i:03d}" for i,s in enumerate(sorted({_expected_source(q) for q in questions}),1)}
    metrics={mode:{name:{"hits":report[h],"total":report[t]} for name,(h,t) in m9.METRICS.items()}
             for mode,report in reports.items()}
    rows=[]
    for i,q in enumerate(questions,1):
        maps={mode:{chunk_identity(c):j for j,c in enumerate(reports[mode]["cases"][i-1]["diagnostic_results"],1)}
              for mode in ("dense","bm25")}
        modes={}
        for mode,report in reports.items():
            case=report["cases"][i-1]; candidates=case["diagnostic_results"][:20]
            modes[mode]={"evidence_rank":_evidence_rank_in_results(case,candidates),
                **{name:case[field] for name,field in {
                "top1_source_hit":"top1_source_hit","top3_source_hit":"top3_source_hit",
                "top1_page_hit":"top1_page_hit","top3_page_hit":"top3_page_hit",
                "normalized_evidence_hit":"normalized_keyword_hit","strict_evidence_hit":"keyword_hit"}.items()},
                "candidates":[m9._safe_candidate(c,j,sources,q["question"],maps["dense"],maps["bm25"])
                              for j,c in enumerate(candidates,1)]}
        rows.append({"question_id":i,"qa_category":q["category"],"known_gt_uncertain":bool(q.get("review_required",False)),
            "query_features":m9.query_features(q["question"]),"evidence_in_chunks":m9._evidence_available(q,chunks),"modes":modes})
    for mode in reports:
        available=[r for r in rows if r["evidence_in_chunks"]]
        metrics[mode]["recall_at20"]={"hits":sum(r["modes"][mode]["evidence_rank"] is not None for r in rows),"total":len(rows)}
        metrics[mode]["recall_at20_evidence_available"]={"hits":sum(r["modes"][mode]["evidence_rank"] is not None for r in available),"total":len(available)}
    predicates={"normalized_evidence":lambda c:c.get("normalized_keyword_hit") is True,
                "top3_page":lambda c:c.get("top3_page_hit") is True,"joint_success":m9._joint_success}
    transitions={name:{direction:m9._transition(reports,predicate,before,after)
        for direction,before,after in (("dense_bge_to_hybrid_bge","dense_reranker","hybrid_reranker"),
                                      ("hybrid_to_hybrid_bge","hybrid","hybrid_reranker"),
                                      ("dense_to_hybrid","dense","hybrid"))}
        for name,predicate in predicates.items()}
    matrix={}
    for dense,bm25 in ((True,False),(False,True),(True,True),(False,False)):
        group=[r for r in rows if r["modes"]["dense"]["normalized_evidence_hit"]==dense and r["modes"]["bm25"]["normalized_evidence_hit"]==bm25]
        matrix[f"dense_{'yes' if dense else 'no'}_bm25_{'yes' if bm25 else 'no'}"]={"count":len(group),
            "question_ids":[r["question_id"] for r in group],
            "hybrid_success_ids":[r["question_id"] for r in group if r["modes"]["hybrid"]["normalized_evidence_hit"]]}
    return {"experiment":"M9.2 Independent Validation","question_count":len(questions),"metrics":metrics,
            "complementarity":matrix,"transitions":transitions,"cases":rows}


def duplicate_diagnostics(candidates):
    texts=[_norm(c.get("text","")) for c in candidates]
    exact=[]; near=[]; same_block=[]
    for i in range(len(candidates)):
        for j in range(i+1,len(candidates)):
            if texts[i] and texts[i]==texts[j]: exact.append([i+1,j+1])
            elif texts[i] and texts[j] and SequenceMatcher(None,texts[i],texts[j],autojunk=False).ratio()>=.90:
                near.append([i+1,j+1])
            a,b=candidates[i].get("metadata",{}),candidates[j].get("metadata",{})
            if all(a.get(k)==b.get(k) for k in ("source","page","block_type","block_index")):
                same_block.append([i+1,j+1])
    return {"structural_duplicate_count":len(candidates)-len({chunk_identity(c) for c in candidates}),
            "exact_text_duplicate_pairs":exact,"near_text_duplicate_pairs":near,
            "same_block_pairs":same_block,"near_threshold":.90}


def audit_case(question,candidates,question_id):
    sources={s:f"S{i:03d}" for i,s in enumerate(sorted({_expected_source(question)} | {c.get('source',c.get('metadata',{}).get('source')) for c in candidates}),1)}
    return {"question_id":question_id,"evidence_rank":_evidence_rank_in_results(question,candidates),
            "top3_in_scope_count":sum(_matches_case_scope(question,c) for c in candidates[:3]),
            "duplicates":duplicate_diagnostics(candidates),
            "candidates":[dict(m9._safe_candidate(c,i,sources,question['question'],{},{}),
                               character_count=len(c.get('text','')),in_expected_scope=_matches_case_scope(question,c))
                          for i,c in enumerate(candidates,1)]}


def ranking_audit(args,original,documents,manifest):
    prior=json.loads(args.m9_summary.read_text(encoding='utf-8'))
    if prior['configuration']['input_inventory']!=manifest['original_input_inventory']:
        raise ValueError('M9.1 corpus inventory differs; ordinal reconstruction unsafe')
    for name,digest in prior['configuration']['core_sha256'].items():
        if m9._sha256_file(PROJECT_ROOT/name)!=digest: raise ValueError('M9.1 controlled source changed')
    chunks=split_documents(documents,max_chars=500,overlap=80)
    if len(chunks)!=prior['configuration']['chunks']: raise ValueError('Corpus ordinal mismatch')
    root=safe_root(AUDIT_ROOT); root.mkdir(parents=True,exist_ok=True)
    public=[]; private=[]
    for qid in (7,19,24,25):
        item=original[qid-1]; old=prior['cases'][qid-1]; modes={}; private_modes={}
        for mode in MODES:
            candidates=[]
            for row in old['modes'][mode]['candidates']:
                index=row['corpus_index']; c=dict(chunks[index],corpus_index=index)
                for key in ('score','dense_rank','bm25_rank','rrf_score','reranker_score'):
                    if key in row: c[key]=row[key]
                candidates.append(c)
            result=audit_case(item,candidates,qid)
            if result['evidence_rank']!=old['modes'][mode]['evidence_rank']:
                raise ValueError('Historical evidence rank did not reproduce')
            modes[mode]=result
            private_modes[mode]={'candidates':candidates,'diagnostics':result}
        public.append({'question_id':qid,'modes':modes})
        private.append({'question_id':qid,'qa':item,'modes':private_modes})
    m9._write_json(root/'anonymous_audit.json',{'cases':public,'configuration':frozen_configuration(),'m9_summary_sha256':m9._sha256_file(args.m9_summary)})
    m9._write_json(root/'private_candidates.local.json',private)
    print('Four historical candidate pools reconstructed and evidence ranks verified.',flush=True)
    return public


def run(args):
    questions,original,documents,manifest=load_inputs(args)
    require_manifest(json.loads(args.manifest.read_text(encoding='utf-8')),manifest)
    ranking_audit(args,original,documents,manifest)
    if args.audit_only: return
    chunks=split_documents(documents,max_chars=500,overlap=80)
    indices={chunk_identity(c):i for i,c in enumerate(chunks)}
    if len(indices)!=len(chunks): raise ValueError('Duplicate structural identities')
    root=safe_root(args.output_dir); root.mkdir(parents=True,exist_ok=True)
    run_dir=root/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ'); run_dir.mkdir(exist_ok=False)
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    timings={key:[] for key in ('query_embedding','dense_query','bm25_query','rrf_fusion','hybrid_query_without_embedding','dense_reranker','hybrid_reranker')}
    started=time.perf_counter(); bm25=BM25Index(chunks); index_cost=time.perf_counter()-started
    started=time.perf_counter(); model=load_model(); embeddings=embed_chunks(chunks,model); embedding_cost=time.perf_counter()-started
    started=time.perf_counter(); bge=load_reranker(); bge_cost=time.perf_counter()-started
    rankings={m:{} for m in MODES}
    for qid,item in enumerate(questions,1):
        query=item['question']; started=time.perf_counter()
        vector=model.encode(query,convert_to_numpy=True,show_progress_bar=False)
        timings['query_embedding'].append(time.perf_counter()-started)
        started=time.perf_counter(); dense=retrieve(vector,chunks,embeddings,top_k=20); dt=time.perf_counter()-started
        timings['dense_query'].append(dt)
        for c in dense: c['corpus_index']=indices[chunk_identity(c)]
        started=time.perf_counter(); lexical=bm25.search(query,20); bt=time.perf_counter()-started
        timings['bm25_query'].append(bt)
        started=time.perf_counter(); hybrid=rrf_fuse(dense,lexical,top_k=40)[:20]; ft=time.perf_counter()-started
        timings['rrf_fusion'].append(ft); timings['hybrid_query_without_embedding'].append(dt+bt+ft)
        pools={'dense':dense,'bm25':lexical,'hybrid':hybrid}
        for mode in MODES:
            if mode.endswith('reranker'):
                pool=pools[mode.removesuffix('_reranker')]; started=time.perf_counter()
                rankings[mode][query]=rerank(query,pool,bge,top_k=len(pool))
                timings[mode].append(time.perf_counter()-started)
            else: rankings[mode][query]=pools[mode]
        print(f'I{qid:02d}/{len(questions)}: five frozen arms complete',flush=True)
    reports={mode:evaluate_questions(questions,None,chunks,[],documents=documents,diagnostic_k=20,
        candidate_provider=lambda query,limit,rows=rankings[mode]:rows[query][:limit]) for mode in MODES}
    summary=build_summary(questions,reports,documents,chunks)
    _,_,_,after=load_inputs(args); require_manifest(manifest,after)
    summary['frozen_manifest']=manifest; summary['post_run_manifest_matches']=True
    summary['corpus']={'documents':len(documents),'chunks':len(chunks),'mineru_calls':0}
    summary['runtime']={'bm25_index_build_seconds':index_cost,'embedding_load_and_index_seconds':embedding_cost,
                        'reranker_load_seconds':bge_cost,'stages':{k:m9._timing_summary(v) for k,v in timings.items()}}
    m9._write_json(run_dir/'summary.json',summary)
    print(json.dumps({'metrics':summary['metrics'],'transitions':summary['transitions']},indent=2),flush=True)
    print('Anonymous output: '+str(run_dir.relative_to(PROJECT_ROOT)/'summary.json'),flush=True)
    return summary


def build_argument_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--freeze',action='store_true'); action.add_argument('--audit-only',action='store_true')
    parser.add_argument('--dataset',type=Path,default=ROOT/'m9_independent_qa.local.json')
    parser.add_argument('--manifest',type=Path,default=ROOT/'frozen_manifest.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT)
    parser.add_argument('--m9-summary',type=Path,default=PROJECT_ROOT/'outputs/m9_hybrid_retrieval/run_20261002T093708Z/summary.json')
    return parser


def main():
    args=build_argument_parser().parse_args()
    try:
        if args.freeze: freeze(args)
        else: run(args)
    except (OSError,ValueError,RuntimeError) as exc:
        print(f'M9.2 stopped: {exc}',file=sys.stderr); return 1
    return 0

if __name__=='__main__':
    raise SystemExit(main())
