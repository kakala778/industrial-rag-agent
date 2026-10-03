"""Offline ID selection, quote fidelity and host-rendered counterfactual metrics."""
from copy import deepcopy
import re
import unicodedata

from evaluation.agent01_metrics import review_finding, finding_fingerprint


def normalize_quote(text):
    """Formatting only: preserve case, numbers, comparisons and unknown macros."""
    text=re.sub(r'(?<=\d)([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)',
                lambda m:'^'+unicodedata.normalize('NFKC',m[1]),text)
    text=unicodedata.normalize('NFKC',text)
    for command,value in ((r'\geqslant','≥'),(r'\leqslant','≤'),(r'\geq','≥'),
                          (r'\leq','≤'),(r'\ge','≥'),(r'\le','≤'),(r'\times','x')):
        text=re.sub(re.escape(command)+r'(?![A-Za-z])',lambda _:value,text)
    text=re.sub(r'\\(?:mathrm|text|mathbf|operatorname)\{([^{}]*)\}',r'\1',text)
    text=text.replace('×','x').replace(r'\,','').replace(r'\;','').replace(r'\!','')
    text=text.replace('<=','≤').replace('>=','≥')
    text=re.sub(r'(?<![A-Za-z])(mm|cm|km|m)\^(?:\{([23])\}|([23]))(?![0-9])',
                lambda m:m[1]+(m[2] or m[3]),text)
    text=re.sub(r'[\s{}$]','',text)
    return re.sub(r'(?<=[0-9m])\*(?=[0-9])','x',text)


def _contains(text,quote):
    if not quote: return False
    start=0
    while (index:=text.find(quote,start))>=0:
        left=text[index-1] if index else ''
        right=text[index+len(quote):index+len(quote)+1]
        bad_left=quote[0].isdigit() and left and left in '0123456789.x×'
        end_pattern=r'[0-9.]' if quote[-1].isdigit() else r'[A-Za-z0-9.]'
        bad_right=bool(re.match(r'[A-Za-z0-9.]',quote[-1])) and bool(re.match(end_pattern,right))
        if not bad_left and not bad_right: return True
        start=index+1
    return False


def quote_fidelity(quote,text):
    if not isinstance(quote,str) or not isinstance(text,str): return 'NOT_FAITHFUL'
    if _contains(text,quote): return 'EXACT'
    if _contains(normalize_quote(text),normalize_quote(quote)): return 'NORMALIZED_EQUIVALENT'
    return 'NOT_FAITHFUL'


def render_selected_ids(selection,observed_ids,looked_up):
    """Experimental offline projection, never used to repair a runtime quote."""
    rendered=[]
    seen=set()
    for item in selection:
        if not isinstance(item,dict) or set(item)!={'scope','evidence_id'}:
            raise ValueError('ID-only selection requires scope and ID')
        eid=item['evidence_id']
        evidence=looked_up.get(eid)
        if eid in seen or eid not in observed_ids or evidence is None or evidence['source']!=item['scope']:
            raise ValueError('unobserved/unlooked-up/wrong-scope/duplicate selection')
        seen.add(eid)
        rendered.append(dict(scope=item['scope'],evidence_id=eid,quote=evidence['text'],
                             citation={k:deepcopy(evidence.get(k)) for k in
                                       ('source','page','block_type','block_index','chunk_id')}))
    return rendered


def candidate_group(oracle,candidate_reviews,*,candidate_ids=None):
    if not oracle: return 'PREFLIGHT_CONTROL'
    if any(o.get('expected_available') is None for o in oracle.values()): return 'UNASSESSED'
    required={s for s,o in oracle.items() if o.get('expected_available') is True}
    explicit_ids=any('expected_evidence_ids' in o for o in oracle.values())
    if explicit_ids:
        if any(not isinstance(oracle[s].get('expected_evidence_ids'),list)
               or not oracle[s]['expected_evidence_ids'] for s in required):
            return 'UNASSESSED'
        observed=set(candidate_reviews) if candidate_ids is None else set(candidate_ids)
        for scope in required:
            expected=set(oracle[scope]['expected_evidence_ids'])
            if not any(evidence_id in observed
                       and candidate_reviews.get(evidence_id,{}).get('scope')==scope
                       and candidate_reviews.get(evidence_id,{}).get('label')=='RELEVANT'
                       for evidence_id in expected):
                return 'RETRIEVAL_BOUND'
        return 'CANDIDATE_AVAILABLE'
    available={r['scope'] for r in candidate_reviews.values() if r['label']=='RELEVANT'}
    return 'CANDIDATE_AVAILABLE' if required<=available else 'RETRIEVAL_BOUND'


def score_selection(state,actions,oracle,candidate_reviews,*,finding_reviews=None):
    finish=next((a for a in reversed(actions) if a['action']=='FINISH'),None)
    findings=finish['findings'] if finish else []
    observed={r['evidence_id'] for search in state.get('search_history',[])
              for r in search.get('results',[]) if isinstance(r,dict) and 'evidence_id' in r}
    explicit_ids=bool(oracle) and any('expected_evidence_ids' in o for o in oracle.values())
    rows=[]
    for f in findings:
        evidence=state['looked_up_evidence'].get(f['evidence_id'])
        eligible=(evidence is not None and f['evidence_id'] in state['evidence_ids']
                  and evidence['source']==f['scope']
                  and (not explicit_ids or f['evidence_id'] in observed))
        candidate=candidate_reviews.get(f['evidence_id'],{})
        expected_ids=(oracle or {}).get(f['scope'],{}).get('expected_evidence_ids')
        expected_id_match=(f['evidence_id'] in expected_ids
                           if isinstance(expected_ids,list) else None)
        expected_id_correct=(expected_id_match is True if explicit_ids
                             else expected_id_match is not False)
        row=dict(scope=f['scope'],evidence_id=f['evidence_id'],eligible_id=eligible,
                 id_label=candidate.get('label','UNCERTAIN') if eligible else 'INVALID_ID',
                 expected_id_match=expected_id_match,
                 correct_expected_id=bool(eligible and candidate.get('label')=='RELEVANT'
                                          and expected_id_correct),
                 id_alignment={k:candidate.get(k) for k in ('field_alignment','unit_alignment','condition_alignment','scope_alignment')},
                 quote_fidelity=quote_fidelity(f['quote'],evidence['text']) if eligible else 'NOT_FAITHFUL',
                 quote_review=review_finding(f,evidence,(oracle or {}).get(f['scope'])))
        rows.append(row)
    if finding_reviews is not None:
        if len(finding_reviews)!=len(findings):
            raise ValueError('review must cover every attempted finding')
        for f,row,review in zip(findings,rows,finding_reviews):
            if review.get('finding_sha256')!=finding_fingerprint(f) or review.get('label') not in ('RELEVANT','PARTIAL','IRRELEVANT','UNCERTAIN'):
                raise ValueError('stale or invalid review')
            for key in ('field_alignment','unit_alignment','condition_alignment','scope_alignment'):
                value=review.get(key)
                if value is not None and type(value) is not bool:
                    raise ValueError('invalid alignment review')
            row['quote_review']={k:review[k] for k in ('label','field_alignment','unit_alignment','condition_alignment','scope_alignment')}
    required={s for s,o in (oracle or {}).items() if o.get('expected_available') is True}
    correct={r['scope'] for r in rows if r['correct_expected_id']}
    id_success=bool(finish and required<=correct and all(r['correct_expected_id'] for r in rows))
    full=bool(state['status'] in ('finished','incomplete') and id_success and all(
        r['quote_fidelity']=='EXACT' and r['quote_review']['label']=='RELEVANT' for r in rows))
    prototype=[]
    try:
        prototype=render_selected_ids([{k:f[k] for k in ('scope','evidence_id')} for f in findings],
                                       state['evidence_ids'],state['looked_up_evidence'])
    except ValueError:
        id_success=False
        full=False
    return dict(group=candidate_group(oracle,candidate_reviews,candidate_ids=observed),
                expected_id_mode=explicit_ids,
                correct_evidence_ids=sum(r['correct_expected_id'] for r in rows),
                attempted_evidence_ids=len(rows),finish_attempted=finish is not None,
                review_method='offline_finding_review' if finding_reviews is not None else 'frozen_lexical_screen',
                id_selection_success=id_success,fully_relevant_task_success=full,
                quote_fidelity=[r['quote_fidelity'] for r in rows],findings=rows,
                evidence_omission=sorted(required-correct),
                irrelevant_evidence_selected=sum(r['id_label']=='IRRELEVANT' for r in rows),
                host_rendered_selection_success=bool(id_success and prototype),
                host_rendered_prototype=prototype)
