"""Secondary, post hoc output diagnostics. Never repair labels or replace primary metrics."""
import argparse
import json
from pathlib import Path
from evaluate import DECISIONS, align, load_gold, load_predictions, parse_output, ratio, _unique_pairs, _reject_constant


def extract_one_object(raw):
    """Return one unambiguous object; reject duplicate keys and multiple objects."""
    if not isinstance(raw,str):return None
    # A complete non-object JSON value is not silently transformed into an object.
    try:
        complete=json.loads(raw,object_pairs_hook=_unique_pairs,parse_constant=_reject_constant)
        return complete if isinstance(complete,dict) else None
    except ValueError:
        pass
    objects=[];index=0
    while index<len(raw):
        start=raw.find('{',index)
        if start<0:break
        depth=0;quoted=False;escaped=False;end=None
        for position in range(start,len(raw)):
            ch=raw[position]
            if quoted:
                if escaped:escaped=False
                elif ch=='\\':escaped=True
                elif ch=='"':quoted=False
            elif ch=='"':quoted=True
            elif ch=='{':depth+=1
            elif ch=='}':
                depth-=1
                if depth==0:end=position+1;break
        if end is None:return None
        try:
            value=json.loads(raw[start:end],object_pairs_hook=_unique_pairs,parse_constant=_reject_constant)
        except ValueError:return None
        objects.append(value);index=end

    return objects[0] if len(objects)==1 else None


def score(gold,predictions):
    # Reuse fatal duplicate/unknown-ID checks; missing IDs remain in denominators.
    align(gold,predictions)
    by_id={p['id']:p for p in predictions};n=len(gold)
    syntax=objects=schema=decision_correct=joint_correct=valid_decisions=0
    confusion={g:{p:0 for p in (*DECISIONS,'invalid')} for g in DECISIONS}
    for row in gold:
        raw=by_id.get(row['id'],{}).get('raw_output')
        if isinstance(raw,str):
            try:json.loads(raw,parse_constant=_reject_constant);syntax+=1
            except ValueError:pass
        obj=extract_one_object(raw);objects+=obj is not None
        schema+=parse_output(raw)[1] is None
        g=json.loads(row['output']);p=obj.get('decision') if obj is not None else None
        p=p if isinstance(p,str) and p in DECISIONS else 'invalid'
        valid_decisions+=p!='invalid';confusion[g['decision']][p]+=1
        decision_correct+=p==g['decision']
        joint_correct+=bool(obj is not None and p==g['decision'] and 'skill' in obj and obj['skill']==g['skill'])
    f1=[]
    for label in DECISIONS:
        tp=confusion[label][label]
        fp=sum(confusion[g][label] for g in DECISIONS if g!=label)
        fn=sum(v for p,v in confusion[label].items() if p!=label)
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return {'n':n,'raw_json_syntax_validity':ratio(syntax,n),'extracted_object_rate':ratio(objects,n),
            'schema_validity':ratio(schema,n),'valid_decision_rate':ratio(valid_decisions,n),
            'decision_accuracy':ratio(decision_correct,n),'decision_macro_f1':sum(f1)/len(f1),
            'joint_accuracy':ratio(joint_correct,n),'confusion':confusion,
            'interpretation':'Secondary post hoc diagnostic, added after observing base schema failures; not a replacement for frozen primary metrics. Decision is scored independently of skill semantics. Joint requires the stated decision and skill, with no inferred null, label repair, or gold-guided extraction. Exactly one unambiguous JSON object may be extracted from surrounding text. Duplicate-key objects are rejected. Missing outputs count as failures.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gold',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();gold=load_gold(a.gold);predictions=load_predictions(a.predictions)
    result=score(gold,predictions)
    for name,is_challenge in [('challenge',True),('controlled',False)]:
        subset=[r for r in gold if r['family'].startswith('challenge_')==is_challenge]
        ids={r['id'] for r in subset}
        if subset:result.setdefault('subsets',{})[name]=score(subset,[r for r in predictions if r['id'] in ids])
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['n','raw_json_syntax_validity','schema_validity','decision_accuracy','joint_accuracy']}))

if __name__=='__main__':main()
