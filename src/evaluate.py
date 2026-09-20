import json,re
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np, soundfile as sf, torch
from .context_layer import AudioContextLayer
from .audio_utils import logmel

ROOT=Path(__file__).resolve().parents[1]; SR=16000

def norm(s): return re.sub(r'[^a-z0-9 ]','',s.lower()).strip()
def token_f1(a,b):
    A=norm(a).split(); B=norm(b).split();
    if not A or not B:return 1.0 if A==B else 0.0
    from collections import Counter
    ca,cb=Counter(A),Counter(B); common=sum((ca&cb).values());
    if common==0:return 0.0
    p=common/len(A); r=common/len(B); return 2*p*r/(p+r)

def model_acc(layer,records):
    ep,et=[],[]; yp,yt=[],[]; yp2,yt2=[],[]
    for r in records:
        c=layer.build_context(ROOT/r['audio_path'])
        yt.append(r['environment']); yp.append(c['environment'])
        for x,e in zip(c['timeline'],range(12)):
            gt=next((z['event'] for z in r['events'] if int(z['start'])==e),'background')
            yt2.append(gt); yp2.append(x['event'])
    return {
      'environment_accuracy':accuracy_score(yt,yp),'environment_macro_f1':f1_score(yt,yp,average='macro'),
      'event_accuracy':accuracy_score(yt2,yp2),'event_macro_f1':f1_score(yt2,yp2,average='macro')}

def main():
    records=json.loads((ROOT/'data/test.json').read_text()); layer=AudioContextLayer(); m=model_acc(layer,records)
    rows=[]; by=defaultdict(list)
    for r in records:
        for qa in r['qa']:
            pred,c=layer.answer(ROOT/r['audio_path'],qa['question']); em=float(norm(pred)==norm(qa['answer'])); f1=token_f1(pred,qa['answer'])
            row={'clip_id':r['clip_id'],'type':qa['type'],'question':qa['question'],'gold':qa['answer'],'prediction':pred,'exact_match':em,'token_f1':f1}
            rows.append(row); by[qa['type']].append(row)
    metrics={'model_metrics':m,'qa_exact_match':float(np.mean([x['exact_match'] for x in rows])),'qa_token_f1':float(np.mean([x['token_f1'] for x in rows])),'by_type':{k:{'n':len(v),'exact_match':float(np.mean([x['exact_match'] for x in v])),'token_f1':float(np.mean([x['token_f1'] for x in v]))} for k,v in by.items()}}
    errors=[x for x in rows if not x['exact_match']][:10]
    metrics['error_examples']=errors
    (ROOT/'results/metrics.json').write_text(json.dumps(metrics,indent=2)); (ROOT/'results/qa_predictions.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
