import json,re
from pathlib import Path
import numpy as np, torch, soundfile as sf
import joblib
from .audio_utils import logmel
from .model import AudioCNN

ROOT=Path(__file__).resolve().parents[1]; SR=16000
META=json.loads((ROOT/'data/dataset_meta.json').read_text())
EVENTS=META['events']; ENVS=META['environments']; EVENT_LABELS=['background']+EVENTS
CAUSAL={
 'home':'Because the audio contains a quiet indoor household ambience.',
 'office':'Because the audio contains a steady mechanical indoor ambience.',
 'street':'Because the audio contains a broad noisy background with traffic-like sounds.',
 'park':'Because the audio contains quiet outdoor ambience with occasional high-pitched natural sounds.',
 'rain':'Because the audio contains a continuous broadband rainfall-like texture.'}

def load_model(name, classes):
    ck=torch.load(ROOT/f'models/{name}.pt',map_location='cpu',weights_only=False)
    m=AudioCNN(classes); m.load_state_dict(ck['state_dict']); m.eval(); return m

class AudioContextLayer:
    def __init__(self):
        self.event_model=joblib.load(ROOT/'models/event_rf.joblib')
        self.env_model=joblib.load(ROOT/'models/environment_model.joblib')
    def build_context(self,audio_path):
        wav,_=sf.read(audio_path); wav=np.asarray(wav,dtype=np.float32)
        feats=[]
        for s in range(12):
            seg=wav[s*SR:(s+1)*SR]; m=logmel(seg)
            feats.append(np.r_[m.mean(1),m.std(1),m.max(1),np.mean(np.abs(seg)),np.std(seg)])
        probs=self.event_model['model'].predict_proba(np.array(feats))
        event_timeline=[]
        for s,p in enumerate(probs):
            idx=int(np.argmax(p)); event_timeline.append({'start':s,'end':s+1,'event':self.event_model['classes'][idx],'confidence':float(p[idx])})
        mel=logmel(wav); feat=np.concatenate([mel.mean(axis=1),mel.std(axis=1)])[None]
        p=self.env_model['model'].predict_proba(feat)[0]; ei=int(np.argmax(p)); env=self.env_model['classes'][ei]
        return {'environment':env,'environment_confidence':float(p[ei]),'timeline':event_timeline}
    def answer(self,audio_path,question):
        c=self.build_context(audio_path); q=question.lower().strip()
        if 'why' in q or 'explain' in q:
            return CAUSAL[c['environment']],c
        if ('what sounds' in q) or ('sounds are present' in q) or ('what sound' in q and 'environment' not in q):
            seen=[]
            for x in c['timeline']:
                if x['event']!='background' and x['confidence']>=.5 and x['event'] not in seen: seen.append(x['event'])
            return ', '.join(pretty_event(e) for e in seen) if seen else 'no distinct event sounds detected',c
        if 'environment' in q or 'where' in q or 'setting' in q:
            return ENVS_TEXT(c['environment']),c
        if q.startswith('is there') or 'present' in q or 'heard' in q:
            target=self._target_event(q); found=sum(1 for x in c['timeline'] if x['event']==target and x['confidence']>=.5)
            return ('yes' if found else 'no'),c
        if 'how many' in q or 'times' in q or 'occur' in q or 'occurs' in q:
            target=self._target_event(q); count=sum(1 for x in c['timeline'] if x['event']==target and x['confidence']>=.5)
            return str(count),c
        if 'before' in q or 'after' in q:
            target=self._target_event(q); hits=[x for x in c['timeline'] if x['event']==target and x['confidence']>=.5]
            if 'before' in q:
                first=hits[-1] if hits else None
                prev=[x for x in c['timeline'][:first['start']] if x['event']!='background' and x['confidence']>=.5] if first else []
                ans=pretty_event(prev[-1]['event']) if prev else 'no clearly detected event'
            else:
                last=hits[-1] if hits else None
                nxt=[x for x in c['timeline'][last['end']:] if x['event']!='background' and x['confidence']>=.5] if last else []
                ans=pretty_event(nxt[0]['event']) if nxt else 'no clearly detected event'
            return ans,c
        target=self._target_event(q)
        return pretty_event(target),c
    def _target_event(self,q):
        aliases={'knock':'door_knock','door':'door_knock','bark':'dog_bark','dog':'dog_bark','clap':'clap','bell':'bell','footstep':'footsteps','footsteps':'footsteps'}
        for k,v in aliases.items():
            if k in q:return v
        return 'background'

def pretty_event(e): return {'door_knock':'a door knock','dog_bark':'a dog bark','clap':'a clap','bell':'a bell','footsteps':'footsteps','background':'background ambience'}.get(e,e)
def ENVS_TEXT(e): return {'home':'home','office':'an office','street':'a street','park':'a park','rain':'a rainy environment'}[e]
