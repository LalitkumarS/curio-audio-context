import json, random, math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

ROOT=Path(__file__).resolve().parents[1]
AUDIO=ROOT/'data/audio'
AUDIO.mkdir(parents=True, exist_ok=True)
SR=16000; DUR=12; N=SR*DUR
random.seed(42); np.random.seed(42)

EVENTS=['door_knock','dog_bark','clap','bell','footsteps']
ENVS=['home','office','street','park','rain']
EVENT_TEXT={
 'door_knock':'a door knock','dog_bark':'a dog bark','clap':'a clap','bell':'a bell','footsteps':'footsteps'}
ENV_TEXT={'home':'home','office':'an office','street':'a street','park':'a park','rain':'a rainy environment'}

def band_noise(n, lo, hi):
    x=np.random.randn(n)
    sos=butter(4,[lo/(SR/2),hi/(SR/2)],btype='bandpass',output='sos')
    return sosfilt(sos,x)

def event_wave(name, length=1.0):
    n=int(SR*length); t=np.arange(n)/SR; y=np.zeros(n)
    if name=='door_knock':
        for center in [0.24,0.48]:
            k=int(center*SR); m=int(0.07*SR); idx=np.arange(max(0,k-m),min(n,k+m)); tt=(idx-k)/SR
            y[idx]+=np.exp(-abs(tt)*45)*(np.sin(2*np.pi*230*tt)+0.45*np.sin(2*np.pi*510*tt))
    elif name=='dog_bark':
        centers=[0.30,0.62];
        for c in centers:
            k=int(c*SR); m=int(.16*SR); idx=np.arange(max(0,k-m),min(n,k+m)); tt=(idx-k)/SR
            env=np.exp(-((tt)/.10)**2)
            y[idx]+=env*(np.sin(2*np.pi*(380+110*tt)*tt)+.5*np.sin(2*np.pi*760*tt))
    elif name=='clap':
        k=int(.45*SR); m=int(.10*SR); idx=np.arange(max(0,k-m),min(n,k+m)); tt=(idx-k)/SR
        y[idx]+=np.random.randn(len(idx))*np.exp(-abs(tt)*30)
        y += .15*band_noise(n,1500,6000)*np.exp(-np.maximum(t-.35,0)*12)
    elif name=='bell':
        k=int(.45*SR); idx=np.arange(k,n); tt=(idx-k)/SR
        env=np.exp(-tt*2.4)
        y[idx]+=env*(np.sin(2*np.pi*880*tt)+.55*np.sin(2*np.pi*1760*tt)+.3*np.sin(2*np.pi*2640*tt))
    elif name=='footsteps':
        for c in [.22,.58,.88]:
            k=int(c*SR); m=int(.09*SR); idx=np.arange(max(0,k-m),min(n,k+m)); tt=(idx-k)/SR
            y[idx]+=np.exp(-abs(tt)*32)*(np.sin(2*np.pi*85*tt)+.35*np.sin(2*np.pi*170*tt))
    y=y/(np.max(np.abs(y))+1e-8)
    return y.astype(np.float32)

def environment(name):
    t=np.arange(N)/SR
    if name=='home':
        y=.055*band_noise(N,120,260)+.012*np.random.randn(N)+.015*np.sin(2*np.pi*60*t)
    elif name=='office':
        y=.055*band_noise(N,300,700)+.012*np.random.randn(N)+.012*np.sin(2*np.pi*120*t)
    elif name=='street':
        y=.065*band_noise(N,500,1400)+.025*np.random.randn(N)
        for c in [2.2,6.8,10.1]:
            k=int(c*SR); m=int(.12*SR); idx=np.arange(max(0,k-m),min(N,k+m)); tt=(idx-k)/SR
            y[idx]+=.12*np.exp(-abs(tt)*18)*np.sin(2*np.pi*420*tt)
    elif name=='park':
        y=.018*np.random.randn(N)
        for c,f in [(2.0,2200),(5.5,3200),(9.0,1800)]:
            k=int(c*SR); m=int(.16*SR); idx=np.arange(max(0,k-m),min(N,k+m)); tt=(idx-k)/SR
            y[idx]+=.35*np.exp(-abs(tt)*18)*np.sin(2*np.pi*f*tt)
    elif name=='rain':
        y=.075*band_noise(N,2200,7000)+.018*np.random.randn(N)
        for c in np.linspace(.2,11.8,20):
            k=int(c*SR); idx=np.arange(k,min(N,k+int(.025*SR))); tt=(idx-k)/SR
            y[idx]+=.12*np.exp(-tt*90)
    return y.astype(np.float32)

def make_questions(cid, env, events):
    by_event={e:[] for e in EVENTS}
    for e in events: by_event[e['event']].append(e)
    qs=[]
    # perceptual/environment
    qs.append({'type':'what','question':'What environment does this audio suggest?','answer':ENV_TEXT[env]})
    unique=[]
    for e in events:
        if e['event'] not in unique: unique.append(e['event'])
    sounds=', '.join(EVENT_TEXT[e] for e in unique)
    qs.append({'type':'what_sound','question':'What sounds are present in the audio?','answer':sounds})
    # event presence + count
    target=max(EVENTS,key=lambda e:len(by_event[e]))
    qs.append({'type':'presence','question':f'Is there {EVENT_TEXT[target]} in the audio?','answer':'yes' if by_event[target] else 'no'})
    qs.append({'type':'counting','question':f'How many times does {EVENT_TEXT[target]} occur?','answer':str(len(by_event[target]))})
    # temporal relation if >=2 distinct events
    ordered=sorted(events,key=lambda x:x['start']); last=ordered[-1]; prev=ordered[-2]
    qs.append({'type':'temporal','question':f'What happens immediately before {EVENT_TEXT[last["event"]]}?','answer':EVENT_TEXT[prev['event']]})
    # causal/environment reasoning based on designed scene
    causal={
      'home':'Because the audio contains a quiet indoor household ambience.',
      'office':'Because the audio contains a steady mechanical indoor ambience.',
      'street':'Because the audio contains a broad noisy background with traffic-like sounds.',
      'park':'Because the audio contains quiet outdoor ambience with occasional high-pitched natural sounds.',
      'rain':'Because the audio contains a continuous broadband rainfall-like texture.'}
    qs.append({'type':'why','question':'Why does the audio appear to be in this environment?','answer':causal[env]})
    return qs

clips=[]; total=90
for i in range(total):
    cid=f'clip_{i:04d}'
    env=ENVS[i%len(ENVS)]
    audio=environment(env)
    n_events=random.randint(2,4)
    starts=random.sample([1,2,3,4,5,6,7,8,9],n_events)
    chosen=random.choices(EVENTS,k=n_events)
    events=[]
    for s,e in sorted(zip(starts,chosen)):
        wav=event_wave(e,1.0)
        a=int(s*SR); b=a+SR
        audio[a:b]+=0.55*wav
        events.append({'event':e,'start':s,'end':s+1})
    audio=np.clip(audio,-1,1)
    path=AUDIO/f'{cid}.wav'; sf.write(path,audio,SR)
    clips.append({'clip_id':cid,'audio_path':str(path.relative_to(ROOT)),'environment':env,'events':events,'qa':make_questions(cid,env,events)})

random.shuffle(clips)
ntr=int(total*0.7); nv=int(total*0.15); train=clips[:ntr]; val=clips[ntr:ntr+nv]; test=clips[ntr+nv:]
for split,data in [('train',train),('val',val),('test',test)]:
    (ROOT/'data'/f'{split}.json').write_text(json.dumps(data,indent=2))
meta={'sample_rate':SR,'duration_seconds':DUR,'events':EVENTS,'environments':ENVS,'num_clips':total,'splits':{'train':len(train),'val':len(val),'test':len(test)},'seed':42}
(ROOT/'data/dataset_meta.json').write_text(json.dumps(meta,indent=2))
print('Generated',total,'clips:',{k:len(v) for k,v in [('train',train),('val',val),('test',test)]})
