import json, random
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import joblib
from .audio_utils import logmel
from .model import AudioCNN

ROOT=Path(__file__).resolve().parents[1]; SR=16000
random.seed(42); np.random.seed(42); torch.manual_seed(42)

def load_split(name): return json.loads((ROOT/'data'/f'{name}.json').read_text())

def make_event_data(records, events):
    labels=['background']+events; X=[]; y=[]
    for r in records:
        wav,_=sf.read(ROOT/r['audio_path']); wav=wav.astype(np.float32)
        event_by_sec={int(e['start']):e['event'] for e in r['events']}
        for s in range(12):
            seg=wav[s*SR:(s+1)*SR]
            X.append(logmel(seg)[None]); y.append(labels.index(event_by_sec.get(s,'background')))
    return np.stack(X),np.array(y),labels

def make_env_data(records, envs):
    X=[]; y=[]
    for r in records:
        wav,_=sf.read(ROOT/r['audio_path']); wav=wav.astype(np.float32)
        X.append(logmel(wav)[None]); y.append(envs.index(r['environment']))
    return np.stack(X),np.array(y)

def train_model(Xtr,ytr,Xv,yv,n_classes,name,epochs=10):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=AudioCNN(n_classes).to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3); counts=np.bincount(ytr,minlength=n_classes).astype(np.float32); weights=counts.sum()/(n_classes*np.maximum(counts,1)); loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    tr=DataLoader(TensorDataset(torch.tensor(Xtr),torch.tensor(ytr)),batch_size=32,shuffle=True)
    va=DataLoader(TensorDataset(torch.tensor(Xv),torch.tensor(yv)),batch_size=64)
    hist={'train_loss':[],'val_loss':[],'val_acc':[]}; best=None; bestacc=-1
    for ep in range(epochs):
        model.train(); losses=[]
        for xb,yb in tr:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval(); vl=[]; correct=0; total=0
        with torch.no_grad():
            for xb,yb in va:
                out=model(xb.to(device)); vl.append(loss_fn(out,yb.to(device)).item()); correct+=(out.argmax(1).cpu()==yb).sum().item(); total+=len(yb)
        acc=correct/total; hist['train_loss'].append(float(np.mean(losses))); hist['val_loss'].append(float(np.mean(vl))); hist['val_acc'].append(acc)
        if acc>bestacc: bestacc=acc; best={k:v.cpu() for k,v in model.state_dict().items()}
        print(name,ep+1,'train_loss',round(hist['train_loss'][-1],4),'val_acc',round(acc,4))
    model.load_state_dict(best); torch.save({'state_dict':model.state_dict(),'classes':n_classes},ROOT/f'models/{name}.pt')
    plt.figure(); plt.plot(hist['train_loss'],label='train loss'); plt.plot(hist['val_loss'],label='val loss'); plt.xlabel('epoch'); plt.ylabel('loss'); plt.title(name+' loss'); plt.legend(); plt.tight_layout(); plt.savefig(ROOT/f'results/{name}_loss.png'); plt.close()
    (ROOT/f'results/{name}_history.json').write_text(json.dumps(hist,indent=2))
    return model

if __name__=='__main__':
    tr,va=load_split('train'),load_split('val'); meta=json.loads((ROOT/'data/dataset_meta.json').read_text())
    Xtr,ytr,labels=make_event_data(tr,meta['events']); Xv,yv,_=make_event_data(va,meta['events'])
    np.savez_compressed(ROOT/'data/event_features.npz',X=Xtr,y=ytr)
    train_model(Xtr,ytr,Xv,yv,len(labels),'event_model',epochs=10)
    Xtr,ytr=make_env_data(tr,meta['environments']); Xv,yv=make_env_data(va,meta['environments'])
    # A compact spectral baseline is intentionally used for environment inference; it is fast, interpretable, and robust for this tiny PoC dataset.
    def env_feat(X): return np.concatenate([X.mean(axis=(1,3)),X.std(axis=(1,3))],axis=1)
    clf=RandomForestClassifier(n_estimators=200,random_state=42)
    clf.fit(env_feat(Xtr),ytr); print('environment_model val_acc',clf.score(env_feat(Xv),yv)); joblib.dump({'model':clf,'classes':meta['environments']},ROOT/'models/environment_model.joblib')
    (ROOT/'results/environment_model_history.json').write_text(json.dumps({'val_acc':[float(clf.score(env_feat(Xv),yv))]},indent=2))
