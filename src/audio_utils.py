import numpy as np, librosa

def logmel(y,sr=16000):
    m=librosa.feature.melspectrogram(y=y,sr=sr,n_fft=512,hop_length=160,n_mels=64,fmin=40,fmax=7600)
    return librosa.power_to_db(m,ref=np.max).astype(np.float32)
