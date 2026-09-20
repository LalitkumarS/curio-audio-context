import argparse, json
from .context_layer import AudioContextLayer

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--audio',required=True); ap.add_argument('--question',required=True); a=ap.parse_args()
    layer=AudioContextLayer(); ans,c=layer.answer(a.audio,a.question)
    print('\nANSWER:',ans)
    print('\nAUDIO CONTEXT')
    print('Environment:',c['environment'],f"({c['environment_confidence']:.2f})")
    for x in c['timeline']:
        if x['event']!='background': print(f"{x['start']:.0f}-{x['end']:.0f}s: {x['event']} ({x['confidence']:.2f})")
if __name__=='__main__': main()
