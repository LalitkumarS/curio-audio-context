from flask import Flask,request,jsonify
from tempfile import NamedTemporaryFile
from .context_layer import AudioContextLayer
app=Flask(__name__); layer=AudioContextLayer()
@app.get('/')
def root(): return '<h2>Curio Next Audio Context Layer</h2><p>POST multipart/form-data to /answer with fields audio and question.</p>'
@app.post('/answer')
def answer():
    if 'audio' not in request.files or 'question' not in request.form: return jsonify({'error':'audio and question are required'}),400
    f=request.files['audio']
    with NamedTemporaryFile(suffix='.wav') as t:
        f.save(t.name); ans,c=layer.answer(t.name,request.form['question'])
    return jsonify({'answer':ans,'context':c})
if __name__=='__main__': app.run(host='0.0.0.0',port=8000)
