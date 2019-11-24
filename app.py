from flask import Flask, jsonify, request, json, abort
import json
from embapi.model_inference import load_model, get_sentence_embedding
from logging import getLogger
import os
import nltk
nltk.download("punkt")

# create logger
logger = getLogger()

# setup app
app = Flask(__name__)

# load model
print("Loading Model...")
if not os.path.exists("chkpt"):
    os.makedirs("chkpt")
model = load_model('chkpt/german.model')

@app.route("/api/", methods = ["POST"])
def get_embeddings():
    
    # check inputs
    if not request.json or not "sent" in request.json:
        abort(400)

    # get embeddings
    emb, oov = get_sentence_embedding(
        model, request.json["sent"], pooling="max"
        )

    # return result
    res = {
        'embedding' : emb.tolist(),
        'oov-words' : oov
    }

    # return result
    return jsonify(res)     

if __name__ == '__main__':
    app.run(debug=True,  port=8000)  