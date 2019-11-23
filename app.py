import locale
locale.setlocale(locale.LC_ALL, 'de_DE')

from flask import Flask, jsonify, request, json
import json
from embapi.model_inference import load_model, get_sentence_embedding
from logging import getLogger
import os

# catch case sensitive import for linux
try:
    from flask_restful import Resource, Api

except ModuleNotFoundError:
    from Flask_RESTful import Resource, Api

# create logger
logger = getLogger()

# setup app
app = Flask(__name__)
app.config['DEBUG'] = True
api = Api(app)

# load model
print("Loading Model...")
if not os.path.exists("chkpt"):
    os.makedirs("chkpt")

model = load_model('chkpt/german.model')

class FlaskRestApi(Resource):


    def post(self):
        res = {}
        
        # get the posted query
        userdata = request.data
        userModel = json.loads(userdata)

        # debug
        print("post:", userModel["sent"])

        # get embeddings
        emb, oov = get_sentence_embedding(
            model, userModel["sent"], pooling="max"
            )

        # return result
        res = {
            'embedding' : emb.tolist(),
            'oov-words' : oov
        }

        # error handling
            
        # return result
        return jsonify(res)     

api.add_resource(FlaskRestApi, '/embapi', endpoint = 'flaskrest')

if __name__ == '__main__':
    app.run(debug=True,  port=8000)  