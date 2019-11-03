from flask import Flask, jsonify, request, json
from flask_restful import Resource, Api
import json
from embapi.model_inference import load_model, get_sentence_embedding
from logging import getLogger
import os

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
    
print(os.listdir())
print(os.listdir("chkpt"))
print(os.getenv("embapi_storage_name"))

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