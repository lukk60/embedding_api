from flask import Flask, jsonify, request, json
from flask_restful import Resource, Api
import json
from embapi.model_inference import load_model, get_sentence_embedding

app = Flask(__name__)
app.config['DEBUG'] = True
api = Api(app)

model = load_model('chkpt/german.model')

class FlaskRestApi(Resource):


    def post(self):
        res = {}
        
        # get the posted query
        userdata = request.data
        userModel = json.loads(userdata)
                
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
    app.run(debug=True,  port=8080)