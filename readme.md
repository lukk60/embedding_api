# REST API for Model Inference 
Demonstrate how to create a REST API from a Word2Vec model using Flask.
REST API in Flask adapted from: 
- https://mparsec.com/2017/07/18/rest-api-using-flask-restful-and-azure-storage/
German Word2Vec Model from:
- https://devmount.github.io/GermanWordEmbeddings/#download

## Files and directories
myfirstrestapi
|- chkpt: files for trained model 
|- embapi: modules and utils to load models and make predictions
app.py: Flask app that loads model and listens to inference requests
embapi_inference.py: Class to load model and run inference

## Run App
# example:
python app.py

## Call api: 
# example: 
import requests
res = requests.post("http://localhost:8080/embapi", json={"sent":"Dies ist ein deutscher Satz"})
res.json()