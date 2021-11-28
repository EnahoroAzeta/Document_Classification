from flask import Flask, request
from flask_restful import Resource, Api
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import json
import pandas as pd


app = Flask(__name__)
api = Api(app)

# Upickle model and vectorizer
model = pickle.load(open('docClassficationModel.pkl', 'rb'))
vector = pickle.load(open('vectorizer.pickle', 'rb'))
vectorizer = CountVectorizer(analyzer="word")


class Records(Resource):
   def get(self):
      # get request that returns the JSON format for API request
      return {"JSON data format": {"Document": "Documents string"}}, 200

   def post(self):
   # post request
      global model
      global vectorizer
   
      # get json data as sent to API
      data = request.get_json(force=True)

      # extract document for classification
      request_data = data["JSON"]["Document"]

      # convert document to list for vectorizer
      request_dat = [request_data]

      # vectorizing the document string
      vectorized_data = vector.transform(request_dat)

      # lookup dictionary
      classification = {0: 'tech',
                        1 : 'business',
                        2: 'sport',
                        3: 'entertainment',
                        4: 'politics'
                        }

      # prediction
      prediction = model.predict(vectorized_data)

      # convert array to list
      pred_list = prediction.tolist()

      #get prediction value for lookup
      val = pred_list[0]



      # get corresponding value from the classification dictionary (using the model prediction as the key)
      result = classification.get(val)


      return {'classification': result}, 200








api.add_resource(Records, '/')
app.run()


