import numpy as np
import pickle
import sklearn
from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3
model = pickle.load(open('models.pkl', 'rb')) 
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('Heart_Disease_Classifier.html', title='Home')



@app.route('/Heart_Disease_Classifier')
def Heart_Disease_Classifier():
       return render_template('Heart_Disease_Classifier.html')   

@app.route('/predict', methods =['POST'])
def predict():
  
  features = [float(i) for i in request.form.values()]
  #Convert features to array
  array_features = [np.array(features)]
  #Predict features
  prediction = model.predict(array_features)
  output = prediction
  if output == 1:
    return render_template('Heart_Disease_Classifier.html', result = 'The patient is not likely to have heart disease!')
  else:
    return render_template('Heart_Disease_Classifier.html', result = 'The patient is likely to have heart disease!')

if __name__ == '__main__':
   app.run(debug=True,port=5100)