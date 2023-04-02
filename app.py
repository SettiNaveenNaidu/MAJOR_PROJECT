from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('new.html')
@app.route('/neural.html')
def index():
    return render_template('neural.html')

@app.route('/theory.html')
def about():
    return render_template('theory.html')

@app.route('/linear.html')
def menu():
    return render_template('linear.html')

standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Feed = float(request.form['Feed'])
        Speed=float(request.form['Speed'])
        Doc=float(request.form['Doc'])
        Woc=float(request.form['Woc'])
        Dir=float(request.form['Dir'])
        Coolant=float(request.form['Coolant'])
        
        prediction=model.predict([[Feed,Speed,Doc,Woc,Dir,Coolant]])
        output=round(prediction[0],6)
        
        return render_template('new.html',prediction_text="The value of roughness is {}".format(output))
    else:
        return render_template('new.html')

if __name__=="__main__":
    app.run(debug=True)
