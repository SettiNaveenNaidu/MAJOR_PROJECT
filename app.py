from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model1 = pickle.load(open('models/linear.pkl', 'rb'))
model2 = pickle.load(open('models/lasso.pkl', 'rb'))
model3 = pickle.load(open('models/ridge.pkl', 'rb'))
model4 = pickle.load(open('models/svr.pkl', 'rb'))
model5 = pickle.load(open('models/neural.pkl', 'rb'))


@app.route('/',methods=['GET'])

def Home():
    return render_template('linear.html')

standard_to = StandardScaler()
@app.route("/predict1", methods=['POST'])
def predict1():
    if request.method == 'POST':
        Feed = float(request.form['Feed'])
        Speed=float(request.form['Speed'])
        Doc=float(request.form['Doc'])
        Woc=float(request.form['Woc'])
        Dir=float(request.form['Dir'])
        Coolant=float(request.form['Coolant'])
        
        prediction=model.predict1([[Feed,Speed,Doc,Woc,Dir,Coolant]])
        output=round(prediction[0],6)
        
        return render_template('linear_reg.html',prediction_text="The value of roughness is {}".format(output))
    else:
        return render_template('linear_reg.html')
   
@app.route("/predict2", methods=['POST'])
def predict2():
    if request.method == 'POST':
        Feed = float(request.form['Feed'])
        Speed=float(request.form['Speed'])
        Doc=float(request.form['Doc'])
        Woc=float(request.form['Woc'])
        Dir=float(request.form['Dir'])
        Coolant=float(request.form['Coolant'])
        
        prediction=model.predict2([[Feed,Speed,Doc,Woc,Dir,Coolant]])
        output=round(prediction[0],6)
        
        return render_template('lasso.html',prediction_text="The value of roughness is {}".format(output))
    else:
        return render_template('lasso.html')
    
@app.route("/predict3", methods=['POST'])
def predict3():
    if request.method == 'POST':
        Feed = float(request.form['Feed'])
        Speed=float(request.form['Speed'])
        Doc=float(request.form['Doc'])
        Woc=float(request.form['Woc'])
        Dir=float(request.form['Dir'])
        Coolant=float(request.form['Coolant'])
        
        prediction=model.predict3([[Feed,Speed,Doc,Woc,Dir,Coolant]])
        output=round(prediction[0],6)
        
        return render_template('ridge.html',prediction_text="The value of roughness is {}".format(output))
    else:
        return render_template('ridge.html')
    
@app.route("/predict4", methods=['POST'])
def predict4():
    if request.method == 'POST':
        Feed = float(request.form['Feed'])
        Speed=float(request.form['Speed'])
        Doc=float(request.form['Doc'])
        Woc=float(request.form['Woc'])
        Dir=float(request.form['Dir'])
        Coolant=float(request.form['Coolant'])
        
        prediction=model.predict4([[Feed,Speed,Doc,Woc,Dir,Coolant]])
        output=round(prediction[0],6)
        
        return render_template('svr.html',prediction_text="The value of roughness is {}".format(output))
    else:
        return render_template('svr.html')
    
@app.route("/predict5", methods=['POST'])
def predict5():
    if request.method == 'POST':
        Feed = float(request.form['Feed'])
        Speed=float(request.form['Speed'])
        Doc=float(request.form['Doc'])
        Woc=float(request.form['Woc'])
        Dir=float(request.form['Dir'])
        Coolant=float(request.form['Coolant'])
        
        prediction=model.predict5([[Feed,Speed,Doc,Woc,Dir,Coolant]])
        output=round(prediction[0],6)
        
        return render_template('neural.html',prediction_text="The value of roughness is {}".format(output))
    else:
        return render_template('neural.html')
    


if __name__=="__main__":
    app.run(debug=True)
