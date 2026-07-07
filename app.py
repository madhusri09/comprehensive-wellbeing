from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/hdi_model.pkl","rb"))
encoder = pickle.load(open("model/label_encoder.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    life=float(request.form["life"])
    mean=float(request.form["mean"])
    expected=float(request.form["expected"])
    gni=float(request.form["gni"])

    prediction=model.predict([[life,mean,expected,gni]])

    result=encoder.inverse_transform(prediction)

    return render_template("result.html",prediction=result[0])

if __name__=="__main__":
    app.run(debug=True)