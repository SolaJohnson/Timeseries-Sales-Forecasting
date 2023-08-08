import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

# Create flask app
flask_app = Flask(__name__)
filename = 'model.pkl'
model = joblib.load(filename)

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The predicted total sales for that day is {} NGN".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)