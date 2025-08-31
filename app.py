from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load trained pipeline
with open("loan_approval_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")  # render your HTML form

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])  # make DataFrame for pipeline
    prediction = model.predict(df)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
