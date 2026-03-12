from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():

    amount = float(request.form["amount"])
    merchant_id = float(request.form["merchant_id"])
    transaction_hour = float(request.form["transaction_hour"])
    transaction_day = float(request.form["transaction_day"])
    transaction_month = float(request.form["transaction_month"])

    # Convert to array
    features = np.array([[amount, merchant_id, transaction_hour, transaction_day, transaction_month]])

    # Scale input
    scaled_features = scaler.transform(features)

    # Predict
    prediction = model.predict(scaled_features)

    if prediction[0] == 1:
        result = "⚠️ Fraudulent Transaction Detected"
    else:
        result = "✅ Legitimate Transaction"

    return render_template("home.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)