from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import json

app = Flask(__name__)

# Load model
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load feature names
with open("features.json", "r") as f:
    feature_names = json.load(f)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get data from JSON (Postman) or form (HTML)
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()

    # Handle Location dropdown
    location = data.pop("Location", None)

    if location:
        location_feature = f"Location_{location}"
        if location_feature in feature_names:
            data[location_feature] = 1

    # Convert numeric inputs
    processed_data = {}
    for key, value in data.items():
        try:
            processed_data[key] = float(value)
        except:
            pass

    # Create full feature vector
    features = [0] * len(feature_names)

    for key, value in processed_data.items():
        if key in feature_names:
            index = feature_names.index(key)
            features[index] = value

    features = np.array(features).reshape(1, -1)

    # Scale features
    scaled_features = scaler.transform(features)

    # Predict
    prediction = model.predict(scaled_features)

    if prediction[0] == 1:
        result = " Fraudulent Transaction"
    else:
        result = " Legitimate Transaction"

    # If HTML request → return page
    if not request.is_json:
        return render_template("home.html", prediction_text=result)

    # If API request → return JSON
    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)