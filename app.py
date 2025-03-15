#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify
import pickle
import os
import numpy as np

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/best_xgb_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

@app.route("/", methods=["GET"])
def home():
    return "House Price Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Define expected features
        expected_features = [
            "housing_median_age", "population", "median_income",
            "ocean_proximity", "rooms_per_household",
            "bedrooms_per_house", "population_per_household",
            "location_cluster"
        ]

        # Ensure all expected features are in the request
        if not all(feature in data for feature in expected_features):
            return jsonify({"error": "Missing one or more required features"}), 400

        # Prepare input data
        input_data = np.array([data[feature] for feature in expected_features]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Convert NumPy float32 to Python float to avoid JSON serialization error
        return jsonify({"predicted_price": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Ensure Flask runs on 0.0.0.0 for Docker
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

