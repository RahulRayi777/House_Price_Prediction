from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd

# Load trained model, encoders, and scaler
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")

# Define the app
app = Flask(__name__)

# Feature names (should match train.py)
categorical_cols = ["street", "city", "statezip", "country"]
numerical_cols = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", 
    "waterfront", "view", "condition", "sqft_above", "sqft_basement", 
    "yr_built", "yr_renovated"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from form
        input_data = request.form.to_dict()

        # Convert numerical inputs
        for col in numerical_cols:
            input_data[col] = float(input_data[col])

        # Encode categorical inputs
        for col in categorical_cols:
            input_data[col] = label_encoders[col].transform([input_data[col]])[0]

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Scale numerical features
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Make prediction
        predicted_price = model.predict(input_df)[0]

        # Risk Assessment Logic
        avg_price = 500000  # Adjust based on dataset insights
        risk_level = "High Risk" if predicted_price < avg_price * 0.7 else (
            "Medium Risk" if predicted_price < avg_price * 1.2 else "Low Risk"
        )

        return render_template(
            "index.html",
            prediction=f"Predicted House Price: ${predicted_price:,.2f}",
            risk=f"Risk Level: {risk_level}",
        )

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
