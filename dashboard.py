"""
Customer Churn Prediction — REST API
--------------------------------------
Flask API that loads the trained model and serves real-time churn
predictions for a given customer profile.

Run:
    python train_model.py     # first, to generate the model artifacts
    python api.py
"""

import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load("churn_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_names = joblib.load("feature_names.joblib")
model_name = joblib.load("model_name.joblib")


def build_feature_row(payload: dict) -> pd.DataFrame:
    row = {
        "tenure_months": payload.get("tenure_months", 0),
        "monthly_charges": payload.get("monthly_charges", 0.0),
        "support_tickets": payload.get("support_tickets", 0),
        "payment_delay_days": payload.get("payment_delay_days", 0.0),
        "usage_drop_pct": payload.get("usage_drop_pct", 0.0),
        "contract_type_one_year": 1 if payload.get("contract_type") == "one_year" else 0,
        "contract_type_two_year": 1 if payload.get("contract_type") == "two_year" else 0,
    }
    df = pd.DataFrame([row])
    return df.reindex(columns=feature_names, fill_value=0)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": model_name})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    required = ["tenure_months", "monthly_charges", "support_tickets", "contract_type"]
    missing = [f for f in required if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    X = build_feature_row(payload)

    if model_name == "Logistic Regression":
        X_input = scaler.transform(X)
    else:
        X_input = X

    prob = float(model.predict_proba(X_input)[0][1])
    risk = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"

    return jsonify({
        "churn_probability": round(prob, 4),
        "risk_level": risk,
        "model_used": model_name,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
