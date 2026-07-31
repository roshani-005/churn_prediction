# Customer Churn Prediction & Analytics Dashboard

Trains and compares Logistic Regression, Random Forest, and XGBoost to predict
customer churn, serves predictions via a Flask API, and visualizes risk in a
Streamlit dashboard.

## Setup
```bash
pip install -r requirements.txt
python train_model.py      # trains models, picks best by ROC-AUC, saves artifacts
python api.py               # serves predictions at http://localhost:5001/predict
streamlit run dashboard.py  # interactive UI at http://localhost:8501
```

## Example API call
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure_months": 5, "monthly_charges": 80, "support_tickets": 4, "contract_type": "month_to_month", "payment_delay_days": 10, "usage_drop_pct": 30}'
```

```json
{
  "churn_probability": 0.7885,
  "risk_level": "High",
  "model_used": "Logistic Regression"
}
```

> Note: `train_model.py` uses synthetic data generation so the project runs
> end-to-end without a real dataset — swap `generate_synthetic_data()` for a
> `pd.read_csv(...)` of your actual customer data to use in production.
