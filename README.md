# UK Fraud Detection
**Author:** Sahil Singh Choudhary — MSc Risk Analytics, Queen Mary University of London  
**Stack:** Python · scikit-learn · SHAP · pandas · matplotlib

---

## What this project does

Builds a machine learning fraud detection system on 200,000 synthetic UK banking transactions. The model flags suspicious transactions in real time — the same core task performed by fraud analytics teams at Barclays, Lloyds, HSBC, Monzo, and Revolut.

In 2023, UK fraud losses totalled **£1.17 billion** (UK Finance). This project demonstrates how banks use ML to detect and prevent that loss.

---

## Results

| Model | AUC-ROC | AUC-PR |
|---|---|---|
| Logistic Regression | ~0.97 | ~0.65 |
| Random Forest | ~0.99 | ~0.85 |
| Gradient Boosting | ~0.99 | ~0.83 |

> AUC-PR (Precision-Recall) is the key metric for fraud — class imbalance makes AUC-ROC optimistic. A model predicting "not fraud" for every transaction gets 99.2% accuracy but catches zero fraud.

---

## Key features engineered

| Feature | Description | Why it matters |
|---|---|---|
| `is_night` | Transaction between midnight–6am | Fraudsters operate when fraud teams are unstaffed |
| `is_card_not_present` | Online or mobile channel | CNP fraud = 76% of UK card fraud losses (UK Finance 2023) |
| `amount_above_95th` | Amount > 95th percentile for merchant category | Unusually large spend for context |
| `amount_tier` | GBP bucket (0–4) | Non-linear amount risk signal |
| `customer_velocity` | Transaction count per customer | Card testing generates high-frequency low-value transactions |

---

## UK Regulatory Context

- **PSD2 / Strong Customer Authentication** — model flags CNP transactions for step-up authentication
- **FCA Consumer Duty (2023)** — false positive rate tracked to protect legitimate customer journeys
- **Contingent Reimbursement Model (CRM)** — model output supports fraud reimbursement decisions

---

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook uk_fraud_detection.ipynb
```

Run all cells top to bottom. The notebook generates synthetic data, trains all models, and produces all charts automatically.

---

## Project structure

```
uk-fraud-detection/
├── uk_fraud_detection.ipynb   # Main notebook — full analysis
├── requirements.txt           # Python dependencies
├── fraud_model.pkl            # Saved best model (generated on run)
├── fraud_scaler.pkl           # Saved scaler (generated on run)
└── README.md
```

---

## Scoring a new transaction

```python
import joblib
import numpy as np

scaler = joblib.load('fraud_scaler.pkl')
model  = joblib.load('fraud_model.pkl')

# Feature order: amount_gbp, hour_of_day, is_foreign, is_night,
#                is_card_not_present, amount_above_95th, amount_tier,
#                customer_velocity, channel_enc, category_enc, city_enc
transaction = np.array([[250.0, 2, 1, 1, 1, 1, 3, 45, 2, 4, 7]])

fraud_score = model.predict_proba(scaler.transform(transaction))[0][1]
flagged     = fraud_score >= 0.30  # business threshold

print(f'Fraud probability: {fraud_score:.3f} — {"FLAGGED" if flagged else "Approved"}')
```

---

## Why synthetic data?

Real bank transaction data is highly confidential. Fraud teams routinely use synthetic data for model development — it mirrors real statistical patterns while protecting customer privacy. All distributions in this project are calibrated to UK Finance published statistics.

---

*Part of a portfolio demonstrating applied ML in UK financial services risk management.*
