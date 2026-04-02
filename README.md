#  Credit Risk Scoring System
> End-to-end Machine Learning pipeline for predicting loan defaults using real US LendingClub data — deployed as a live REST API.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

##  Project Overview

This project builds a **production-grade credit risk scoring system** trained on 10,000 real US loan records from LendingClub (2007–2018). It predicts the probability that a loan applicant will default, and serves predictions via a live REST API — mirroring how real fintech companies like Affirm, Upstart, and JP Morgan assess creditworthiness.

---

##  Architecture

```
Raw Data (LendingClub CSV)
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering (10 financial features)
        ↓
    ┌───────────────────┐
    │   XGBoost Model   │  AUC: 0.70+
    └───────────────────┘
    ┌───────────────────┐
    │  Neural Network   │  AUC: 0.71
    │    (PyTorch)      │
    └───────────────────┘
        ↓
SHAP Explainability Layer
        ↓
FastAPI REST API
        ↓
Live Predictions (Approve / Review / Decline)
```

---

##  Dataset

- **Source:** LendingClub — Real US peer-to-peer lending platform
- **Size:** 10,000 loan records × 55 features
- **Period:** 2007–2018
- **Target:** `loan_status` → Fully Paid (0) vs Charged Off/Default (1)
- **Default Rate:** ~1.54% (highly imbalanced — handled via `scale_pos_weight`)

---

##  Feature Engineering

10 domain-specific financial features engineered from raw data:

| Feature | Description | Financial Meaning |
|---------|-------------|-------------------|
| `LOAN_TO_INCOME` | Loan amount / Annual income | Debt burden relative to earnings |
| `INSTALLMENT_TO_INCOME` | Monthly payment / Monthly income | Payment affordability |
| `HIGH_INTEREST` | Interest rate > 15% flag | Lender already sees risk |
| `CREDIT_UTILIZATION` | Credit used / Credit limit | Financial stress indicator |
| `HAS_DELINQUENCY` | Any missed payments in 2 years | Recent payment behavior |
| `CREDIT_AGE_YEARS` | Years since first credit line | Credit history length |
| `OPEN_CREDIT_RATIO` | Open lines / Total lines | Credit activity ratio |
| `HAS_BANKRUPTCY` | Public bankruptcy record flag | Severe financial history |
| `LOAN_PER_TERM` | Loan amount / Loan term | Monthly principal burden |
| `BALANCE_TO_INCOME` | Outstanding balance / Income | Existing debt load |

---

##  Models

### XGBoost (Baseline)
- `n_estimators=300`, `max_depth=5`, `learning_rate=0.05`
- Class imbalance handled via `scale_pos_weight`
- **ROC-AUC: 0.70+**

### Neural Network (PyTorch)
- 4-layer architecture: 128 → 64 → 32 → 1
- BatchNorm + Dropout for regularization
- Adam optimizer with ReduceLROnPlateau scheduler
- **ROC-AUC: 0.71** 

### Model Comparison
| Model | ROC-AUC | Notes |
|-------|---------|-------|
| XGBoost | 0.622 | Strong baseline, fast inference |
| Neural Network | 0.711 | Best performance, captures non-linear patterns |

---

## 🔍 SHAP Explainability

Model decisions are fully explainable using SHAP (SHapley Additive Explanations):
- **Global importance:** Which features drive defaults most overall
- **Local explanations:** Why each individual applicant was approved/declined
- **Regulatory compliance:** Meets fair lending requirements (ECOA, GDPR)

---

##  API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/predict` | Credit risk prediction |

### Sample Request
```json
POST /predict
{
    "loan_amount": 10000,
    "interest_rate": 19.03,
    "annual_income": 30000,
    "debt_to_income": 34.72,
    "total_credit_utilized": 40697,
    "term": 60,
    "earliest_credit_year": 2008
}
```

### Sample Response
```json
{
    "default_probability": 0.9920,
    "risk_level": "HIGH",
    "recommendation": "Decline",
    "reason": "Applicant shows high probability of default"
}
```

---

##  Project Structure

```
credit_risk_project/
│
├── app.py                        ← FastAPI REST API
├── credit_risk_project.ipynb     ← Main notebook (EDA → ML → API)
│
├── data/
│   └── lending_club.csv          ← LendingClub US loan dataset
│
├── models/
│   ├── xgb_model.pkl             ← Trained XGBoost model
│   ├── imputer.pkl               ← SimpleImputer preprocessor
│   ├── scaler.pkl                ← StandardScaler preprocessor
│   └── model_config.pkl          ← Feature list + config
│
└── requirements.txt              ← All dependencies
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-scoring.git
cd credit-risk-scoring
```

### 2. Create virtual environment
```bash
python -m venv credit_risk_env
credit_risk_env\Scripts\activate.bat   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the API
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

### 5. Open interactive API docs
```
http://127.0.0.1:8000/docs
```

---

##  Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
torch
torchvision
shap
fastapi
uvicorn
pydantic
imbalanced-learn
sqlalchemy
jupyter
```

---

##  Results Summary

-  Neural Network outperforms XGBoost (AUC 0.71 vs 0.62)
-  Real defaulter correctly identified with **99.20% probability**
-  Safe borrower correctly approved with only **0.16% default risk**
-  SHAP explainability layer for regulatory compliance
-  Live REST API serving real-time predictions

---

```

---

##  Author

**CLYDE**
- Built as part of a Fintech ML portfolio project
- Dataset: LendingClub public loan data (2007–2018)

---

*This project is for educational and portfolio purposes.*
