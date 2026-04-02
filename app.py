import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import traceback

app = FastAPI(title="Credit Risk Scoring API  LendingClub", version="1.0")

models_path = r"C:\Users\CLYDE\credit_risk_project\models"

with open(f"{models_path}/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Rebuild preprocessors fresh
df = pd.read_csv(r"C:\Users\CLYDE\credit_risk_project\data\lending_club.csv")
df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
df["LOAN_TO_INCOME"]        = df["loan_amount"] / (df["annual_income"] + 1)
df["INSTALLMENT_TO_INCOME"] = df["installment"] / (df["annual_income"] / 12 + 1)
df["HIGH_INTEREST"]         = (df["interest_rate"] > 15).astype(int)
df["CREDIT_UTILIZATION"]    = df["total_credit_utilized"] / (df["total_credit_limit"] + 1)
df["HAS_DELINQUENCY"]       = (df["delinq_2y"] > 0).astype(int)
df["CREDIT_AGE_YEARS"]      = 2024 - pd.to_datetime(df["earliest_credit_line"]).dt.year
df["OPEN_CREDIT_RATIO"]     = df["open_credit_lines"] / (df["total_credit_lines"] + 1)
df["HAS_BANKRUPTCY"]        = (df["public_record_bankrupt"] > 0).astype(int)
df["LOAN_PER_TERM"]         = df["loan_amount"] / df["term"]
df["BALANCE_TO_INCOME"]     = df["balance"] / (df["annual_income"] + 1)

FEATURES = [
    "loan_amount", "interest_rate", "installment", "annual_income",
    "debt_to_income", "total_credit_lines", "open_credit_lines",
    "total_credit_limit", "total_credit_utilized", "delinq_2y",
    "public_record_bankrupt", "balance", "term",
    "LOAN_TO_INCOME", "INSTALLMENT_TO_INCOME", "HIGH_INTEREST",
    "CREDIT_UTILIZATION", "HAS_DELINQUENCY", "CREDIT_AGE_YEARS",
    "OPEN_CREDIT_RATIO", "HAS_BANKRUPTCY", "LOAN_PER_TERM", "BALANCE_TO_INCOME"
]

X = df[FEATURES].copy()
imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()
imputer.fit(X)
scaler.fit(imputer.transform(X))

print("Preprocessors rebuilt successfully!")

class ApplicantData(BaseModel):
    loan_amount:            float
    interest_rate:          float
    installment:            float
    annual_income:          float
    debt_to_income:         float
    total_credit_lines:     float
    open_credit_lines:      float
    total_credit_limit:     float
    total_credit_utilized:  float
    delinq_2y:              float
    public_record_bankrupt: float
    balance:                float
    term:                   float
    earliest_credit_year:   int

@app.get("/")
def home():
    return {"message": "Credit Risk Scoring API is live!"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost"}

@app.post("/predict")
def predict(data: ApplicantData):
    try:
        loan_to_income        = data.loan_amount / (data.annual_income + 1)
        installment_to_income = data.installment / (data.annual_income / 12 + 1)
        high_interest         = int(data.interest_rate > 15)
        credit_utilization    = data.total_credit_utilized / (data.total_credit_limit + 1)
        has_delinquency       = int(data.delinq_2y > 0)
        credit_age_years      = 2024 - data.earliest_credit_year
        open_credit_ratio     = data.open_credit_lines / (data.total_credit_lines + 1)
        has_bankruptcy        = int(data.public_record_bankrupt > 0)
        loan_per_term         = data.loan_amount / (data.term + 1)
        balance_to_income     = data.balance / (data.annual_income + 1)

        features = np.array([[
            data.loan_amount, data.interest_rate, data.installment,
            data.annual_income, data.debt_to_income, data.total_credit_lines,
            data.open_credit_lines, data.total_credit_limit, data.total_credit_utilized,
            data.delinq_2y, data.public_record_bankrupt, data.balance, data.term,
            loan_to_income, installment_to_income, high_interest,
            credit_utilization, has_delinquency, credit_age_years,
            open_credit_ratio, has_bankruptcy, loan_per_term, balance_to_income
        ]])

        features   = imputer.transform(features)
        features   = scaler.transform(features)
        risk_score = model.predict_proba(features)[0][1]

        if risk_score > 0.05:
            risk_level     = "HIGH"
            recommendation = "Decline"
            reason         = "Applicant shows high probability of default"
        elif risk_score > 0.02:
            risk_level     = "MEDIUM"
            recommendation = "Manual Review"
            reason         = "Applicant shows moderate default risk"
        else:
            risk_level     = "LOW"
            recommendation = "Approve"
            reason         = "Applicant shows low probability of default"

        return {
            "default_probability": round(float(risk_score), 4),
            "risk_level":          risk_level,
            "recommendation":      recommendation,
            "reason":              reason
        }

    except Exception as e:
        return JSONResponse(status_code=500,
                           content={"error": str(e),
                                    "detail": traceback.format_exc()})