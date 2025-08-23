from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path
from typing import Optional

ARTIFACT_DIR = Path('credit_risk') / 'artifacts'

def _load_artifacts():
    """Load model artifacts, with a clearer error if missing."""
    required = [
        ARTIFACT_DIR / 'model.pkl',
        ARTIFACT_DIR / 'scaler.pkl',
        ARTIFACT_DIR / 'feature_names.pkl',
        ARTIFACT_DIR / 'impute_values.pkl'
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "Artifacts missing: " + ', '.join(missing) +
            "\nGenerate them by running 'python credit_risk/train_credit_model.py' "
            "or create synthetic ones via 'python credit_risk/create_synthetic_artifacts.py'."
        )
    model_ = joblib.load(required[0])
    scaler_ = joblib.load(required[1])
    feature_names_ = joblib.load(required[2])
    impute_values_ = joblib.load(required[3])
    return model_, scaler_, feature_names_, impute_values_

model, scaler, feature_names, impute_values = _load_artifacts()

app = FastAPI(title='Credit Risk Scoring API', version="1.1.0")

# Allow all origins for hackathon demo (tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend -> model feature mapping (subset, extend as needed)
FIELD_MAP = {
    "age": "TLTimeFirst",  # NOTE: adjust if you actually trained with Age
    "annual_income": "Income",
    "employment_years": "TLTimeLast",  # placeholder mapping
    "derogatory_marks": "DerogCnt",
    "inquiries_last6m": "InqCnt06",
    "inquiries_finance_24m": "InqFinanceCnt24",
    "total_accounts": "TLCnt",
    "active_accounts": "TLSatCnt",
    "high_credit_util_75": "TL75UtilCnt",
    "util_50_plus": "TL50UtilCnt",
    "balance_high_credit_pct": "TLBalHCPct",
    "satisfied_pct": "TLSatPct",
    "delinquency_30_60_24m": "TLDel3060Cnt24",
    "delinquency_90d_24m": "TLDel90Cnt24",
    "delinquencies_60d": "TLDel60Cnt",
    "chargeoffs_last24m": "TLBadCnt24",
    "derog_or_bad_cnt": "TLBadDerogCnt",
    "accounts_open_last24m": "TLOpen24Pct",
    "max_account_balance": "TLMaxSum",
    "total_balance": "TLSum"
}

THRESHOLD = 0.5

class CreditForm(BaseModel):
    age: Optional[float] = None
    annual_income: Optional[float] = None
    employment_years: Optional[float] = None
    derogatory_marks: Optional[float] = None
    inquiries_last6m: Optional[float] = None
    inquiries_finance_24m: Optional[float] = None
    total_accounts: Optional[float] = None
    active_accounts: Optional[float] = None
    high_credit_util_75: Optional[float] = None
    util_50_plus: Optional[float] = None
    balance_high_credit_pct: Optional[float] = None
    satisfied_pct: Optional[float] = None
    delinquency_30_60_24m: Optional[float] = None
    delinquency_90d_24m: Optional[float] = None
    delinquencies_60d: Optional[float] = None
    chargeoffs_last24m: Optional[float] = None
    derog_or_bad_cnt: Optional[float] = None
    accounts_open_last24m: Optional[float] = None
    max_account_balance: Optional[float] = None
    total_balance: Optional[float] = None

def _coerce_float(val, feat):
    if val is None:
        return float(impute_values.get(feat, 0.0))
    if isinstance(val, (int, float)):
        return float(val)
    # string cleaning
    if isinstance(val, str):
        cleaned = val.replace('$','').replace('%','').replace(',','').strip()
        try:
            return float(cleaned)
        except ValueError:
            return float(impute_values.get(feat, 0.0))
    return float(impute_values.get(feat, 0.0))

@app.post('/predict')
def predict(form: CreditForm):
    incoming = form.dict(exclude_unset=True)
    ordered = []
    # Build dict keyed by model feature names
    mapped = {}
    for user_key, feature in FIELD_MAP.items():
        if feature not in feature_names:
            continue
        value = incoming.get(user_key)
        mapped[feature] = _coerce_float(value, feature)
    # Fill remainder using impute values
    for feat in feature_names:
        if feat not in mapped:
            mapped[feat] = float(impute_values.get(feat, 0.0))
        ordered.append(mapped[feat])

    X = np.array(ordered).reshape(1, -1)
    Xs = scaler.transform(X)
    prob_bad = float(model.predict_proba(Xs)[0,1])
    prob_good = 1 - prob_bad
    is_bad = int(prob_bad >= THRESHOLD)
    return {
        "status": "High Risk" if is_bad else "Likely Eligible",
        "prediction": "Bad Credit" if is_bad else "Good Credit",
        "probability_bad": round(prob_bad, 4),
        "probability_good": round(prob_good, 4),
        "threshold_used": THRESHOLD,
        "model_version": "1.1.0"
    }

@app.get('/')
def root():
    return {"message": "Credit Risk Scoring API. POST JSON to /predict"}

@app.get('/health')
def health():
    return {"ok": True}
