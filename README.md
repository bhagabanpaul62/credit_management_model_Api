# Credit Risk Scoring System

End‑to‑end project for training a credit risk (good / bad) classification model, exposing it via a FastAPI service, and interacting through a Next.js frontend form.

---

## 1. Repository Structure

```
fast-api/                  # Backend (FastAPI) service + model code
  api.py                   # REST API (load artifacts, /predict endpoint)
  train_credit_model.py    # Full training pipeline (real dataset)
  create_synthetic_artifacts.py  # Generate placeholder artifacts (demo/testing)
  artifacts/               # Serialized model + scaler + metadata (generated)
frontend/                  # Next.js React UI form
  app/page.jsx             # Form posting to API /predict
  .env                     # NEXT_PUBLIC_API_BASE_URL config
clening/                   # (Legacy/working) raw & merged CSV files
train.py                   # (Legacy / prior experiments)
README.md                  # This file
```

> NOTE: Some legacy folders (e.g. `clening/`, `train.py`) are kept for reference. Core, current code paths are `fast-api/` and `frontend/`.

---

## 2. Data & Feature Engineering Pipeline

Implemented in `fast-api/train_credit_model.py`.

Steps:

1. Load CSV (configured at `DATA_PATH`).
2. Drop identifier column (`ID`) if present.
3. Clean textual numeric columns:
   - Remove `$`, `%`, commas, and whitespace.
   - Coerce to numeric; invalid strings -> NaN.
4. Target (`TARGET`) coerced to numeric; rows with missing target dropped.
5. Features: all non-target columns.
6. Missing feature values imputed with mean of each column (stored as `impute_values.pkl`).
7. Split: Train / Val / Test (70% / 20% / 10% overall via 70% then 30% split, and internal 33% of temp for test) with stratification.
8. Scale features with `StandardScaler` (fitted on train only, saved as `scaler.pkl`).
9. Train `LogisticRegression` (lbfgs, `max_iter=1000`, `class_weight='balanced'`).
10. Evaluate on validation & test sets (Accuracy, ROC AUC, Confusion Matrix, Classification Report printed to console).
11. Persist artifacts:
    - `model.pkl` (fitted logistic regression)
    - `scaler.pkl` (fitted scaler)
    - `feature_names.pkl` (ordered list of model input feature names)
    - `impute_values.pkl` (Series of mean values used for imputation)

---

## 3. Synthetic Artifacts (Optional)

If you do not have the original dataset you can still bring up the API/UI using synthetic artifacts:

```
cd fast-api
python create_synthetic_artifacts.py
```

This script fabricates numeric feature distributions and a synthetic target, trains a logistic model, and writes artifacts. These are for **demo/testing only** – not production credit decisions.

> IMPORTANT: The synthetic script currently writes to `credit_risk/artifacts` in code comments; ensure artifacts end up in `fast-api/artifacts/` (adjust if needed).

---

## 4. Feature Mapping (Frontend -> Model)

The API maps human-friendly form keys to internal model feature names (subset shown):

```
age -> TLTimeFirst
annual_income -> Income
employment_years -> TLTimeLast
derogatory_marks -> DerogCnt
inquiries_last6m -> InqCnt06
inquiries_finance_24m -> InqFinanceCnt24
total_accounts -> TLCnt
active_accounts -> TLSatCnt
high_credit_util_75 -> TL75UtilCnt
util_50_plus -> TL50UtilCnt
balance_high_credit_pct -> TLBalHCPct
satisfied_pct -> TLSatPct
delinquency_30_60_24m -> TLDel3060Cnt24
delinquency_90d_24m -> TLDel90Cnt24
delinquencies_60d -> TLDel60Cnt
chargeoffs_last24m -> TLBadCnt24
derog_or_bad_cnt -> TLBadDerogCnt
accounts_open_last24m -> TLOpen24Pct
max_account_balance -> TLMaxSum
total_balance -> TLSum
```

Any remaining model features missing from request are filled with mean impute values.

---

## 5. Prediction Flow (Runtime)

1. Backend starts, loads artifacts from `fast-api/artifacts`.
2. Frontend form collects user inputs (numbers or yes/no for booleans).
3. Frontend POSTs JSON to `/predict` with fields above.
4. API:
   - Builds a feature vector ordered by `feature_names.pkl`.
   - Coerces / cleans inputs (strips symbols, converts yes/no to 1/0 in frontend).
   - Fills missing with imputed means.
   - Scales with `scaler.pkl`.
   - Logistic regression outputs probability of the "bad" class.
   - Applies threshold (default 0.5) -> status + probabilities returned.

Response JSON example:

```json
{
  "status": "Likely Eligible",
  "prediction": "Good Credit",
  "probability_bad": 0.1342,
  "probability_good": 0.8658,
  "threshold_used": 0.5,
  "model_version": "1.1.0"
}
```

---

## 6. API Endpoints

Base URL (default dev): `http://127.0.0.1:8000`

| Method | Path       | Description                   |
| ------ | ---------- | ----------------------------- |
| GET    | `/`        | Basic info message            |
| GET    | `/health`  | Health check `{ "ok": true }` |
| POST   | `/predict` | Score a credit application    |

Test quickly with PowerShell:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

---

## 7. Frontend (Next.js)

Location: `frontend/`

- Uses `axios` to POST to `process.env.NEXT_PUBLIC_API_BASE_URL + /predict`.
- Environment file: `frontend/.env` (add `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000`).
- Boolean selects (Yes/No) converted client-side to 1/0.

Run dev server:

```powershell
cd frontend
npm install   # first time
npm run dev
```

Default: http://localhost:3000

---

## 8. Running the Backend

```powershell
cd fast-api
# (Optional) create venv
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# If you have real data:
python train_credit_model.py
# Or create synthetic artifacts:
python create_synthetic_artifacts.py

# Start API
uvicorn api:app --reload --port 8000
```

Visit: http://127.0.0.1:8000/docs for interactive Swagger UI.

---

## 9. Retraining / Updating the Model

1. Place or point `DATA_PATH` in `train_credit_model.py` to your dataset.
2. Ensure target column name matches (`TARGET`).
3. Run the training script.
4. Verify metrics printed in console meet expectations.
5. (Optional) version artifacts by copying to a date/versioned folder before overwriting.
6. Restart the API process to load new artifacts.

> Keep `scikit-learn` version consistent between training and serving to avoid `InconsistentVersionWarning` when unpickling.

---

## 10. Environment Variables

Backend (optional future):

```
MODEL_THRESHOLD=0.5
ALLOW_ORIGINS=*
```

Frontend:

```
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

---

## 11. Troubleshooting

| Issue                           | Cause                                      | Fix                                             |
| ------------------------------- | ------------------------------------------ | ----------------------------------------------- |
| RuntimeError: Artifacts missing | No model files in artifacts dir            | Run training or synthetic script                |
| InconsistentVersionWarning      | Different sklearn versions for pickle load | Align versions or retrain under current version |
| 404 on /predict                 | Wrong base URL in frontend env             | Update `.env` and restart dev server            |
| CORS errors                     | Origin restrictions (if tightened)         | Add frontend origin to `allow_origins`          |

---

## 12. Security & Compliance Notes

This demo does not implement:

- Authentication / authorization
- Audit logging
- Bias / fairness audits
- PII handling & encryption

Add these before any production or regulated deployment.

---

## 13. Roadmap / Improvement Ideas

- Add model versioning & metadata endpoint (`/model-info`).
- Threshold tuning via ROC / F1 optimization script.
- Add Dockerfile & docker-compose for one-command spin‑up.
- Batch scoring job script.
- Monitoring: log inputs + latency + drift stats.
- Replace logistic regression with calibrated gradient boosting / XGBoost.

---

## 14. License

Add your chosen license (e.g., MIT) here.

---

## 15. Quick Start (Shortest Path)

```powershell
# Backend
cd fast-api
python create_synthetic_artifacts.py   # or python train_credit_model.py
uvicorn api:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:3000, submit form, see prediction.

---

## 16. Disclaimer



---

Happy building!
