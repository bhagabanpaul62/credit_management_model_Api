"""Utility to create synthetic training artifacts when the original dataset
is unavailable. This lets the FastAPI service start for UI / integration
testing. DO NOT use these artifacts for real credit risk decisions.

Run:
  python credit_risk/create_synthetic_artifacts.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

ARTIFACT_DIR = Path('credit_risk') / 'artifacts'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Mirror FIELD_MAP feature names used by the API (values side)
FIELD_MAP_VALUES = [
    'TLTimeFirst', 'Income', 'TLTimeLast', 'DerogCnt', 'InqCnt06', 'InqFinanceCnt24',
    'TLCnt', 'TLSatCnt', 'TL75UtilCnt', 'TL50UtilCnt', 'TLBalHCPct', 'TLSatPct',
    'TLDel3060Cnt24', 'TLDel90Cnt24', 'TLDel60Cnt', 'TLBadCnt24', 'TLBadDerogCnt',
    'TLOpen24Pct', 'TLMaxSum', 'TLSum'
]

feature_names = FIELD_MAP_VALUES  # order as listed
n_features = len(feature_names)

rows = 500
rng = np.random.default_rng(42)

# Generate synthetic numeric features with varied scales
data = {}
for name in feature_names:
    if 'Pct' in name or 'Util' in name:
        data[name] = rng.normal(40, 20, rows).clip(0, 100)  # percentage like
    elif 'Income' in name:
        data[name] = rng.lognormal(mean=10, sigma=0.4, size=rows) / 1e5  # scaled
    elif 'Sum' in name or 'Max' in name:
        data[name] = rng.lognormal(mean=8, sigma=0.6, size=rows)
    else:
        data[name] = rng.poisson(2, rows)

df = pd.DataFrame(data)

# Synthetic target: probability increases with some risk proxies
risk_score = (
    0.02 * df['TL75UtilCnt'] +
    0.02 * df['TL50UtilCnt'] +
    0.01 * df['TLDel60Cnt'] +
    0.03 * df['TLBadCnt24'] +
    0.02 * df['DerogCnt'] +
    0.000002 * df['TLMaxSum'] -
    0.000001 * df['Income']
)
prob = 1 / (1 + np.exp(-risk_score))
y = (rng.random(rows) < prob).astype(int)

impute_values = df.mean()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
model = LogisticRegression(max_iter=500, class_weight='balanced')
model.fit(X_scaled, y)

joblib.dump(model, ARTIFACT_DIR / 'model.pkl')
joblib.dump(scaler, ARTIFACT_DIR / 'scaler.pkl')
joblib.dump(feature_names, ARTIFACT_DIR / 'feature_names.pkl')
joblib.dump(impute_values, ARTIFACT_DIR / 'impute_values.pkl')

print('Synthetic artifacts created in', ARTIFACT_DIR.resolve())
