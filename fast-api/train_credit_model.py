import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import joblib

DATA_PATH = Path('01traning') / 'a_Dataset_CreditScoring.xlsx - Sheet1.csv'
ARTIFACT_DIR = Path('credit_risk') / 'artifacts'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = 'TARGET'
ID_COL = 'ID'

print(f'Loading data from {DATA_PATH}')
df = pd.read_csv(DATA_PATH)
print('Raw shape:', df.shape)

# Drop ID
if ID_COL in df.columns:
    df = df.drop(columns=[ID_COL])

# Clean currency and percentage columns
for col in df.columns:
    if df[col].dtype == object:
        df[col] = (df[col].astype(str)
                          .str.replace('$', '', regex=False)
                          .str.replace('%', '', regex=False)
                          .str.replace(',', '', regex=False)
                          .str.strip())
        # Empty strings to NaN
        df[col] = df[col].replace({'': np.nan, 'nan': np.nan, 'None': np.nan})

# Convert all non-target to numeric
feature_cols = [c for c in df.columns if c != TARGET_COL]
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Handle target
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')

# Missing values -> mean imputation
impute_values = df[feature_cols].mean()
df[feature_cols] = df[feature_cols].fillna(impute_values)

# Drop rows with missing target
before_target_drop = df.shape[0]
df = df.dropna(subset=[TARGET_COL])
print(f'Dropped {before_target_drop - df.shape[0]} rows with null target.')

X = df[feature_cols]
y = df[TARGET_COL].astype(int)

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42, stratify=y_temp)
print(f'Splits -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
model.fit(X_train_s, y_train)

# Validation metrics
val_preds = model.predict(X_val_s)
val_probs = model.predict_proba(X_val_s)[:,1]
print('\nValidation Metrics:')
print('Accuracy:', accuracy_score(y_val, val_preds))
print('ROC AUC:', roc_auc_score(y_val, val_probs))

# Test metrics
test_preds = model.predict(X_test_s)
test_probs = model.predict_proba(X_test_s)[:,1]
print('\nTest Metrics:')
print('Accuracy:', accuracy_score(y_test, test_preds))
print('ROC AUC:', roc_auc_score(y_test, test_probs))
print('Confusion Matrix:\n', confusion_matrix(y_test, test_preds))
print('Classification Report:\n', classification_report(y_test, test_preds))

# Save artifacts
joblib.dump(model, ARTIFACT_DIR / 'model.pkl')
joblib.dump(scaler, ARTIFACT_DIR / 'scaler.pkl')
joblib.dump(feature_cols, ARTIFACT_DIR / 'feature_names.pkl')
joblib.dump(impute_values, ARTIFACT_DIR / 'impute_values.pkl')
print('\nSaved artifacts to', ARTIFACT_DIR.resolve())
