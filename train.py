import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import os

# ---------- Load Data ----------
train = pd.read_csv("data/train/train.csv")
val   = pd.read_csv("data/val/val.csv")
test  = pd.read_csv("data/test/test.csv")

# Split Features & Target
X_train, y_train = train.drop("Default", axis=1), train["Default"]
X_val, y_val     = val.drop("Default", axis=1), val["Default"]
X_test, y_test   = test.drop("Default", axis=1), test["Default"]

# Ensure all numeric (important for XGBoost/LightGBM)
X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
X_val   = X_val.apply(pd.to_numeric, errors="coerce").fillna(0)
X_test  = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

# ---------- Train XGBoost ----------
print("\n🚀 Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    use_label_encoder=False,
    random_state=42,
    early_stopping_rounds=50
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# ---------- Train LightGBM ----------
print("\n🚀 Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=10)]
)

# ---------- Evaluate Models ----------
print("\n📊 Evaluating Models...")

def evaluate_model(name, model):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]
    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))
    return roc_auc_score(y_test, probs)

xgb_auc = evaluate_model("XGBoost", xgb_model)
lgb_auc = evaluate_model("LightGBM", lgb_model)

# ---------- Save Best Model ----------
os.makedirs("models", exist_ok=True)

if xgb_auc >= lgb_auc:
    joblib.dump(xgb_model, "models/best_model.pkl")
    print("\n✅ Saved best model: XGBoost → models/best_model.pkl")
else:
    joblib.dump(lgb_model, "models/best_model.pkl")
    print("\n✅ Saved best model: LightGBM → models/best_model.pkl")
