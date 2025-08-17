import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ---------- Load Data ----------
df = pd.read_csv(
    r"c:/Users/bhaga/OneDrive/Desktop/credit_managment/clening/combined_credit_data.csv",
    low_memory=False
)

# ---------- Clean Target Column ----------
# Drop rows where target ("Default") is missing
df = df.dropna(subset=["Default"])

# If "Default" column is not numeric, convert it (example: Yes/No → 1/0)
if df["Default"].dtype == "object":
    df["Default"] = df["Default"].map({"Yes": 1, "No": 0}).fillna(df["Default"])

# Features (X) and target (y)
X = df.drop("Default", axis=1)
y = df["Default"]

# ---------- Split Data ----------
# Train (70%) and Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Validation (20%) and Test (10%) from Temp
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.33, random_state=42, stratify=y_temp
)

# ---------- Recombine Features + Target ----------
train_set = pd.concat([X_train, y_train], axis=1)
val_set   = pd.concat([X_val, y_val], axis=1)
test_set  = pd.concat([X_test, y_test], axis=1)

# ---------- Create Folders ----------
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/val", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

# ---------- Save Files ----------
train_set.to_csv("data/train/train.csv", index=False)
val_set.to_csv("data/val/val.csv", index=False)
test_set.to_csv("data/test/test.csv", index=False)

print("✅ Data split completed and saved in 'data/' folder")
print("Train shape:", train_set.shape)
print("Validation shape:", val_set.shape)
print("Test shape:", test_set.shape)
print("Unique values in target:", y.unique())
