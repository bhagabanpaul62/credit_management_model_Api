import pandas as pd
import os

# Check working directory
print("Current dir:", os.getcwd())

# ---------- Load Datasets ----------
df1 = pd.read_csv(r"c:/Users/bhaga/OneDrive/Desktop/credit_managment/clening/credit_score.csv")   # spending/behavior dataset
df2 = pd.read_csv(r"c:/Users/bhaga/OneDrive/Desktop/credit_managment/clening/Loan_default.csv")   # loan application dataset
df3 = pd.read_csv(r"c:/Users/bhaga/OneDrive/Desktop/credit_managment/clening/Loan.csv")           # simple loan dataset

# ---------- Standardize Column Names ----------
df1.rename(columns={
    "CUST_ID": "CustomerID",
    "INCOME": "Income",
    "SAVINGS": "Savings",
    "DEBT": "Debt",
    "CREDIT_SCORE": "CreditScore",
    "DEFAULT": "Default"
}, inplace=True)

df2.rename(columns={
    "ID": "CustomerID",
    "income": "Income",
    "Credit_Score": "CreditScore",
    "loan_amount": "LoanAmount",
    "dtir1": "DTIRatio",
    "Status": "Default"
}, inplace=True)

df3.rename(columns={
    "LoanID": "CustomerID",
    "Age": "Age",
    "Income": "Income",
    "LoanAmount": "LoanAmount",
    "CreditScore": "CreditScore",
    "DTIRatio": "DTIRatio",
    "Default": "Default"
}, inplace=True)

# ---------- Create Unified Schema ----------
unified_cols = [
    "CustomerID", "Age", "Gender", "Income", "Savings", "Debt", "LoanAmount",
    "CreditScore", "MonthsEmployed", "NumCreditLines", "InterestRate", "LoanTerm",
    "DTIRatio", "Education", "EmploymentType", "MaritalStatus", "HasMortgage",
    "HasDependents", "LoanPurpose", "HasCoSigner", "LTV", "Region",
    "Spending_Clothing", "Spending_Education", "Spending_Entertainment",
    "Spending_Gambling", "Spending_Groceries", "Spending_Health", "Spending_Housing",
    "Spending_Tax", "Spending_Travel", "Spending_Utilities",
    "CAT_Gambling", "CAT_Debt", "CAT_CreditCard", "CAT_Mortgage", "CAT_SavingsAccount",
    "Default"
]

# Add missing columns
for col in unified_cols:
    if col not in df1.columns: df1[col] = None
    if col not in df2.columns: df2[col] = None
    if col not in df3.columns: df3[col] = None

# Align to schema
df1 = df1[unified_cols]
df2 = df2[unified_cols]
df3 = df3[unified_cols]

# ---------- Merge All ----------
combined = pd.concat([df1, df2, df3], ignore_index=True)

# ---------- Export ----------
out_path = r"c:/Users/bhaga/OneDrive/Desktop/credit_managment/clening/combined_credit_data.csv"
combined.to_csv(out_path, index=False)

print("✅ Combined dataset created with shape:", combined.shape)
print("📂 Saved to:", out_path)
