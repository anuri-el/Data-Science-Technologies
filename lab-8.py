import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

SEP = "=" * 67

OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(OUTPUT_DIR, "sample_data.xlsx")
DESC_PATH = os.path.join(OUTPUT_DIR, "Datdata_descriptiona_Set_7.xlsx")

REF_DATE  = pd.Timestamp("2021-02-01")

def main():
    print(f"\n{SEP}")
    print("  SCORING")

    df = load_data()
    print(f"  Loaded: {df.shape[0]} applications * {df.shape[1]} features")

    n_good = df["target"].sum()
    n_bad = len(df) - n_good

    df = feature_engineering(df)

    print(f"\nTarget:")
    print(f"  1 = The loan was returned on time: {n_good} ({n_good/len(df)*100:.1f}%)")
    print(f"  0 = Delayed: {n_bad}  ({n_bad/len(df)*100:.1f}%)")

    print(f"\n{SEP}")
    print(f"New Features:")
    new_feats = ["age","job_tenure_months","addr_tenure_months","dti", "loan_to_income","daily_payment","expense_ratio",
                 "net_income","has_active_loans","is_repeat_client", "applied_night","applied_weekend"]
    for f in new_feats:
        print(f"  {f:<20}: mean={df[f].mean():<10.3f}  std={df[f].std():.3f}")


    print(f"\n{SEP}")
    df = detect_fraud(df)

    print(f"Fraud detection results:")
    print(f"{'Rule':<45} {'Trigger count':>12} {'%':>8}")
    for key, desc in FRAUD_RULES.items():
        cnt = df["fraud_flags"].str.contains(key, regex=False).sum()
        pct = cnt / len(df) * 100
        if cnt > 0:
            print(f"{desc:<45} {cnt:>12} {pct:>7.1f}%")


    print(f"\nIsolation Forest: {df['iso_anomaly'].sum()} anomalies")
    print(f"\nFraud risk categories:")
    fr_counts = df["fraud_risk"].value_counts()
    for cat, cnt in fr_counts.items():
        print(f"    {cat:<10}: {cnt:>5} ({cnt/len(df)*100:.1f}%)")


    print(f"\n{SEP}")





def load_data():
    df = pd.read_excel(DATA_PATH)

    for col in ["applied_at", "birth_date", "employment_date", "fact_addr_start_date", "created_at", "closed_at",
                "prior_employment_start_date", "prior_employment_end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["target"] = ((df["loan_overdue"] == 0) & (df["overdue_days"].fillna(0) <= 2)).astype(int)

    df.rename(columns={"Marital status": "marital_status_id", "Application": "id"}, inplace=True)

    return df


def feature_engineering(df: pd.DataFrame):
    df["age"] = ((REF_DATE - df["birth_date"]).dt.days / 365.25).round(1)

    df["job_tenure_months"] = ((REF_DATE - df["employment_date"]).dt.days / 30.44).fillna(0).clip(lower=0).round(1)

    df["addr_tenure_months"] = ((REF_DATE - df["fact_addr_start_date"]).dt.days / 30.44).fillna(0).clip(lower=0).round(1)

    df["dti"] = ((df["monthly_expenses"] + df["other_loans_about_monthly"].fillna(0)) / df["monthly_income"].replace(0, np.nan)).fillna(0).clip(0, 10)

    df["loan_to_income"] = (df["loan_amount"] / df["monthly_income"].replace(0, np.nan)).fillna(0).clip(0, 50)

    df["daily_payment"] = (df["loan_amount"] / df["loan_days"].replace(0, np.nan)).fillna(0)

    df["expense_ratio"] = (df["monthly_expenses"] / df["monthly_income"].replace(0, np.nan)).fillna(0).clip(0, 10)

    df["net_income"] = df["monthly_income"] - df["monthly_expenses"]

    df["has_active_loans"] = (df["other_loans_active"].fillna(0) > 0).astype(int)

    df["prolongation_count"] = df["prolongation_number"].fillna(0)

    dup_users = df["user_id"].value_counts()
    df["is_repeat_client"] = (df["user_id"].map(dup_users) > 1).astype(int)

    df["applied_hour"] = df["applied_at"].dt.hour.fillna(12).astype(int)
    df["applied_night"] = ((df["applied_hour"] >= 23) | (df["applied_hour"] < 6)).astype(int)

    df["applied_weekend"] = (df["applied_at"].dt.dayofweek >= 5).astype(int)

    return df


FRAUD_RULES = {
    "income_lt_expenses": "Income less than expenses",
    "impossible_seniority": "Work experience exceeds age-18",
    "extreme_income": "Anomalously high income (>5std)",
    "extreme_low_income": "Anomalously low income (<500 UAH)",
    "active_loans_no_payment": "Active loans with no monthly payments",
    "future_employment": "Employment start date in the future",
    "future_birth": "Date of birth in the future",
    "dti_extreme": "DTI > 3.0 (critical debt burden)",
    "loan_to_income_extreme": "Loan > 20 monthly incomes",
    "night_large_loan": "Night application + loan amount > 4000 UAH",
    "income_outlier_iqr": "Income — statistical outlier (IQR)",
    "inconsistent_overdue": "overdue_days > 0 but loan_overdue = 0",
    "seniority_years_zero_job_tenure": "Work experience > 0, but job start date missing",
}


def detect_fraud(df: pd.DataFrame):
    df["fraud_score"] = 0
    df["fraud_flags"]  = ""

    def add_flag(mask, key, weight=1):
        df.loc[mask, "fraud_flags"] += (key + "|")
        df.loc[mask, "fraud_score"] += weight

    mask = df["monthly_income"] < df["monthly_expenses"]
    add_flag(mask, "income_lt_expenses", weight=2)

    mask = df["seniority_years"] > (df["age"] - 18).clip(lower=0)
    add_flag(mask, "impossible_seniority", weight=3)

    mu, sigma = df["monthly_income"].mean(), df["monthly_income"].std()
    mask = df["monthly_income"] > mu + 5 * sigma
    add_flag(mask, "extreme_income", weight=2)

    mask = df["monthly_income"] < 500
    add_flag(mask, "extreme_low_income", weight=2)

    mask = (df["other_loans_active"].fillna(0) > 0) & (df["other_loans_about_monthly"].fillna(0) == 0)
    add_flag(mask, "active_loans_no_payment", weight=2)

    mask = df["employment_date"].notna() & (df["employment_date"] > REF_DATE)
    add_flag(mask, "future_employment", weight=3)

    mask = df["birth_date"].notna() & (df["birth_date"] > REF_DATE)
    add_flag(mask, "future_birth", weight=5)

    mask = df["dti"] > 3.0
    add_flag(mask, "dti_extreme", weight=1)

    mask = df["loan_to_income"] > 20
    add_flag(mask, "loan_to_income_extreme", weight=2)

    mask = (df["applied_night"] == 1) & (df["loan_amount"] > 4000)
    add_flag(mask, "night_large_loan", weight=1)

    q1, q3 = df["monthly_income"].quantile(0.25), df["monthly_income"].quantile(0.75)
    iqr = q3 - q1
    mask = (df["monthly_income"] < q1 - 3*iqr) | (df["monthly_income"] > q3 + 3*iqr)
    add_flag(mask, "income_outlier_iqr", weight=2)

    mask = (df["overdue_days"] > 2) & (df["loan_overdue"] == 0)
    add_flag(mask, "inconsistent_overdue", weight=3)

    mask = (df["seniority_years"] > 0) & (df["employment_date"].isna())
    add_flag(mask, "seniority_years_zero_job_tenure", weight=1)

    iso_features = ["monthly_income","loan_amount","loan_days", "monthly_expenses","dti","age","seniority_years", "loan_to_income"]
    X_iso = df[iso_features].fillna(df[iso_features].median())
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["iso_anomaly"] = (iso.fit_predict(X_iso) == -1).astype(int)
    df.loc[df["iso_anomaly"] == 1, "fraud_flags"] += "iso_forest|"
    df.loc[df["iso_anomaly"] == 1, "fraud_score"] += 1

    df["fraud_risk"] = pd.cut(df["fraud_score"], bins=[-1, 0, 2, 5, 100], labels=["Clean", "Low", "Medium", "Critical"])

    return df




if __name__ == "__main__":
    main()