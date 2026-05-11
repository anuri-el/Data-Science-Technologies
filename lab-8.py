import os
import numpy as np
import pandas as pd

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
    print(f"  0 = Delayed:                       {n_bad}  ({n_bad/len(df)*100:.1f}%)")

    print(f"\n{SEP}")
    print(f"New Features:")
    new_feats = ["age","job_tenure_months","addr_tenure_months","dti",
                 "loan_to_income","daily_payment","expense_ratio",
                 "net_income","has_active_loans","is_repeat_client",
                 "applied_night","applied_weekend"]
    for f in new_feats:
        print(f"  {f:<20}: mean={df[f].mean():<10.3f}  std={df[f].std():.3f}")


    print(f"\n{SEP}")


def load_data():
    df = pd.read_excel(DATA_PATH)

    for col in ["applied_at", "birth_date", "employment_date",
                "fact_addr_start_date", "created_at", "closed_at",
                "prior_employment_start_date", "prior_employment_end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["target"] = ((df["loan_overdue"] == 0) & (df["overdue_days"].fillna(0) <= 2)).astype(int)

    df.rename(columns={"Marital status": "marital_status_id", "Application": "id"}, inplace=True)

    return df


def feature_engineering(df: pd.DataFrame):
    print(f"\n{SEP}\n  БЛОК 2: FEATURE ENGINEERING\n{SEP}")

    df["age"] = ((REF_DATE - df["birth_date"]).dt.days / 365.25).round(1)

    df["job_tenure_months"] = (
        (REF_DATE - df["employment_date"]).dt.days / 30.44
    ).fillna(0).clip(lower=0).round(1)

    df["addr_tenure_months"] = (
        (REF_DATE - df["fact_addr_start_date"]).dt.days / 30.44
    ).fillna(0).clip(lower=0).round(1)

    df["dti"] = (
        (df["monthly_expenses"] + df["other_loans_about_monthly"].fillna(0))
        / df["monthly_income"].replace(0, np.nan)
    ).fillna(0).clip(0, 10)

    df["loan_to_income"] = (
        df["loan_amount"] / df["monthly_income"].replace(0, np.nan)
    ).fillna(0).clip(0, 50)

    df["daily_payment"] = (
        df["loan_amount"] / df["loan_days"].replace(0, np.nan)
    ).fillna(0)

    df["expense_ratio"] = (
        df["monthly_expenses"] / df["monthly_income"].replace(0, np.nan)
    ).fillna(0).clip(0, 10)

    df["net_income"] = df["monthly_income"] - df["monthly_expenses"]

    df["has_active_loans"] = (df["other_loans_active"].fillna(0) > 0).astype(int)

    df["prolongation_count"] = df["prolongation_number"].fillna(0)

    dup_users = df["user_id"].value_counts()
    df["is_repeat_client"] = (df["user_id"].map(dup_users) > 1).astype(int)

    df["applied_hour"] = df["applied_at"].dt.hour.fillna(12).astype(int)
    df["applied_night"] = ((df["applied_hour"] >= 23) | (df["applied_hour"] < 6)).astype(int)

    df["applied_weekend"] = (df["applied_at"].dt.dayofweek >= 5).astype(int)

    return df



if __name__ == "__main__":
    main()