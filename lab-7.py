import os
import pandas as pd

SEP = "=" * 67

OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(OUTPUT_DIR, "Data_Set_7.xlsx")

def main():
    print(f"\n{SEP}")

    df = load_and_clean()
    print(f"Loaded: {df.shape[0]} rows * {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    print(df.info())


    print(f"\n{SEP}")
    print(f"Anomalies in Revenue (3*IQR): {df['IsAnomaly'].sum()} ({df['IsAnomaly'].mean()*100:.1f}%)")
    print(f"Date range: {df['OrderDate'].min().date()} - {df['OrderDate'].max().date()}")
    print(f"Unique customers: {df['CustomerID'].nunique()}")
    print(f"Unique products: {df['ProductName'].nunique()}")
    print(f"Counties: {df['CustomerCountry'].nunique()}")
    print(f"Employees: {df['EmployeeName'].nunique()}")
    print(f"Paid: {df['IsPaid'].sum()} ({df['IsPaid'].mean()*100:.1f}%) / Not paid: {(~df['IsPaid']).sum()}")

    print(f"\n  Revenue: μ={df['Revenue'].mean():.2f}  std={df['Revenue'].std():.2f} min={df['Revenue'].min():.2f} max={df['Revenue'].max():.2f}")
    
    print(f"\n{SEP}")
    olap = build_olap(df)

    print(f"\n  Monthly data: {len(olap['monthly'])} months")
    print(f"\n  {'Month':<10} {'Revenue':>12} {'Orders':>10} {'Customers':>10} {'Paid %':>9}")
    for _, r in olap["monthly"].iterrows():
        if r["Revenue"] > 0:
            print(f"  {r['YM_str']:<10} {r['Revenue']:>12,.2f} {int(r['Orders']):>10} {int(r['Customers']):>10} {r['PaidPct']:>8.1f}%")
    



def load_and_clean():
    df = pd.read_excel(DATA_PATH, sheet_name="qrySales")

    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

    df["Year"] = df["OrderDate"].dt.year
    df["Month"] = df["OrderDate"].dt.month
    df["Quarter"] = df["OrderDate"].dt.quarter
    df["YearMonth"] = df["OrderDate"].dt.to_period("M")
    df["YearQuarter"]= df["OrderDate"].dt.to_period("Q")
    df["DayOfWeek"] = df["OrderDate"].dt.day_name()
    df["WeekNo"] = df["OrderDate"].dt.isocalendar().week.astype(int)

    df["IsPaid"] = df["Paid?"].str.strip().str.upper() == "YES"

    df.drop_duplicates(inplace=True)
    df.dropna(subset=["Revenue", "OrderDate"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    q1, q3 = df["Revenue"].quantile(0.25), df["Revenue"].quantile(0.75)
    iqr = q3 - q1
    anom = ((df["Revenue"] < q1 - 3*iqr) | (df["Revenue"] > q3 + 3*iqr))
    df["IsAnomaly"] = anom

    return df


def build_olap(df: pd.DataFrame):
    monthly = (df.groupby("YearMonth")
                 .agg(Revenue=("Revenue", sum), 
                      Orders=("Revenue", "count"), 
                      AvgOrder=("Revenue", "mean"), 
                      Customers=("CustomerID", "nunique"),
                      PaidRev=("Revenue", lambda x: x[df.loc[x.index, "IsPaid"]].sum()))
                 .reset_index()
                 .sort_values("YearMonth"))
    monthly["YM_str"] = monthly["YearMonth"].astype(str)
    monthly["UnpaidRev"] = monthly["Revenue"] - monthly["PaidRev"]
    monthly["PaidPct"] = monthly["PaidRev"] / monthly["Revenue"] * 100

    full_idx = pd.period_range(monthly["YearMonth"].min(), monthly["YearMonth"].max(), freq="M")
    monthly = monthly.set_index("YearMonth").reindex(full_idx)

    num_cols = ["Revenue", "Orders", "AvgOrder", "Customers", "PaidRev", "UnpaidRev", "PaidPct"]
    for col in num_cols:
        if col in monthly.columns:
            monthly[col] = monthly[col].fillna(0)
    monthly = monthly.reset_index().rename(columns={"index": "YearMonth"})
    monthly["YM_str"] = monthly["YearMonth"].astype(str)
    

    qurterly = (df.groupby("YearQuarter")
                  .agg(Revenue=("Revenue", "sum"), Orders=("Revenue", "count"))
                  .reset_index())
    qurterly["YQ_str"] = qurterly["YearQuarter"].astype(str)


    annual = (df.groupby("Year")
                .agg(Revenue=("Revenue", "sum"), Orders=("Revenue", "count"))
                .reset_index())

    return dict(monthly=monthly, qurterly=qurterly, annual=annual)



if __name__ == "__main__":
    main()
