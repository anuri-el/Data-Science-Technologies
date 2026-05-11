import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

SEP = "=" * 67

OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(OUTPUT_DIR, "Data_Set_7.xlsx")


C = dict(
    bg="#0D1117", panel="#161B22", grid="#30363D",
    text="#E6EDF3", sub="#8B949E",
    c0="#2196F3", c1="#FF9800", c2="#4CAF50", c3="#E91E63",
    c4="#9C27B0", c5="#00BCD4", c6="#FF5722", c7="#8BC34A",
    gold="#FFD700", best="#66BB6A", worst="#EF5350",
    paid="#4CAF50", unpaid="#EF5350",
)
PAL8 = [C["c0"],C["c1"],C["c2"],C["c3"],C["c4"],C["c5"],C["c6"],C["c7"]]


def main():
    print(f"\n{SEP}")

    df = load_and_clean()
    print(f"Loaded: {df.shape[0]} rows * {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")


    print(f"\n{SEP}")
    print(f"Anomalies in Revenue (3*IQR): {df['IsAnomaly'].sum()} ({df['IsAnomaly'].mean()*100:.1f}%)")
    print(f"Date range: {df['OrderDate'].min().date()} - {df['OrderDate'].max().date()}")
    print(f"Unique customers: {df['CustomerID'].nunique()}")
    print(f"Unique products: {df['ProductName'].nunique()}")
    print(f"Counties: {df['CustomerCountry'].nunique()}")
    print(f"Employees: {df['EmployeeName'].nunique()}")
    print(f"Paid: {df['IsPaid'].sum()} ({df['IsPaid'].mean()*100:.1f}%) / Not paid: {(~df['IsPaid']).sum()}")

    print(f"\nRevenue: mu={df['Revenue'].mean():.2f} | std={df['Revenue'].std():.2f} | min={df['Revenue'].min():.2f} | max={df['Revenue'].max():.2f}")
    

    print(f"\n{SEP}")
    olap = build_olap(df)
    monthly = olap["monthly"]
    quarterly = olap["quarterly"]

    print(f"Monthly data: {len(monthly)} months")
    print(f"\n{'Month':<10} {'Revenue':>12} {'Orders':>10} {'Customers':>10} {'Paid %':>9}")
    for _, r in monthly.iterrows():
        if r["Revenue"] > 0:
            print(f"{r['YM_str']:<10} {r['Revenue']:>12,.2f} {int(r['Orders']):>10} {int(r['Customers']):>10} {r['PaidPct']:>8.1f}%")
    

    print(f"\n{SEP}")
    kpi = compute_kpi(df)
    employees = kpi["employees"]
    countries = kpi["countries"]
    products = kpi["products"]
    customers = kpi["customers"]
    
    print(f"\n  Ефективність менеджерів:")
    print(f"  {'Employee':<22} {'Revenue':>12} {'Orders':>8} {'AvgOrder':>10} {'Customers':>8} {'Paid%':>8}")
    for name, r in employees.iterrows():
        print(f"  {name:<22} {r['Revenue']:>12,.0f} {int(r['Orders']):>8} {r['AvgOrder']:>10,.1f} {int(r['Customers']):>8} {r['PaidPct']:>7.1f}%")

    n_A = (customers["ABC"]=="A").sum()
    n_B = (customers["ABC"]=="B").sum()
    n_C = (customers["ABC"]=="C").sum()
    print(f"\nABC customer analysis:")
    print(f"  A (80% revenue): {n_A} customers ({n_A/len(customers)*100:.1f}%)")
    print(f"  B (15% revenue): {n_B} customers ({n_B/len(customers)*100:.1f}%)")
    print(f"  C (5% revenue):  {n_C} customers ({n_C/len(customers)*100:.1f}%)")

    unpaid_rev = df[~df["IsPaid"]]["Revenue"].sum()
    total_rev = df["Revenue"].sum()
    print(f"Risk (unpaid revenue):")
    print(f"Paid: ${df[df['IsPaid']]['Revenue'].sum():>12,.2f} ({df['IsPaid'].mean()*100:.1f}%)")
    print(f"Not Paid: ${unpaid_rev:>12,.2f} ({unpaid_rev/total_rev*100:.1f}%)")


    print(f"\n{SEP}")
    decomp_res = decompose_series(monthly)
    ts = decomp_res["ts"]

    print(f"\nADF test for stationarity:")
    print(f"  Statistic: {decomp_res['adf_stat']:.4f}  p-value: {decomp_res['adf_p']:.4f}")
    print(f"  Conclusion: {'Stationary' if decomp_res['adf_p'] < 0.05 else 'Non-stationary'} series")

    print(f"\nADF after 1st differencing:")
    print(f"  Statistic: {decomp_res['adf2']:.4f}  p-value: {decomp_res['p2']:.4f}")
    print(f"  Conclusion: {'Stationary' if decomp_res['p2'] < 0.05 else 'Non-stationary'} series")

    print(f"\nACF (lags 1-5):  {[f'{v:.3f}' for v in decomp_res['acf'][1:6]]}")
    print(f"PACF (lags 1-5): {[f'{v:.3f}' for v in decomp_res['pacf'][1:6]]}")


    plot_monthly_revenue_stacked(monthly, "l7_monthly_revenue.png")
    plot_top_countries(countries, "l7_top_countries.png")
    plot_manager_revenue(employees, "l7_manager_revenue.png")
    plot_manager_payment_percentage(employees, "l7_manager_payment_pct.png")
    plot_top_products(products, "l7_top_products.png")
    plot_abc_customer_pie(customers, "l7_abc_pie.png")
    plot_quarterly_revenue(quarterly, "l7_quarterly_revenue.png")

    plot_time_series_with_trend(ts, "l7_time_series.png")
    plot_acf_pacf(ts, decomp_res, "l7_acf_pacf.png")
    plot_decomposition_components(ts, decomp_res, "l7_decomposition.png")
    



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
                 .agg(Revenue=("Revenue", "sum"), 
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
    

    quarterly = (df.groupby("YearQuarter")
                  .agg(Revenue=("Revenue", "sum"), Orders=("Revenue", "count"))
                  .reset_index())
    quarterly["YQ_str"] = quarterly["YearQuarter"].astype(str)


    annual = (df.groupby("Year")
                .agg(Revenue=("Revenue", "sum"), Orders=("Revenue", "count"))
                .reset_index())

    return dict(monthly=monthly, quarterly=quarterly, annual=annual)


def compute_kpi(df: pd.DataFrame):
    emp = (df.groupby("EmployeeName")
             .agg(Revenue=("Revenue", "sum"),
                  Orders=("Revenue", "count"),
                  AvgOrder=("Revenue", "mean"),
                  Customers=("CustomerID", "nunique"),
                  PaidPct=("IsPaid", "mean"))
             .sort_values("Revenue", ascending=False))
    emp["PaidPct"] *= 100

    country = (df.groupby("CustomerCountry")
                 .agg(Revenue=("Revenue", "sum"),
                      Orders=("Revenue", "count"),
                      Customers=("CustomerID", "nunique"))
                 .sort_values("Revenue", ascending=False))
    
    product = (df.groupby("ProductName")
                 .agg(Revenue=("Revenue", "sum"),
                       Qty=("Quantity", "sum"),
                       Orders=("Revenue", "count"))
                 .sort_values("Revenue", ascending=False))

    customers = (df.groupby("CustomerName")
                   .agg(Revenue=("Revenue", "sum"),
                        Orders=("Revenue", "count"),
                        Country=("CustomerCountry", "first"))
                   .sort_values("Revenue", ascending=False))
    customers = customers.sort_values("Revenue", ascending=False)
    customers["CumPct"] = customers["Revenue"].cumsum() / customers["Revenue"].sum() * 100
    customers["ABC"] = pd.cut(customers["CumPct"], bins=[0,80,95,100], labels=["A","B","C"])

    return dict(employees = emp, countries=country, products=product, customers=customers)


def decompose_series(monthly: pd.DataFrame):
    ts = monthly.set_index("YearMonth")["Revenue"]
    ts.index = ts.index.to_timestamp()

    adf_stat, adf_p, *_ = adfuller(ts[ts > 0].values)

    ts_diff = ts.diff().dropna()
    adf2, p2, *_ = adfuller(ts_diff.values)

    try:
        decomp = seasonal_decompose(ts, model="additive", period=4, extrapolate_trend="freq")
    except Exception:
        decomp = None

    ts_vals = ts.values
    acf_vals = acf(ts_vals,  nlags=10, fft=False)
    pacf_vals = pacf(ts_vals, nlags=10, method="ols")

    return dict(ts=ts, adf_stat=adf_stat, adf_p=adf_p, 
                ts_diff=ts_diff, adf2=adf2, p2=p2,
                decomp=decomp, acf=acf_vals, pacf=pacf_vals)










def fmt_usd(x, _=None):
    return f"${x/1000:.0f}K" if abs(x) >= 1000 else f"${x:.0f}"


def set_axis_style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, color=C["text"], pad=5)
    if xlabel:
        ax.set_xlabel(xlabel, color=C["text"])
    if ylabel:
        ax.set_ylabel(ylabel, color=C["text"])
    ax.tick_params(colors=C["sub"])
    ax.set_facecolor(C["panel"])
    ax.grid(True, alpha=0.15, color=C["grid"], linestyle="-", linewidth=0.4)


def plot_monthly_revenue_stacked(monthly, fname):
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=C["bg"])
    
    ym_labels = monthly["YM_str"].tolist()
    paid = monthly["PaidRev"].values
    unpaid = monthly["UnpaidRev"].values
    x_pos = np.arange(len(ym_labels))
    
    ax.bar(x_pos, paid, color=C["paid"], alpha=0.85, label="Paid", edgecolor=C["grid"])
    ax.bar(x_pos, unpaid, color=C["unpaid"], alpha=0.75, label="Unpaid", bottom=paid, edgecolor=C["grid"])
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ym_labels, rotation=45, ha="right", color=C["text"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
    set_axis_style(ax, "Monthly Revenue (Paid / Unpaid)", "Month", "Revenue, $")
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_top_countries(countries, fname):
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=C["bg"])
    
    top_c = countries.head(10)
    bars = ax.barh(top_c.index[::-1], top_c["Revenue"].values[::-1], color=PAL8[:len(top_c)], alpha=0.85, edgecolor=C["grid"])
    
    for bar, v in zip(bars, top_c["Revenue"].values[::-1]):
        ax.text(v + 1000, bar.get_y() + bar.get_height()/2, f"${v/1000:.0f}K", va="center", color=C["text"])
    
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
    set_axis_style(ax, "Top-10 Countries by Revenue")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_manager_revenue(employees, fname):
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=C["bg"])
    
    emp_names = [n.split()[-1] for n in employees.index]
    bars = ax.bar(emp_names, employees["Revenue"].values, color=PAL8[:len(emp_names)], alpha=0.85, edgecolor=C["grid"])
    
    for bar, v in zip(bars, employees["Revenue"].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, f"${v/1000:.0f}K", ha="center", va="bottom", color=C["text"])
    
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
    set_axis_style(ax, "Revenue by Manager")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_manager_payment_percentage(employees, fname):
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=C["bg"])
    
    emp_names = [n.split()[-1] for n in employees.index]
    paid_pcts = employees["PaidPct"].values
    
    bars = ax.barh(emp_names[::-1], paid_pcts[::-1],
                   color=[C["best"] if v >= 60 else C["worst"] for v in paid_pcts[::-1]],
                   alpha=0.85, edgecolor=C["grid"])
    
    ax.axvline(60, color=C["gold"], lw=1.5, ls="--", label="Threshold 60%")
    
    for bar, v in zip(bars, paid_pcts[::-1]):
        ax.text(v + 0.5, bar.get_y() + bar.get_height()/2, f"{v:.1f}%", va="center", color=C["text"])
    
    set_axis_style(ax, "% Paid Orders by Manager", "%", "")
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_top_products(products, fname):
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=C["bg"])
    
    top_p = products.head(12)
    short_names = [n[:20] for n in top_p.index]
    
    ax.barh(short_names[::-1], top_p["Revenue"].values[::-1], color=C["c4"], alpha=0.85, edgecolor=C["grid"])
    
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
    set_axis_style(ax, "Top-12 Products by Revenue")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_abc_customer_pie(customers, fname):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=C["bg"])
    
    abc_data = customers.groupby("ABC")["Revenue"].sum()
    abc_colors = {"A": C["best"], "B": C["gold"], "C": C["worst"]}
    
    wedges, texts, autos = ax.pie(
        abc_data.values,
        labels=[f"Class {k}" for k in abc_data.index],
        autopct="%1.1f%%",
        colors=[abc_colors.get(k, C["c0"]) for k in abc_data.index],
        startangle=90,
        wedgeprops=dict(edgecolor=C["bg"], linewidth=2),
    )
    
    for t in texts:
        t.set_color(C["text"])
        t.set_fontsize(9)
    for a in autos:
        a.set_color("white")
        a.set_fontsize(8)
    
    ax.set_facecolor(C["panel"])
    ax.set_title("ABC Customer Analysis (by Revenue)", color=C["text"], pad=5)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_quarterly_revenue(quarterly, fname):
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=C["bg"])
    
    q_labels = quarterly["YQ_str"].tolist()
    ax.bar(q_labels, quarterly["Revenue"].values,
           color=[PAL8[i % len(PAL8)] for i in range(len(q_labels))],
           alpha=0.85, edgecolor=C["grid"])
    
    for i, v in enumerate(quarterly["Revenue"].values):
        ax.text(i, v + 1000, f"${v/1000:.0f}K", ha="center", va="bottom", color=C["text"])
    
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
    set_axis_style(ax, "Quarterly Revenue", "Quarter")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_time_series_with_trend(ts, fname):
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=C["bg"])
    
    ax.plot(ts.index, ts.values, color=C["c0"], lw=1.8, alpha=0.85, label="Monthly Revenue", marker="o", ms=5)
    
    x_num = np.arange(len(ts))
    lr = LinearRegression().fit(x_num.reshape(-1, 1), ts.values)
    trend_line = lr.predict(x_num.reshape(-1, 1))
    ax.plot(ts.index, trend_line, "--", color=C["gold"], lw=2.0, label="Trend (LR)")
    
    ma3 = ts.rolling(3).mean()
    ax.plot(ts.index, ma3.values, color=C["c2"], lw=1.8, ls="-.", label="MA(3)")
    
    set_axis_style(ax, "Monthly Revenue + Trend + MA(3)", "Date", "$")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_acf_pacf(ts, decomp_res, fname):
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=C["bg"])
    
    lags = np.arange(len(decomp_res["acf"]))
    ax.bar(lags, decomp_res["acf"], color=C["c0"], alpha=0.8, label="ACF", width=0.35, align="edge")
    ax.bar(lags + 0.35, decomp_res["pacf"], color=C["c1"], alpha=0.8, label="PACF", width=0.35, align="edge")
    
    ax.axhline(0, color=C["text"], lw=0.8, alpha=0.5)
    ax.axhline(1.96 / np.sqrt(len(ts)), color=C["gold"], lw=1.2, ls="--", alpha=0.7)
    ax.axhline(-1.96 / np.sqrt(len(ts)), color=C["gold"], lw=1.2, ls="--", alpha=0.7)
    
    set_axis_style(ax, "ACF / PACF of Time Series", "Lag", "Correlation")
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_decomposition_components(ts, decomp_res, fname):
    if decomp_res["decomp"] is None:
        return None
    
    dec = decomp_res["decomp"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=C["bg"])
    
    components = [dec.trend, dec.seasonal, dec.resid]
    titles = ["Trend", "Seasonality", "Residuals"]
    colors = [C["c2"], C["c3"], C["c4"]]
    
    for ax, comp, title, color_k in zip(axes, components, titles, colors):
        ax.plot(ts.index, comp.values, color=color_k, lw=1.8, alpha=0.85)
        ax.axhline(0, color=C["sub"], lw=0.8, ls="--", alpha=0.5)
        set_axis_style(ax, f"Decomposition: {title}", "Date", "$")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    main()
