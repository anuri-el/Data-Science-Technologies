import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

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


    print(f"\n{SEP}")
    arima_res = fit_arima(monthly)

    arima = arima_res["ARIMA"]
    sarima = arima_res["SARIMA"]
    linear_reg = arima_res["LinearReg"]
    n_test = arima_res["n_test"]

    print(f" ARIMA(1,1,1):  RMSE={arima['rmse']:>10,.2f}  R2={arima['r2']:.4f}  AIC={arima['fit'].aic:.1f}")
    print(f" SARIMA(1,1,1)(1,0,1,4): RMSE={sarima['rmse']:>10,.2f}  R2={sarima['r2']:.4f}  AIC={sarima['fit'].aic:.1f}")
    print(f" Linear Regression (baseline): RMSE={linear_reg['rmse']:>10,.2f}  R2={linear_reg['r2']:.4f}")
    
    print(f"\n  Forecast ({arima_res['best_key']}) 3 months ahead:")
    idx = pd.date_range(arima_res["ts_pos"].index[-1] + pd.DateOffset(months=1), periods=3, freq="MS")
    for d, v in zip(idx, arima_res["forecast_3m"][-3:]):
        print(f"  {d.strftime('%Y-%m')}: ${v:>10,.2f}")


    print(f"\n{SEP}")
    ann_res = fit_ann(monthly, n_test)
    print(f"\n  Architecture |  RMSE      |  R2")
    for arch in ["MLP", "LSTM"]:
        print(f"  {arch:<12} | {ann_res[arch]['rmse']:>10,.2f} | {ann_res[arch]['r2']:.4f}")

    print(f"\n  Best ANN-architecture: {ann_res['best']}")



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
    
    plot_arima_forecast(arima_res, "l7_arima_forecast.png")

    plot_ann_learning_curve(ann_res, "MLP", "l7_mlp_learning.png")
    plot_ann_learning_curve(ann_res, "LSTM", "l7_lstm_learning.png")
    plot_ann_forecast(ann_res, "MLP", monthly, "l7_mlp_forecast.png")
    plot_ann_forecast(ann_res, "LSTM", monthly, "l7_lstm_forecast.png")
    plot_models_rmse_comparison(arima_res, ann_res, "l7_models_comparison.png")

    plot_manager_month_heatmap(df, "l7_manager_month_heatmap.png")
    plot_top_customers(customers, "l7_top_customers.png")


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


def fit_arima(monthly: pd.DataFrame):
    ts = monthly.set_index("YearMonth")["Revenue"]
    ts.index = ts.index.to_timestamp()
    ts_pos = ts[ts > 0].copy()

    n_test = 4
    train = ts_pos.iloc[:-n_test]
    test = ts_pos.iloc[-n_test:]

    results = {}

    try:
        arima = SARIMAX(train, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
        arima_fit = arima.fit(disp=False)
        pred_a = arima_fit.forecast(n_test)
        rmse_a = np.sqrt(mean_squared_error(test, pred_a))
        r2_a = r2_score(test, pred_a)
        results["ARIMA"] = dict(fit=arima_fit, pred=pred_a, rmse=rmse_a, r2=r2_a, aic=arima_fit.aic)
    except Exception as e:
        print(f"  ARIMA: {e}")

    try:
        sarima = SARIMAX(train, order=(1,1,1), seasonal_order=(1,0,1,4), enforce_stationarity=False, enforce_invertibility=False)
        sarima_fit = sarima.fit(disp=False)
        pred_s = sarima_fit.forecast(n_test)
        rmse_s = np.sqrt(mean_squared_error(test, pred_s))
        r2_s = r2_score(test, pred_s)
        results["SARIMA"] = dict(fit=sarima_fit, pred=pred_s, rmse=rmse_s, r2=r2_s, aic=sarima_fit.aic)
    except Exception as e:
        print(f"  SARIMA: {e}")

    X_all = np.arange(len(ts_pos)).reshape(-1,1)
    lr = LinearRegression().fit(X_all[:len(train)], train.values)
    pred_lr = lr.predict(X_all[len(train):len(train)+n_test])
    rmse_lr = np.sqrt(mean_squared_error(test, pred_lr))
    r2_lr = r2_score(test, pred_lr)
    results["LinearReg"] = dict(pred=pred_lr, rmse=rmse_lr, r2=r2_lr)

    best_key = min([k for k in results if k != "LinearReg"], key=lambda k: results[k]["rmse"], default="ARIMA")
    if best_key in results:
        best_fit = results[best_key]["fit"]
        forecast_3 = best_fit.forecast(n_test + 3)
        results["forecast_3m"] = forecast_3[-3:]
        results["best_key"] = best_key

    results["train"] = train
    results["test"] = test
    results["ts_pos"]= ts_pos
    results["n_test"]= n_test
    return results


WINDOW = 4

def make_sequences(arr: np.ndarray, w: int):
    X, y = [], []
    for i in range(len(arr) - w):
        X.append(arr[i:i+w])
        y.append(arr[i+w])
    return np.array(X), np.array(y)


def train_ann(ts_values: np.ndarray, n_test: int, arch: str):
    scaler = MinMaxScaler(feature_range=(0,1))
    ts_sc = scaler.fit_transform(ts_values.reshape(-1,1)).ravel()

    X, y = make_sequences(ts_sc, WINDOW)
    X_train, y_train = X[:-n_test], y[:-n_test]
    X_test, y_test = X[-n_test:], y[-n_test:]

    X_train_3d = X_train[:, :, np.newaxis]
    X_test_3d = X_test[:, :, np.newaxis]

    if arch == "MLP":
        model = keras.Sequential([
            keras.Input(shape=(WINDOW,)),
            layers.Dense(32, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ], name="MLP")
        Xtr, Xte = X_train, X_test
    else: 
        model = keras.Sequential([
            keras.Input(shape=(WINDOW,1)),
            layers.LSTM(32, return_sequences=True),
            layers.LSTM(16),
            layers.Dense(8, activation="relu"),
            layers.Dense(1),
        ], name="LSTM")
        Xtr, Xte = X_train_3d, X_test_3d

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    cb = [callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0)]
    hist = model.fit(Xtr, y_train, epochs=200, batch_size=4, validation_split=0.2, callbacks=cb, verbose=0)

    pred_sc = model.predict(Xte, verbose=0).ravel()
    pred = scaler.inverse_transform(pred_sc.reshape(-1,1)).ravel()
    true = scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)

    return dict(model=model, hist=hist, pred=pred, true=true,
                rmse=rmse, r2=r2, scaler=scaler,
                X_train=Xtr, y_train=y_train)


def fit_ann(monthly: pd.DataFrame, n_test: int):
    ts = monthly.set_index("YearMonth")["Revenue"]
    ts.index = ts.index.to_timestamp()
    ts_pos = ts[ts > 0].values

    results = {}
    for arch in ["MLP", "LSTM"]:
        r = train_ann(ts_pos, n_test, arch)
        results[arch] = r

    best = min(results, key=lambda k: results[k]["rmse"])
    results["best"] = best
    results["ts_pos"] = ts_pos
    results["n_test"] = n_test
    return results


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


def plot_arima_forecast(arima_res, fname):
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=C["bg"])
    
    train_s = arima_res["train"]
    test_s = arima_res["test"]
    
    ax.plot(train_s.index, train_s.values, color=C["c0"], lw=1.8, label="Train", marker="o", ms=4)
    ax.plot(test_s.index, test_s.values, color=C["c2"], lw=2.0, label="Actual (Test)", marker="s", ms=6, zorder=5)
    
    method_styles = {"ARIMA": "--", "SARIMA": "-.", "LinearReg": ":"}
    method_colors = {"ARIMA": C["c1"], "SARIMA": C["c3"], "LinearReg": C["c5"]}
    
    for mkey in ["ARIMA", "SARIMA", "LinearReg"]:
        if mkey in arima_res:
            r_ = arima_res[mkey]
            pred_idx = test_s.index if isinstance(r_["pred"], pd.Series) else test_s.index[:len(r_["pred"])]
            pred_vals = r_["pred"].values if isinstance(r_["pred"], pd.Series) else r_["pred"]
            ax.plot(pred_idx, pred_vals, ls=method_styles[mkey], color=method_colors[mkey], lw=2.0, label=f"{mkey} (R2={r_['r2']:.3f})", marker="^", ms=5)
    
    if "forecast_3m" in arima_res:
        last_date = test_s.index[-1]
        fut_idx = pd.date_range(last_date + pd.DateOffset(months=1), periods=3, freq="MS")
        ax.plot(fut_idx, arima_res["forecast_3m"], "o--", color=C["gold"], lw=2.2, ms=8, label="Forecast +3m", zorder=6)
        ax.fill_between(fut_idx, arima_res["forecast_3m"] * 0.85, arima_res["forecast_3m"] * 1.15, alpha=0.12, color=C["gold"])
    
    ax.axvspan(test_s.index[0], test_s.index[-1], alpha=0.06, color=C["c2"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
    set_axis_style(ax, "ARIMA/SARIMA Forecast vs Actual", "Date", "$")
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"], ncol=3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_ann_learning_curve(ann_res, architecture, fname):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=C["bg"])
    
    r_ = ann_res[architecture]
    hist = r_["hist"]
    ep = range(1, len(hist.history["loss"]) + 1)
    
    ax.plot(ep, hist.history["loss"], color=C["c2"], lw=1.8, label="Train")
    ax.plot(ep, hist.history["val_loss"], color=C["c3"], lw=1.8, ls="--", label="Validation")
    
    set_axis_style(ax, f"{architecture}: Learning Curve (MSE Loss)", "Epoch", "Loss")
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_ann_forecast(ann_res, architecture, monthly, fname):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=C["bg"])
    
    ts_obj = monthly.set_index("YearMonth")["Revenue"]
    ts_obj.index = ts_obj.index.to_timestamp()
    ts_obj = ts_obj[ts_obj > 0]
    
    r_ = ann_res[architecture]
    n_show = len(r_["true"])
    idx = ts_obj.index[-n_show:]
    
    ax.plot(idx, r_["true"], color=C["c0"], lw=2.0, marker="o", ms=6, label="Actual")
    ax.plot(idx, r_["pred"], color=C["c1"], lw=2.0, ls="--", marker="s", ms=6, label=f"{architecture} (R2={r_['r2']:.4f})")
    ax.fill_between(idx, r_["true"], r_["pred"], alpha=0.15, color=C["c3"])
    
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
    set_axis_style(ax, f"{architecture}: Forecast vs Actual (Test)", "Date", "$")
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_models_rmse_comparison(arima_res, ann_res, fname):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=C["bg"])
    
    all_methods = {}
    for k in ["ARIMA", "SARIMA", "LinearReg"]:
        if k in arima_res:
            all_methods[k] = {"rmse": arima_res[k]["rmse"], "r2": arima_res[k]["r2"]}
    for k in ["MLP", "LSTM"]:
        all_methods[k] = {"rmse": ann_res[k]["rmse"], "r2": ann_res[k]["r2"]}
    
    names_all = list(all_methods.keys())
    rmses_all = [all_methods[k]["rmse"] for k in names_all]
    best_all = names_all[np.argmin(rmses_all)]
    
    clrs_all = [C["best"] if n == best_all else PAL8[i % len(PAL8)] for i, n in enumerate(names_all)]
    
    bars = ax.bar(names_all, rmses_all, color=clrs_all, alpha=0.85, edgecolor=C["grid"])
    for bar, v in zip(bars, rmses_all):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f"${v:,.0f}", ha="center", va="bottom", color=C["text"])
    
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
    set_axis_style(ax, "RMSE Comparison: ARIMA vs ANN", "", "RMSE ($)")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_manager_month_heatmap(df, fname):
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=C["bg"])
    cmap_heat = LinearSegmentedColormap.from_list("rev", ["#0D1117", "#1565C0", "#FF8F00", "#E53935"])
    
    emp_short = {n: n.split()[-1] for n in df["EmployeeName"].unique()}
    df["EmpShort"] = df["EmployeeName"].map(emp_short)
    
    pivot_em = (df.groupby(["EmpShort", df["OrderDate"].dt.month])["Revenue"].sum().unstack(fill_value=0))
    im2 = ax.imshow(pivot_em.values, cmap=cmap_heat, aspect="auto")
    
    ax.set_xticks(range(12))
    ax.set_xticklabels([f"M{m+1}" for m in range(12)], color=C["text"])
    ax.set_yticks(range(len(pivot_em.index)))
    ax.set_yticklabels(pivot_em.index.tolist(), color=C["text"])
    
    plt.colorbar(im2, ax=ax, fraction=0.04)
    set_axis_style(ax, "Manager * Month (Revenue $)")
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


def plot_top_customers(customers, fname):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=C["bg"])
    
    top15 = customers.head(15)
    short_cust = [n.split()[-1] for n in top15.index]
    abc_clr = {"A": C["best"], "B": C["gold"], "C": C["worst"]}
    bar_colors = [abc_clr.get(str(top15.iloc[i]["ABC"]), C["c0"]) for i in range(len(top15))]
    
    ax.barh(short_cust[::-1], top15["Revenue"].values[::-1], color=bar_colors[::-1], alpha=0.85, edgecolor=C["grid"])
    
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
    set_axis_style(ax, "Top-15 Customers", "$", "")
    
    handles = [plt.Rectangle((0, 0), 1, 1, color=abc_clr[k], alpha=0.85) for k in ["A", "B", "C"]]
    ax.legend(handles, ["A (80%)", "B (15%)", "C (5%)"], facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    main()