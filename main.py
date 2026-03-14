import requests
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import os


def main():
    CURRENCIES = ["USD", "EUR", "GBP"]
    DAYS_BACK = 365
    OUTPUT_CSV = "outputs/nbu_exchange_rates.csv"

    df = parse_nbu_data(currencies=CURRENCIES, days_back=DAYS_BACK)

    save_to_csv(df, OUTPUT_CSV)

    x = np.arange(len(df))
    trends = {}
    for cur in CURRENCIES:
        y = df[cur].values
        tr = fit_trend(x, y)
        trends[cur] = tr
        print_trend_info(cur, tr)

    all_stats = {}
    for cur in CURRENCIES:
        st = compute_statistics(df[cur].values, cur)
        all_stats[cur] = st
        print_statistics(st)


def fetch_nbu_rate(currency: str, date: datetime):
    date_str = date.strftime("%Y%m%d")
    url = f"https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?valcode={currency}&date={date_str}&json"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data:
            rate = data[0]["rate"]
            return rate
    except Exception:
        pass


def parse_nbu_data(currencies: list[str], days_back: int):
    today = datetime.today()
    dates = [today - timedelta(days=i) for i in range(days_back, -1, -1)]
    records = []
    for d in dates:
        row = {"date": d.date()}
        for cur in currencies:
            row[cur] = fetch_nbu_rate(cur, d)
        records.append(row)
    df = pd.DataFrame(records)
    df.dropna(inplace=True)

    df.reset_index(drop=True, inplace=True)
    print("-" * 64)
    print(f"Rows: {len(df)}")
    print(f"Dates: {df["date"].iloc[0]} - {df["date"].iloc[-1]}")
    for cur in currencies:
        print(f"{cur}: min={df[cur].min()}, max={df[cur].max()}, mean={df[cur].mean()}")
    print("-" * 64)

    return df


def save_to_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

    print(f"File: {path}")
    print(f"Dataset size: {df.shape[0]} rows * {df.shape[1]} columns")


def compute_statistics(series: np.ndarray, label: str):
    n = len(series)
    mu = np.mean(series)
    med = np.median(series)
    var = np.var(series, ddof=1)
    std = np.std(series, ddof=1)
    ske = stats.skew(series)
    kur = stats.kurtosis(series)
    cv = std / mu * 100 if mu != 0 else np.nan
    ci = stats.t.interval(0.95, df=n-1, loc=mu, scale=stats.sem(series))

    result = dict(label=label, n=n, mean=mu, median=med, variance=var, std=std, skewness=ske, kurtosis=kur, cv_pct=cv, ci95_lo=ci[0], ci95_hi=ci[1])
    return result


def print_statistics(stats_dict: dict):
    print(f"""
    ------------------{stats_dict["label"]}------------------
    Кількість спостережень: {stats_dict["n"]}
    Математичне очікування: {stats_dict["mean"]:.3f}
    Медіана:                {stats_dict["median"]:.3f}
    Дисперсія:              {stats_dict["variance"]:.3f}
    Середньокв. відхилення: {stats_dict["std"]:.3f}
    Коефіцієнт варіації:    {stats_dict["cv_pct"]:.3f}
    Асиметрія:              {stats_dict["skewness"]:.3f}
    Ексцес:                 {stats_dict["kurtosis"]:.3f}
    ДІ 95% (mean):          {stats_dict["ci95_lo"]:.3f} - {stats_dict["ci95_hi"]:.3f}""")


def fit_trend(x: np.ndarray, y: np.ndarray):
    p1 = np.polyfit(x, y, 1)
    y_lin = np.polyval(p1, x)
    r2_lin = 1 - np.sum((y - y_lin) ** 2) / np.sum((y - np.mean(y)) ** 2)

    p2 = np.polyfit(x, y, 2)
    y_quad = np.polyval(p2, x)
    r2_quad = 1 - np.sum((y - y_quad) ** 2) / np.sum((y - np.mean(y)) ** 2)

    return dict(p_linear=p1, y_linear=y_lin, r2_linear=r2_lin, 
                p_quad=p2, y_quad=y_quad, r2_quad=r2_quad)


def print_trend_info(cur: str, tr: dict):
    p1, p2 = tr["p_linear"], tr["p_quad"]
    best = "quad" if tr["r2_quad"] > tr["r2_linear"] else "linear"
    print(f"""
    ------------------{cur}------------------
    Лінійний:       y = {p1[0]:.6f} * x + {p1[1]:.6f}
                    r2 = {tr["r2_linear"]}

    Квадратичний:   y = {p2[0]:.6f} * x2 + {p2[1]:.6f} * x + {p2[2]:.6f}
                    r2 = {tr["r2_quad"]}

    Кращий тренд: {best}""")


if __name__ == "__main__":
    main()