import requests
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os


def main():
    CURRENCIES = ["USD", "EUR", "GBP"]
    DAYS_BACK = 365
    OUTPUT_CSV = "outputs/nbu_exchange_rates.csv"
    COLORS = {"USD": "#2196F3", "EUR": "#4CAF50", "GBP": "#FF9800"}

    # df = parse_nbu_data(currencies=CURRENCIES, days_back=DAYS_BACK)

    # df = save_to_csv(df, OUTPUT_CSV)

    df = load_or_fetch_data(OUTPUT_CSV, CURRENCIES, DAYS_BACK)

    x = np.arange(len(df))
    trends = {}
    for cur in CURRENCIES:
        y = df[cur].values
        tr = fit_trend(x, y)
        trends[cur] = tr
        print_trend_info(cur, tr)

    all_stats = {}
    for cur in CURRENCIES:
        st = compute_statistics(df[cur].values, f"{cur}_real")
        all_stats[cur] = st
        print_statistics(st)

    synthetics = {}
    for cur in CURRENCIES:
        y_real = df[cur].values
        y_synth = synthesize_model(y_real, trends[cur], dist="normal")
        synthetics[cur] = y_synth

        st_synth = compute_statistics(y_synth, f"{cur}_synth")
        print_statistics(st_synth)

        kolmogorov_smirnov_test(y_real, y_synth, cur)

        st_real = all_stats[cur]
        print(f"Різниця mean:   {abs(st_real["mean"] - st_synth["mean"]):.3f} UAH")
        print(f"Різниця std:    {abs(st_real["std"] - st_synth["std"]):.3f} UAH")

    # plot_currency_trends(df, trends, "outputs/currency_trends.png", CURRENCIES, COLORS)
    # plot_residuals(df, "outputs/residuals.png", trends, CURRENCIES, COLORS)
    # plot_real_synth_comparison(df, synthetics, "outputs/real_synth.png", CURRENCIES, COLORS)
    plot_histograms(df, synthetics, "outputs/histograms.png", CURRENCIES, COLORS)
    plot_histograms_detrended(df, trends, synthetics, "outputs/histograms.png", CURRENCIES, COLORS)


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


def load_or_fetch_data(csv_path: str, currencies: list, days_back: int):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = parse_nbu_data(currencies, days_back)
        save_to_csv(df, csv_path)
    
    return df


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
                    r2 = {tr["r2_linear"]:.6f}

    Квадратичний:   y = {p2[0]:.6f} * x2 + {p2[1]:.6f} * x + {p2[2]:.6f}
                    r2 = {tr["r2_quad"]:.6f}

    Кращий тренд: {best}""")


def synthesize_model(y_real: np.ndarray, trend_params: dict, dist: str = "normal"):
    x = np.arange(len(y_real))
    p2 = trend_params["p_quad"]
    trend = np.polyval(p2, x)
    resid = y_real - trend
    sigma = np.std(resid)

    if dist == "normal":
        noise = np.random.normal(loc=np.mean(resid), scale=sigma, size=len(y_real))
    else:
        a, b = np.min(resid), np.max(resid)
        noise = np.random.uniform(a, b, size=len(y_real))

    return trend + noise


def kolmogorov_smirnov_test(real: np.ndarray, synth: np.ndarray, label: str):
    ks_stats, p_val = stats.ks_2samp(real, synth)
    verdict = "models are similar" if p_val > 0.05 else "significant difference"
    print(f"KS-test [{label}]: D={ks_stats:.6f} p={p_val:.6f} - {verdict}")


def plot_currency_trends(df: pd.DataFrame, trends: dict, output_path: str, currencies: list, colors: dict):
    n_cur = len(currencies)
    fig, axes = plt.subplots(n_cur, 1, figsize=(12, 9), facecolor="#0F1117")

    dates = pd.to_datetime(df["date"])

    for i, cur in enumerate(currencies):
        ax = axes[i]
        y = df[cur].values
        tr = trends[cur]

        ax.plot(dates, y, color=colors[cur], alpha=0.8, label="real_data")
        ax.plot(dates, tr["y_linear"], linestyle="--", color="#FF5252", alpha=0.9, label=f"linear r2={tr["r2_linear"]:.3f}")
        ax.plot(dates, tr["y_quad"], linestyle="-.", color="#FFD700", alpha=0.9, label=f"quad r2={tr["r2_quad"]:.3f}")

        ax_style(ax, f"{cur}/UAH - курс та тренди")
        ax.set_ylabel("UAH", color="#AAAAAA", fontsize=8)
        # ax.legend(fontsize=7, facecolor="#1A1D27", edgecolor="#444444", labelcolor="#FFFFFF", loc="upper left")
    
    plt.tight_layout(pad=3)
    plt.savefig(output_path)
    plt.show()
    print(f"Currency trends were saved to {output_path}.")


def plot_residuals(df: pd.DataFrame, output_path: str, trends: dict, currencies: list, colors: dict):
    n_cur = len(currencies)
    fig, axes = plt.subplots(n_cur, 1, figsize=(12, 9), facecolor="#0F1117")
    
    dates = pd.to_datetime(df["date"])

    for i, cur in enumerate(currencies):
        ax = axes[i]
        resid = df[cur].values - trends[cur]["y_quad"]

        ax.bar(dates, resid, width=1.5, color=colors[cur], alpha=0.7)

        ax.axhline(0, color="#FFFFFF", linewidth=0.8, alpha=0.5)
        
        sigma = np.std(resid)
        ax.axhline(sigma, linestyle="--", color="#FF5252", linewidth=1, alpha=0.8, label=f"+sigma ({sigma:.3f})")
        ax.axhline(-sigma, linestyle="--", color="#FF5252", linewidth=1, alpha=0.8, label=f"-sigma ({-sigma:.3f})")

        ax_style(ax, f"{cur} - residuals (real - quad trend)")
        ax.set_ylabel("delta UAH", color="#AAAAAA", fontsize=8)
    
    plt.tight_layout(pad=3)
    plt.savefig(output_path)
    plt.show()
    print(f"Residuals were saved to {output_path}.")


def plot_real_synth_comparison(df: pd.DataFrame, synthetics: dict, output_path: str, currencies: list, colors: dict):
    n_cur = len(currencies)
    fig, axes = plt.subplots(n_cur, 1, figsize=(12, 9), facecolor="#0F1117")

    dates = pd.to_datetime(df["date"])

    for i, cur in enumerate(currencies):
        ax = axes[i]
        y_real = df[cur].values
        y_synth = synthetics[cur]

        ax.plot(dates, y_real, color=colors[cur], linewidth=1.2, alpha=0.85, label="real")
        ax.plot(dates, y_synth, color="#E040FB", linewidth=1, alpha=0.75, linestyle="--", label="synth")

        ax_style(ax, f"{cur} - real vs. synth")
        ax.set_ylabel("UAH", color="#AAAAAA", fontsize=8)
    
    plt.tight_layout(pad=3)
    plt.savefig(output_path)
    plt.show()
    print(f"real vs synth plot was saved to {output_path}.")


def plot_histograms(df: pd.DataFrame, synthetics: dict, output_path: str, currencies: list, colors: dict):
    n_cur = len(currencies)
    fig, axes = plt.subplots(1, n_cur, figsize=(18, 7), facecolor="#0F1117")

    for i, cur in enumerate(currencies):
        ax = axes[i]

        y_real = df[cur].values
        y_synth = synthetics[cur]
        n_bins = min(35, max(10, len(y_real) // 6))

        ax.hist(y_real, bins=n_bins, density=True, color=colors[cur], alpha=0.7, label="real")
        ax.hist(y_synth, bins=n_bins, density=True, color="#E040FB", alpha=0.5, label="synth")

        kde_real = stats.gaussian_kde(y_real)
        kde_synth = stats.gaussian_kde(y_synth)
        xr = np.linspace(min(y_real.min(), y_synth.min()), max(y_real.max(), y_synth.max()), 300)
        ax.plot(xr, kde_real(xr), color="#FFFFFF", linewidth=1.5, label="kde_real")
        ax.plot(xr, kde_synth(xr), color="#FF80AB", linewidth=1.5, linestyle="--", label="kde_synth")

        ax_style(ax, f"{cur} - histogram / kde")
        ax.set_ylabel("Density", color="#AAAAAA", fontsize=8)

    plt.tight_layout(pad=3)
    plt.savefig(output_path)
    plt.show()


def plot_histograms_detrended(df: pd.DataFrame, trends: dict, synthetics: dict, output_path: str, currencies: list, colors: dict):
    n_cur = len(currencies)
    fig, axes = plt.subplots(1, n_cur, figsize=(18, 7), facecolor="#0F1117")

    for i, cur in enumerate(currencies):
        ax = axes[i]

        y_real = df[cur].values
        real_trend = trends[cur]["y_quad"]
        real_residuals = y_real - real_trend

        y_synth = synthetics[cur]
        synth_residuals = y_synth - real_trend

        n_bins = min(35, max(10, len(y_real) // 6))

        ax.hist(real_residuals, bins=n_bins, density=True, color=colors[cur], alpha=0.7, label="real")
        ax.hist(synth_residuals, bins=n_bins, density=True, color="#E040FB", alpha=0.5, label="synth")

        kde_real = stats.gaussian_kde(real_residuals)
        kde_synth = stats.gaussian_kde(synth_residuals)
        xr = np.linspace(min(real_residuals.min(), synth_residuals.min()), max(real_residuals.max(), synth_residuals.max()))
        ax.plot(xr, kde_real(xr), color="#FFFFFF", linewidth=1.5, label="kde_real")
        ax.plot(xr, kde_synth(xr), color="#FF80AB", linewidth=1.5, linestyle="--", label="kde_synth")

        ax_style(ax, f"{cur} - histogram / kde")
        ax.set_ylabel("Density", color="#AAAAAA", fontsize=8)

    plt.tight_layout(pad=3)
    plt.savefig(output_path)
    plt.show()


def plot_qq():
    pass


def ax_style(ax, title=""):
    ax.set_facecolor("#1A1D27")
    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    if title:
        ax.set_title(title, color="#FFFFFF", fontsize=10, fontweight="bold", pad=5)
    ax.margins(x=0.01)
    ax.legend(fontsize=7, facecolor="#1A1D27", edgecolor="#444444", labelcolor="#FFFFFF", loc="upper left")
    ax.grid(linestyle="--", color="#555566", alpha=0.15, linewidth=0.6)


if __name__ == "__main__":
    main()