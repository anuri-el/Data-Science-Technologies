import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


SEP = "=" * 67

OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "l5_nbu_exchange_rates_2y.csv")


CURRENCIES = ["USD", "EUR", "GBP"]
DAYS_BACK = 365 * 2
COLORS = {"USD": "#2196F3", "EUR": "#4CAF50", "GBP": "#FF9800"}



def main():
    print(f"\n{SEP}")
    print(f"Level 3")
    df = load_or_fetch_data(CSV_PATH, CURRENCIES, DAYS_BACK)

    print(f"\n{SEP}")
    print(f"Dates: {df["date"].iloc[0]} - {df["date"].iloc[-1]}")
    for cur in CURRENCIES:
        print(f"{cur}: min={df[cur].min()}, max={df[cur].max()}, mean={df[cur].mean():.4f}")

    print(f"\nFile: {CSV_PATH}")
    print(f"Dataset size: {df.shape[0]} rows * {df.shape[1]} columns")

    print(f"\n{SEP}")
    print("GTV-1: Clustering of Exchange Rates")
    results = gtv1_clustering(df)


    print(f"\nPCA: {results["pca"].explained_variance_ratio_.cumsum()*100}")
    print(f"PCA: {results["pca"].explained_variance_ratio_.sum()*100:.1f}% dispersion in 4 components")


    print(f"\nBest k: K={results['best_k']}")
    print(f" K  {'Silhouette':<15} {'Davies-B.':<15} {'Calinski':<15}")
    for idx, k in enumerate(results["K_RANGE"]):
        print(f" {k}   {results["sils"][idx]:<15.4f} {results["dbs_scores"][idx]:<15.4f} {results["chs"][idx]:<15.4f} ")



    # plot_currency_trends(df, "l5_currency_trends.png", CURRENCIES, COLORS)


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


def compute_rsi(prices: np.ndarray, period):
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period).mean().values
    avg_l = pd.Series(loss).ewm(span=period).mean().values
    rs = np.where(avg_l == 0, 100, avg_g / (avg_l + 1e-9))
    return 100 - 100 / (1 + rs)


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

    for cur in currencies:
        df[f"{cur}_ret"] = df[cur].pct_change().fillna(0)
        df[f"{cur}_vol5"] = df[cur].rolling(5).std().fillna(0)
        df[f"{cur}_ma20"] = df[cur].rolling(20).mean().fillna(df[cur])
        df[f"{cur}_rsi"] = compute_rsi(df[cur].values, 14)

    return df


def load_or_fetch_data(csv_path: str, currencies: list, days_back: int):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = parse_nbu_data(currencies, days_back)
        df.to_csv(csv_path, index=False, encoding="utf-8")
    return df


def gtv1_clustering(df: pd.DataFrame):
    feature_cols = [f"{c}_ret" for c in CURRENCIES] + [f"{c}_vol5" for c in CURRENCIES] + [f"{c}_rsi" for c in CURRENCIES] 
    X_raw = df[feature_cols].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    pca = PCA(n_components=4, random_state=42)
    X_pca = pca.fit_transform(X)

    K_RANGE = range(2, 9)
    inertias, sils, dbs, chs = [], [], [], []
    for k in K_RANGE:
        km =  KMeans(n_clusters=k, n_init=15, random_state=42)
        lbl = km.fit_predict(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, lbl))
        dbs.append(davies_bouldin_score(X, lbl))
        chs.append(calinski_harabasz_score(X, lbl))

    best_k = list(K_RANGE)[np.argmax(sils)]

    results = {"X_pca" : X_pca, 
               "X" : X_pca, 
               "df" : df, 
               "best_k" : best_k, 
               "K_RANGE": K_RANGE, 
               "inertias": inertias, 
               "sils": sils, 
               "dbs_scores" : dbs, 
               "chs" : chs, 
               "feature_cols" : feature_cols, 
               "pca" : pca,
               "scaler" : scaler}

    km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    lbl_km = km.fit_predict(X)
    results["K-Means"] = dict(
        labels=lbl_km, k=best_k,
        sil=silhouette_score(X, lbl_km),
        db=davies_bouldin_score(X, lbl_km),
        ch=calinski_harabasz_score(X, lbl_km), 
        centers=km.cluster_centers_,
    )

    agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    lbl_a = agg.fit_predict(X)
    results["Hierarchical"] = dict(
        labels=lbl_a, k=best_k,
        sil=silhouette_score(X, lbl_a),
        db=davies_bouldin_score(X, lbl_a),
        ch=calinski_harabasz_score(X, lbl_a),
    )
    Z_link = linkage(X[:80], method="ward")
    results["Z_link"] = Z_link

        

    return results




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



def plot_currency_trends(df: pd.DataFrame, filename: str, currencies: list, colors: dict):
    output_path = os.path.join(OUTPUT_DIR, filename)

    n_cur = len(currencies)
    fig, axes = plt.subplots(n_cur, 1, figsize=(12, 9), facecolor="#0F1117")

    dates = pd.to_datetime(df["date"])

    for i, cur in enumerate(currencies):
        ax = axes[i]
        y = df[cur].values

        ax.plot(dates, y, color=colors[cur], alpha=0.8, label="real_data")
        ax_style(ax, f"{cur}/UAH — курс та тренди")
        ax.set_ylabel("UAH", color="#AAAAAA", fontsize=8)
        # ax.legend(fontsize=7, facecolor="#1A1D27", edgecolor="#444444", labelcolor="#FFFFFF", loc="upper left")
    
    plt.tight_layout(pad=3)
    plt.savefig(output_path)
    plt.show()
    print(f"Trend analysis: {output_path}")



if __name__ == "__main__":
    main()