import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from matplotlib.colors import LinearSegmentedColormap


SEP = "=" * 67

OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(OUTPUT_DIR, "sample_data.xlsx")
DESC_PATH = os.path.join(OUTPUT_DIR, "Datdata_descriptiona_Set_7.xlsx")

REF_DATE  = pd.Timestamp("2021-02-01")

SCORE_BASE = 300
SCORE_MIN = 0
SCORE_MAX = 850


C = dict(
    bg="#0D1117", panel="#161B22", grid="#30363D",
    text="#E6EDF3", sub="#8B949E",
    c0="#2196F3", c1="#FF9800", c2="#4CAF50", c3="#E91E63",
    c4="#9C27B0", c5="#00BCD4", c6="#FF5722", c7="#8BC34A",
    good="#4CAF50", bad="#EF5350", fraud="#FF1744",
    gold="#FFD700", neutral="#78909C",
)
PAL8 = [C["c0"],C["c1"],C["c2"],C["c3"],C["c4"],C["c5"],C["c6"],C["c7"]]


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
    
    df = compute_scorecard(df)
    print(f"Score range: {df['score_raw'].min():.0f} - {df['score_raw'].max():.0f}")
    print(f"Average Score: {df['score_raw'].mean():.1f} +- {df['score_raw'].std():.1f}")
    
    print(f"\nFeature contribution to average score::")
    score_details = df["score_details"].iloc[0]
    for feat, val in sorted(score_details.items(), key=lambda x: abs(x[1]), reverse=True):
        bar = "=" * int(abs(val) * 1.5) if abs(val) > 0 else ""
        print(f"{feat:<20} {val:>+7.2f}  {bar}")

    print(f"\nRating categories:")
    rating_dist = df["rating"].value_counts().sort_index()
    for r, cnt in rating_dist.items():
        pct = cnt / len(df) * 100
        bar = "=" * int(pct / 3)
        print(f"  {r:>4}: {cnt:>4} ({pct:>5.1f}%)  {bar}")

    print(f"\nRecommendations:")
    for rec, cnt in df["recommendation"].value_counts().items():
        print(f"  {rec:<25}: {cnt:>4} ({cnt/len(df)*100:.1f}%)")


    print(f"\n{SEP}")

    X, X_sc, y, scaler = prepare_ml_data(df)
    clf_res  = train_classifiers(X_sc, y)

    print(f"{'Model':<18} {'Acc%':>8} {'F1':>8} {'AUC':>8} {'CV-Acc%':>10} {'Time,ms':>8}")
    for name, res in clf_res.items():
        if name in ("best", "importances", "X_tr", "X_te", "y_tr", "y_te"):
            continue
        print(f"{name:<18} {res['acc']:>8.2f} {res['f1']:>8.4f} {res['auc']:>8.4f} {res['cv']:>10.2f} {res['t_ms']:>8.1f}")

    print(f"\nBest model (AUC): {clf_res['best']} (AUC={clf_res[clf_res['best']]['auc']:.4f})")

    print(f"\nTop-10 features (Random Forest importance):")
    for feat, imp in clf_res['importances'].nlargest(10).items():
        bar = "=" * int(imp * 100)
        print(f"  {feat:<20} {imp:.4f}  {bar}")


    print(f"\n{SEP}")

    clu_res  = cluster_borrowers(X_sc, df)

    print(f"PCA: {clu_res['pca'].explained_variance_ratio_.cumsum()[-1]*100:.1f}% variance in 3 components")
    print(f"Optimal K (Silhouette): {clu_res['best_k']}")

    print(f"\n{'Cluster':>9} {'N':>5}  {'Income':>10}  {'DTI':>8}  {'Score':>8}  {'Bad%':>7}  Profile")
    for cl in range(clu_res['best_k']):
        sub = df[df["cluster"] == cl]
        profile_info = clu_res['profiles'][cl]
        print(f"{cl:>9} {profile_info['n']:>5}  {sub['monthly_income'].mean():>10,.0f}  {sub['dti'].mean():>8.3f}  {sub['score_raw'].mean():>8.1f}  {profile_info['bad_rate']:>6.1f}%  {profile_info['profile']}")

    best_model = clf_res[clf_res["best"]]["model"]
    df["ml_prediction"] = best_model.predict(X_sc)
    df["ml_probability"] = best_model.predict_proba(X_sc)[:, 1] if hasattr(best_model, "predict_proba") else df["ml_prediction"].astype(float)
    df["binary_decision"] = df["ml_prediction"]


    plot_eda_score_distribution(df, "l8_score_distribution.png")
    plot_eda_income_vs_score(df, "l8_income_vs_score.png")
    plot_eda_dti_vs_score(df, "l8_dti_vs_score.png")

    plot_fraud_score_distribution(df, "l8_fraud_score_distribution.png")
    plot_fraud_isolation_forest(df, "l8_fraud_isolation_forest.png")

    plot_ml_confusion_matrix(clf_res, "l8_ml_confusion_matrix.png")

    plot_clustering_pca_target(df, clu_res, "l8_clustering_pca_target.png")
    plot_clustering_dendrogram(clu_res, "l8_clustering_dendrogram.png")
    plot_clustering_pca_3d(clu_res, "l8_clustering_pca_3d.png")



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


SCORECARD = {
    "age": [
        (lambda x: x < 22, -10, "< 22 years"),
        (lambda x: (x >= 22) & (x < 25), 5, "22-24"),
        (lambda x: (x >= 25) & (x < 35), 20, "25-34"),
        (lambda x: (x >= 35) & (x < 50), 18, "35-49"),
        (lambda x: x >= 50, 10, "50+"),
    ],

    "monthly_income": [
        (lambda x: x < 5000, -15, "< 5000"),
        (lambda x: (x >= 5000) & (x < 8000), 5, "5000-8000"),
        (lambda x: (x >= 8000) & (x < 12000), 15, "8000-12000"),
        (lambda x: (x >= 12000) & (x < 20000), 20, "12000-20000"),
        (lambda x: x >= 20000, 15, "20000+"),
    ],

    "dti": [
        (lambda x: x <= 0.30, 25, "≤ 0.30"),
        (lambda x: (x > 0.30) & (x <= 0.50), 18, "0.30-0.50"),
        (lambda x: (x > 0.50) & (x <= 0.70), 10, "0.50-0.70"),
        (lambda x: (x > 0.70) & (x <= 1.0), 0, "0.70-1.0"),
        (lambda x: x > 1.0, -20, "> 1.0"),
    ],

    "seniority_years": [
        (lambda x: x < 1, -5, "< 1 year"),
        (lambda x: (x >= 1) & (x < 3), 10, "1-2 years"),
        (lambda x: (x >= 3) & (x < 7), 18, "3-6 years"),
        (lambda x: x >= 7, 22, "7+ years"),
    ],

    "education_id": [
        (lambda x: x == 1, -5, "None"),
        (lambda x: x == 2, 5, "Below secondary"),
        (lambda x: x == 3, 10, "Secondary"),
        (lambda x: x == 4, 14, "Vocational/Technical"),
        (lambda x: x == 5, 18, "Higher"),
        (lambda x: x == 6, 22, "Academic degree"),
    ],

    "has_immovables": [
        (lambda x: x == 1, 10, "Yes"),
        (lambda x: x == 0, -5, "No"),
    ],

    "has_movables": [
        (lambda x: x == 1, 8, "Yes"),
        (lambda x: x == 0, -3, "No"),
    ],

    "other_loans_active": [
        (lambda x: x == 0, 12, "None"),
        (lambda x: (x > 0) & (x <= 1), 0, "1"),
        (lambda x: x > 1, -10, "2+"),
    ],

    "loan_to_income": [
        (lambda x: x <= 0.5, 20, "≤ 0.5"),
        (lambda x: (x > 0.5) & (x <= 1.0), 15, "0.5-1.0"),
        (lambda x: (x > 1.0) & (x <= 2.0), 8, "1.0-2.0"),
        (lambda x: (x > 2.0) & (x <= 4.0), 0, "2.0-4.0"),
        (lambda x: x > 4.0, -15, "> 4.0"),
    ],

    "applied_night": [
        (lambda x: x == 1, -5, "Night application"),
        (lambda x: x == 0, 5, "Day application"),
    ],

    "fraud_score": [
        (lambda x: x == 0, 10, "No fraud flags"),
        (lambda x: x == 1, 0, "1 flag"),
        (lambda x: (x >= 2) & (x <= 3), -10, "2-3 flags"),
        (lambda x: x > 3, -25, "> 3 flags"),
    ],
}


def compute_scorecard(df: pd.DataFrame):
    df["score_raw"] = SCORE_BASE
    score_details = {}

    for feature, bins in SCORECARD.items():
        if feature not in df.columns:
            continue
        col = df[feature]
        pts = pd.Series(0, index=df.index)
        for cond_fn, points, _ in bins:
            mask = cond_fn(col)
            pts[mask] = points
        df["score_raw"] += pts
        score_details[feature] = pts.mean()

    df["score_raw"] = df["score_raw"].clip(SCORE_MIN, SCORE_MAX)

    center = df["score_raw"].mean()
    scale = df["score_raw"].std() + 1e-9
    df["pd_estimate"] = 1 / (1 + np.exp((df["score_raw"] - center) / scale))

    df["rating"] = pd.cut(
        df["score_raw"],
        bins=[SCORE_MIN, 300, 370, 430, 500, 580, SCORE_MAX],
        labels=["HR","D","C","B","A","AA"],
        include_lowest=True
    )

    conditions = [
        df["score_raw"] >= 500,
        df["score_raw"] >= 430,
        df["score_raw"] >= 370,
        df["score_raw"] < 370,
    ]
    choices = ["Approve", "Conditionally approve", "Requires manual review", "Reject"]
    df["recommendation"] = np.select(conditions, choices, default="Reject")

    df["score_details"] = [score_details] * len(df)

    return df


ML_FEATURES = [
    "loan_amount", "loan_days", "age", "gender_id",
    "marital_status_id", "children_count_id", "education_id",
    "has_immovables", "has_movables", "employment_type_id",
    "seniority_years", "monthly_income", "monthly_expenses",
    "other_loans_active", "other_loans_about_monthly",
    "dti", "loan_to_income", "net_income", "expense_ratio",
    "job_tenure_months", "addr_tenure_months", "daily_payment",
    "applied_night", "applied_weekend", "is_repeat_client",
    "fraud_score", "has_active_loans", "prolongation_count",
    "fact_addr_owner_type_id",
]

def prepare_ml_data(df: pd.DataFrame):
    X = df[ML_FEATURES].copy()
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())

    y = df["target"].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    return X, X_sc, y, scaler


def train_classifiers(X_sc: np.ndarray, y: np.ndarray):
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.25, random_state=42, stratify=y)

    models = {
        "LogisticReg": LogisticRegression(max_iter=500, random_state=42, C=0.8),
        "KNN(k=7)": KNeighborsClassifier(n_neighbors=7, metric="euclidean"),
        "SVM(RBF)": SVC(kernel="rbf", probability=True, random_state=42, C=1.0),
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42, max_depth=8, class_weight="balanced"),
        "GradientBoost":  GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
        t_ms = (time.perf_counter() - t0) * 1000

        acc = accuracy_score(y_te, y_pred) * 100
        f1 = f1_score(y_te, y_pred, zero_division=0)
        auc = roc_auc_score(y_te, y_proba) if y_proba is not None else 0.0
        cv_s = cross_val_score(model, X_sc, y, cv=cv, scoring="accuracy").mean() * 100

        results[name] = dict(
            model=model, acc=acc, f1=f1, auc=auc,
            cv=cv_s, y_pred=y_pred, y_proba=y_proba,
            y_te=y_te, t_ms=t_ms,
        )

    best_name = max(results, key=lambda k: results[k]["auc"])

    rf = models["RandomForest"]
    importances = pd.Series(rf.feature_importances_, index=ML_FEATURES)

    results["best"] = best_name
    results["importances"] = importances
    results["X_tr"] = X_tr
    results["X_te"] = X_te
    results["y_tr"] = y_tr
    results["y_te"] = y_te
    return results


def cluster_borrowers(X_sc: np.ndarray, df: pd.DataFrame):
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    K_range = range(2, 9)
    inertias, sils = [], []
    for k in K_range:
        km = KMeans(n_clusters=k, n_init=15, random_state=42)
        lbl = km.fit_predict(X_sc)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X_sc, lbl, sample_size=400))

    best_k = list(K_range)[np.argmax(sils)]

    km_final = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    km_labels = km_final.fit_predict(X_sc)

    agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    agg_labels= agg.fit_predict(X_sc)

    nbrs = NearestNeighbors(n_neighbors=8).fit(X_sc)
    dists, _ = nbrs.kneighbors(X_sc)
    eps = float(np.percentile(dists[:, -1], 90))
    dbs = DBSCAN(eps=eps, min_samples=8)
    dbs_labels = dbs.fit_predict(X_sc)
    n_dbs = len(set(dbs_labels)) - (1 if -1 in dbs_labels else 0)

    df["cluster"] = km_labels
    cluster_profiles = {}
    for cl in range(best_k):
        sub = df[df["cluster"] == cl]
        bad_rate = (1 - sub["target"].mean()) * 100
        profile = ("Risky" if bad_rate > 40 else "Reliable" if bad_rate < 15 else "Medium")
        cluster_profiles[cl] = dict(n=len(sub), bad_rate=bad_rate, profile=profile)

    Z_link = linkage(X_sc[:100], method="ward")

    return dict(
        X_pca=X_pca, km_labels=km_labels, agg_labels=agg_labels,
        dbs_labels=dbs_labels, n_dbs=n_dbs, best_k=best_k,
        inertias=inertias, sils=sils, K_range=list(K_range),
        Z_link=Z_link, pca=pca, profiles=cluster_profiles,
    )


def plot_eda_score_distribution(df, fname):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=C["bg"])
    for tgt, lbl, clr in [(1, "Returned", C["good"]), (0, "Overdue", C["bad"])]:
        vals = df[df["target"] == tgt]["score_raw"]
        ax.hist(vals, bins=40, density=True, alpha=0.65, color=clr, edgecolor=C["grid"], label=f"{lbl} (N={len(vals)})")
    ax.axvline(430, color=C["gold"], lw=2.0, ls="--", label="Approval threshold")
    ax.set_title("Score distribution: Returned vs Overdue", color=C["text"])
    ax.set_xlabel("Scoring Score", color=C["sub"])
    ax.set_ylabel("Density", color=C["sub"])
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["sub"])
    for spine in ax.spines.values():
        spine.set_edgecolor(C["grid"])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.close(fig)


def plot_eda_income_vs_score(df, fname):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=C["bg"])
    for tgt, clr, lbl in [(1, C["good"], "Returned"), (0, C["bad"], "Overdue")]:
        sub = df[df["target"] == tgt]
        ax.scatter(sub["monthly_income"].clip(0, 50000), sub["score_raw"], c=clr, s=15, alpha=0.55, label=lbl)
    ax.axhline(430, color=C["gold"], lw=1.5, ls="--")
    ax.set_title("Income vs Scoring Score", color=C["text"])
    ax.set_xlabel("Monthly income, UAH", color=C["sub"])
    ax.set_ylabel("Score", color=C["sub"])
    ax.set_facecolor(C["panel"])
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])
    ax.tick_params(colors=C["sub"])
    for spine in ax.spines.values(): 
        spine.set_edgecolor(C["grid"])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.close(fig)


def plot_eda_dti_vs_score(df, fname):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=C["bg"])
    sc = ax.scatter(df["dti"].clip(0, 3), df["score_raw"], c=df["target"], cmap=plt.cm.colors.ListedColormap([C["bad"], C["good"]]), s=15, alpha=0.65, edgecolors="none")
    ax.axhline(430, color=C["gold"], lw=1.5, ls="--")
    ax.axvline(0.7, color=C["c6"], lw=1.5, ls=":")
    ax.set_title("DTI vs Scoring Score", color=C["text"])
    ax.set_xlabel("DTI (debt/income)", color=C["sub"])
    ax.set_ylabel("Score", color=C["sub"])
    ax.set_facecolor(C["panel"])
    plt.colorbar(sc, ax=ax, fraction=0.04, label="0=Overdue / 1=Returned")
    ax.tick_params(colors=C["sub"])
    for spine in ax.spines.values():
        spine.set_edgecolor(C["grid"])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.close(fig)




def plot_fraud_score_distribution(df, fname):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=C["bg"])
    ax.hist(df["fraud_score"], bins=range(0, df["fraud_score"].max() + 2), color=C["c1"], alpha=0.85, edgecolor=C["grid"], align="left", rwidth=0.8)
    ax.axvline(2, color=C["bad"], lw=2.0, ls="--", label="Risk threshold = 2")
    ax.axvline(5, color=C["fraud"], lw=2.0, ls="-.", label="Critical risk = 5")
    ax.set_title("Fraud Score distribution (number of triggered rules)", color=C["text"])
    ax.set_xlabel("Fraud Score", color=C["sub"])
    ax.set_ylabel("Number of applications", color=C["sub"])
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["sub"])
    for spine in ax.spines.values():
        spine.set_edgecolor(C["grid"])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.close(fig)


def plot_fraud_isolation_forest(df, fname):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=C["bg"])
    iso_fraud = df[df["iso_anomaly"] == 1]["fraud_score"]
    iso_normal = df[df["iso_anomaly"] == 0]["fraud_score"]
    ax.hist(iso_normal, bins=range(0, 10), density=True, alpha=0.65,
            color=C["c0"], edgecolor=C["grid"], label="Normal (ISO)", align="left")
    ax.hist(iso_fraud, bins=range(0, 10), density=True, alpha=0.65,
            color=C["fraud"], edgecolor=C["grid"], label="Anomaly (ISO)", align="left")
    ax.set_title("Isolation Forest: Rule-based Fraud Score (Normal vs Anomaly)", color=C["text"])
    ax.set_xlabel("Rule-based Fraud Score", color=C["sub"])
    ax.set_ylabel("Density", color=C["sub"])
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["sub"])
    for spine in ax.spines.values(): spine.set_edgecolor(C["grid"])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.close(fig)


def plot_ml_confusion_matrix(clf_res, fname):
    best = clf_res["best"]
    y_te = clf_res[best]["y_te"]
    cm = confusion_matrix(y_te, clf_res[best]["y_pred"])
    
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=C["bg"])
    cmap_cm = LinearSegmentedColormap.from_list("cm", ["#0D1117", C["c0"]])
    im = ax.imshow(cm, cmap=cmap_cm)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() * 0.4 else C["sub"])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Overdue", "Returned"], color=C["text"])
    ax.set_yticklabels(["Overdue", "Returned"], color=C["text"])
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(f"Confusion Matrix: {best}", color=C["text"])
    ax.set_xlabel("Prediction", color=C["sub"])
    ax.set_ylabel("Actual", color=C["sub"])
    ax.tick_params(colors=C["sub"])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.close(fig)


def plot_clustering_pca_target(df, clu_res, fname):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=C["bg"])
    X_pca = clu_res["X_pca"]
    
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df["target"].values, cmap=ListedColormap([C["bad"], C["good"]]), s=20, alpha=0.65, edgecolors="none")
    ax.set_title("PCA: Returned (green) vs Overdue (red)", color=C["text"])
    ax.set_xlabel("PC1", color=C["sub"])
    ax.set_ylabel("PC2", color=C["sub"])
    ax.set_facecolor(C["panel"])
    plt.colorbar(sc, ax=ax, fraction=0.04, label="0=Overdue / 1=Returned")
    ax.tick_params(colors=C["sub"])
    for spine in ax.spines.values():
        spine.set_edgecolor(C["grid"])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.close(fig)


def plot_clustering_dendrogram(clu_res, fname):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=C["bg"])
    
    dendrogram(clu_res["Z_link"], ax=ax, truncate_mode="lastp", p=15, show_leaf_counts=True, color_threshold=clu_res["Z_link"][-clu_res["best_k"] + 1, 2], above_threshold_color=C["sub"])
    
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["sub"])
    for spine in ax.spines.values(): spine.set_edgecolor(C["grid"])
    ax.set_title("Dendrogram (Ward linkage, 100 observations)", color=C["text"], pad=10)
    ax.set_xlabel("Application ID", color=C["sub"])
    ax.set_ylabel("Distance", color=C["sub"])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.close(fig)


def plot_clustering_pca_3d(clu_res, fname):
    fig = plt.figure(figsize=(12, 10), facecolor=C["bg"])
    ax = fig.add_subplot(111, projection="3d")
    
    X_pca = clu_res["X_pca"]
    km_labels = clu_res["km_labels"]
    best_k = clu_res["best_k"]
    
    cmap_k = ListedColormap(PAL8[:best_k])
    
    ax.set_facecolor(C["panel"])
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(C["grid"])
    ax.yaxis.pane.set_edgecolor(C["grid"])
    ax.zaxis.pane.set_edgecolor(C["grid"])
    
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=km_labels, cmap=cmap_k, s=15, alpha=0.65)
    
    ax.set_xlabel("PC1", color=C["sub"])
    ax.set_ylabel("PC2", color=C["sub"])
    ax.set_zlabel("PC3", color=C["sub"])
    ax.tick_params(colors=C["sub"], labelsize=8)
    ax.set_title(f"3D PCA: K-Means (k={best_k})", color=C["text"])
    ax.view_init(elev=20, azim=45)
    ax.grid(color=C["grid"], alpha=0.2)
    
    plt.colorbar(sc, ax=ax, fraction=0.04, label="Cluster")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path)
    plt.close(fig)



if __name__ == "__main__":
    main()