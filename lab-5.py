import os
import requests
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.colors import ListedColormap


SEP = "=" * 67

OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "l5_nbu_exchange_rates_2y.csv")
IMG_PATH = os.path.join(OUTPUT_DIR, "l5_img.jpg")

CURRENCIES = ["USD", "EUR", "GBP"]
DAYS_BACK = 365 * 2

C = dict(
    bg="#0D1117", panel="#161B22", grid="#30363D",
    text="#E6EDF3", sub="#8B949E",
    c0="#2196F3", c1="#FF9800", c2="#4CAF50", c3="#E91E63",
    c4="#9C27B0", c5="#00BCD4", c6="#FF5722"
)
PALETTE_7 = [C["c0"],C["c1"],C["c2"],C["c3"],C["c4"],C["c5"],C["c6"]]


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
    r1 = gtv1_clustering(df)


    print(f"\nPCA: {r1["pca"].explained_variance_ratio_.cumsum()*100}")
    print(f"PCA: {r1["pca"].explained_variance_ratio_.sum()*100:.1f}% dispersion in 4 components")


    print(f"\nK-Means best k: K={r1['best_k']}")
    print(f" K  {'Silhouette':<15} {'Davies-B.':<15} {'Calinski':<15}")
    for idx, k in enumerate(r1["K_RANGE"]):
        print(f" {k}   {r1["sils"][idx]:<15.4f} {r1["dbs_scores"][idx]:<15.4f} {r1["chs"][idx]:<15.4f} ")

    print(f"{'Method':<15} {'K':>4}  {'Silhouette':>10}  {'Davies-B.':>10}  {'Calinski':>10}")
    for m_name in ["K-Means", "Hierarchical", "GMM", "DBSCAN"]:
        r = r1[m_name]
        print(f"{m_name:<15} {r['k']:>4}  {r['sil']:>10.4f}  {r['db']:>10.4f}  {r['ch']:>10.1f}")
    print(f"KNeighborsClassifier - accuracy: {r1['KNN']['acc']*100:.2f}%")



    print(f"\n{SEP}")
    print("GTV-2: Clustering of img")

    img = cv.imread(IMG_PATH)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    enh_dict = enhance_image(img)
    enh_img = enh_dict["enhanced"]
    enh_img_rgb = cv.cvtColor(enh_img, cv.COLOR_BGR2RGB)

    r2 = gtv2_color_clustering(img_rgb, enh_img_rgb, n_clusters=3)

    print(f"\nK-Means best k: K={r2['best_k2']}")
    print(f"\n  {'Cluster':>8}  {'Pixels':>9}  {'%':>6} {'L':>6}  {'a':>6}  {'b':>6}  Dominant color")
    for seg_stat in r2['stats']:
        rgb_values = tuple(int(x) for x in seg_stat['rgb'])
        print(f"  {seg_stat['k']:>8}  {seg_stat['count']:>9}  {seg_stat['pct']:>6.2f} {seg_stat['L']:>6.1f}  {seg_stat['a']:>6.1f}  {seg_stat['b']:>6.1f}  {rgb_values}")



    print(f"\n{SEP}")
    print("GTV-3: Counting objects on an image ")
    coins_img, placed = detect_coins_from_image("./outputs/l5_coins.jpg")

    r3 = gtv3_object_counting(coins_img, placed)

    print(f"True number of coins: {r3['gt_count']}")
    print(f"{'Method':<28} {'Count':<8} Error")
    for name, cnt in [
        ("HoughCircles", r3['n_hough']),
        ("Watershed + CC", r3['n_ws']),
        ("Ensemble (fused)", r3['n_fused']),
    ]:
        err = abs(cnt - r3['gt_count'])
        pct = err / max(r3['gt_count'], 1) * 100
        print(f"  {name:<28} {cnt:<8} {pct:5.1f}%")

    print(f"Radius (fused):")
    print(f"  Small  (r <  60 px) : {r3['small']}")
    print(f"  Medium (60 ≤ r < 75): {r3['medium']}")
    print(f"  Large  (r ≥  75 px) : {r3['large']}")
    if r3['radii']:
        print(f" min={min(r3['radii'])}px  max={max(r3['radii'])}px  avg={sum(r3['radii'])/len(r3['radii']):.1f}px")


    plot_pca_methods(r1, "l5_pca_methods.png")
    plot_dendrogram(r1, "l5_dedrogram.png")

    plot_original_vs_enhanced(img_rgb, enh_img_rgb, "l5_original_vs_enhanced.png")
    plot_segmented_image(r2, "l5_segmented_image.png")
    plot_cluster_masks(enh_dict, r2, "l5_cluster_masks.png")
    
    plot_detection_results(r3['img_bgr'], r3['opened'], r3['ws_det'], r3['fused'], r3['markers_ws'], r3['gt_count'], "l5_detection_results.png")
    plot_radii_histogram(r3, "l5_radii_histogram.png")


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

    
    gmm = GaussianMixture(n_components=best_k, covariance_type="full", n_init=5, random_state=42)
    lbl_g = gmm.fit_predict(X)
    proba = gmm.predict_proba(X)
    results["GMM"] = dict(
        labels=lbl_g, k=best_k, proba=proba,
        sil=silhouette_score(X, lbl_g),
        db=davies_bouldin_score(X, lbl_g),
        ch=calinski_harabasz_score(X, lbl_g),
        bic=gmm.bic(X), aic=gmm.aic(X),
    )


    nbrs = NearestNeighbors(n_neighbors=5).fit(X)
    dists, _ = nbrs.kneighbors(X)
    eps = float(np.percentile(dists[:, -1], 90))
    dbs_m = DBSCAN(eps=eps, min_samples=5)
    lbl_d = dbs_m.fit_predict(X)
    n_dbscan = len(set(lbl_d)) - (1 if -1 in lbl_d else 0)
    noise_n = int((lbl_d == -1).sum())
    sil_d = silhouette_score(X, lbl_d) if n_dbscan > 1 else -1.0
    results["DBSCAN"] = dict(
        labels=lbl_d, k=n_dbscan, eps=eps,
        sil=sil_d, noise=noise_n,
        db=davies_bouldin_score(X, lbl_d) if n_dbscan>1 else 99,
        ch=calinski_harabasz_score(X, lbl_d) if n_dbscan>1 else 0,
    )


    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = lbl_km[:split], lbl_km[split:]
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(X_tr, y_tr)
    acc = knn.score(X_te, y_te)
    lbl_knn = knn.predict(X)
    results["KNN"] = dict(
        labels=lbl_knn, acc=acc, k=best_k,
        sil=silhouette_score(X, lbl_knn),
    )

    return results


def enhance_image(img_rgb: np.ndarray):
    img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)

    img_bil = cv.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)

    img_lab = cv.cvtColor(img_bil, cv.COLOR_BGR2LAB)
    clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    img_clahe = cv.cvtColor(img_lab, cv.COLOR_LAB2BGR)

    img_hsv = cv.cvtColor(img_clahe, cv.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * 1.35, 0, 255)
    img_enhanced_bgr = cv.cvtColor(
        img_hsv.astype(np.uint8), cv.COLOR_HSV2BGR
    )
    img_enhanced = cv.cvtColor(img_enhanced_bgr, cv.COLOR_BGR2RGB)

    return dict(
        original=img_rgb,
        bilateral=cv.cvtColor(img_bil, cv.COLOR_BGR2RGB),
        enhanced=img_enhanced,
    )


def gtv2_color_clustering(img_rgb: np.ndarray, img_enhanced: np.ndarray, n_clusters: int):
    h, w = img_enhanced.shape[:2]

    img_lab = cv.cvtColor(img_enhanced, cv.COLOR_RGB2LAB).astype(np.float32)

    pixels_lab = img_lab.reshape(-1, 3)
    yy, xx = np.mgrid[0:h, 0:w]
    coords  = np.stack([xx.ravel() / w * 30, yy.ravel() / h * 30], axis=1).astype(np.float32)
    pixels  = np.concatenate([pixels_lab, coords], axis=1)

    sample_idx = np.random.choice(len(pixels), 3000, replace=False)
    X_s = pixels[sample_idx]

    inertias_2, sils_2 = [], []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=5, max_iter=100, random_state=42)
        lbl_s = km.fit_predict(X_s)
        inertias_2.append(km.inertia_)
        sils_2.append(silhouette_score(X_s, lbl_s, sample_size=1000, random_state=42))

    best_k2 = 2 + np.argmax(sils_2)

    km_img = KMeans(n_clusters=n_clusters, n_init=12, max_iter=300, random_state=42)
    labels = km_img.fit_predict(pixels)
    labels_img = labels.reshape(h, w)

    centers_lab = km_img.cluster_centers_[:, :3]
    seg_lab = np.zeros_like(img_lab)
    for k in range(n_clusters):
        mask = labels_img == k
        seg_lab[mask] = centers_lab[k]
    seg_rgb = cv.cvtColor(np.clip(seg_lab, 0, 255).astype(np.uint8), cv.COLOR_LAB2RGB)

    seg_stats = []
    total_px   = h * w
    for k in range(n_clusters):
        cnt  = int((labels == k).sum())
        pct  = cnt / total_px * 100
        L, a, b = centers_lab[k]

        ctr_lab = np.array([[[L, a, b]]], dtype=np.uint8)
        ctr_rgb = cv.cvtColor(ctr_lab, cv.COLOR_LAB2RGB)[0, 0]
        r, g, bv = ctr_rgb

        seg_stats.append(dict(k=k, count=cnt, pct=pct, L=L, a=a, b=b, rgb=tuple(ctr_rgb)))

    sil_img = silhouette_score(pixels[::50], labels[::50])
    print(f"\n  Silhouette score (к=7): {sil_img:.4f}")

    return dict( 
        best_k2=best_k2,
        labels=labels_img, seg_rgb=seg_rgb,
        centers_lab=centers_lab, stats=seg_stats,
        n_clusters=n_clusters, sil=sil_img,
        inertias=inertias_2, sils=sils_2,
    )


def detect_coins_from_image(image_path: str):
    img = cv.imread(image_path)
   
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.medianBlur(img_gray, 7)
    blurred = cv.GaussianBlur(blurred, (9, 9), 2.5)

    circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, dp=1.15, minDist=60, param1=60, param2=35, minRadius=50, maxRadius=90)
    
    placed = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        
        for (x, y, r) in circles:
            mask = np.zeros_like(blurred)
            cv.circle(mask, (x, y), r, 255, -1)
            
            mean_color = cv.mean(img_rgb, mask=mask)[:3]
            mean_color = [c / 255.0 for c in mean_color]
            
            placed.append((x, y, r, mean_color))
    return img_rgb, placed


def gtv3_object_counting(img_rgb: np.ndarray, ground_truth: list):
    h, w = img_rgb.shape[:2]
    img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    blurred_h = cv.medianBlur(img_gray, 7)
    blurred_h = cv.GaussianBlur(blurred_h, (9, 9), 2.5)

    raw_circles = cv.HoughCircles(blurred_h, cv.HOUGH_GRADIENT, dp=1.15, minDist=60, param1=60, param2=35, minRadius=50, maxRadius=90)
    hough_det = []
    if raw_circles is not None:
        for (x, y, r) in np.round(raw_circles[0]).astype(int):
            mask = np.zeros(img_gray.shape, dtype=np.uint8)
            cv.circle(mask, (x, y), r, 255, -1)
            mc = cv.mean(img_rgb, mask=mask)[:3]
            hough_det.append((int(x), int(y), int(r), tuple(c / 255.0 for c in mc)))


    bil  = cv.bilateralFilter(img_gray, d=11, sigmaColor=80, sigmaSpace=80)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enh  = clahe.apply(bil)

    _, thr_otsu = cv.threshold(enh, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    thr_adapt = cv.adaptiveThreshold(enh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 4)
    thr_combined = cv.bitwise_and(thr_otsu, thr_adapt)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    closed = cv.morphologyEx(thr_combined, cv.MORPH_CLOSE, kernel, iterations=2)
    opened = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel, iterations=1)

    dist   = cv.distanceTransform(opened, cv.DIST_L2, 5)
    dist_n = cv.normalize(dist, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    _, fg_sure = cv.threshold(dist_n, int(0.45 * dist_n.max()), 255, cv.THRESH_BINARY)
    fg_sure = fg_sure.astype(np.uint8)

    bg_sure = cv.dilate(opened, kernel, iterations=3)
    unknown = cv.subtract(bg_sure, fg_sure)

    n_markers, markers = cv.connectedComponents(fg_sure)
    markers = markers + 1
    markers[unknown == 255] = 0
    img_ws = img_bgr.copy()
    markers_ws = cv.watershed(img_ws, markers)

    n_cc, cc_labels, cc_stats, cc_centroids = cv.connectedComponentsWithStats(fg_sure, connectivity=8)
    min_area = 500
    max_area = h * w // 20
    valid_cc = [i for i in range(1, n_cc) if min_area < cc_stats[i, cv.CC_STAT_AREA] < max_area]

    ws_det = []
    for i in valid_cc:
        cx = int(cc_centroids[i][0])
        cy = int(cc_centroids[i][1])
        area = cc_stats[i, cv.CC_STAT_AREA]
        r_eq = int(np.sqrt(area / np.pi))
        mask = np.zeros(img_gray.shape, dtype=np.uint8)
        cv.circle(mask, (cx, cy), r_eq, 255, -1)
        mc = cv.mean(img_rgb, mask=mask)[:3]
        ws_det.append((cx, cy, r_eq, tuple(c / 255.0 for c in mc)))


    def circle_iou(c1, c2):
        x1, y1, r1 = c1[0], c1[1], c1[2]
        x2, y2, r2 = c2[0], c2[1], c2[2]
        d = np.hypot(x1 - x2, y1 - y2)
        if d >= r1 + r2:
            return 0.0
        if d <= abs(r1 - r2):
            smaller = min(r1, r2)
            return (np.pi * smaller**2) / (np.pi * max(r1, r2)**2)
        
        a1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2*d*r1))
        a2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2*d*r2))
        tri = 0.5 * np.sqrt((-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2))
        inter = a1 + a2 - tri
        union  = np.pi*(r1**2 + r2**2) - inter
        return inter / union if union > 0 else 0.0

    IOU_THRESH = 0.30

    fused = list(hough_det)
    for ws_c in ws_det:
        overlaps = any(circle_iou(ws_c, h_c) > IOU_THRESH for h_c in fused)
        if not overlaps:
            fused.append(ws_c)

    n_hough = len(hough_det)
    n_ws = len(ws_det)
    n_fused = len(fused)
    gt_count = len(ground_truth)

    radii_f = [c[2] for c in fused]
    small = sum(1 for r in radii_f if r < 60)
    medium = sum(1 for r in radii_f if 60 <= r < 75)
    large = sum(1 for r in radii_f if r >= 75)

    return dict(
        img_bgr=img_bgr,
        hough_det=hough_det, ws_det=ws_det, fused=fused,
        n_hough=n_hough, n_ws=n_ws, n_fused=n_fused,
        gt_count=gt_count,
        blurred_h=blurred_h, enh=enh,
        thr_combined=thr_combined, opened=opened,
        dist_n=dist_n, fg_sure=fg_sure, markers_ws=markers_ws,
        radii=radii_f, small=small, medium=medium, large=large,
    )


def plot_pca_methods(r1: dict, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig = plt.figure(figsize=(18, 10), facecolor=C["bg"])
    best_k = r1["best_k"]
    X_pca = r1["X_pca"]
    cmap_k = ListedColormap(PALETTE_7[:best_k])
    
    methods_plot = [
        ("K-Means", r1["K-Means"]["labels"]),
        ("Hierarchical", r1["Hierarchical"]["labels"]),
        ("GMM", r1["GMM"]["labels"]),
        ("DBSCAN", r1["DBSCAN"]["labels"]),
    ]
    
    for i, (mname, lbls) in enumerate(methods_plot, 1):
        ax = fig.add_subplot(2, 2, i)
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=lbls, cmap=cmap_k, s=14, alpha=0.75)
        
        if mname == "K-Means":
            ctr_pca = r1["pca"].transform(r1["K-Means"]["centers"])
            ax.scatter(ctr_pca[:, 0], ctr_pca[:, 1], marker="*", s=200, c="white", edgecolors="black", linewidths=0.8, zorder=10)
        
        k_show = r1[mname]["k"] if mname != "DBSCAN" else r1["DBSCAN"]["k"]
        sil_show = r1[mname]["sil"]
        ax.set_title(f"{mname} (k={k_show}) Sil={sil_show:.3f}", color=C["text"])
        ax.set_xlabel("PC1", color=C["sub"])
        ax.set_ylabel("PC2", color=C["sub"])
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["sub"])
        plt.colorbar(sc, ax=ax, fraction=0.04)
    
    fig.suptitle("Comparison of clustering methods in PCA space", color=C["text"])
    fig.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_dendrogram(r1: dict, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=C["bg"])
    best_k = r1["best_k"]
    
    dendrogram(r1["Z_link"], ax=ax, truncate_mode="lastp", p=20, show_leaf_counts=True, color_threshold=r1["Z_link"][-best_k+1, 2], above_threshold_color=C["sub"])
    
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["sub"])
    for sp in ax.spines.values():
        sp.set_edgecolor(C["grid"])
    
    ax.set_title("Dendrogram (Ward linkage)", color=C["text"])
    ax.set_xlabel("Index", color=C["sub"])
    ax.set_ylabel("Відстань злиття", color=C["sub"])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_original_vs_enhanced(original_img: np.ndarray, enhanced_img: np.ndarray, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor=C["bg"])
    
    axes[0].imshow(original_img)
    axes[0].set_title("Original", color=C["text"])
    axes[0].axis("off")
    axes[0].set_facecolor(C["panel"])
    
    axes[1].imshow(enhanced_img)
    axes[1].set_title("Enhanced", color=C["text"])
    axes[1].axis("off")
    axes[1].set_facecolor(C["panel"])
    
    fig.suptitle("Image comparison", color=C["text"])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_segmented_image(r2: dict, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=C["bg"])
    ax.imshow(r2["seg_rgb"])
    ax.set_title(f"K-Means segmentation (K={r2['n_clusters']}) Sil={r2['sil']:.4f}", color=C["text"])
    ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_cluster_masks(enh_dict: dict, r2: dict, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    n_show = min(r2["n_clusters"], 4)
    fig, axes = plt.subplots(1, n_show, figsize=(5*n_show, 4), facecolor=C["bg"])
    if n_show == 1:
        axes = [axes]
    
    stats_sorted = sorted(r2["stats"], key=lambda x: x["count"], reverse=True)
    
    for ax, s in zip(axes, stats_sorted[:n_show]):
        mask = r2["labels"] == s["k"]
        img = cv.cvtColor(enh_dict["enhanced"], cv.COLOR_RGB2BGR)
        overlay = img.copy()
        overlay[~mask] = (overlay[~mask] * 0.20).astype(np.uint8)
        ax.imshow(overlay)
        r_c, g_c, b_c = s["rgb"]
        ax.set_title(f"Cluster {s['k']}: {s['pct']:.1f}% RGB({r_c},{g_c},{b_c})", color=C["text"])
        ax.axis("off")
        ax.set_facecolor(C["panel"])
    
    fig.suptitle("Visualization of individual color clusters", color=C["text"])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_detection_results(img_bgr: np.ndarray, opened: np.ndarray, ws_det: list, fused: list, markers_ws: np.ndarray, gt_count: int, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    img_ws_vis = img_bgr.copy()
    for (x, y, r, _) in ws_det:
        cv.circle(img_ws_vis, (x, y), r, (255, 140, 0), 2)
        cv.circle(img_ws_vis, (x, y), 3, (0, 0, 200), -1)

    img_fused = img_bgr.copy()
    for idx, (x, y, r, _) in enumerate(fused):
        cv.circle(img_fused, (x, y), r, (0, 255, 0), 2)
        cv.circle(img_fused, (x, y), 3, (0, 0, 255), -1)
        cv.putText(img_fused, str(idx + 1), (x - 10, y + 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    ws_boundary = img_bgr.copy()
    ws_boundary[markers_ws == -1] = [0, 0, 255]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    panels = [
        (opened, "Watershed: морфологія", True),
        (img_ws_vis, f"Watershed+CC: {len(ws_det)}/{gt_count}"),
        (img_fused, f"Ensemble fused: {len(fused)}/{gt_count}"),
    ]
    
    for ax, (panel, title, *gray) in zip(axes.flat, panels):
        if gray:
            ax.imshow(panel, cmap="gray")
        else:
            ax.imshow(cv.cvtColor(panel, cv.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_radii_histogram(r3: dict, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=C["bg"])
    
    if r3["radii"]:
        bins_r = np.linspace(min(r3["radii"]), max(r3["radii"]), 12)
        ax.hist(r3["radii"], bins=bins_r, color=C["c1"], alpha=0.85, edgecolor=C["grid"])
        ax.axvline(60, color=C["c3"], ls="--", lw=2, label="small/medium")
        ax.axvline(75, color=C["c0"], ls="--", lw=2, label="medium/large")
    
    ax.set_title("Distribution of coin radii (pixels)", color=C["text"])
    ax.set_xlabel("Radius, px", color=C["sub"])
    ax.set_ylabel("Quantity", color=C["sub"])
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["sub"])
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    main()