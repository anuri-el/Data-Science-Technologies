import os
import numpy as np
import pandas as pd
import alphashape
import geopandas as gpd
import geonamescache
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, LogNorm, Normalize
from shapely.geometry import Point, MultiPoint, Polygon
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.ndimage import gaussian_filter


from pathlib import Path
OUT_DIR = Path("outputs")



gc = geonamescache.GeonamesCache()


SEP = "=" * 67

OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


C = dict(
    bg="#0A0E1A", ocean="#0D1B2E", land="#1C2E4A",
    border="#4A6080", text="#E8EDF8", sub="#7A8BA0",
    grid="#1A2438", panel="#101828",
    c0="#2196F3", c1="#FF9800", c2="#4CAF50", c3="#E91E63",
    c4="#9C27B0", c5="#00BCD4", c6="#FF5722", c7="#8BC34A",
    gold="#FFD700", good="#4CAF50", bad="#EF5350",
)
PAL8   = [C[f"c{i}"] for i in range(8)]

CLUST_COLORS = ["#2196F3","#FF9800","#4CAF50","#E91E63","#9C27B0","#00BCD4"]
US_EXTENT = (-125, -65, 24, 50)
EU_EXTENT = (-12,  35,  34, 72)


def main():
    print(f"\n{SEP}")
    print("Level II")

    us_cities_gdf, eu_cities_gdf, us_states_meta, eu_countries = load_real_data()

    print(f"Loaded US cities (pop>5K): {len(us_cities_gdf):>6}")
    print(f"Loaded European cities (pop>30K): {len(eu_cities_gdf):>6}")
    print(f"US states: {len(us_states_meta):>6}")
    print(f"European countries: {len(eu_countries):>6}")

    print(f"\nTop-5 US cities by population:")
    for _, r in us_cities_gdf.nlargest(5, "population").iterrows():
        print(f" {r['name']:<20} {r['state']:<4} lat={r['lat']:>8.4f}  lon={r['lon']:>9.4f}  pop={r['population']:>10,}")
        
    print(f"\nTop-5 European cities by population:")
    for _, r in eu_cities_gdf.nlargest(5, "population").iterrows():
        print(f" {r['name']:<20} lat={r['lat']:>8.4f}  lon={r['lon']:>7.4f}  pop={r['population']:>10,}")


    print(f"\n{SEP}")
    print("State Polygons")

    gdf = build_state_polygons(us_cities_gdf, us_states_meta)

    print(f"Unit verification (official vs geometric area):")
    print(f"{'State':<22} {'Official km2':>12}  {'Computed km2':>13}  {'Δ%':>7}  {'Density people/km2':>14}")
    for _, r in gdf.nlargest(15, "density").iterrows():
        if r["area_official"] > 0:
            delta = abs(r["area_computed_km2"] - r["area_official"]) / r["area_official"] * 100
            print(f"{r['state_name']:<22} {r['area_official']:>12,}  {r['area_computed_km2']:>13,.0f}  {delta:>7.1f}%  {r['density']:>14.2f}")


    print(f"\n{SEP}")
    gdf, ml_res = cluster_states(gdf)
    best_k = ml_res['best_k']
    sils = ml_res['sils']
    knn_acc = ml_res.get('knn_acc', 0) 

    print(f"\nOptimal K (Silhouette max={max(sils):.4f}): K={best_k}")
    print(f"KNN classifier (k=5): accuracy {knn_acc:.1f}%")
    print(f"\nK-Means Cluster Profile (K={best_k}):")
    print(f"  {'Cluster':>8} {'N':>4}  {'Avg.Dens':>11}  {'Avg.Pop':>12}  Representatives")

    for cl in range(best_k):
        sub = gdf[gdf["cluster"] == cl]
        dens_m = sub["density"].mean()
        pop_m = sub["pop_official"].mean()
        reps = ", ".join(sub.nlargest(3, "pop_official")["state_code"].tolist())
        print(f"  {cl:>8} {len(sub):>4}  {dens_m:>11.2f}  {pop_m:>12,.0f}  {reps}")
    
    print(f"\nAverage KNN distance between state centroids: {gdf['knn_avg_dist_km'].mean():.0f} km")


    print(f"\n{SEP}")
    print("Population Density of European Counties")
    eu_analysis = build_europe_analysis(eu_cities_gdf, eu_countries)
    
    print(f"Countries in sample: {len(eu_analysis)}")
    print(f"\nTop-10 by density:")
    for _, r in eu_analysis.nlargest(10, "density").iterrows():
        print(f"{r['name']:<17}: {r['density']:>6.1f} people/km2  (pop. {r['population']/1e6:.2f}M, area {r['area_km2']:,.0f} km2)")


    plot_us_density(gdf, us_cities_gdf, "l9_us_density.png")
    plot_us_clusters(gdf, ml_res, "l9_us_clusters.png")
    plot_us_centroids_knn(gdf, ml_res, "l9_us_centroids_knn.png")
    plot_us_kde(gdf, us_cities_gdf, "l9_us_kde.png")




def load_real_data():
    all_cities  = gc.get_cities()
    all_countries = gc.get_countries()
    us_states_raw = gc.get_us_states()

    us_rows = []
    for v in all_cities.values():
        if v["countrycode"] == "US" and v["population"] > 5000:
            us_rows.append(dict(
                geonameid = v["geonameid"],
                name = v["name"],
                state = v["admin1code"],
                lat = float(v["latitude"]),
                lon = float(v["longitude"]),
                population= int(v["population"]),
                timezone = v["timezone"],
                geometry = Point(float(v["longitude"]), float(v["latitude"])),
            ))
    us_cities_gdf = gpd.GeoDataFrame(us_rows, crs="EPSG:4326")

    us_states_meta = {k: v["name"] for k, v in us_states_raw.items()}

    eu_iso = {k for k, v in all_countries.items() if v["continentcode"] == "EU"}
    eu_rows = []
    for v in all_cities.values():
        if v["countrycode"] in eu_iso and v["population"] > 30000:
            cdata = all_countries.get(v["countrycode"], {})
            eu_rows.append(dict(
                geonameid = v["geonameid"],
                name = v["name"],
                countrycode = v["countrycode"],
                country = cdata.get("name", ""),
                lat = float(v["latitude"]),
                lon = float(v["longitude"]),
                population = int(v["population"]),
                geometry = Point(float(v["longitude"]), float(v["latitude"])),
            ))
    eu_cities_gdf = gpd.GeoDataFrame(eu_rows, crs="EPSG:4326")
    
    eu_countries = {k: v for k, v in all_countries.items() if v["continentcode"] == "EU" and v["population"] > 100_000 and v["areakm2"] > 100}

    return us_cities_gdf, eu_cities_gdf, us_states_meta, eu_countries



US_STATE_STATS = {
    "AL":{"area":135767,"pop":5024279},"AK":{"area":1723337,"pop":733391},
    "AZ":{"area":295234,"pop":7151502},"AR":{"area":137732,"pop":3011524},
    "CA":{"area":423967,"pop":39538223},"CO":{"area":269601,"pop":5773714},
    "CT":{"area":14357, "pop":3605944},"DE":{"area":6446,  "pop":989948},
    "FL":{"area":170312,"pop":21538187},"GA":{"area":153910,"pop":10711908},
    "HI":{"area":28313, "pop":1455271},"ID":{"area":216443,"pop":1839106},
    "IL":{"area":149995,"pop":12812508},"IN":{"area":94326, "pop":6785528},
    "IA":{"area":145746,"pop":3190369},"KS":{"area":213100,"pop":2937880},
    "KY":{"area":104656,"pop":4505836},"LA":{"area":135659,"pop":4657757},
    "ME":{"area":91633, "pop":1362359},"MD":{"area":32131, "pop":6177224},
    "MA":{"area":27336, "pop":7029917},"MI":{"area":250487,"pop":10077331},
    "MN":{"area":225163,"pop":5706494},"MS":{"area":125438,"pop":2961279},
    "MO":{"area":180540,"pop":6154913},"MT":{"area":380831,"pop":1084225},
    "NE":{"area":200330,"pop":1961504},"NV":{"area":286380,"pop":3104614},
    "NH":{"area":24214, "pop":1377529},"NJ":{"area":22591, "pop":9288994},
    "NM":{"area":314917,"pop":2117522},"NY":{"area":141297,"pop":20201249},
    "NC":{"area":139391,"pop":10439388},"ND":{"area":183108,"pop":779094},
    "OH":{"area":116098,"pop":11799448},"OK":{"area":181037,"pop":3959353},
    "OR":{"area":254799,"pop":4237256},"PA":{"area":119280,"pop":13002700},
    "RI":{"area":4001,  "pop":1097379},"SC":{"area":82933, "pop":5118425},
    "SD":{"area":199729,"pop":886667}, "TN":{"area":109153,"pop":6910840},
    "TX":{"area":695662,"pop":29145505},"UT":{"area":219882,"pop":3271616},
    "VT":{"area":24906, "pop":643077}, "VA":{"area":110787,"pop":8631393},
    "WA":{"area":184661,"pop":7705281},"WV":{"area":62756, "pop":1793716},
    "WI":{"area":169635,"pop":5893718},"WY":{"area":253335,"pop":576851},
    "DC":{"area":177,   "pop":689545},
}


def build_state_polygons(us_cities_gdf: gpd.GeoDataFrame, us_states_meta: dict):
    records = []
    skipped = []

    for state_code, state_name in us_states_meta.items():
        sub = us_cities_gdf[us_cities_gdf["state"] == state_code]
        stats = US_STATE_STATS.get(state_code, {})

        if len(sub) < 3:
            skipped.append(state_code)
            continue

        pts = list(zip(sub["lon"], sub["lat"]))

        area_est = stats.get("area", 100000)
        alpha = 8.0 if area_est < 50000 else 4.0 if area_est < 150000 else 2.0 if area_est < 300000 else 1.5

        try:
            polygon = alphashape.alphashape(pts, alpha)
            if polygon is None or polygon.geom_type not in ("Polygon", "MultiPolygon"):
                polygon = MultiPoint(pts).convex_hull
        except Exception:
            polygon = MultiPoint(pts).convex_hull

        if polygon is None or polygon.is_empty:
            polygon = MultiPoint(pts).convex_hull

        total_pop = sub["population"].sum()
        cx_w = (sub["lon"] * sub["population"]).sum() / (total_pop + 1)
        cy_w = (sub["lat"] * sub["population"]).sum() / (total_pop + 1)

        records.append(dict(
            state_code = state_code,
            state_name = state_name,
            n_cities = len(sub),
            centroid_lon= cx_w,
            centroid_lat= cy_w,
            pop_official = stats.get("pop", 0),
            area_official = stats.get("area", 0),
            density = stats.get("pop", 0) / max(stats.get("area", 1), 1),
            geometry = polygon,
        ))

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    gdf_m = gdf.copy().to_crs("EPSG:3857")
    gdf["area_computed_km2"] = gdf_m.geometry.area / 1e6

    return gdf


def cluster_states(gdf: gpd.GeoDataFrame):
    gdf["log_density"] = np.log1p(gdf["density"])
    gdf["log_pop"] = np.log1p(gdf["pop_official"])
    gdf["log_area"] = np.log1p(gdf["area_official"])
    gdf["n_cities_log"] = np.log1p(gdf["n_cities"])

    features = ["log_density","log_pop","log_area", "centroid_lon","centroid_lat","n_cities_log"]
    X = gdf[features].fillna(0).values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    K_range = range(2, 8)
    sils = []
    for k in K_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        lbl = km.fit_predict(X_sc)
        sils.append(silhouette_score(X_sc, lbl))
    best_k = list(K_range)[np.argmax(sils)]

    km_final = KMeans(n_clusters=best_k, n_init=30, random_state=42)
    gdf["cluster"] = km_final.fit_predict(X_sc)

    agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    gdf["cluster_agg"] = agg.fit_predict(X_sc)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sc)
    gdf["pca1"] = X_pca[:, 0]
    gdf["pca2"] = X_pca[:, 1]

    coords_rad = np.radians(gdf[["centroid_lat","centroid_lon"]].values)
    knn_k = 4
    nbrs = NearestNeighbors(n_neighbors=knn_k+1, metric="haversine").fit(coords_rad)
    dists_knn, idx_knn = nbrs.kneighbors(coords_rad)
    
    gdf["knn_avg_dist_km"] = dists_knn[:, 1:].mean(axis=1) * 6371

    split = int(len(X_sc) * 0.75)
    knn_clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn_clf.fit(X_sc[:split], gdf["cluster"].values[:split])
    gdf["knn_pred"] = knn_clf.predict(X_sc)
    knn_acc = (gdf["knn_pred"] == gdf["cluster"]).mean() * 100

    cluster_names = []
    for cl in range(best_k):
        sub = gdf[gdf["cluster"] == cl]
        dens_m = sub["density"].mean()
        cname = ("Metropolitan" if dens_m > 150 else "Industrial" if dens_m > 60 else "Rural" if dens_m < 20 else "Suburban")
        cluster_names.append(cname)
    gdf["cluster_name"] = gdf["cluster"].map({i: cluster_names[i] for i in range(best_k)})

    ml_res = dict(
        best_k=best_k, K_range=list(K_range), sils=sils,
        X_sc=X_sc, X_pca=X_pca, pca=pca, km=km_final,
        idx_knn=idx_knn, dists_knn=dists_knn, knn_acc=knn_acc
    )
    return gdf, ml_res


def compute_us_kde(us_cities_gdf: gpd.GeoDataFrame, grid_res: int = 300):
    lon_min, lon_max = -125, -65
    lat_min, lat_max = 24, 50

    lon_g = np.linspace(lon_min, lon_max, grid_res)
    lat_g = np.linspace(lat_min, lat_max, grid_res)
    LON, LAT = np.meshgrid(lon_g, lat_g)
    KDE = np.zeros((grid_res, grid_res), dtype=float)

    mask = ((us_cities_gdf["lon"] > lon_min) &
            (us_cities_gdf["lon"] < lon_max) &
            (us_cities_gdf["lat"] > lat_min) &
            (us_cities_gdf["lat"] < lat_max))
    cities_cont = us_cities_gdf[mask]

    sigma = 1.5
    for _, row in cities_cont.iterrows():
        w = row["population"] / 1e5
        KDE += w * np.exp(-(((LON - row["lon"])**2 + (LAT - row["lat"])**2) / (2 * sigma**2)))

    KDE = gaussian_filter(KDE, sigma=3)
    return LON, LAT, KDE


def build_europe_analysis(eu_cities_gdf: gpd.GeoDataFrame, eu_countries: dict):
    records = []
    for iso, cdata in eu_countries.items():
        sub = eu_cities_gdf[eu_cities_gdf["countrycode"] == iso]
        if len(sub) < 2:
            continue
        area = cdata["areakm2"]
        pop = cdata["population"]
        records.append(dict(
            iso = iso,
            iso3 = cdata.get("iso3",""),
            name = cdata["name"],
            capital = cdata["capital"],
            population = pop,
            area_km2 = area,
            density = pop / max(area, 1),
            n_cities = len(sub),
            city_pop_sum = sub["population"].sum(),
            centroid_lon = sub["lon"].mean(),
            centroid_lat = sub["lat"].mean(),
        ))

    gdf = pd.DataFrame(records)
    gdf["log_density"] = np.log1p(gdf["density"])

    return gdf



def sax(ax, title="", xl="", yl=""):
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["sub"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(C["grid"])

    ax.grid(alpha=0.18, color=C["grid"], ls="--", lw=0.6)
    if title: ax.set_title(title, color=C["text"], pad=5)
    if xl:
        ax.set_xlabel(xl, color=C["sub"], fontsize=8)
    if yl:
        ax.set_ylabel(yl, color=C["sub"], fontsize=8)


def setup_map_ax(ax, extent, title=""):
    lon_min, lon_max, lat_min, lat_max = extent

    ax.set_facecolor(C["ocean"])
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    for sp in ax.spines.values():
        sp.set_edgecolor(C["border"])

    step_lon = 15 if (lon_max - lon_min) > 40 else 10
    step_lat = 5

    lon_ticks = np.arange(np.ceil(lon_min / step_lon) * step_lon, lon_max + 1, step_lon)
    lat_ticks = np.arange(np.ceil(lat_min / step_lat) * step_lat, lat_max + 1, step_lat)

    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)

    ax.grid(color=C["grid"], linestyle=":", linewidth=0.5, alpha=0.45)

    def lon_fmt(x, pos):
        return f"{abs(int(x))}°{'W' if x < 0 else 'E'}"

    def lat_fmt(y, pos):
        return f"{abs(int(y))}°{'S' if y < 0 else 'N'}"

    ax.xaxis.set_major_formatter(FuncFormatter(lon_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_fmt))

    ax.tick_params(axis="both", colors=C["sub"])

    if title:
        ax.set_title(title, color=C["text"], pad=6)


def draw_state_polygons(ax, gdf, column, cmap, norm, label_codes=True, alpha=0.88):
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        color = cmap(norm(row[column]))

        def draw_poly(poly):
            if isinstance(poly, Polygon) and not poly.is_empty:
                xs, ys = poly.exterior.xy
                ax.fill(xs, ys, color=color, alpha=alpha, zorder=2)
                ax.plot(xs, ys, color=C["border"], lw=0.5,
                        alpha=0.7, zorder=3)

        if geom.geom_type == "Polygon":
            draw_poly(geom)
        elif geom.geom_type == "MultiPolygon":
            for g in geom.geoms:
                draw_poly(g)

        if label_codes:
            cx, cy = row["centroid_lon"], row["centroid_lat"]
            ax.text(cx, cy, row["state_code"], ha="center", va="center", color="white", zorder=6, path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])


def plot_us_density(gdf: gpd.GeoDataFrame, us_cities_gdf: gpd.GeoDataFrame, fname):
    path = os.path.join(OUTPUT_DIR, fname)
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), facecolor=C["bg"])
    fig.suptitle("Choropleth: Population Density (people/km2)", fontsize=16, color=C["text"], y=0.97)

    setup_map_ax(ax, US_EXTENT, "")

    cmap_d = LinearSegmentedColormap.from_list("us_dens", ["#0D2A4E","#1565C0","#FFA000","#E53935","#7B1FA2"])
    norm_d = LogNorm(vmin=gdf["density"].clip(lower=0.5).min(), vmax=gdf["density"].max())
    draw_state_polygons(ax, gdf, "density", cmap_d, norm_d, label_codes=True, alpha=0.90)

    top_cities = us_cities_gdf.nlargest(30, "population")
    ax.scatter(top_cities["lon"], top_cities["lat"], s=top_cities["population"]/50000,
               c="white", edgecolors=C["gold"], linewidths=0.8, zorder=8, alpha=0.9)
    
    for _, r in us_cities_gdf.nlargest(8, "population").iterrows():
        if US_EXTENT[0] < r["lon"] < US_EXTENT[1] and US_EXTENT[2] < r["lat"] < US_EXTENT[3]:
            ax.annotate(r["name"], (r["lon"], r["lat"]), xytext=(3,3), textcoords="offset points",
                        color=C["gold"], zorder=9, path_effects=[pe.withStroke(linewidth=1.2, foreground=C["bg"])])

    sm = ScalarMappable(cmap=cmap_d, norm=norm_d)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Density (people/km2)", color=C["text"])
    cb.ax.yaxis.set_tick_params(color=C["sub"])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=C["sub"])

    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.07, top=0.93)
    plt.savefig(path)
    plt.close(fig)


def plot_us_clusters(gdf: gpd.GeoDataFrame, ml_res: dict, fname):    
    path = os.path.join(OUTPUT_DIR, fname)
    best_k = ml_res["best_k"]
    
    fig, ax_map = plt.subplots(1, 1, figsize=(20, 12), facecolor=C["bg"])
    fig.suptitle(f"K-Means (K={best_k}): geo-demographic clusters", fontsize=16, color=C["text"], y=0.97)

    setup_map_ax(ax_map, US_EXTENT, "")

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty: continue
        color = CLUST_COLORS[int(row["cluster"]) % len(CLUST_COLORS)]

        def draw_poly_k(poly):
            if isinstance(poly, Polygon) and not poly.is_empty:
                xs, ys = poly.exterior.xy
                ax_map.fill(xs, ys, color=color, alpha=0.85, zorder=2)
                ax_map.plot(xs, ys, color=C["border"], lw=0.5, alpha=0.7, zorder=3)

        if geom.geom_type == "Polygon":
            draw_poly_k(geom)
        elif geom.geom_type == "MultiPolygon":
            for g in geom.geoms:
                draw_poly_k(g)

        ax_map.text(row["centroid_lon"], row["centroid_lat"],
                    row["state_code"], ha="center", va="center", color="white", zorder=6,
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])

    handles = [mpatches.Patch(
        color=CLUST_COLORS[c % len(CLUST_COLORS)], alpha=0.85,
        label=f"Cluster{c}: {gdf[gdf['cluster']==c]['cluster_name'].iloc[0]} "
              f"(N={len(gdf[gdf['cluster']==c])})")
        for c in range(best_k)]
    ax_map.legend(handles=handles, loc="lower left", facecolor=C["panel"], edgecolor=C["border"], labelcolor=C["text"], framealpha=0.9)

    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.07, top=0.93)
    plt.savefig(path)
    plt.close(fig)


def plot_us_centroids_knn(gdf: gpd.GeoDataFrame, ml_res: dict, fname):
    path = os.path.join(OUTPUT_DIR, fname)
    fig, axes = plt.subplots(1, 2, figsize=(26, 10), facecolor=C["bg"])
    fig.suptitle("Centroid = population-weighted mean center of cities", fontsize=16, color=C["text"])

    ax_map = axes[0]
    setup_map_ax(ax_map, US_EXTENT, "State Centroids + KNN (k=4)")

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty: 
            continue
        
        def draw_bg(poly):
            if isinstance(poly, Polygon) and not poly.is_empty:
                xs, ys = poly.exterior.xy
                ax_map.fill(xs, ys, color=C["land"], alpha=0.50, zorder=1)
                ax_map.plot(xs, ys, color=C["border"], lw=0.5, alpha=0.7, zorder=2)
        
        if geom.geom_type == "Polygon":
            draw_bg(geom)
        elif geom.geom_type == "MultiPolygon":
            for g in geom.geoms:
                draw_bg(g)

    idx_knn = ml_res["idx_knn"]
    dists_knn = ml_res["dists_knn"]
    
    for i, (_, row) in enumerate(gdf.iterrows()):
        for j_idx in idx_knn[i][1:3]:
            if j_idx >= len(gdf): 
                continue
            nb = gdf.iloc[j_idx]
            d_km = dists_knn[i, idx_knn[i].tolist().index(j_idx)] * 6371
            lw = max(0.4, 1.5 - d_km/1500)
            ax_map.plot([row["centroid_lon"], nb["centroid_lon"]], [row["centroid_lat"], nb["centroid_lat"]], color=C["c5"], lw=lw, alpha=0.35, zorder=3)

    pop_sz = MinMaxScaler((20, 400)).fit_transform(gdf["pop_official"].values.reshape(-1, 1)).ravel()
    dens_n = Normalize(vmin=gdf["density"].min(), vmax=gdf["density"].max())
    cmap_sc = plt.cm.plasma

    sc = ax_map.scatter(gdf["centroid_lon"], gdf["centroid_lat"], s=pop_sz, c=gdf["density"], cmap=cmap_sc, norm=dens_n, alpha=0.90, edgecolors="white", linewidths=0.8, zorder=7)
    
    for _, r in gdf.nlargest(10, "pop_official").iterrows():
        if US_EXTENT[0] < r["centroid_lon"] < US_EXTENT[1]:
            ax_map.annotate(r["state_code"], (r["centroid_lon"], r["centroid_lat"]), xytext=(3,3), textcoords="offset points",
                            color=C["text"], path_effects=[pe.withStroke(linewidth=1.2, foreground=C["bg"])], zorder=9)
    
    cb = fig.colorbar(sc, ax=ax_map, fraction=0.025, pad=0.01)
    cb.set_label("Density (people/km2)", color=C["text"])
    cb.ax.yaxis.set_tick_params(color=C["sub"])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=C["sub"])
    
    ax_bar = axes[1]
    ax_bar.set_facecolor(C["panel"])
    
    knn_df = gdf[["state_code", "knn_avg_dist_km", "cluster"]].sort_values("knn_avg_dist_km", ascending=False).head(30)
    clrs_b = [CLUST_COLORS[int(c) % len(CLUST_COLORS)] for c in knn_df["cluster"]]
    
    ax_bar.barh(knn_df["state_code"], knn_df["knn_avg_dist_km"], color=clrs_b, alpha=0.85, edgecolor=C["grid"])
    ax_bar.axvline(gdf["knn_avg_dist_km"].mean(), color=C["gold"], lw=1.8, ls="--", label=f"Average: {gdf['knn_avg_dist_km'].mean():.0f} km")
    
    sax(ax_bar, "KNN (k=4): Average distance to neighbors", "km (haversine)", "State")
    ax_bar.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])

    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.07, top=0.93)
    plt.savefig(path)
    plt.close(fig)


def plot_us_kde(gdf: gpd.GeoDataFrame, us_cities_gdf: gpd.GeoDataFrame, fname):
    path = os.path.join(OUTPUT_DIR, fname)
    LON, LAT, KDE = compute_us_kde(us_cities_gdf, grid_res=280)

    fig, ax_map = plt.subplots(1, 1, figsize=(20, 12), facecolor=C["bg"])
    fig.suptitle("KDE surface (population-weighted cities)", fontsize=16, color=C["text"], y=0.97)

    setup_map_ax(ax_map, US_EXTENT, "")

    cmap_kde = LinearSegmentedColormap.from_list("kde_us", ["#050A14","#0D2A4E","#1565C0","#FFA000","#E53935","#7B1FA2"])
    kde_masked = np.ma.masked_where(KDE < KDE.max()*0.005, KDE)
    im = ax_map.pcolormesh(LON, LAT, kde_masked, cmap=cmap_kde, alpha=0.88, shading="auto", zorder=3)

    levels = np.percentile(KDE[KDE > 0], [60, 80, 92, 98])
    ax_map.contour(LON, LAT, KDE, levels=levels, colors=[C["text"]], linewidths=0.7, alpha=0.45, zorder=5)

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        
        def draw_outline(poly):
            if isinstance(poly, Polygon) and not poly.is_empty:
                xs, ys = poly.exterior.xy
                ax_map.plot(xs, ys, color=C["border"], lw=0.6, alpha=0.5, zorder=6)
        
        if geom.geom_type == "Polygon":
            draw_outline(geom)
        elif geom.geom_type == "MultiPolygon":
            for g in geom.geoms:
                draw_outline(g)

    for _, r in us_cities_gdf.nlargest(8, "population").iterrows():
        if US_EXTENT[0] < r["lon"] < US_EXTENT[1] and US_EXTENT[2] < r["lat"] < US_EXTENT[3]:
            ax_map.scatter(r["lon"], r["lat"], s=60, c=C["gold"], edgecolors="white", linewidths=0.8, zorder=9)
            ax_map.annotate(r["name"], (r["lon"], r["lat"]), xytext=(4,4), textcoords="offset points", color=C["gold"],
                            zorder=10, path_effects=[pe.withStroke(linewidth=1.5, foreground=C["bg"])])

    cb = fig.colorbar(im, ax=ax_map, fraction=0.025, pad=0.01)
    cb.set_label("KDE density (weighted)", color=C["text"])
    cb.ax.yaxis.set_tick_params(color=C["sub"])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=C["sub"])

    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.07, top=0.93)
    plt.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()