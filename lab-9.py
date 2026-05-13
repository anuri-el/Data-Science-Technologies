import os
import numpy as np
import pandas as pd
import alphashape
import geopandas as gpd
import geonamescache
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from shapely.geometry import Point, MultiPoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.decomposition import PCA

gc = geonamescache.GeonamesCache()


SEP = "=" * 67

OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(OUTPUT_DIR, "sample_data.xlsx")
DESC_PATH = os.path.join(OUTPUT_DIR, "Datdata_descriptiona_Set_7.xlsx")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "l8_scoring_results.csv")



def main() -> None:
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


def build_state_polygons(us_cities_gdf: gpd.GeoDataFrame, us_states_meta: dict):
    records = []
    skipped = []

    for state_code, state_name in us_states_meta.items():
        sub = us_cities_gdf[us_cities_gdf["state"] == state_code]

        if len(sub) < 3:
            skipped.append(state_code)
            continue

        pts = list(zip(sub["lon"], sub["lat"]))
        alpha = 2.0

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
            total_population = total_pop,
            geometry = polygon,
        ))

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    gdf_m = gdf.copy().to_crs("EPSG:3857")
    gdf["area_computed_km2"] = gdf_m.geometry.area / 1e6

    return gdf


if __name__ == "__main__":
    main()