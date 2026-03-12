from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import requests


ROOT = Path("/Users/cheng80/Desktop/ddri_work")
RAW_DIR = ROOT / "3조 공유폴더"
OUTPUT_DIR = ROOT / "works" / "01_clustering" / "08_integrated" / "intermediate" / "environment_enrichment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NATUREPARK_SHP = ROOT / "3조 공유폴더" / "UQ142_용도구역(도시자연공원구역)_20250805" / "shp파일" / "UPIS_C_UQ142.shp"
RIVER_SHP = ROOT / "3조 공유폴더" / "국토정보지리원 연속수치지형도 하천경계 데이터" / "N3A_E0010001" / "N3A_E0010001.shp"


def haversine_matrix(lat1, lon1, lat2, lon2):
    r = 6371000.0
    lat1 = np.radians(lat1)[:, None]
    lon1 = np.radians(lon1)[:, None]
    lat2 = np.radians(lat2)[None, :]
    lon2 = np.radians(lon2)[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def parse_area_to_float(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).replace("㎡", "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return np.nan


def load_sources():
    stations = pd.read_csv(
        ROOT / "works" / "01_clustering" / "08_integrated" / "source_data" / "ddri_common_station_master.csv"
    )
    stations = stations.rename(columns={"대여소번호": "station_id", "위도": "station_lat", "경도": "station_lon"})
    stations["station_id"] = pd.to_numeric(stations["station_id"], errors="coerce").astype("Int64")
    stations["station_lat"] = pd.to_numeric(stations["station_lat"], errors="coerce")
    stations["station_lon"] = pd.to_numeric(stations["station_lon"], errors="coerce")
    stations = stations.dropna(subset=["station_id", "station_lat", "station_lon"]).copy()
    stations["station_id"] = stations["station_id"].astype(int)

    subway = pd.read_csv(
        RAW_DIR / "[교통데이터] 지하철 정보/서울시 역사마스터 정보/서울시 역사마스터 정보.csv",
        encoding="cp949",
    )
    subway = subway.rename(columns={"역사명": "subway_name", "위도": "subway_lat", "경도": "subway_lon"})
    subway["subway_lat"] = pd.to_numeric(subway["subway_lat"], errors="coerce")
    subway["subway_lon"] = pd.to_numeric(subway["subway_lon"], errors="coerce")
    subway = subway.dropna(subset=["subway_lat", "subway_lon"]).copy()

    bus = pd.read_csv(
        RAW_DIR / "서울시 버스정류소 위치정보/2024년/2024년1~4월1일기준_서울시버스정류소위치정보.csv",
        encoding="cp949",
    )
    bus = bus.rename(columns={"STTN_NM": "bus_stop_name", "CRDNT_Y": "bus_lat", "CRDNT_X": "bus_lon"})
    bus["bus_lat"] = pd.to_numeric(bus["bus_lat"], errors="coerce")
    bus["bus_lon"] = pd.to_numeric(bus["bus_lon"], errors="coerce")
    bus = bus.dropna(subset=["bus_lat", "bus_lon"]).copy()

    park = pd.read_csv(RAW_DIR / "서울시 강남구 공원 정보.csv")
    park = park.rename(columns={"공원명": "park_name", "위도": "park_lat", "경도": "park_lon"})
    park["park_lat"] = pd.to_numeric(park["park_lat"], errors="coerce")
    park["park_lon"] = pd.to_numeric(park["park_lon"], errors="coerce")
    park["park_area_sqm"] = park["면적"].map(parse_area_to_float)
    park = park.dropna(subset=["park_lat", "park_lon"]).copy()

    return stations, subway, bus, park


def add_nearest_context(stations, subway, bus, park):
    station_lat = stations["station_lat"].to_numpy()
    station_lon = stations["station_lon"].to_numpy()

    subway_dist = haversine_matrix(station_lat, station_lon, subway["subway_lat"].to_numpy(), subway["subway_lon"].to_numpy())
    bus_dist = haversine_matrix(station_lat, station_lon, bus["bus_lat"].to_numpy(), bus["bus_lon"].to_numpy())
    park_dist = haversine_matrix(station_lat, station_lon, park["park_lat"].to_numpy(), park["park_lon"].to_numpy())

    stations = stations.copy()
    subway_idx = subway_dist.argmin(axis=1)
    bus_idx = bus_dist.argmin(axis=1)
    park_idx = park_dist.argmin(axis=1)

    stations["nearest_subway_name"] = subway.iloc[subway_idx]["subway_name"].to_numpy()
    stations["nearest_subway_distance_m"] = subway_dist.min(axis=1)
    stations["nearest_subway_lat"] = subway.iloc[subway_idx]["subway_lat"].to_numpy()
    stations["nearest_subway_lon"] = subway.iloc[subway_idx]["subway_lon"].to_numpy()

    stations["nearest_bus_stop_name"] = bus.iloc[bus_idx]["bus_stop_name"].to_numpy()
    stations["nearest_bus_stop_distance_m"] = bus_dist.min(axis=1)
    stations["nearest_bus_stop_lat"] = bus.iloc[bus_idx]["bus_lat"].to_numpy()
    stations["nearest_bus_stop_lon"] = bus.iloc[bus_idx]["bus_lon"].to_numpy()

    stations["nearest_park_name"] = park.iloc[park_idx]["park_name"].to_numpy()
    stations["nearest_park_distance_m"] = park_dist.min(axis=1)
    stations["nearest_park_area_sqm"] = park.iloc[park_idx]["park_area_sqm"].to_numpy()
    stations["nearest_park_lat"] = park.iloc[park_idx]["park_lat"].to_numpy()
    stations["nearest_park_lon"] = park.iloc[park_idx]["park_lon"].to_numpy()
    return stations


def batched(seq: Iterable, size: int):
    seq = list(seq)
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def fetch_elevations(coords: pd.DataFrame, lat_col: str, lon_col: str, prefix: str) -> pd.DataFrame:
    work = coords[[lat_col, lon_col]].drop_duplicates().copy()
    work["coord_key"] = work.apply(lambda row: f"{row[lat_col]:.6f},{row[lon_col]:.6f}", axis=1)
    results = []

    for chunk in batched(work.to_dict("records"), 100):
        latitudes = ",".join(f"{row[lat_col]:.6f}" for row in chunk)
        longitudes = ",".join(f"{row[lon_col]:.6f}" for row in chunk)
        resp = requests.get(
            "https://api.open-meteo.com/v1/elevation",
            params={"latitude": latitudes, "longitude": longitudes},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        elevations = data.get("elevation", [])
        for row, elevation in zip(chunk, elevations):
            results.append(
                {
                    "coord_key": f"{row[lat_col]:.6f},{row[lon_col]:.6f}",
                    f"{prefix}_elevation_m": elevation,
                }
            )

    result_df = pd.DataFrame(results).drop_duplicates("coord_key")
    return result_df


def attach_elevations(stations: pd.DataFrame) -> pd.DataFrame:
    out = stations.copy()

    station_elev = fetch_elevations(out, "station_lat", "station_lon", "station")
    subway_elev = fetch_elevations(out, "nearest_subway_lat", "nearest_subway_lon", "subway")
    bus_elev = fetch_elevations(out, "nearest_bus_stop_lat", "nearest_bus_stop_lon", "bus")
    park_elev = fetch_elevations(out, "nearest_park_lat", "nearest_park_lon", "park")

    for lat_col, lon_col, elev_df, prefix in [
        ("station_lat", "station_lon", station_elev, "station"),
        ("nearest_subway_lat", "nearest_subway_lon", subway_elev, "subway"),
        ("nearest_bus_stop_lat", "nearest_bus_stop_lon", bus_elev, "bus"),
        ("nearest_park_lat", "nearest_park_lon", park_elev, "park"),
    ]:
        out["coord_key"] = out.apply(lambda row: f"{row[lat_col]:.6f},{row[lon_col]:.6f}", axis=1)
        out = out.merge(elev_df, on="coord_key", how="left")
        out = out.drop(columns=["coord_key"])

    out["elevation_diff_nearest_subway_m"] = out["station_elevation_m"] - out["subway_elevation_m"]
    out["elevation_diff_nearest_bus_stop_m"] = out["station_elevation_m"] - out["bus_elevation_m"]
    out["elevation_diff_nearest_park_m"] = out["station_elevation_m"] - out["park_elevation_m"]
    return out


def attach_naturepark_features(stations: pd.DataFrame) -> pd.DataFrame:
    station_gdf = gpd.GeoDataFrame(
        stations.copy(),
        geometry=gpd.points_from_xy(stations["station_lon"], stations["station_lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:5174")

    naturepark = gpd.read_file(NATUREPARK_SHP).to_crs("EPSG:5174")
    nature_union = naturepark.union_all()

    station_gdf["distance_naturepark_m"] = station_gdf.geometry.distance(nature_union)
    station_gdf["inside_naturepark"] = station_gdf.geometry.within(nature_union).astype(int)
    station_gdf["distance_naturepark_m"] = station_gdf["distance_naturepark_m"].round(3)

    return pd.DataFrame(station_gdf.drop(columns=["geometry"]))


def attach_river_features(stations: pd.DataFrame) -> pd.DataFrame:
    station_gdf = gpd.GeoDataFrame(
        stations.copy(),
        geometry=gpd.points_from_xy(stations["station_lon"], stations["station_lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:5179")

    river = gpd.read_file(RIVER_SHP).to_crs("EPSG:5179")
    minx, miny, maxx, maxy = station_gdf.total_bounds
    buffer_m = 3000
    river = river.cx[minx - buffer_m : maxx + buffer_m, miny - buffer_m : maxy + buffer_m].copy()
    river_union = river.union_all()

    station_gdf["distance_river_boundary_m"] = station_gdf.geometry.distance(river_union).round(3)
    return pd.DataFrame(station_gdf.drop(columns=["geometry"]))


def build_enrichment_features():
    stations, subway, bus, park = load_sources()
    features = add_nearest_context(stations, subway, bus, park)
    features = attach_elevations(features)
    features = attach_naturepark_features(features)
    features = attach_river_features(features)

    keep_cols = [
        "station_id",
        "대여소명",
        "주소",
        "station_lat",
        "station_lon",
        "station_elevation_m",
        "nearest_subway_name",
        "nearest_subway_distance_m",
        "subway_elevation_m",
        "elevation_diff_nearest_subway_m",
        "nearest_bus_stop_name",
        "nearest_bus_stop_distance_m",
        "bus_elevation_m",
        "elevation_diff_nearest_bus_stop_m",
        "nearest_park_name",
        "nearest_park_distance_m",
        "nearest_park_area_sqm",
        "park_elevation_m",
        "elevation_diff_nearest_park_m",
        "distance_naturepark_m",
        "inside_naturepark",
        "distance_river_boundary_m",
    ]
    features = features[keep_cols].sort_values("station_id").reset_index(drop=True)
    features.to_csv(OUTPUT_DIR / "ddri_environment_enrichment_features.csv", index=False)

    summary = features[
        [
            "station_elevation_m",
            "nearest_subway_distance_m",
            "elevation_diff_nearest_subway_m",
            "nearest_bus_stop_distance_m",
            "elevation_diff_nearest_bus_stop_m",
            "nearest_park_distance_m",
            "nearest_park_area_sqm",
            "elevation_diff_nearest_park_m",
            "distance_naturepark_m",
            "distance_river_boundary_m",
        ]
    ].describe().round(2)
    summary.to_csv(OUTPUT_DIR / "ddri_environment_enrichment_summary.csv")


if __name__ == "__main__":
    build_enrichment_features()
