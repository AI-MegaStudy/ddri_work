from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "data" / "ddri_return_station_cluster_features.csv"
MAP_PATH = BASE_DIR / "outputs" / "maps" / "ddri_return_station_cluster_300m_map.html"
SHARED_DIR = BASE_DIR.parent / "3조 공유폴더"
SUBWAY_PATH = SHARED_DIR / "[교통데이터] 지하철 정보" / "서울시 역사마스터 정보" / "서울시 역사마스터 정보.csv"
BUS_PATH = SHARED_DIR / "서울시 버스정류소 위치정보" / "2023년" / "2023년각월1일기준_서울시버스정류소위치정보.csv"
PARK_PATH = SHARED_DIR / "서울시 강남구 공원 정보.csv"


COLOR_MAP = {
    "업무·학교 도착형": "#1f77b4",
    "생활도착형": "#ff7f0e",
    "교통거점형": "#d62728",
}


def read_csv_with_fallback(path: Path, encodings: list[str] | None = None, **kwargs) -> pd.DataFrame:
    fallback_encodings = encodings or ["utf-8", "utf-8-sig", "cp949", "euc-kr"]
    last_error: Exception | None = None
    for encoding in fallback_encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path, **kwargs)


def haversine_distance(lat1, lon1, lat2, lon2) -> np.ndarray:
    earth_radius_m = 6_371_000
    lat1_rad = np.radians(np.asarray(lat1))[:, None]
    lon1_rad = np.radians(np.asarray(lon1))[:, None]
    lat2_rad = np.radians(np.asarray(lat2))[None, :]
    lon2_rad = np.radians(np.asarray(lon2))[None, :]
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return earth_radius_m * c


def load_subway() -> pd.DataFrame:
    subway = read_csv_with_fallback(SUBWAY_PATH)
    subway["lat"] = pd.to_numeric(subway["위도"], errors="coerce")
    subway["lon"] = pd.to_numeric(subway["경도"], errors="coerce")
    subway = subway.dropna(subset=["lat", "lon"]).copy()
    subway["name"] = subway["역사명"].astype(str)
    subway["line"] = subway["호선"].astype(str)
    return subway[["name", "line", "lat", "lon"]]


def load_bus() -> pd.DataFrame:
    bus = read_csv_with_fallback(BUS_PATH)
    bus["lat"] = pd.to_numeric(bus["CRDNT_Y"], errors="coerce")
    bus["lon"] = pd.to_numeric(bus["CRDNT_X"], errors="coerce")
    bus = bus.dropna(subset=["lat", "lon"]).copy()
    bus["name"] = bus["STTN_NM"].astype(str)
    return bus[["name", "lat", "lon"]]


def load_park() -> pd.DataFrame:
    park = read_csv_with_fallback(PARK_PATH)
    park["lat"] = pd.to_numeric(park["위도"], errors="coerce")
    park["lon"] = pd.to_numeric(park["경도"], errors="coerce")
    park = park.dropna(subset=["lat", "lon"]).copy()
    park["name"] = park["공원명"].astype(str)
    return park[["name", "lat", "lon"]]


def filter_to_gangnam_extent(
    infra_df: pd.DataFrame,
    station_df: pd.DataFrame,
    lat_margin: float = 0.01,
    lon_margin: float = 0.01,
) -> pd.DataFrame:
    min_lat = station_df["lat"].min() - lat_margin
    max_lat = station_df["lat"].max() + lat_margin
    min_lon = station_df["lon"].min() - lon_margin
    max_lon = station_df["lon"].max() + lon_margin
    return infra_df[
        infra_df["lat"].between(min_lat, max_lat) & infra_df["lon"].between(min_lon, max_lon)
    ].copy()


def nearby_names(row: pd.Series, infra_df: pd.DataFrame, radius_m: int, label: str) -> str:
    if infra_df.empty:
        return f"{label}: 없음"
    distances = haversine_distance([row["lat"]], [row["lon"]], infra_df["lat"].to_numpy(), infra_df["lon"].to_numpy())[0]
    nearby = infra_df.loc[distances <= radius_m].copy()
    if nearby.empty:
        return f"{label}: 없음"
    if label == "지하철":
        names = (nearby["name"] + " (" + nearby["line"] + ")").head(5).tolist()
    else:
        names = nearby["name"].head(5).tolist()
    suffix = "" if len(nearby) <= 5 else f" 외 {len(nearby) - 5}개"
    return f"{label}: " + ", ".join(names) + suffix


def build_popup(row: pd.Series, subway_df: pd.DataFrame, bus_df: pd.DataFrame, park_df: pd.DataFrame) -> str:
    subway_text = nearby_names(row, subway_df, 300, "지하철")
    bus_text = nearby_names(row, bus_df, 300, "버스")
    park_text = nearby_names(row, park_df, 300, "공원")
    return f"""
    <div style="width: 320px;">
      <b>{row['station_name']}</b><br>
      station_id: {row['station_id']}<br>
      profile: {row['cluster_profile']}<br>
      morning_ratio: {float(row['weekday_morning_ratio']):.3f}<br>
      evening_ratio: {float(row['weekday_evening_ratio']):.3f}<br>
      weekend_day_ratio: {float(row['weekend_day_ratio']):.3f}<br>
      subway_m: {float(row['nearest_subway_m']):.1f}<br>
      bus_300m: {int(float(row['bus_stop_count_300m']))}<br>
      park_m: {float(row['nearest_park_m']):.1f}<br>
      note: {row['station_note']}<br><br>
      {subway_text}<br>
      {bus_text}<br>
      {park_text}
    </div>
    """


def add_infra_layers(fmap: folium.Map, subway_df: pd.DataFrame, bus_df: pd.DataFrame, park_df: pd.DataFrame) -> None:
    subway_group = folium.FeatureGroup(name="지하철 마커", show=False)
    for _, row in subway_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color="#6a3d9a",
            fill=True,
            fill_color="#6a3d9a",
            fill_opacity=0.8,
            tooltip=f"{row['name']} ({row['line']})",
        ).add_to(subway_group)
    subway_group.add_to(fmap)

    bus_group = folium.FeatureGroup(name="버스정류소 마커", show=False)
    for _, row in bus_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2,
            color="#1b9e77",
            fill=True,
            fill_color="#1b9e77",
            fill_opacity=0.7,
            tooltip=row["name"],
        ).add_to(bus_group)
    bus_group.add_to(fmap)

    park_group = folium.FeatureGroup(name="공원 마커", show=False)
    for _, row in park_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color="#33a02c",
            fill=True,
            fill_color="#33a02c",
            fill_opacity=0.8,
            tooltip=row["name"],
        ).add_to(park_group)
    park_group.add_to(fmap)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    subway_df = filter_to_gangnam_extent(load_subway(), df)
    bus_df = filter_to_gangnam_extent(load_bus(), df)
    park_df = filter_to_gangnam_extent(load_park(), df)

    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)

    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron")
    add_infra_layers(fmap, subway_df, bus_df, park_df)

    for profile, group in df.groupby("cluster_profile"):
        feature_group = folium.FeatureGroup(name=profile, show=True)
        color = COLOR_MAP.get(profile, "#7f7f7f")

        for _, row in group.iterrows():
            popup_html = build_popup(row, subway_df, bus_df, park_df)
            folium.Circle(
                location=[row["lat"], row["lon"]],
                radius=300,
                color=color,
                weight=1.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.12,
                popup=folium.Popup(popup_html, max_width=360),
                tooltip=f"{row['station_name']} | {row['cluster_profile']}",
            ).add_to(feature_group)

            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.95,
                popup=folium.Popup(popup_html, max_width=360),
            ).add_to(feature_group)

        feature_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(MAP_PATH)
    print(MAP_PATH)


if __name__ == "__main__":
    main()
