from pathlib import Path

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import koreanize_matplotlib


BASE_DIR = Path("/Users/cheng80/Desktop/ddri_work")
RAW_DIR = BASE_DIR / "3조 공유폴더"
DATA_DIR = BASE_DIR / "works" / "clustering" / "data"
IMG_DIR = BASE_DIR / "works" / "clustering" / "images"

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")


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


def load_sources():
    stations = pd.read_csv(DATA_DIR / "ddri_common_station_master.csv")
    clusters = pd.read_csv(DATA_DIR / "ddri_station_cluster_features_train_with_labels.csv")
    park = pd.read_csv(RAW_DIR / "서울시 강남구 공원 정보.csv")
    subway = pd.read_csv(
        RAW_DIR / "[교통데이터] 지하철 정보/서울시 역사마스터 정보/서울시 역사마스터 정보.csv",
        encoding="cp949",
    )
    bus = pd.read_csv(
        RAW_DIR / "서울시 버스정류소 위치정보/2024년/2024년1~4월1일기준_서울시버스정류소위치정보.csv",
        encoding="cp949",
    )

    stations = stations.rename(columns={"대여소번호": "station_id", "위도": "station_lat", "경도": "station_lon"})
    park = park.rename(columns={"공원명": "park_name", "위도": "park_lat", "경도": "park_lon"})
    subway = subway.rename(columns={"역사명": "subway_name", "위도": "subway_lat", "경도": "subway_lon"})
    bus = bus.rename(columns={"STTN_NM": "bus_stop_name", "CRDNT_Y": "bus_lat", "CRDNT_X": "bus_lon"})

    for frame, cols in [
        (stations, ["station_lat", "station_lon"]),
        (park, ["park_lat", "park_lon"]),
        (subway, ["subway_lat", "subway_lon"]),
        (bus, ["bus_lat", "bus_lon"]),
    ]:
        for col in cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    stations = stations.dropna(subset=["station_lat", "station_lon"])
    park = park.dropna(subset=["park_lat", "park_lon"])
    subway = subway.dropna(subset=["subway_lat", "subway_lon"])
    bus = bus.dropna(subset=["bus_lat", "bus_lon"])

    merged = clusters.merge(stations, on="station_id", how="left")
    return merged, park, subway, bus


def build_environment_features():
    stations, park, subway, bus = load_sources()

    station_lat = stations["station_lat"].to_numpy()
    station_lon = stations["station_lon"].to_numpy()

    park_dist = haversine_matrix(
        station_lat, station_lon, park["park_lat"].to_numpy(), park["park_lon"].to_numpy()
    )
    subway_dist = haversine_matrix(
        station_lat, station_lon, subway["subway_lat"].to_numpy(), subway["subway_lon"].to_numpy()
    )
    bus_dist = haversine_matrix(
        station_lat, station_lon, bus["bus_lat"].to_numpy(), bus["bus_lon"].to_numpy()
    )

    stations["park_distance_m"] = park_dist.min(axis=1)
    stations["nearest_park_name"] = park.iloc[park_dist.argmin(axis=1)]["park_name"].to_numpy()
    stations["subway_distance_m"] = subway_dist.min(axis=1)
    stations["nearest_subway_name"] = subway.iloc[subway_dist.argmin(axis=1)]["subway_name"].to_numpy()
    stations["bus_stop_count_300m"] = (bus_dist <= 300).sum(axis=1)
    stations["bus_stop_count_500m"] = (bus_dist <= 500).sum(axis=1)

    stations["cluster_name"] = stations["cluster"].map({0: "일반수요형", 1: "고수요형"})
    stations.to_csv(DATA_DIR / "ddri_cluster_environment_features.csv", index=False)

    summary = (
        stations.groupby(["cluster", "cluster_name"])[
            ["park_distance_m", "subway_distance_m", "bus_stop_count_300m", "bus_stop_count_500m", "avg_rental"]
        ]
        .mean()
        .round(2)
        .reset_index()
    )
    summary.to_csv(DATA_DIR / "ddri_cluster_environment_summary.csv", index=False)

    plot_df = summary.melt(
        id_vars=["cluster", "cluster_name"],
        value_vars=["park_distance_m", "subway_distance_m", "bus_stop_count_300m", "avg_rental"],
        var_name="feature",
        value_name="value",
    )
    feature_labels = {
        "park_distance_m": "공원 거리(m)",
        "subway_distance_m": "지하철 거리(m)",
        "bus_stop_count_300m": "300m 버스정류장 수",
        "avg_rental": "평균 대여량",
    }
    plot_df["feature_label"] = plot_df["feature"].map(feature_labels)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_df, x="feature_label", y="value", hue="cluster_name", ax=ax)
    ax.set_title("군집별 환경/수요 비교")
    ax.set_xlabel("지표")
    ax.set_ylabel("평균값")
    ax.legend(title="군집")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(IMG_DIR / "ddri_cluster_environment_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    reps = (
        stations.sort_values(["cluster", "avg_rental"], ascending=[True, False])
        .groupby("cluster")
        .head(10)
        .copy()
    )
    reps.to_csv(DATA_DIR / "ddri_cluster_representative_stations.csv", index=False)

    print("saved:", DATA_DIR / "ddri_cluster_environment_features.csv")
    print("saved:", DATA_DIR / "ddri_cluster_environment_summary.csv")
    print("saved:", DATA_DIR / "ddri_cluster_representative_stations.csv")
    print("saved:", IMG_DIR / "ddri_cluster_environment_comparison.png")


if __name__ == "__main__":
    build_environment_features()
