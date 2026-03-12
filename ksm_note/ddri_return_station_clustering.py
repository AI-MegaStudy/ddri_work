from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager, rcParams
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
SHARED_DIR = ROOT_DIR / "3조 공유폴더"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
IMAGE_DIR = OUTPUT_DIR / "images"
DATA_DIR = OUTPUT_DIR / "data"

RIDE_PATH = SHARED_DIR / "2023 강남구 따릉이 이용정보" / "2023_강남구_따릉이_이용정보_통합.csv"
STATION_PATH = SHARED_DIR / "강남구 대여소 정보 (2023~2025)" / "2023_강남구_대여소.csv"
SUBWAY_PATH = SHARED_DIR / "[교통데이터] 지하철 정보" / "서울시 역사마스터 정보" / "서울시 역사마스터 정보.csv"
BUS_PATH = SHARED_DIR / "서울시 버스정류소 위치정보" / "2023년" / "2023년각월1일기준_서울시버스정류소위치정보.csv"
PARK_PATH = SHARED_DIR / "서울시 강남구 공원 정보.csv"

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL",
    category=UserWarning,
)


def configure_korean_font() -> None:
    font_candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic"]
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in font_candidates:
        if font_name in available_fonts:
            rcParams["font.family"] = font_name
            break
    rcParams["axes.unicode_minus"] = False


def ensure_output_dirs() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


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


def normalize_station_id(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype("Int64").astype(str).replace("<NA>", np.nan)


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


def load_station_master() -> pd.DataFrame:
    station = read_csv_with_fallback(STATION_PATH)
    station["station_id"] = normalize_station_id(station["대여소번호"])
    station["station_name"] = station["대여소명"].astype(str)
    station["lat"] = pd.to_numeric(station["위도"], errors="coerce")
    station["lon"] = pd.to_numeric(station["경도"], errors="coerce")
    station = station.dropna(subset=["station_id", "lat", "lon"]).drop_duplicates("station_id")
    return station[["station_id", "station_name", "lat", "lon"]]


def load_rides() -> pd.DataFrame:
    rides = read_csv_with_fallback(RIDE_PATH)
    rides["station_id"] = normalize_station_id(rides["반납대여소번호"])
    rides["return_at"] = pd.to_datetime(rides["반납일시"], errors="coerce")
    rides["duration_min"] = pd.to_numeric(rides["이용시간(분)"], errors="coerce")
    rides["distance_m"] = pd.to_numeric(rides["이용거리(M)"], errors="coerce")
    rides = rides.dropna(subset=["station_id", "return_at"])
    rides = rides[(rides["duration_min"] > 0) & (rides["distance_m"] > 0)]

    rides["date"] = rides["return_at"].dt.date
    rides["hour"] = rides["return_at"].dt.hour
    rides["is_weekend"] = rides["return_at"].dt.dayofweek >= 5
    rides["is_weekday"] = ~rides["is_weekend"]
    rides["is_weekday_morning"] = rides["is_weekday"] & rides["hour"].between(7, 10)
    rides["is_weekday_evening"] = rides["is_weekday"] & rides["hour"].between(17, 20)
    rides["is_weekend_day"] = rides["is_weekend"] & rides["hour"].between(11, 18)
    rides["is_night"] = (rides["hour"] >= 21) | (rides["hour"] <= 5)
    return rides


def build_return_pattern_features(rides: pd.DataFrame) -> pd.DataFrame:
    grouped = rides.groupby("station_id")
    daily_counts = rides.groupby(["station_id", "date"]).size().rename("daily_returns").reset_index()
    daily_counts["date"] = pd.to_datetime(daily_counts["date"])
    daily_counts["is_weekend"] = daily_counts["date"].dt.dayofweek >= 5

    weekday_daily_avg = (
        daily_counts.loc[~daily_counts["is_weekend"]]
        .groupby("station_id")["daily_returns"]
        .mean()
        .rename("weekday_avg_returns")
    )
    weekend_daily_avg = (
        daily_counts.loc[daily_counts["is_weekend"]]
        .groupby("station_id")["daily_returns"]
        .mean()
        .rename("weekend_avg_returns")
    )
    overall_daily = (
        daily_counts.groupby("station_id")["daily_returns"]
        .agg(avg_daily_returns="mean", return_std="std", max_daily_returns="max")
        .fillna(0)
    )

    pattern = grouped.agg(
        total_returns=("station_id", "size"),
        avg_duration_min=("duration_min", "mean"),
        avg_distance_m=("distance_m", "mean"),
    )
    pattern = pattern.join(weekday_daily_avg, how="left").join(weekend_daily_avg, how="left").join(overall_daily, how="left")
    pattern["weekday_return_ratio"] = grouped["is_weekday"].mean()
    pattern["weekday_morning_ratio"] = grouped["is_weekday_morning"].mean()
    pattern["weekday_evening_ratio"] = grouped["is_weekday_evening"].mean()
    pattern["weekend_day_ratio"] = grouped["is_weekend_day"].mean()
    pattern["night_ratio"] = grouped["is_night"].mean()
    pattern["weekday_weekend_gap"] = pattern["weekday_avg_returns"] - pattern["weekend_avg_returns"]
    pattern["morning_evening_gap"] = pattern["weekday_evening_ratio"] - pattern["weekday_morning_ratio"]
    return pattern.reset_index().fillna(0)


def load_subway() -> pd.DataFrame:
    subway = read_csv_with_fallback(SUBWAY_PATH)
    subway["lat"] = pd.to_numeric(subway["위도"], errors="coerce")
    subway["lon"] = pd.to_numeric(subway["경도"], errors="coerce")
    subway = subway.dropna(subset=["lat", "lon"])
    return subway[["역사명", "호선", "lat", "lon"]]


def load_bus() -> pd.DataFrame:
    bus = read_csv_with_fallback(BUS_PATH)
    bus["lat"] = pd.to_numeric(bus["CRDNT_Y"], errors="coerce")
    bus["lon"] = pd.to_numeric(bus["CRDNT_X"], errors="coerce")
    bus = bus.dropna(subset=["lat", "lon"])
    return bus[["STTN_NM", "lat", "lon"]]


def load_park() -> pd.DataFrame:
    park = read_csv_with_fallback(PARK_PATH)
    park["lat"] = pd.to_numeric(park["위도"], errors="coerce")
    park["lon"] = pd.to_numeric(park["경도"], errors="coerce")
    park = park.dropna(subset=["lat", "lon"])
    return park[["공원명", "lat", "lon"]]


def build_environment_features(station: pd.DataFrame) -> pd.DataFrame:
    subway = load_subway()
    bus = load_bus()
    park = load_park()

    station_coords = station[["lat", "lon"]].to_numpy()
    subway_distance = haversine_distance(
        station_coords[:, 0], station_coords[:, 1], subway["lat"].to_numpy(), subway["lon"].to_numpy()
    )
    bus_distance = haversine_distance(
        station_coords[:, 0], station_coords[:, 1], bus["lat"].to_numpy(), bus["lon"].to_numpy()
    )
    park_distance = haversine_distance(
        station_coords[:, 0], station_coords[:, 1], park["lat"].to_numpy(), park["lon"].to_numpy()
    )

    env = station[["station_id"]].copy()
    env["nearest_subway_m"] = subway_distance.min(axis=1)
    env["bus_stop_count_300m"] = (bus_distance <= 300).sum(axis=1)
    env["nearest_park_m"] = park_distance.min(axis=1)
    env["park_count_300m"] = (park_distance <= 300).sum(axis=1)
    return env


def prepare_model_table() -> tuple[pd.DataFrame, list[str], list[str]]:
    station = load_station_master()
    rides = load_rides()
    rides = rides[rides["station_id"].isin(station["station_id"])]

    pattern = build_return_pattern_features(rides)
    environment = build_environment_features(station)
    model_df = station.merge(pattern, on="station_id", how="inner").merge(environment, on="station_id", how="left")

    clustering_features = [
        "weekday_morning_ratio",
        "weekday_evening_ratio",
        "weekend_day_ratio",
        "night_ratio",
        "weekday_return_ratio",
        "morning_evening_gap",
        "nearest_subway_m",
        "bus_stop_count_300m",
        "nearest_park_m",
        "park_count_300m",
    ]
    summary_features = [
        "total_returns",
        "avg_daily_returns",
        "weekday_avg_returns",
        "weekend_avg_returns",
        "weekday_weekend_gap",
        "weekday_morning_ratio",
        "weekday_evening_ratio",
        "weekend_day_ratio",
        "night_ratio",
        "nearest_subway_m",
        "bus_stop_count_300m",
        "nearest_park_m",
        "park_count_300m",
    ]
    fill_columns = sorted(set(clustering_features + summary_features))
    model_df[fill_columns] = model_df[fill_columns].fillna(0)
    return model_df, clustering_features, summary_features


def run_k_search(scaled_features: np.ndarray, k_values: list[int]) -> pd.DataFrame:
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(scaled_features)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(scaled_features, labels),
            }
        )
    return pd.DataFrame(rows)


def infer_cluster_label(row: pd.Series) -> str:
    if (
        row["weekday_morning_ratio"] >= row["weekday_evening_ratio"] + 0.04
        and row["weekday_return_ratio"] >= 0.7
    ):
        return "업무·학교 도착형"
    if (
        row["weekday_evening_ratio"] >= row["weekday_morning_ratio"]
        and row["weekday_return_ratio"] >= 0.7
        and row["nearest_subway_m"] <= 400
        and row["bus_stop_count_300m"] >= 40
    ):
        return "교통거점형"
    if row["weekend_day_ratio"] >= 0.18 and (row["nearest_park_m"] <= 500 or row["park_count_300m"] >= 1):
        return "생활도착형"
    if row["weekday_evening_ratio"] >= row["weekday_morning_ratio"] + 0.04:
        return "생활도착형"
    return "생활도착형"


def build_station_notes(model_df: pd.DataFrame) -> pd.DataFrame:
    note_df = model_df.copy()
    cluster_profiles = (
        note_df.groupby("cluster_label")[
            [
                "weekday_evening_ratio",
                "weekday_morning_ratio",
                "weekday_return_ratio",
                "nearest_subway_m",
                "bus_stop_count_300m",
                "weekend_day_ratio",
                "nearest_park_m",
                "park_count_300m",
            ]
        ]
        .mean()
        .apply(infer_cluster_label, axis=1)
        .to_dict()
    )
    note_df["cluster_profile"] = note_df["cluster_label"].map(cluster_profiles)

    def station_note(row: pd.Series) -> str:
        if row["cluster_profile"] == "업무·학교 도착형":
            return (
                f"평일 아침 반납 비중({row['weekday_morning_ratio']:.3f})이 저녁보다 높고 "
                f"평일 비중({row['weekday_return_ratio']:.3f})도 높아 업무·학교 도착형으로 볼 수 있음"
            )
        if row["cluster_profile"] == "교통거점형":
            return (
                f"평일 저녁 반납 비중({row['weekday_evening_ratio']:.3f})이 높고 "
                f"300m 내 버스정류소 {int(row['bus_stop_count_300m'])}개, "
                f"최근접 지하철 {row['nearest_subway_m']:.0f}m라서 교통거점형으로 볼 수 있음"
            )
        if row["cluster_profile"] == "생활도착형":
            return (
                f"저녁 또는 생활시간대 반납 비중이 높고 "
                f"생활권 안에서 도착 수요가 모이는 생활도착형으로 볼 수 있음"
            )
        return (
            f"아침/저녁/주말 반납 패턴이 한쪽으로 강하게 치우치지 않고 "
            f"300m 버스정류소 {int(row['bus_stop_count_300m'])}개 수준의 생활도착형으로 볼 수 있음"
        )

    note_df["station_note"] = note_df.apply(station_note, axis=1)
    return note_df


def plot_elbow_silhouette(metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(data=metrics_df, x="k", y="inertia", marker="o", ax=axes[0], color="#1f77b4")
    axes[0].set_title("엘보우 플롯")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("군집 내 제곱합")

    sns.lineplot(data=metrics_df, x="k", y="silhouette", marker="o", ax=axes[1], color="#ff7f0e")
    axes[1].set_title("실루엣 점수")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("점수")

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "ddri_return_cluster_elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca(model_df: pd.DataFrame, scaled_features: np.ndarray) -> None:
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled_features)
    plot_df = model_df.copy()
    plot_df["pca_x"] = components[:, 0]
    plot_df["pca_y"] = components[:, 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="pca_x",
        y="pca_y",
        hue="cluster_profile",
        palette="Set2",
        s=80,
        ax=ax,
    )
    ax.set_title("반납 대여소 패턴 군집 (PCA 2차원)")
    ax.set_xlabel("주성분 1")
    ax.set_ylabel("주성분 2")
    ax.legend(title="군집 유형")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "ddri_return_cluster_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_feature_means(model_df: pd.DataFrame, summary_features: list[str]) -> None:
    cluster_means = model_df.groupby("cluster_profile")[summary_features].mean().reset_index()
    scaled = cluster_means.copy()
    scaler = MinMaxScaler()
    scaled[summary_features] = scaler.fit_transform(cluster_means[summary_features])
    long_df = scaled.melt(id_vars=["cluster_profile"], var_name="feature", value_name="scaled_mean")

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.barplot(data=long_df, x="feature", y="scaled_mean", hue="cluster_profile", ax=ax)
    feature_label_map = {
        "total_returns": "총 반납건수",
        "avg_daily_returns": "일평균 반납량",
        "weekday_avg_returns": "주중 일평균 반납량",
        "weekend_avg_returns": "주말 일평균 반납량",
        "weekday_weekend_gap": "주중-주말 격차",
        "weekday_morning_ratio": "주중 아침 비중",
        "weekday_evening_ratio": "주중 저녁 비중",
        "weekend_day_ratio": "주말 낮 비중",
        "night_ratio": "야간 비중",
        "nearest_subway_m": "최근접 지하철 거리",
        "bus_stop_count_300m": "300m 버스정류소 수",
        "nearest_park_m": "최근접 공원 거리",
        "park_count_300m": "300m 공원 수",
    }
    ax.set_title("군집별 패턴 및 300m 환경 특성 비교")
    ax.set_xlabel("")
    ax.set_ylabel("정규화 평균값")
    current_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([feature_label_map.get(label, label) for label in current_labels], rotation=45, ha="right")
    ax.legend(title="군집 유형")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "ddri_return_cluster_feature_means.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_outputs(model_df: pd.DataFrame, metrics_df: pd.DataFrame, summary_features: list[str]) -> None:
    model_df.to_csv(DATA_DIR / "ddri_return_station_cluster_features.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(DATA_DIR / "ddri_return_k_search_metrics.csv", index=False, encoding="utf-8-sig")

    cluster_summary = model_df.groupby(["cluster_label", "cluster_profile"])[summary_features].mean().round(3)
    cluster_summary["station_count"] = model_df.groupby(["cluster_label", "cluster_profile"]).size()
    cluster_summary.to_csv(DATA_DIR / "ddri_return_cluster_summary.csv", encoding="utf-8-sig")


def main() -> None:
    ensure_output_dirs()
    sns.set_theme(style="whitegrid")
    configure_korean_font()

    model_df, clustering_features, summary_features = prepare_model_table()
    scaled_features = StandardScaler().fit_transform(model_df[clustering_features])

    metrics_df = run_k_search(scaled_features, [2, 3, 4, 5, 6])
    best_k = int(metrics_df.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])

    model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    model_df["cluster_label"] = model.fit_predict(scaled_features)
    model_df = build_station_notes(model_df)

    plot_elbow_silhouette(metrics_df)
    plot_pca(model_df, scaled_features)
    plot_cluster_feature_means(model_df, summary_features)
    save_outputs(model_df, metrics_df, summary_features)

    print(f"best_k={best_k}")
    print(f"stations={len(model_df)}")
    print(f"profiles={sorted(model_df['cluster_profile'].unique().tolist())}")


if __name__ == "__main__":
    main()
