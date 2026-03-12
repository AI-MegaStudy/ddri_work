from __future__ import annotations

from pathlib import Path

import branca.colormap as cm
import folium
import pandas as pd


ROOT = Path("/Users/cheng80/Desktop/ddri_work")
RAW_ROOT = ROOT / "3조 공유폴더"
INTEGRATED_DIR = ROOT / "works" / "01_clustering" / "08_integrated"
OUTPUT_DIR = INTEGRATED_DIR / "intermediate" / "return_time_district"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_WINDOWS = {
    "7_10": list(range(7, 11)),
    "11_14": list(range(11, 15)),
    "17_20": list(range(17, 21)),
}


def year_trip_dir(year: int) -> Path:
    return RAW_ROOT / f"{year} 강남구 따릉이 이용정보"


def year_station_path(year: int) -> Path:
    return RAW_ROOT / "강남구 대여소 정보 (2023~2025)" / f"{year}_강남구_대여소.csv"


def active_station_ids(split: str) -> set[int]:
    if split == "train":
        path = (
            ROOT
            / "works"
            / "01_clustering"
            / "archive_1st"
            / "06_data"
            / "ddri_station_cluster_features_train_2023_2024.csv"
        )
    else:
        path = (
            ROOT
            / "works"
            / "01_clustering"
            / "archive_1st"
            / "06_data"
            / "ddri_station_cluster_features_test_2025.csv"
        )
    return set(pd.read_csv(path)["station_id"].astype(int))


def load_station_master(year: int) -> pd.DataFrame:
    station = pd.read_csv(year_station_path(year))
    station = station.rename(
        columns={
            "대여소번호": "station_id",
            "대여소명": "station_name",
            "위도": "latitude",
            "경도": "longitude",
        }
    )
    station["station_id"] = pd.to_numeric(station["station_id"], errors="coerce").astype("Int64")
    station["latitude"] = pd.to_numeric(station["latitude"], errors="coerce")
    station["longitude"] = pd.to_numeric(station["longitude"], errors="coerce")
    station = (
        station.dropna(subset=["station_id", "latitude", "longitude"])
        .drop_duplicates(subset=["station_id"])
        [["station_id", "station_name", "주소", "latitude", "longitude"]]
        .copy()
    )
    station["station_id"] = station["station_id"].astype(int)
    return station


def iter_trip_files(year: int) -> list[Path]:
    return sorted(year_trip_dir(year).glob("*.csv"))


def read_trip_file(path: Path) -> pd.DataFrame:
    usecols = [
        "대여일시",
        "대여 대여소번호",
        "반납일시",
        "반납대여소번호",
        "이용시간(분)",
        "이용거리(M)",
    ]
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def preprocess_trip_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["대여 대여소번호", "반납대여소번호", "이용시간(분)", "이용거리(M)"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(
        subset=["대여일시", "반납일시", "대여 대여소번호", "반납대여소번호", "이용시간(분)", "이용거리(M)"]
    ).copy()
    out = out[(out["이용시간(분)"] > 0) & (out["이용거리(M)"] > 0)].copy()
    out["대여 대여소번호"] = out["대여 대여소번호"].astype(int)
    out["반납대여소번호"] = out["반납대여소번호"].astype(int)
    out["대여일시"] = pd.to_datetime(out["대여일시"], errors="coerce")
    out["반납일시"] = pd.to_datetime(out["반납일시"], errors="coerce")
    out = out.dropna(subset=["대여일시", "반납일시"]).copy()

    same_station_short = (
        (out["대여 대여소번호"] == out["반납대여소번호"]) & (out["이용시간(분)"] <= 5)
    )
    out = out.loc[~same_station_short].copy()
    out["return_hour"] = out["반납일시"].dt.hour
    return out


def summarize_return_windows(year: int) -> pd.DataFrame:
    month_frames = []
    for trip_file in iter_trip_files(year):
        df = preprocess_trip_df(read_trip_file(trip_file))
        summary = (
            df.groupby("반납대여소번호")
            .agg(
                total_return_count=("반납대여소번호", "size"),
                return_7_10_count=("return_hour", lambda s: s.isin(TIME_WINDOWS["7_10"]).sum()),
                return_11_14_count=("return_hour", lambda s: s.isin(TIME_WINDOWS["11_14"]).sum()),
                return_17_20_count=("return_hour", lambda s: s.isin(TIME_WINDOWS["17_20"]).sum()),
            )
            .reset_index()
            .rename(columns={"반납대여소번호": "station_id"})
        )
        month_frames.append(summary)

    yearly = pd.concat(month_frames, ignore_index=True).groupby("station_id", as_index=False).sum()
    yearly["arrival_7_10_ratio"] = yearly["return_7_10_count"] / yearly["total_return_count"]
    yearly["arrival_11_14_ratio"] = yearly["return_11_14_count"] / yearly["total_return_count"]
    yearly["arrival_17_20_ratio"] = yearly["return_17_20_count"] / yearly["total_return_count"]
    yearly["dominant_window"] = yearly[
        ["return_7_10_count", "return_11_14_count", "return_17_20_count"]
    ].idxmax(axis=1)
    yearly["district_hypothesis"] = yearly["dominant_window"].map(
        {
            "return_7_10_count": "업무/상업지구 후보",
            "return_11_14_count": "점심 상권/업무지구 후보",
            "return_17_20_count": "주거지구 후보",
        }
    )

    station = load_station_master(year)
    merged = yearly.merge(station, on="station_id", how="left")
    merged["source_year"] = year
    return merged[
        [
            "source_year",
            "station_id",
            "station_name",
            "주소",
            "latitude",
            "longitude",
            "total_return_count",
            "return_7_10_count",
            "return_11_14_count",
            "return_17_20_count",
            "arrival_7_10_ratio",
            "arrival_11_14_ratio",
            "arrival_17_20_ratio",
            "dominant_window",
            "district_hypothesis",
        ]
    ].sort_values("station_id")


def add_dominant_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dominant_ratio"] = out[
        ["arrival_7_10_ratio", "arrival_11_14_ratio", "arrival_17_20_ratio"]
    ].max(axis=1)
    out["district_hypothesis"] = out[
        ["arrival_7_10_ratio", "arrival_11_14_ratio", "arrival_17_20_ratio"]
    ].idxmax(axis=1).map(
        {
            "arrival_7_10_ratio": "업무/상업지구 후보",
            "arrival_11_14_ratio": "점심 상권/업무지구 후보",
            "arrival_17_20_ratio": "주거지구 후보",
        }
    )
    return out


def build_train_test_features(yearly_frames: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ids = active_station_ids("train")
    test_ids = active_station_ids("test")

    train = pd.concat(
        [yearly_frames[2023][yearly_frames[2023]["station_id"].isin(train_ids)], yearly_frames[2024][yearly_frames[2024]["station_id"].isin(train_ids)]],
        ignore_index=True,
    )
    train = (
        train.groupby("station_id", as_index=False)
        .agg(
            station_name=("station_name", "last"),
            주소=("주소", "last"),
            latitude=("latitude", "last"),
            longitude=("longitude", "last"),
            total_return_count=("total_return_count", "sum"),
            return_7_10_count=("return_7_10_count", "sum"),
            return_11_14_count=("return_11_14_count", "sum"),
            return_17_20_count=("return_17_20_count", "sum"),
        )
    )
    train["arrival_7_10_ratio"] = train["return_7_10_count"] / train["total_return_count"]
    train["arrival_11_14_ratio"] = train["return_11_14_count"] / train["total_return_count"]
    train["arrival_17_20_ratio"] = train["return_17_20_count"] / train["total_return_count"]
    train = add_dominant_context(train)
    train["split"] = "train_2023_2024"

    test = yearly_frames[2025][yearly_frames[2025]["station_id"].isin(test_ids)].copy()
    test = add_dominant_context(test)
    test["split"] = "test_2025"

    final_cols = [
        "split",
        "station_id",
        "station_name",
        "주소",
        "latitude",
        "longitude",
        "total_return_count",
        "return_7_10_count",
        "return_11_14_count",
        "return_17_20_count",
        "arrival_7_10_ratio",
        "arrival_11_14_ratio",
        "arrival_17_20_ratio",
        "dominant_ratio",
        "district_hypothesis",
    ]
    return train[final_cols].sort_values("station_id"), test[final_cols].sort_values("station_id")


def build_merged_second_clustering_features(train_return: pd.DataFrame, test_return: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_train = pd.read_csv(
        INTEGRATED_DIR / "final" / "features" / "ddri_final_district_clustering_features_train_2023_2024.csv"
    )
    base_test = pd.read_csv(
        INTEGRATED_DIR / "final" / "features" / "ddri_final_district_clustering_features_test_2025.csv"
    )
    drop_old_ratio_cols = ["arrival_7_10_ratio", "arrival_11_14_ratio", "arrival_17_20_ratio"]
    base_train = base_train.drop(columns=drop_old_ratio_cols, errors="ignore")
    base_test = base_test.drop(columns=drop_old_ratio_cols, errors="ignore")

    keep_cols = [
        "station_id",
        "station_name",
        "주소",
        "latitude",
        "longitude",
        "total_return_count",
        "return_7_10_count",
        "return_11_14_count",
        "return_17_20_count",
        "arrival_7_10_ratio",
        "arrival_11_14_ratio",
        "arrival_17_20_ratio",
        "dominant_ratio",
        "district_hypothesis",
    ]

    train = base_train.merge(train_return[keep_cols], on="station_id", how="inner")
    test = base_test.merge(test_return[keep_cols], on="station_id", how="inner")
    return train.sort_values("station_id"), test.sort_values("station_id")


def build_map(df: pd.DataFrame, value_col: str, title: str, output_path: Path) -> None:
    map_df = df.dropna(subset=["latitude", "longitude", value_col]).copy()
    if map_df.empty:
        return

    center = [map_df["latitude"].mean(), map_df["longitude"].mean()]
    fmap = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")
    vmax = float(map_df[value_col].max())
    vmin = float(map_df[value_col].min())
    colormap = cm.linear.YlOrRd_09.scale(vmin, vmax if vmax > vmin else vmin + 1)
    colormap.caption = title
    colormap.add_to(fmap)

    for _, row in map_df.iterrows():
        color = colormap(row[value_col])
        popup = folium.Popup(
            (
                f"대여소번호: {int(row['station_id'])}<br>"
                f"대여소명: {row['station_name']}<br>"
                f"주소: {row['주소']}<br>"
                f"{value_col}: {row[value_col]:,.0f}<br>"
                f"07-10 비율: {row['arrival_7_10_ratio']:.3f}<br>"
                f"11-14 비율: {row['arrival_11_14_ratio']:.3f}<br>"
                f"17-20 비율: {row['arrival_17_20_ratio']:.3f}<br>"
                f"지구 가설: {row['district_hypothesis']}"
            ),
            max_width=350,
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4 + (row[value_col] / vmax * 12 if vmax else 4),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=1,
            popup=popup,
        ).add_to(fmap)

    fmap.save(output_path)


def main() -> None:
    yearly_frames: dict[int, pd.DataFrame] = {}
    for year in [2023, 2024, 2025]:
        yearly = summarize_return_windows(year)
        yearly_path = OUTPUT_DIR / f"ddri_return_time_features_{year}.csv"
        yearly.to_csv(yearly_path, index=False)
        yearly_frames[year] = yearly

        build_map(
            yearly,
            "return_7_10_count",
            f"{year} 07-10시 반납 분포",
            OUTPUT_DIR / f"ddri_return_map_{year}_7_10.html",
        )
        build_map(
            yearly,
            "return_11_14_count",
            f"{year} 11-14시 반납 분포",
            OUTPUT_DIR / f"ddri_return_map_{year}_11_14.html",
        )
        build_map(
            yearly,
            "return_17_20_count",
            f"{year} 17-20시 반납 분포",
            OUTPUT_DIR / f"ddri_return_map_{year}_17_20.html",
        )

    train, test = build_train_test_features(yearly_frames)
    train.to_csv(OUTPUT_DIR / "ddri_second_cluster_return_features_train_2023_2024.csv", index=False)
    test.to_csv(OUTPUT_DIR / "ddri_second_cluster_return_features_test_2025.csv", index=False)

    merged_train, merged_test = build_merged_second_clustering_features(train, test)
    merged_train.to_csv(OUTPUT_DIR / "ddri_second_cluster_merged_features_train_2023_2024.csv", index=False)
    merged_test.to_csv(OUTPUT_DIR / "ddri_second_cluster_merged_features_test_2025.csv", index=False)


if __name__ == "__main__":
    main()
