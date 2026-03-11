from pathlib import Path

import pandas as pd


BASE_DIR = Path("/Users/cheng80/Desktop/ddri_work")
RAW_DIR = BASE_DIR / "3조 공유폴더"
CLUSTER_DIR = BASE_DIR / "works" / "clustering" / "data"
CALENDAR_DIR = BASE_DIR / "works" / "calendar" / "data"
OUTPUT_DIR = BASE_DIR / "works" / "prediction" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RENTAL_COLS = [
    "대여일시",
    "대여 대여소번호",
    "반납일시",
    "반납대여소번호",
    "이용시간(분)",
    "이용거리(M)",
]


def load_common_station_ids():
    master = pd.read_csv(CLUSTER_DIR / "ddri_common_station_master.csv")
    return set(master["대여소번호"].tolist()), master


def load_clean_events(paths, valid_return_ids, common_ids):
    frames = []
    for path in paths:
        df = pd.read_csv(path, usecols=RENTAL_COLS)
        df["대여일시"] = pd.to_datetime(df["대여일시"], errors="coerce")
        df["반납일시"] = pd.to_datetime(df["반납일시"], errors="coerce")
        df["대여 대여소번호"] = pd.to_numeric(df["대여 대여소번호"], errors="coerce")
        df["반납대여소번호"] = pd.to_numeric(df["반납대여소번호"], errors="coerce")
        df["이용시간(분)"] = pd.to_numeric(df["이용시간(분)"], errors="coerce")
        df["이용거리(M)"] = pd.to_numeric(df["이용거리(M)"], errors="coerce")

        mask_complete = df[
            ["대여일시", "반납일시", "대여 대여소번호", "반납대여소번호", "이용시간(분)", "이용거리(M)"]
        ].notna().all(axis=1)
        mask_positive = (df["이용시간(분)"] > 0) & (df["이용거리(M)"] > 0)
        mask_rent_common = df["대여 대여소번호"].isin(common_ids)
        mask_return_valid = df["반납대여소번호"].isin(valid_return_ids)

        clean_df = df.loc[mask_complete & mask_positive & mask_rent_common & mask_return_valid].copy()
        clean_df["station_id"] = clean_df["대여 대여소번호"].astype(int)
        clean_df["return_station_id"] = clean_df["반납대여소번호"].astype(int)
        clean_df["date"] = clean_df["대여일시"].dt.normalize()
        clean_df["return_date"] = clean_df["반납일시"].dt.normalize()
        frames.append(clean_df[["station_id", "date", "return_station_id", "return_date"]])

    return pd.concat(frames, ignore_index=True)


def build_station_day_metrics(event_df):
    rental_df = (
        event_df.groupby(["station_id", "date"])
        .size()
        .reset_index(name="rental_count")
    )
    return_df = (
        event_df.groupby(["return_station_id", "return_date"])
        .size()
        .reset_index(name="return_count")
        .rename(columns={"return_station_id": "station_id", "return_date": "date"})
    )
    same_station_df = event_df[
        (event_df["station_id"] == event_df["return_station_id"]) & (event_df["date"] == event_df["return_date"])
    ]
    same_station_df = (
        same_station_df.groupby(["station_id", "date"])
        .size()
        .reset_index(name="same_station_return_count")
    )

    metrics_df = (
        rental_df.merge(return_df, on=["station_id", "date"], how="left")
        .merge(same_station_df, on=["station_id", "date"], how="left")
        .sort_values(["station_id", "date"])
        .reset_index(drop=True)
    )
    metrics_df["return_count"] = metrics_df["return_count"].fillna(0).astype(int)
    metrics_df["same_station_return_count"] = metrics_df["same_station_return_count"].fillna(0).astype(int)
    metrics_df["same_station_return_ratio"] = (
        metrics_df["same_station_return_count"] / metrics_df["rental_count"]
    ).fillna(0.0)
    metrics_df["net_flow"] = metrics_df["rental_count"] - metrics_df["return_count"]
    return metrics_df


def build_weather_daily():
    weather_paths = [
        RAW_DIR / "2023-2024년 강남구 날씨데이터(00시-24시)" / "gangnam_weather_1year_2023.csv",
        RAW_DIR / "2023-2024년 강남구 날씨데이터(00시-24시)" / "gangnam_weather_1year_2024.csv",
    ]
    extra_dir = BASE_DIR / "works" / "weather" / "data"
    extra_paths = sorted(extra_dir.glob("ddri_weather_*_hourly.csv"))
    weather_paths.extend(extra_paths)
    frames = []
    for path in weather_paths:
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.normalize()
        frames.append(df)
    weather = pd.concat(frames, ignore_index=True)
    daily = (
        weather.groupby("date")
        .agg(
            temperature_mean=("temperature", "mean"),
            temperature_min=("temperature", "min"),
            temperature_max=("temperature", "max"),
            humidity_mean=("humidity", "mean"),
            wind_speed_mean=("wind_speed", "mean"),
            precipitation_sum=("precipitation", "sum"),
        )
        .reset_index()
    )
    daily["is_rainy_day"] = (daily["precipitation_sum"] > 0).astype(int)
    return daily


def main():
    common_ids, station_master = load_common_station_ids()
    cluster_labels = pd.read_csv(CLUSTER_DIR / "ddri_station_cluster_features_train_with_labels.csv")[
        ["station_id", "cluster"]
    ].rename(columns={"cluster": "cluster_label"})
    calendar = pd.read_csv(CALENDAR_DIR / "ddri_calendar_daily_2023_2025.csv")
    calendar["date"] = pd.to_datetime(calendar["date"])

    station_2023 = pd.read_csv(RAW_DIR / "강남구 대여소 정보 (2023~2025)" / "2023_강남구_대여소.csv")
    station_2024 = pd.read_csv(RAW_DIR / "강남구 대여소 정보 (2023~2025)" / "2024_강남구_대여소.csv")
    station_2025 = pd.read_csv(RAW_DIR / "강남구 대여소 정보 (2023~2025)" / "2025_강남구_대여소.csv")
    valid_return_ids = {
        2023: set(pd.to_numeric(station_2023["대여소번호"], errors="coerce").dropna().astype(int)),
        2024: set(pd.to_numeric(station_2024["대여소번호"], errors="coerce").dropna().astype(int)),
        2025: set(pd.to_numeric(station_2025["대여소번호"], errors="coerce").dropna().astype(int)),
    }

    train_events_2023 = load_clean_events(
        sorted((RAW_DIR / "2023 강남구 따릉이 이용정보").glob("*.csv")),
        valid_return_ids[2023],
        common_ids,
    )
    train_events_2024 = load_clean_events(
        sorted((RAW_DIR / "2024 강남구 따릉이 이용정보").glob("*.csv")),
        valid_return_ids[2024],
        common_ids,
    )
    test_events_2025 = load_clean_events(
        sorted((RAW_DIR / "2025 강남구 따릉이 이용정보").glob("*.csv")),
        valid_return_ids[2025],
        common_ids,
    )

    station_day_train_2023 = build_station_day_metrics(train_events_2023)
    station_day_train_2024 = build_station_day_metrics(train_events_2024)
    station_day_test_2025 = build_station_day_metrics(test_events_2025)

    target_train = pd.concat([station_day_train_2023, station_day_train_2024], ignore_index=True)
    weather_daily = build_weather_daily()
    weather_train_daily = weather_daily[weather_daily["date"].dt.year.isin([2023, 2024])].copy()
    weather_test_daily = weather_daily[weather_daily["date"].dt.year == 2025].copy()

    target_train.to_csv(OUTPUT_DIR / "ddri_station_day_target_train_2023_2024.csv", index=False)
    station_day_test_2025.to_csv(OUTPUT_DIR / "ddri_station_day_target_test_2025.csv", index=False)
    weather_train_daily.to_csv(OUTPUT_DIR / "ddri_weather_daily_2023_2024.csv", index=False)
    weather_test_daily.to_csv(OUTPUT_DIR / "ddri_weather_daily_2025.csv", index=False)

    station_day_metrics_summary = pd.DataFrame(
        [
            {
                "dataset": "train_2023_2024",
                "rows": len(target_train),
                "rental_count_sum": int(target_train["rental_count"].sum()),
                "return_count_sum": int(target_train["return_count"].sum()),
                "same_station_return_count_sum": int(target_train["same_station_return_count"].sum()),
                "same_station_return_ratio_mean": round(target_train["same_station_return_ratio"].mean(), 6),
                "net_flow_mean": round(target_train["net_flow"].mean(), 6),
            },
            {
                "dataset": "test_2025",
                "rows": len(station_day_test_2025),
                "rental_count_sum": int(station_day_test_2025["rental_count"].sum()),
                "return_count_sum": int(station_day_test_2025["return_count"].sum()),
                "same_station_return_count_sum": int(station_day_test_2025["same_station_return_count"].sum()),
                "same_station_return_ratio_mean": round(station_day_test_2025["same_station_return_ratio"].mean(), 6),
                "net_flow_mean": round(station_day_test_2025["net_flow"].mean(), 6),
            },
        ]
    )
    station_day_metrics_summary.to_csv(OUTPUT_DIR / "ddri_station_day_flow_metrics_summary.csv", index=False)

    train_dataset = (
        target_train.merge(calendar, on="date", how="left")
        .merge(weather_train_daily, on="date", how="left")
        .merge(cluster_labels, on="station_id", how="left")
        .merge(
            station_master[["대여소번호", "대여소명", "위도", "경도"]].rename(columns={"대여소번호": "station_id"}),
            on="station_id",
            how="left",
        )
        .sort_values(["station_id", "date"])
        .reset_index(drop=True)
    )
    train_dataset["holiday_name"] = train_dataset["holiday_name"].fillna("")
    train_dataset.to_csv(OUTPUT_DIR / "ddri_station_day_train_baseline_dataset.csv", index=False)

    test_dataset = (
        station_day_test_2025.merge(calendar, on="date", how="left")
        .merge(weather_test_daily, on="date", how="left")
        .merge(cluster_labels, on="station_id", how="left")
        .merge(
            station_master[["대여소번호", "대여소명", "위도", "경도"]].rename(columns={"대여소번호": "station_id"}),
            on="station_id",
            how="left",
        )
        .sort_values(["station_id", "date"])
        .reset_index(drop=True)
    )
    test_dataset["holiday_name"] = test_dataset["holiday_name"].fillna("")
    test_dataset.to_csv(OUTPUT_DIR / "ddri_station_day_test_baseline_dataset.csv", index=False)

    test_exception_rows = test_dataset[test_dataset["cluster_label"].isna()].copy()
    test_main_dataset = test_dataset[test_dataset["cluster_label"].notna()].copy()
    test_main_dataset.to_csv(OUTPUT_DIR / "ddri_station_day_test_main_eval_dataset.csv", index=False)
    test_exception_rows.to_csv(OUTPUT_DIR / "ddri_station_day_test_exception_cases.csv", index=False)

    print("saved train target:", OUTPUT_DIR / "ddri_station_day_target_train_2023_2024.csv")
    print("saved test target:", OUTPUT_DIR / "ddri_station_day_target_test_2025.csv")
    print("saved weather daily:", OUTPUT_DIR / "ddri_weather_daily_2023_2024.csv")
    print("saved weather daily:", OUTPUT_DIR / "ddri_weather_daily_2025.csv")
    print("saved flow summary:", OUTPUT_DIR / "ddri_station_day_flow_metrics_summary.csv")
    print("saved baseline train dataset:", OUTPUT_DIR / "ddri_station_day_train_baseline_dataset.csv")
    print("saved baseline test dataset:", OUTPUT_DIR / "ddri_station_day_test_baseline_dataset.csv")
    print("saved test main eval dataset:", OUTPUT_DIR / "ddri_station_day_test_main_eval_dataset.csv")
    print("saved test exception cases:", OUTPUT_DIR / "ddri_station_day_test_exception_cases.csv")
    print("train_rows=", len(train_dataset))
    print("test_rows=", len(test_dataset))
    print("test_main_rows=", len(test_main_dataset))
    print("test_exception_rows=", len(test_exception_rows))
    print("test_target_rows=", len(station_day_test_2025))


if __name__ == "__main__":
    main()
