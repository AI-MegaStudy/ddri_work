from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/Users/cheng80/Desktop/ddri_work")
BASE_DIR = ROOT / "works" / "01_clustering" / "08_integrated" / "intermediate" / "return_time_district"


CORE_FEATURES = [
    "arrival_7_10_ratio",
    "arrival_11_14_ratio",
    "arrival_17_20_ratio",
    "morning_net_inflow",
    "evening_net_inflow",
    "subway_distance_m",
    "bus_stop_count_300m",
]

META_COLUMNS = [
    "station_id",
    "mapped_dong_code",
    "mapped_dong_name",
    "station_name",
    "주소",
    "latitude",
    "longitude",
    "total_return_count",
    "return_7_10_count",
    "return_11_14_count",
    "return_17_20_count",
    "dominant_ratio",
    "district_hypothesis",
    "life_pop_7_10_mean",
    "life_pop_11_14_mean",
    "life_pop_17_20_mean",
]


def build_ready_input(split: str) -> pd.DataFrame:
    if split == "train":
        src = BASE_DIR / "ddri_second_cluster_merged_features_train_2023_2024.csv"
    else:
        src = BASE_DIR / "ddri_second_cluster_merged_features_test_2025.csv"

    df = pd.read_csv(src)
    out = df[META_COLUMNS + CORE_FEATURES].copy()
    return out.sort_values("station_id").reset_index(drop=True)


def main() -> None:
    train = build_ready_input("train")
    test = build_ready_input("test")

    train.to_csv(BASE_DIR / "ddri_second_cluster_ready_input_train_2023_2024.csv", index=False)
    test.to_csv(BASE_DIR / "ddri_second_cluster_ready_input_test_2025.csv", index=False)


if __name__ == "__main__":
    main()
