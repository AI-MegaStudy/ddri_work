from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/cheng80/Desktop/ddri_work")
INTEGRATED_DIR = ROOT / "works" / "01_clustering" / "08_integrated"
BASE_INPUT_DIR = INTEGRATED_DIR / "intermediate" / "return_time_district"
POI_DIR = INTEGRATED_DIR / "intermediate" / "poi_features"
OUTPUT_DIR = INTEGRATED_DIR / "intermediate" / "poi_enriched_clustering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


POI_FEATURES = [
    "restaurant_count_300m",
    "cafe_count_300m",
    "convenience_store_count_300m",
    "pharmacy_count_300m",
    "food_retail_count_1000m",
    "fitness_count_500m",
    "cinema_count_1000m",
]


def build(split: str) -> pd.DataFrame:
    if split == "train":
        base = pd.read_csv(BASE_INPUT_DIR / "ddri_second_cluster_ready_input_train_2023_2024.csv")
    else:
        base = pd.read_csv(BASE_INPUT_DIR / "ddri_second_cluster_ready_input_test_2025.csv")

    poi = pd.read_csv(POI_DIR / "ddri_station_poi_candidate_features.csv")
    poi_cols = ["station_id"] + POI_FEATURES
    merged = base.merge(poi[poi_cols], on="station_id", how="left")

    for col in POI_FEATURES:
        merged[f"log1p_{col}"] = np.log1p(merged[col].fillna(0.0))

    return merged.sort_values("station_id").reset_index(drop=True)


def main() -> None:
    train = build("train")
    test = build("test")
    train.to_csv(OUTPUT_DIR / "ddri_poi_enriched_cluster_ready_input_train_2023_2024.csv", index=False)
    test.to_csv(OUTPUT_DIR / "ddri_poi_enriched_cluster_ready_input_test_2025.csv", index=False)


if __name__ == "__main__":
    main()
