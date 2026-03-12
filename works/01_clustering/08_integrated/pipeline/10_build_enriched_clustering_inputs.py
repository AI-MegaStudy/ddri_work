from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/Users/cheng80/Desktop/ddri_work")
INTEGRATED_DIR = ROOT / "works" / "01_clustering" / "08_integrated"
BASE_INPUT_DIR = INTEGRATED_DIR / "intermediate" / "return_time_district"
ENV_DIR = INTEGRATED_DIR / "intermediate" / "environment_enrichment"
OUTPUT_DIR = INTEGRATED_DIR / "intermediate" / "enriched_second_clustering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


ENRICH_FEATURES = [
    "station_elevation_m",
    "elevation_diff_nearest_subway_m",
    "nearest_park_area_sqm",
    "distance_naturepark_m",
    "distance_river_boundary_m",
]


def build(split: str) -> pd.DataFrame:
    if split == "train":
        base = pd.read_csv(BASE_INPUT_DIR / "ddri_second_cluster_ready_input_train_2023_2024.csv")
    else:
        base = pd.read_csv(BASE_INPUT_DIR / "ddri_second_cluster_ready_input_test_2025.csv")

    env = pd.read_csv(ENV_DIR / "ddri_environment_enrichment_features.csv")
    env_cols = ["station_id"] + ENRICH_FEATURES
    merged = base.merge(env[env_cols], on="station_id", how="left")
    return merged.sort_values("station_id").reset_index(drop=True)


def main() -> None:
    train = build("train")
    test = build("test")
    train.to_csv(OUTPUT_DIR / "ddri_enriched_cluster_ready_input_train_2023_2024.csv", index=False)
    test.to_csv(OUTPUT_DIR / "ddri_enriched_cluster_ready_input_test_2025.csv", index=False)


if __name__ == "__main__":
    main()
