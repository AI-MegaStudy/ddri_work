from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/Users/cheng80/Desktop/ddri_work")
INTEGRATED_DIR = ROOT / "works" / "01_clustering" / "08_integrated"
ENV_DIR = INTEGRATED_DIR / "intermediate" / "environment_enrichment"
CLUSTER_DIR = INTEGRATED_DIR / "final" / "results" / "second_clustering_results" / "data"


def main() -> None:
    env = pd.read_csv(ENV_DIR / "ddri_environment_enrichment_features.csv")
    labeled = pd.read_csv(CLUSTER_DIR / "ddri_second_cluster_train_with_labels.csv")

    merged = labeled.merge(env, on="station_id", how="left", suffixes=("", "_env"))
    merged.to_csv(ENV_DIR / "ddri_environment_enrichment_with_clusters.csv", index=False)

    cluster_summary = (
        merged.groupby("cluster")[
            [
                "station_elevation_m",
                "elevation_diff_nearest_subway_m",
                "elevation_diff_nearest_bus_stop_m",
                "nearest_park_distance_m",
                "nearest_park_area_sqm",
                "elevation_diff_nearest_park_m",
                "distance_naturepark_m",
                "inside_naturepark",
                "distance_river_boundary_m",
            ]
        ]
        .mean()
        .round(2)
    )
    cluster_summary.to_csv(ENV_DIR / "ddri_environment_enrichment_cluster_summary.csv")

    corr_cols = [
        "station_elevation_m",
        "elevation_diff_nearest_subway_m",
        "elevation_diff_nearest_bus_stop_m",
        "nearest_park_distance_m",
        "nearest_park_area_sqm",
        "elevation_diff_nearest_park_m",
        "distance_naturepark_m",
        "inside_naturepark",
        "distance_river_boundary_m",
        "arrival_7_10_ratio",
        "arrival_11_14_ratio",
        "arrival_17_20_ratio",
        "morning_net_inflow",
        "evening_net_inflow",
    ]
    corr = merged[corr_cols].corr().round(3)
    corr.to_csv(ENV_DIR / "ddri_environment_enrichment_correlation.csv")


if __name__ == "__main__":
    main()
