from pathlib import Path

import koreanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = PROJECT_ROOT / "works" / "clustering" / "data" / "ddri_station_cluster_features_train_with_labels.csv"
STATION_PATH = PROJECT_ROOT / "works" / "clustering" / "data" / "ddri_common_station_master.csv"
REPRESENTATIVE_PATH = PROJECT_ROOT / "works" / "clustering" / "data" / "ddri_cluster_representative_stations.csv"
OUTPUT_PATH = PROJECT_ROOT / "works" / "presentation" / "ddri_cluster_static_map.png"


def build_static_map() -> None:
    feature_df = pd.read_csv(FEATURE_PATH)
    station_df = pd.read_csv(STATION_PATH).rename(
        columns={"대여소번호": "station_id", "대여소명": "station_name", "위도": "lat", "경도": "lon"}
    )
    rep_df = pd.read_csv(REPRESENTATIVE_PATH)

    plot_df = feature_df.merge(station_df[["station_id", "station_name", "lat", "lon"]], on="station_id", how="left")
    plot_df["cluster_name"] = plot_df["cluster"].map({0: "일반수요형", 1: "고수요형"})

    colors = {"일반수요형": "#5b8ff9", "고수요형": "#e8684a"}

    fig, ax = plt.subplots(figsize=(14, 9), dpi=180)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#eef4fb")

    for cluster_name, group in plot_df.groupby("cluster_name"):
        ax.scatter(
            group["lon"],
            group["lat"],
            s=group["avg_rental"] * 3.2,
            alpha=0.78,
            color=colors[cluster_name],
            edgecolor="white",
            linewidth=0.8,
            label=cluster_name,
        )

    top_rep_df = (
        rep_df.sort_values(["cluster_name", "avg_rental"], ascending=[True, False])
        .groupby("cluster_name")
        .head(3)
        .copy()
    )
    for _, row in top_rep_df.iterrows():
        ax.scatter(
            row["station_lon"],
            row["station_lat"],
            s=220,
            color=colors[row["cluster_name"]],
            edgecolor="#111111",
            linewidth=1.2,
            zorder=4,
        )
        ax.text(
            row["station_lon"] + 0.0013,
            row["station_lat"] + 0.0008,
            row["대여소명"],
            fontsize=10,
            weight="bold",
            color="#111111",
            zorder=5,
        )

    ax.set_title("강남구 따릉이 대여소 군집 분포도", fontsize=22, weight="bold", pad=14)
    ax.text(
        0.01,
        0.02,
        "점 크기 = 평균 대여량, 강조 표기 = 군집별 대표 대여소",
        transform=ax.transAxes,
        fontsize=11,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#cbd5e1"},
    )
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=11)
    ax.set_xlabel("경도", fontsize=12)
    ax.set_ylabel("위도", fontsize=12)
    ax.grid(alpha=0.18, linestyle="--")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_static_map()
