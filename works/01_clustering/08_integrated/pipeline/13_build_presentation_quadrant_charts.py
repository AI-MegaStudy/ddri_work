from __future__ import annotations

from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/Users/cheng80/Desktop/ddri_work")
INTEGRATED_DIR = ROOT / "works" / "01_clustering" / "08_integrated"
INPUT_PATH = (
    INTEGRATED_DIR / "final" / "results" / "second_clustering_results" / "data" / "ddri_second_cluster_train_with_labels.csv"
)
OUTPUT_DIR = INTEGRATED_DIR / "final" / "results" / "second_clustering_results" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PALETTE = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4", "#7b3294", "#1b9e77"]


def add_quadrant_labels_ratio(ax, x_mean: float, y_mean: float) -> None:
    ax.axvline(x_mean, color="#666666", linestyle="--", linewidth=1)
    ax.axhline(y_mean, color="#666666", linestyle="--", linewidth=1)
    ax.text(x_mean * 0.45, y_mean * 1.28, "저녁 도착형", fontsize=10, color="#555555")
    ax.text(x_mean * 1.06, y_mean * 1.28, "양시간대 혼합형", fontsize=10, color="#555555")
    ax.text(x_mean * 0.45, y_mean * 0.52, "저강도 균형형", fontsize=10, color="#555555")
    ax.text(x_mean * 1.06, y_mean * 0.52, "아침 도착형", fontsize=10, color="#555555")


def add_quadrant_labels_flow(ax) -> None:
    ax.axvline(0, color="#666666", linestyle="--", linewidth=1)
    ax.axhline(0, color="#666666", linestyle="--", linewidth=1)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.text(x0 * 0.9, y1 * 0.82, "저녁만 순유입", fontsize=10, color="#555555")
    ax.text(x1 * 0.2, y1 * 0.82, "양시간대 순유입", fontsize=10, color="#555555")
    ax.text(x0 * 0.9, y0 * 0.75, "양시간대 순유출", fontsize=10, color="#555555")
    ax.text(x1 * 0.2, y0 * 0.75, "아침만 순유입", fontsize=10, color="#555555")


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: arrival ratio quadrant
    ax = axes[0]
    for cluster_id, group in df.groupby("cluster"):
        color = PALETTE[int(cluster_id) % len(PALETTE)]
        ax.scatter(
            group["arrival_7_10_ratio"],
            group["arrival_17_20_ratio"],
            s=30,
            alpha=0.35,
            color=color,
        )
        center_x = group["arrival_7_10_ratio"].mean()
        center_y = group["arrival_17_20_ratio"].mean()
        ax.scatter(center_x, center_y, s=220, color=color, edgecolor="black", linewidth=1.2)
        ax.text(center_x, center_y, f"C{int(cluster_id)}", ha="center", va="center", fontsize=10, weight="bold")

    x_mean = df["arrival_7_10_ratio"].mean()
    y_mean = df["arrival_17_20_ratio"].mean()
    add_quadrant_labels_ratio(ax, x_mean, y_mean)
    ax.set_title("아침 반납 비율 vs 저녁 반납 비율")
    ax.set_xlabel("07~10시 반납 비율")
    ax.set_ylabel("17~20시 반납 비율")

    # Panel 2: net inflow quadrant
    ax = axes[1]
    for cluster_id, group in df.groupby("cluster"):
        color = PALETTE[int(cluster_id) % len(PALETTE)]
        ax.scatter(
            group["morning_net_inflow"],
            group["evening_net_inflow"],
            s=30,
            alpha=0.35,
            color=color,
        )
        center_x = group["morning_net_inflow"].mean()
        center_y = group["evening_net_inflow"].mean()
        ax.scatter(center_x, center_y, s=220, color=color, edgecolor="black", linewidth=1.2)
        ax.text(center_x, center_y, f"C{int(cluster_id)}", ha="center", va="center", fontsize=10, weight="bold")

    add_quadrant_labels_flow(ax)
    ax.set_title("아침 순유입 vs 저녁 순유입")
    ax.set_xlabel("아침 순유입")
    ax.set_ylabel("저녁 순유입")

    fig.suptitle("통합 군집화 핵심 사분면 해석", fontsize=18, weight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ddri_second_cluster_quadrant_views.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Static map with basemap and legend for presentation use.
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))
    handles = []
    labels = []
    for cluster_id, group in gdf.groupby("cluster"):
        color = PALETTE[int(cluster_id) % len(PALETTE)]
        group.plot(
            ax=ax,
            markersize=70,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
            label=f"Cluster {int(cluster_id)}",
        )
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.6,
                markersize=9,
            )
        )
        labels.append(f"Cluster {int(cluster_id)}")

    minx, miny, maxx, maxy = gdf.total_bounds
    pad_x = (maxx - minx) * 0.08
    pad_y = (maxy - miny) * 0.08
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, attribution=False)
    ax.set_title("강남구 따릉이 통합 군집화 지도", fontsize=18, weight="bold")
    ax.set_axis_off()
    ax.legend(handles, labels, title="군집", loc="lower left", frameon=True, fontsize=10, title_fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ddri_second_cluster_static_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
