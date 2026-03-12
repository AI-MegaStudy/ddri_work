from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path("/Users/cheng80/Desktop/ddri_work")
INTEGRATED_DIR = ROOT / "works" / "01_clustering" / "08_integrated"
INPUT_DIR = INTEGRATED_DIR / "intermediate" / "return_time_district"
OUTPUT_DIR = INTEGRATED_DIR / "final" / "results" / "second_clustering_results"
OUTPUT_DATA_DIR = OUTPUT_DIR / "data"
OUTPUT_IMG_DIR = OUTPUT_DIR / "images"

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

FEATURE_COLS = [
    "arrival_7_10_ratio",
    "arrival_11_14_ratio",
    "arrival_17_20_ratio",
    "morning_net_inflow",
    "evening_net_inflow",
    "subway_distance_m",
    "bus_stop_count_300m",
]

FEATURE_LABELS = {
    "arrival_7_10_ratio": "07-10시 반납 비율",
    "arrival_11_14_ratio": "11-14시 반납 비율",
    "arrival_17_20_ratio": "17-20시 반납 비율",
    "morning_net_inflow": "아침 순유입",
    "evening_net_inflow": "저녁 순유입",
    "subway_distance_m": "지하철 거리(m)",
    "bus_stop_count_300m": "300m 버스정류장 수",
}

CLUSTER_NAME_MAP = {
    0: "업무/상업 혼합형",
    1: "초강한 아침 도착 업무 거점형",
    2: "주거 도착형",
    3: "생활·상권 혼합형",
    4: "외곽 주거형",
    5: "확장 군집 5",
    6: "확장 군집 6",
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(INPUT_DIR / "ddri_second_cluster_ready_input_train_2023_2024.csv")
    test = pd.read_csv(INPUT_DIR / "ddri_second_cluster_ready_input_test_2025.csv")
    return train, test


def run_k_search(train: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train[FEATURE_COLS])
    rows = []
    for k in range(5, 8):
        model = KMeans(n_clusters=k, random_state=42, n_init=30)
        labels = model.fit_predict(X_scaled)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X_scaled, labels),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DATA_DIR / "ddri_second_kmeans_search_metrics.csv", index=False)
    return result


def choose_k(k_search: pd.DataFrame) -> int:
    best = k_search.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    return int(best["k"])


def fit_cluster_model(train: pd.DataFrame, test: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, KMeans]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS])
    X_test = scaler.transform(test[FEATURE_COLS])

    model = KMeans(n_clusters=k, random_state=42, n_init=30)
    train_labels = model.fit_predict(X_train)
    test_labels = model.predict(X_test)

    train_out = train.copy()
    test_out = test.copy()
    train_out["cluster"] = train_labels
    test_out["cluster"] = test_labels

    train_out.to_csv(OUTPUT_DATA_DIR / "ddri_second_cluster_train_with_labels.csv", index=False)
    test_out.to_csv(OUTPUT_DATA_DIR / "ddri_second_cluster_test_with_labels.csv", index=False)
    return train_out, test_out, scaler, model


def save_cluster_summary(train_labeled: pd.DataFrame) -> pd.DataFrame:
    summary = train_labeled.groupby("cluster")[FEATURE_COLS].mean().round(4)
    summary["station_count"] = train_labeled.groupby("cluster").size()
    summary.to_csv(OUTPUT_DATA_DIR / "ddri_second_cluster_summary.csv")
    return summary


def save_cluster_hypothesis_crosstab(train_labeled: pd.DataFrame) -> pd.DataFrame:
    crosstab = pd.crosstab(train_labeled["cluster"], train_labeled["district_hypothesis"])
    crosstab.to_csv(OUTPUT_DATA_DIR / "ddri_second_cluster_hypothesis_crosstab.csv")
    return crosstab


def save_representative_stations(train_labeled: pd.DataFrame, scaler: StandardScaler, model: KMeans) -> pd.DataFrame:
    scaled = scaler.transform(train_labeled[FEATURE_COLS])
    center_distances = []
    for idx, row in train_labeled.reset_index(drop=True).iterrows():
        cluster = int(row["cluster"])
        dist = ((scaled[idx] - model.cluster_centers_[cluster]) ** 2).sum() ** 0.5
        center_distances.append(dist)
    rep = train_labeled.copy()
    rep["center_distance"] = center_distances
    rep = rep.sort_values(["cluster", "center_distance"]).groupby("cluster").head(5)
    rep.to_csv(OUTPUT_DATA_DIR / "ddri_second_cluster_representative_stations.csv", index=False)
    return rep


def plot_k_search(k_search: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(data=k_search, x="k", y="inertia", marker="o", ax=axes[0])
    axes[0].set_title("2차 군집화 Elbow")
    sns.lineplot(data=k_search, x="k", y="silhouette", marker="o", ax=axes[1])
    axes[1].set_title("2차 군집화 Silhouette")
    fig.tight_layout()
    fig.savefig(OUTPUT_IMG_DIR / "ddri_second_kmeans_elbow_silhouette.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pca_scatter(train_labeled: pd.DataFrame, scaler: StandardScaler) -> None:
    X_scaled = scaler.transform(train_labeled[FEATURE_COLS])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    plot_df = train_labeled.copy()
    plot_df["PC1"] = coords[:, 0]
    plot_df["PC2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="cluster", palette="tab10", s=70, ax=ax)
    ax.set_title("2차 군집화 PCA 산점도")
    fig.tight_layout()
    fig.savefig(OUTPUT_IMG_DIR / "ddri_second_kmeans_pca_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_means(summary: pd.DataFrame) -> None:
    plot_df = summary.reset_index().melt(id_vars="cluster", value_vars=FEATURE_COLS, var_name="feature", value_name="value")
    plot_df["feature_label"] = plot_df["feature"].map(FEATURE_LABELS)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_df, x="feature_label", y="value", hue="cluster", ax=ax)
    ax.set_title("2차 군집별 Feature 평균")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(OUTPUT_IMG_DIR / "ddri_second_cluster_feature_means.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_profile_heatmap(summary: pd.DataFrame) -> None:
    heat_df = summary[FEATURE_COLS].rename(columns=FEATURE_LABELS)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
    ax.set_title("2차 군집 프로파일 히트맵")
    fig.tight_layout()
    fig.savefig(OUTPUT_IMG_DIR / "ddri_second_cluster_profile_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_size(train_labeled: pd.DataFrame) -> None:
    size_df = train_labeled["cluster"].value_counts().sort_index().reset_index()
    size_df.columns = ["cluster", "count"]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=size_df, x="cluster", y="count", ax=ax)
    ax.set_title("2차 군집별 대여소 수")
    fig.tight_layout()
    fig.savefig(OUTPUT_IMG_DIR / "ddri_second_cluster_size.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_hypothesis_crosstab(crosstab: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(crosstab, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("군집별 지구판단 가설 분포")
    ax.set_xlabel("지구판단 가설")
    ax.set_ylabel("군집")
    fig.tight_layout()
    fig.savefig(OUTPUT_IMG_DIR / "ddri_second_cluster_hypothesis_crosstab.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_cluster_map(train_labeled: pd.DataFrame) -> None:
    plot_df = train_labeled.dropna(subset=["latitude", "longitude"]).copy()
    center = [plot_df["latitude"].mean(), plot_df["longitude"].mean()]
    fmap = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")
    palette = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4", "#7b3294", "#1b9e77"]

    for _, row in plot_df.iterrows():
        cluster = int(row["cluster"])
        color = palette[cluster % len(palette)]
        cluster_name = CLUSTER_NAME_MAP.get(cluster, f"군집 {cluster}")
        popup = folium.Popup(
            (
                f"군집: {cluster} ({cluster_name})<br>"
                f"대여소번호: {int(row['station_id'])}<br>"
                f"대여소명: {row['station_name']}<br>"
                f"07-10 비율: {row['arrival_7_10_ratio']:.3f}<br>"
                f"11-14 비율: {row['arrival_11_14_ratio']:.3f}<br>"
                f"17-20 비율: {row['arrival_17_20_ratio']:.3f}<br>"
                f"대여소별 도착시간 가설: {row['district_hypothesis']}"
            ),
            max_width=320,
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=1,
            popup=popup,
        ).add_to(fmap)

    legend_rows = []
    for cluster in sorted(plot_df["cluster"].unique()):
        color = palette[int(cluster) % len(palette)]
        cluster_name = CLUSTER_NAME_MAP.get(int(cluster), f"군집 {int(cluster)}")
        legend_rows.append(
            f"""
            <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
              <span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:{color}; border:1px solid #555;"></span>
              <span style="font-size:12px; color:#222;">Cluster {int(cluster)}: {cluster_name}</span>
            </div>
            """
        )
    legend_html = f"""
    <div style="
        position: fixed;
        top: 16px;
        right: 16px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);
        padding: 12px 14px;
        min-width: 220px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    ">
      <div style="font-size:13px; font-weight:700; color:#111827; margin-bottom:6px;">군집 범례</div>
      {''.join(legend_rows)}
      <div style="margin-top:8px; font-size:11px; color:#475569;">
        팝업의 지구 가설은 개별 대여소의 도착시간대 우세 패턴입니다.
      </div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    fmap.save(OUTPUT_DIR / "ddri_second_cluster_map.html")


def main() -> None:
    train, test = load_inputs()
    k_search = run_k_search(train)
    selected_k = choose_k(k_search)

    (OUTPUT_DATA_DIR / "ddri_second_selected_k.txt").write_text(str(selected_k), encoding="utf-8")

    train_labeled, test_labeled, scaler, model = fit_cluster_model(train, test, selected_k)
    summary = save_cluster_summary(train_labeled)
    crosstab = save_cluster_hypothesis_crosstab(train_labeled)
    save_representative_stations(train_labeled, scaler, model)

    plot_k_search(k_search)
    plot_pca_scatter(train_labeled, scaler)
    plot_feature_means(summary)
    plot_cluster_profile_heatmap(summary)
    plot_cluster_size(train_labeled)
    plot_cluster_hypothesis_crosstab(crosstab)
    save_cluster_map(train_labeled)


if __name__ == "__main__":
    main()
