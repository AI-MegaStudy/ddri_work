from __future__ import annotations

from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path("/Users/cheng80/Desktop/ddri_work")
INTEGRATED_DIR = ROOT / "works" / "01_clustering" / "08_integrated"
INPUT_DIR = INTEGRATED_DIR / "intermediate" / "poi_enriched_clustering"
OUTPUT_DIR = INTEGRATED_DIR / "intermediate" / "poi_enriched_second_clustering_results"
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
    "log1p_restaurant_count_300m",
    "log1p_cafe_count_300m",
    "log1p_convenience_store_count_300m",
    "log1p_pharmacy_count_300m",
    "log1p_food_retail_count_1000m",
    "log1p_fitness_count_500m",
    "log1p_cinema_count_1000m",
]

FEATURE_LABELS = {
    "arrival_7_10_ratio": "07-10시 반납 비율",
    "arrival_11_14_ratio": "11-14시 반납 비율",
    "arrival_17_20_ratio": "17-20시 반납 비율",
    "morning_net_inflow": "아침 순유입",
    "evening_net_inflow": "저녁 순유입",
    "subway_distance_m": "지하철 거리(m)",
    "bus_stop_count_300m": "300m 버스정류장 수",
    "log1p_restaurant_count_300m": "log1p(300m 일반음식점 수)",
    "log1p_cafe_count_300m": "log1p(300m 커피숍 수)",
    "log1p_convenience_store_count_300m": "log1p(300m 편의점 수)",
    "log1p_pharmacy_count_300m": "log1p(300m 약국 수)",
    "log1p_food_retail_count_1000m": "log1p(1000m 식품판매업(기타) 수)",
    "log1p_fitness_count_500m": "log1p(500m 체력단련장 수)",
    "log1p_cinema_count_1000m": "log1p(1000m 영화상영관 수)",
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(INPUT_DIR / "ddri_poi_enriched_cluster_ready_input_train_2023_2024.csv")
    test = pd.read_csv(INPUT_DIR / "ddri_poi_enriched_cluster_ready_input_test_2025.csv")
    return train, test


def run_k_search(train: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(train[FEATURE_COLS])
    rows = []
    for k in range(5, 8):
        model = KMeans(n_clusters=k, random_state=42, n_init=30)
        labels = model.fit_predict(x_scaled)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette_score(x_scaled, labels)})
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DATA_DIR / "ddri_poi_enriched_kmeans_search_metrics.csv", index=False)
    return out


def choose_k(k_search: pd.DataFrame) -> int:
    best = k_search.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    return int(best["k"])


def fit_model(train: pd.DataFrame, test: pd.DataFrame, k: int):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURE_COLS])
    x_test = scaler.transform(test[FEATURE_COLS])
    model = KMeans(n_clusters=k, random_state=42, n_init=30)
    train_labels = model.fit_predict(x_train)
    test_labels = model.predict(x_test)
    train_out = train.copy()
    test_out = test.copy()
    train_out["cluster"] = train_labels
    test_out["cluster"] = test_labels
    train_out.to_csv(OUTPUT_DATA_DIR / "ddri_poi_enriched_cluster_train_with_labels.csv", index=False)
    test_out.to_csv(OUTPUT_DATA_DIR / "ddri_poi_enriched_cluster_test_with_labels.csv", index=False)
    return train_out, test_out, scaler


def save_summaries(train_labeled: pd.DataFrame) -> None:
    summary = train_labeled.groupby("cluster")[FEATURE_COLS].mean().round(4)
    summary["station_count"] = train_labeled.groupby("cluster").size()
    summary.to_csv(OUTPUT_DATA_DIR / "ddri_poi_enriched_cluster_summary.csv")
    crosstab = pd.crosstab(train_labeled["cluster"], train_labeled["district_hypothesis"])
    crosstab.to_csv(OUTPUT_DATA_DIR / "ddri_poi_enriched_cluster_hypothesis_crosstab.csv")


def plot_k_search(k_search: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(data=k_search, x="k", y="inertia", marker="o", ax=axes[0])
    axes[0].set_title("POI 보강 군집화 Elbow")
    sns.lineplot(data=k_search, x="k", y="silhouette", marker="o", ax=axes[1])
    axes[1].set_title("POI 보강 군집화 Silhouette")
    fig.tight_layout()
    fig.savefig(OUTPUT_IMG_DIR / "ddri_poi_enriched_kmeans_elbow_silhouette.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pca(train_labeled: pd.DataFrame, scaler: StandardScaler) -> None:
    x_scaled = scaler.transform(train_labeled[FEATURE_COLS])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x_scaled)
    plot_df = train_labeled.copy()
    plot_df["PC1"] = coords[:, 0]
    plot_df["PC2"] = coords[:, 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="cluster", palette="tab10", s=70, ax=ax)
    ax.set_title("POI 보강 군집화 PCA 산점도")
    fig.tight_layout()
    fig.savefig(OUTPUT_IMG_DIR / "ddri_poi_enriched_kmeans_pca_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_means(train_labeled: pd.DataFrame, scaler: StandardScaler) -> None:
    scaled = scaler.transform(train_labeled[FEATURE_COLS])
    scaled_df = pd.DataFrame(scaled, columns=FEATURE_COLS)
    scaled_df["cluster"] = train_labeled["cluster"].to_numpy()
    scaled_summary = scaled_df.groupby("cluster")[FEATURE_COLS].mean().reset_index()
    plot_df = scaled_summary.melt(id_vars="cluster", value_vars=FEATURE_COLS, var_name="feature", value_name="value")
    plot_df["feature_label"] = plot_df["feature"].map(FEATURE_LABELS)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=plot_df, x="feature_label", y="value", hue="cluster", ax=ax)
    ax.set_title("POI 보강 군집별 Feature 평균(표준화)")
    ax.tick_params(axis="x", rotation=35)
    ax.set_xlabel("")
    ax.set_ylabel("표준화 평균(z-score)")
    fig.tight_layout()
    fig.savefig(OUTPUT_IMG_DIR / "ddri_poi_enriched_cluster_feature_means.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_cluster_map(train_labeled: pd.DataFrame) -> None:
    plot_df = train_labeled.dropna(subset=["latitude", "longitude"]).copy()
    fmap = folium.Map(location=[plot_df["latitude"].mean(), plot_df["longitude"].mean()], zoom_start=13, tiles="CartoDB positron")
    palette = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4", "#7b3294", "#1b9e77"]
    for _, row in plot_df.iterrows():
        color = palette[int(row["cluster"]) % len(palette)]
        popup = folium.Popup(
            (
                f"군집: {int(row['cluster'])}<br>"
                f"대여소번호: {int(row['station_id'])}<br>"
                f"대여소명: {row['station_name']}<br>"
                f"대여소별 도착시간 가설: {row['district_hypothesis']}<br>"
                f"log1p(커피숍): {row['log1p_cafe_count_300m']:.3f}<br>"
                f"log1p(편의점): {row['log1p_convenience_store_count_300m']:.3f}"
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
    fmap.save(OUTPUT_DIR / "ddri_poi_enriched_cluster_map.html")


def write_comparison(k_search: pd.DataFrame, train_labeled: pd.DataFrame) -> None:
    selected_k = int(k_search.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    best_silhouette = float(k_search.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["silhouette"])
    summary = train_labeled.groupby("cluster")[FEATURE_COLS].mean().round(3)
    lines = [
        "# POI 보강 군집화 비교 메모",
        "",
        "## 선택 기준",
        "",
        "- 사용 피처: `restaurant`, `cafe`, `convenience_store`, `pharmacy`, `food_retail`, `fitness`, `cinema`",
        "- `food_retail`은 `식품판매업(기타)`로, 마트·백화점 식품관·오프라인 판매 거점을 대리하는 축으로 사용",
        "- `hospital`은 후보로 검토했지만 silhouette를 낮춰 최종 적용 세트에서는 제외",
        "- `golf_practice`는 따릉이 목적지 직접성이 낮고, `bakery`는 카페와 중복이 커서 제외",
        "- 카운트 분포 완화를 위해 모든 POI 피처는 `log1p` 변환 후 사용",
        "",
        "## 결과 요약",
        "",
        f"- 선택 k: `{selected_k}`",
        f"- 최고 silhouette: `{best_silhouette:.4f}`",
        "- 기존 메인 통합 군집화와 비교해 분리도 개선 여부를 함께 검토",
        "",
        "## 군집 평균(발췌)",
        "",
        summary.to_markdown(),
        "",
    ]
    (OUTPUT_DIR / "poi_enriched_vs_base_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train, test = load_inputs()
    k_search = run_k_search(train)
    selected_k = choose_k(k_search)
    (OUTPUT_DATA_DIR / "ddri_poi_enriched_selected_k.txt").write_text(str(selected_k), encoding="utf-8")
    train_labeled, test_labeled, scaler = fit_model(train, test, selected_k)
    save_summaries(train_labeled)
    plot_k_search(k_search)
    plot_pca(train_labeled, scaler)
    plot_feature_means(train_labeled, scaler)
    save_cluster_map(train_labeled)
    write_comparison(k_search, train_labeled)


if __name__ == "__main__":
    main()
