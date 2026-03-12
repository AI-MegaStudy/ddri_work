from pathlib import Path

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import koreanize_matplotlib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path("/Users/cheng80/Desktop/ddri_work")
DATA_DIR = BASE_DIR / "3조 공유폴더"
OUTPUT_DATA_DIR = BASE_DIR / "works" / "01_clustering" / "06_data"
OUTPUT_IMG_DIR = BASE_DIR / "works" / "01_clustering" / "07_images"
PREP_DATA_DIR = BASE_DIR / "works" / "01_clustering" / "02_preprocessing" / "data"

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
PREP_DATA_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

RENTAL_COLS = [
    "대여일시",
    "대여 대여소번호",
    "반납일시",
    "반납대여소번호",
    "이용시간(분)",
    "이용거리(M)",
]

FEATURE_COLS = [
    "avg_rental",
    "rental_std",
    "weekday_avg",
    "weekend_avg",
    "peak_ratio",
    "night_ratio",
    "weekday_weekend_gap",
]

FEATURE_LABELS = {
    "avg_rental": "평균 대여량",
    "rental_std": "대여량 표준편차",
    "weekday_avg": "평일 평균 대여량",
    "weekend_avg": "주말 평균 대여량",
    "peak_ratio": "출퇴근 시간 비율",
    "night_ratio": "야간 비율",
    "weekday_weekend_gap": "평일-주말 차이",
}


def load_station_frames():
    station_master_files = {
        2023: DATA_DIR / "강남구 대여소 정보 (2023~2025)" / "2023_강남구_대여소.csv",
        2024: DATA_DIR / "강남구 대여소 정보 (2023~2025)" / "2024_강남구_대여소.csv",
        2025: DATA_DIR / "강남구 대여소 정보 (2023~2025)" / "2025_강남구_대여소.csv",
    }

    station_frames = {}
    station_id_sets = {}
    for year, path in station_master_files.items():
        df = pd.read_csv(path)
        df["대여소번호"] = pd.to_numeric(df["대여소번호"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["대여소번호"]).copy()
        df["대여소번호"] = df["대여소번호"].astype(int)
        station_frames[year] = df
        station_id_sets[year] = set(df["대여소번호"])

    common_station_ids = sorted(set.intersection(*station_id_sets.values()))
    common_station_master = (
        station_frames[2025]
        .loc[
            station_frames[2025]["대여소번호"].isin(common_station_ids),
            ["대여소번호", "대여소명", "자치구", "주소", "위도", "경도"],
        ]
        .drop_duplicates("대여소번호")
        .sort_values("대여소번호")
        .reset_index(drop=True)
    )

    common_station_master.to_csv(OUTPUT_DATA_DIR / "ddri_common_station_master.csv", index=False)
    return station_id_sets, common_station_ids


def file_groups():
    return {
        2023: sorted((DATA_DIR / "2023 강남구 따릉이 이용정보").glob("*.csv")),
        2024: sorted((DATA_DIR / "2024 강남구 따릉이 이용정보").glob("*.csv")),
        2025: sorted((DATA_DIR / "2025 강남구 따릉이 이용정보").glob("*.csv")),
    }


def clean_rental_file(path: Path, valid_station_ids: set[int], common_station_ids: set[int]):
    df = pd.read_csv(path, usecols=RENTAL_COLS)
    before_count = len(df)

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
    mask_short_same_station_return = (
        (df["대여 대여소번호"] == df["반납대여소번호"]) & (df["이용시간(분)"] <= 5)
    )
    mask_rent_common = df["대여 대여소번호"].isin(common_station_ids)
    mask_return_gangnam = df["반납대여소번호"].isin(valid_station_ids)

    clean_df = df.loc[
        mask_complete
        & mask_positive
        & ~mask_short_same_station_return
        & mask_rent_common
        & mask_return_gangnam
    ].copy()
    clean_df["대여 대여소번호"] = clean_df["대여 대여소번호"].astype(int)
    clean_df["반납대여소번호"] = clean_df["반납대여소번호"].astype(int)

    stats = {
        "file_name": path.name,
        "rows_before": before_count,
        "rows_after": len(clean_df),
        "dropped_missing": int((~mask_complete).sum()),
        "dropped_nonpositive": int((mask_complete & ~mask_positive).sum()),
        "dropped_short_same_station_return": int((mask_complete & mask_positive & mask_short_same_station_return).sum()),
        "dropped_noncommon_rent": int(
            (mask_complete & mask_positive & ~mask_short_same_station_return & ~mask_rent_common).sum()
        ),
        "dropped_outside_gangnam_return": int(
            (
                mask_complete
                & mask_positive
                & ~mask_short_same_station_return
                & mask_rent_common
                & ~mask_return_gangnam
            ).sum()
        ),
    }
    return clean_df, stats


def load_clean_group(paths, valid_station_ids, common_station_ids, group_name):
    frames = []
    stats_rows = []
    for idx, path in enumerate(paths, start=1):
        clean_df, stats = clean_rental_file(path, valid_station_ids, common_station_ids)
        frames.append(clean_df)
        stats["group_name"] = group_name
        stats_rows.append(stats)
        print(
            f"[{group_name}] {idx}/{len(paths)} {path.name}: "
            f"{stats['rows_before']:,} -> {stats['rows_after']:,}"
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(stats_rows)


def build_station_features(df):
    work = df.copy()
    work["date"] = work["대여일시"].dt.date
    work["hour"] = work["대여일시"].dt.hour
    work["weekday"] = work["대여일시"].dt.weekday
    work["is_weekend"] = work["weekday"] >= 5
    work["is_peak"] = work["hour"].isin([7, 8, 9, 18, 19, 20])
    work["is_night"] = work["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5])

    daily = (
        work.groupby(["대여 대여소번호", "date"])
        .size()
        .reset_index(name="daily_rental_count")
    )
    avg_rental = daily.groupby("대여 대여소번호")["daily_rental_count"].mean()
    rental_std = daily.groupby("대여 대여소번호")["daily_rental_count"].std().fillna(0)
    weekday_avg = (
        work.loc[~work["is_weekend"]]
        .groupby(["대여 대여소번호", "date"])
        .size()
        .groupby("대여 대여소번호")
        .mean()
    )
    weekend_avg = (
        work.loc[work["is_weekend"]]
        .groupby(["대여 대여소번호", "date"])
        .size()
        .groupby("대여 대여소번호")
        .mean()
    )
    total = work.groupby("대여 대여소번호").size()
    peak = work.loc[work["is_peak"]].groupby("대여 대여소번호").size()
    night = work.loc[work["is_night"]].groupby("대여 대여소번호").size()

    features = pd.DataFrame(
        {
            "station_id": avg_rental.index,
            "avg_rental": avg_rental.values,
            "rental_std": rental_std.reindex(avg_rental.index).values,
            "weekday_avg": weekday_avg.reindex(avg_rental.index).fillna(0).values,
            "weekend_avg": weekend_avg.reindex(avg_rental.index).fillna(0).values,
            "peak_ratio": (peak / total).reindex(avg_rental.index).fillna(0).values,
            "night_ratio": (night / total).reindex(avg_rental.index).fillna(0).values,
        }
    )
    features["weekday_weekend_gap"] = features["weekday_avg"] - features["weekend_avg"]
    return features.sort_values("station_id").reset_index(drop=True)


def save_kmeans_outputs(train_features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_features[FEATURE_COLS])

    search_rows = []
    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_scaled)
        search_rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X_scaled, labels),
            }
        )

    k_search_df = pd.DataFrame(search_rows)
    k_search_df.to_csv(OUTPUT_DATA_DIR / "ddri_kmeans_search_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(data=k_search_df, x="k", y="inertia", marker="o", ax=axes[0])
    axes[0].set_title("엘보우 플롯")
    axes[0].set_xlabel("군집 수 (k)")
    axes[0].set_ylabel("관성")
    sns.lineplot(data=k_search_df, x="k", y="silhouette", marker="o", ax=axes[1])
    axes[1].set_title("군집 수별 실루엣 점수")
    axes[1].set_xlabel("군집 수 (k)")
    axes[1].set_ylabel("실루엣 점수")
    plt.tight_layout()
    fig.savefig(OUTPUT_IMG_DIR / "ddri_kmeans_elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    final_k = int(k_search_df.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    final_model = KMeans(n_clusters=final_k, random_state=42, n_init=20)
    train_features = train_features.copy()
    train_features["cluster"] = final_model.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    pca_points = pca.fit_transform(X_scaled)
    train_features["pca_1"] = pca_points[:, 0]
    train_features["pca_2"] = pca_points[:, 1]
    train_features.to_csv(OUTPUT_DATA_DIR / "ddri_station_cluster_features_train_with_labels.csv", index=False)

    cluster_summary = train_features.groupby("cluster")[FEATURE_COLS].mean().round(3)
    cluster_summary.to_csv(OUTPUT_DATA_DIR / "ddri_cluster_summary.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=train_features, x="pca_1", y="pca_2", hue="cluster", palette="Set2", ax=ax)
    ax.set_title(f"PCA 군집 산점도 (k={final_k})")
    ax.set_xlabel("주성분 1")
    ax.set_ylabel("주성분 2")
    ax.legend(title="군집")
    plt.tight_layout()
    fig.savefig(OUTPUT_IMG_DIR / "ddri_kmeans_pca_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    cluster_plot_df = cluster_summary.reset_index().melt(
        id_vars="cluster", var_name="feature", value_name="value"
    )
    cluster_plot_df["feature_label"] = cluster_plot_df["feature"].map(FEATURE_LABELS)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=cluster_plot_df, x="feature_label", y="value", hue="cluster", ax=ax)
    ax.set_title("군집별 주요 특성 평균")
    ax.set_xlabel("특성")
    ax.set_ylabel("평균값")
    ax.legend(title="군집")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(OUTPUT_IMG_DIR / "ddri_cluster_feature_means.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return final_k, k_search_df, cluster_summary


def save_feature_correlation_heatmap(train_features: pd.DataFrame) -> None:
    corr_df = train_features[FEATURE_COLS].corr().round(2)
    rename_map = {col: FEATURE_LABELS[col] for col in FEATURE_COLS}
    corr_df = corr_df.rename(index=rename_map, columns=rename_map)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_df, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("군집화 입력 특성 상관관계 히트맵")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(OUTPUT_IMG_DIR / "ddri_feature_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_cluster_profile_heatmap(cluster_summary: pd.DataFrame) -> None:
    heatmap_df = cluster_summary.copy()
    heatmap_df.index = [f"군집 {idx}" for idx in heatmap_df.index]
    heatmap_df = heatmap_df.rename(columns=FEATURE_LABELS)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(heatmap_df, annot=True, cmap="OrRd", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("군집별 주요 특성 프로파일 히트맵")
    ax.set_xlabel("특성")
    ax.set_ylabel("군집")
    plt.tight_layout()
    fig.savefig(OUTPUT_IMG_DIR / "ddri_cluster_profile_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_cluster_size_chart(train_features: pd.DataFrame) -> None:
    cluster_count_df = (
        train_features["cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="station_count")
    )
    cluster_count_df["cluster_label"] = cluster_count_df["cluster"].map(lambda x: f"군집 {x}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.barplot(
        data=cluster_count_df,
        x="cluster_label",
        y="station_count",
        hue="cluster_label",
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.set_title("군집별 대여소 수")
    ax.set_xlabel("군집")
    ax.set_ylabel("대여소 수")
    for patch, value in zip(ax.patches, cluster_count_df["station_count"]):
        ax.annotate(f"{value}", (patch.get_x() + patch.get_width() / 2, patch.get_height()), ha="center", va="bottom")
    plt.tight_layout()
    fig.savefig(OUTPUT_IMG_DIR / "ddri_cluster_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_weekday_hour_heatmap(train_df: pd.DataFrame) -> None:
    plot_df = train_df.copy()
    plot_df["weekday"] = plot_df["대여일시"].dt.dayofweek
    plot_df["hour"] = plot_df["대여일시"].dt.hour
    weekday_labels = {
        0: "월",
        1: "화",
        2: "수",
        3: "목",
        4: "금",
        5: "토",
        6: "일",
    }
    heatmap_df = (
        plot_df.groupby(["weekday", "hour"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(7), fill_value=0)
        .rename(index=weekday_labels)
    )

    fig, ax = plt.subplots(figsize=(14, 4.5))
    sns.heatmap(heatmap_df, cmap="Blues", ax=ax)
    ax.set_title("학습 구간 요일-시간대별 대여 건수 히트맵")
    ax.set_xlabel("시간")
    ax.set_ylabel("요일")
    plt.tight_layout()
    fig.savefig(OUTPUT_IMG_DIR / "ddri_weekday_hour_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_monthly_rental_trend(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    frames = []
    for label, df in [("학습(2023~2024)", train_df), ("테스트(2025)", test_df)]:
        monthly = (
            df.assign(month=df["대여일시"].dt.to_period("M").astype(str))
            .groupby("month")
            .size()
            .reset_index(name="rental_count")
        )
        monthly["dataset"] = label
        frames.append(monthly)
    plot_df = pd.concat(frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.lineplot(data=plot_df, x="month", y="rental_count", hue="dataset", marker="o", ax=ax)
    ax.set_title("월별 대여 건수 추이")
    ax.set_xlabel("월")
    ax.set_ylabel("대여 건수")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="구간")
    plt.tight_layout()
    fig.savefig(OUTPUT_IMG_DIR / "ddri_monthly_rental_trend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_coverage_summary(master_ids, train_ids, test_ids):
    summary = pd.DataFrame(
        [
            {"group": "common_station_master", "station_count": len(master_ids)},
            {"group": "train_feature_station_count", "station_count": len(train_ids)},
            {"group": "test_feature_station_count", "station_count": len(test_ids)},
            {"group": "missing_in_train", "station_count": len(master_ids - train_ids)},
            {"group": "missing_in_test", "station_count": len(master_ids - test_ids)},
            {"group": "missing_in_both", "station_count": len(master_ids - train_ids - test_ids)},
        ]
    )
    summary.to_csv(OUTPUT_DATA_DIR / "ddri_station_coverage_summary.csv", index=False)

    flags = []
    for station_id in sorted(master_ids):
        flags.append(
            {
                "station_id": station_id,
                "in_common_master": 1,
                "in_train_features": int(station_id in train_ids),
                "in_test_features": int(station_id in test_ids),
            }
        )
    pd.DataFrame(flags).to_csv(OUTPUT_DATA_DIR / "ddri_station_coverage_flags.csv", index=False)


def main():
    station_id_sets, common_station_ids = load_station_frames()
    common_station_id_set = set(common_station_ids)
    rental_groups = file_groups()

    train_df_2023, stats_2023 = load_clean_group(
        rental_groups[2023], station_id_sets[2023], common_station_id_set, "train_2023"
    )
    train_df_2024, stats_2024 = load_clean_group(
        rental_groups[2024], station_id_sets[2024], common_station_id_set, "train_2024"
    )
    test_df, stats_2025 = load_clean_group(
        rental_groups[2025], station_id_sets[2025], common_station_id_set, "test_2025"
    )

    train_df = pd.concat([train_df_2023, train_df_2024], ignore_index=True)
    clean_stats = pd.concat([stats_2023, stats_2024, stats_2025], ignore_index=True)
    clean_stats.to_csv(PREP_DATA_DIR / "ddri_cleaning_log.csv", index=False)

    train_features = build_station_features(train_df)
    test_features = build_station_features(test_df)
    train_features.to_csv(OUTPUT_DATA_DIR / "ddri_station_cluster_features_train_2023_2024.csv", index=False)
    test_features.to_csv(OUTPUT_DATA_DIR / "ddri_station_cluster_features_test_2025.csv", index=False)

    final_k, k_search_df, cluster_summary = save_kmeans_outputs(train_features)
    labeled_train_features = pd.read_csv(OUTPUT_DATA_DIR / "ddri_station_cluster_features_train_with_labels.csv")
    save_feature_correlation_heatmap(train_features)
    save_cluster_profile_heatmap(cluster_summary)
    save_cluster_size_chart(labeled_train_features)
    save_weekday_hour_heatmap(train_df)
    save_monthly_rental_trend(train_df, test_df)
    save_coverage_summary(
        set(common_station_ids),
        set(train_features["station_id"]),
        set(test_features["station_id"]),
    )

    print("\n=== CLEANING SUMMARY ===")
    print(
        clean_stats.groupby("group_name")[
            ["rows_before", "rows_after", "dropped_missing", "dropped_nonpositive", "dropped_noncommon_rent", "dropped_outside_gangnam_return"]
        ].sum().to_string()
    )
    print("\n=== K SEARCH ===")
    print(k_search_df.to_string(index=False))
    print("\n=== CLUSTER SUMMARY ===")
    print(cluster_summary.to_string())
    print(f"\nFINAL_K={final_k}")


if __name__ == "__main__":
    main()
