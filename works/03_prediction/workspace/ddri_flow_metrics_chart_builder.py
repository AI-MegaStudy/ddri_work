from pathlib import Path

import koreanize_matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path("/Users/cheng80/Desktop/ddri_work")
DATA_DIR = BASE_DIR / "works" / "prediction" / "data"
IMG_DIR = BASE_DIR / "works" / "prediction" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="AppleGothic")


def load_labeled_frames():
    frames = []
    for dataset_name, path in [
        ("학습(2023~2024)", DATA_DIR / "ddri_station_day_train_baseline_dataset.csv"),
        ("테스트(2025)", DATA_DIR / "ddri_station_day_test_baseline_dataset.csv"),
    ]:
        df = pd.read_csv(path)
        df["dataset"] = dataset_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_summary_chart(df: pd.DataFrame) -> None:
    summary = (
        df.groupby("dataset")
        .agg(
            평균_대여량=("rental_count", "mean"),
            평균_반납량=("return_count", "mean"),
            평균_self_return_비율=("same_station_return_ratio", "mean"),
            평균_순유출입=("net_flow", "mean"),
        )
        .reset_index()
    )
    long_summary = summary.melt(id_vars="dataset", var_name="지표", value_name="값")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)

    sns.barplot(data=long_summary, x="지표", y="값", hue="dataset", ax=axes[0])
    axes[0].set_title("학습/테스트 운영 보조 지표 평균 비교")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("평균값")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend(title="데이터셋")

    flow_df = df.copy()
    flow_df["net_flow_clipped"] = flow_df["net_flow"].clip(-20, 20)
    sns.histplot(
        data=flow_df,
        x="net_flow_clipped",
        hue="dataset",
        bins=41,
        stat="density",
        common_norm=False,
        alpha=0.45,
        ax=axes[1],
    )
    axes[1].set_title("순유출입(net flow) 분포 비교")
    axes[1].set_xlabel("순유출입 (20 이하/이상 구간 절단)")
    axes[1].set_ylabel("밀도")

    plt.tight_layout()
    fig.savefig(IMG_DIR / "ddri_flow_metrics_summary.png", bbox_inches="tight")
    plt.close(fig)


def build_ratio_boxplot(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df["same_station_return_ratio_capped"] = plot_df["same_station_return_ratio"].clip(upper=0.6)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    sns.boxplot(
        data=plot_df,
        x="dataset",
        y="same_station_return_ratio_capped",
        hue="dataset",
        palette=["#4e79a7", "#e15759"],
        legend=False,
        ax=ax,
    )
    ax.set_title("동일 대여소 반납 비율 분포")
    ax.set_xlabel("")
    ax.set_ylabel("same_station_return_ratio (0.6 상한 표시)")
    plt.tight_layout()
    fig.savefig(IMG_DIR / "ddri_same_station_return_ratio_boxplot.png", bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_labeled_frames()
    build_summary_chart(df)
    build_ratio_boxplot(df)


if __name__ == "__main__":
    main()
