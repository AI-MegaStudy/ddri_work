from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path("/Users/cheng80/Desktop/ddri_work")
BASE_DIR = ROOT / "works" / "01_clustering" / "08_integrated"
INPUT_CSV = BASE_DIR / "intermediate" / "return_time_district" / "ddri_return_time_features_2025.csv"
OUTPUT_DIR = BASE_DIR / "intermediate" / "return_time_district"


WINDOW_CONFIG = [
    ("return_7_10_count", "07-10시 반납 상위 대여소(2025)", "ddri_return_top_2025_7_10.png"),
    ("return_11_14_count", "11-14시 반납 상위 대여소(2025)", "ddri_return_top_2025_11_14.png"),
    ("return_17_20_count", "17-20시 반납 상위 대여소(2025)", "ddri_return_top_2025_17_20.png"),
]


def main() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False

    df = pd.read_csv(INPUT_CSV)

    for column, title, filename in WINDOW_CONFIG:
        top_df = (
            df[["station_id", "station_name", column]]
            .sort_values(column, ascending=False)
            .head(10)
            .copy()
        )
        top_df["label"] = top_df["station_name"].str.slice(0, 20)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        sns.barplot(
            data=top_df,
            x=column,
            y="label",
            hue="label",
            palette="YlOrRd",
            dodge=False,
            legend=False,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("반납 건수")
        ax.set_ylabel("대여소")

        for idx, value in enumerate(top_df[column]):
            ax.text(value + max(top_df[column]) * 0.01, idx, f"{int(value):,}", va="center", fontsize=10)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
