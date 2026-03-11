from pathlib import Path

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import koreanize_matplotlib


BASE_DIR = Path("/Users/cheng80/Desktop/ddri_work")
DATA_DIR = BASE_DIR / "works" / "clustering" / "data"
IMG_DIR = BASE_DIR / "works" / "clustering" / "images"

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def build_cleaning_charts():
    agg = pd.read_csv(DATA_DIR / "ddri_cleaning_summary_by_group.csv")

    group_labels = {
        "train_2023": "학습 2023",
        "train_2024": "학습 2024",
        "test_2025": "테스트 2025",
    }
    stage_labels = {
        "rows_before": "전처리 전",
        "rows_after": "전처리 후",
    }
    reason_labels = {
        "dropped_missing": "결측치 제거",
        "dropped_nonpositive": "0 이하 시간/거리 제거",
        "dropped_noncommon_rent": "공통 기준 밖 대여소 제거",
        "dropped_outside_gangnam_return": "강남구 외 반납 제거",
    }

    plot_df = agg.melt(
        id_vars="group_name",
        value_vars=["rows_before", "rows_after"],
        var_name="stage",
        value_name="rows",
    )
    plot_df["group_label"] = plot_df["group_name"].map(group_labels)
    plot_df["stage_label"] = plot_df["stage"].map(stage_labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot_df, x="group_label", y="rows", hue="stage_label", ax=ax)
    ax.set_title("전처리 전후 행 수 비교")
    ax.set_xlabel("데이터 구간")
    ax.set_ylabel("행 수")
    ax.legend(title="구간")
    plt.tight_layout()
    fig.savefig(IMG_DIR / "ddri_cleaning_before_after.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    reason_df = agg.melt(
        id_vars="group_name",
        value_vars=[
            "dropped_missing",
            "dropped_nonpositive",
            "dropped_noncommon_rent",
            "dropped_outside_gangnam_return",
        ],
        var_name="reason",
        value_name="count",
    )
    reason_df["group_label"] = reason_df["group_name"].map(group_labels)
    reason_df["reason_label"] = reason_df["reason"].map(reason_labels)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=reason_df, x="group_label", y="count", hue="reason_label", ax=ax)
    ax.set_title("전처리 제거 사유별 건수")
    ax.set_xlabel("데이터 구간")
    ax.set_ylabel("제거 건수")
    ax.legend(title="제거 사유")
    plt.tight_layout()
    fig.savefig(IMG_DIR / "ddri_cleaning_drop_reasons.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    build_cleaning_charts()
    print("saved report charts")


if __name__ == "__main__":
    main()
