# summary_aggregation

목적: `cheng80` 군집별 대리 실험의 루트 집계 산출물을 재생성하는 작업 폴더다.

## 노트북

- `11_ddri_cluster_result_collection.ipynb`
  - 5개 군집 1차 실험 결과를 취합해 `ddri_cluster_model_metrics_collection_template.csv`를 생성
- `12_ddri_cluster_second_round_comparison.ipynb`
  - 1차와 2차 결과를 비교해 `ddri_cluster_second_round_comparison_summary.csv`를 생성
- `13_ddri_cluster01_third_round_progression.ipynb`
  - `cluster01`의 1차/2차/3차 진행표를 만들어 `cluster01_third_round_progression_summary.csv`를 생성

## 출력 경로

- `output/data/ddri_cluster_model_metrics_collection_template.csv`
- `output/data/ddri_cluster_second_round_comparison_summary.csv`
- `output/data/cluster01_third_round_progression_summary.csv`
