# cheng80 군집별 대리 실험 트랙

목적: 팀원별 군집 실험이 지연되거나 누락될 경우를 대비해, `cheng80`가 대표 대여소 15개 기준 `5개 군집`을 각각 별도 폴더에서 대리 실행할 수 있게 준비한 작업 경로다.

## 폴더 구조

- `cluster00`
  - 군집: `업무/상업 혼합형`
- `cluster01`
  - 군집: `아침 도착 업무 집중형`
- `cluster02`
  - 군집: `주거 도착형`
- `cluster03`
  - 군집: `생활권 혼합형`
- `cluster04`
  - 군집: `외곽 주거형`

각 폴더에는 아래 파일이 들어 있다.

- `01_cluster_modeling.ipynb`

이 노트북은 공통 템플릿을 복사한 것이며, `TARGET_STATION_GROUP` 값이 각 군집에 맞게 미리 설정되어 있다.

## 사용 기준

- 공통 규약: `../05_ddri_team_cluster_modeling_protocol.md`
- 공통 템플릿 원본: `../06_ddri_cluster_modeling_template.ipynb`
- 공통 데이터: `3조 공유폴더/대표대여소_예측데이터_15개/raw_data/`

취합/후속 문서:

- `summary_aggregation/output/data/ddri_cluster_model_metrics_collection_template.csv`
- `01_ddri_cluster_result_collection.md`
- `02_ddri_second_round_experiment_criteria.md`
- `03_ddri_cluster_feature_candidate_recommendations.md`
- `summary_aggregation/output/data/ddri_cluster_second_round_comparison_summary.csv`
- `04_ddri_second_round_result_summary.md`
- `summary_aggregation/output/data/cluster01_third_round_progression_summary.csv`
- `05_ddri_cluster01_third_round_summary.md`
- `06_ddri_cluster_experiment_overall_summary.md`
- `07_ddri_cluster_final_recommendation.md`

루트 집계 전용 폴더:

- `summary_aggregation`
  - 루트 집계 CSV 재생성 노트북 보관
  - CSV 산출물은 `summary_aggregation/output/data/`에 저장

별도 분석 폴더:

- `rep15_error_analysis`
  - 대표 15개 스테이션 오류 우선순위와 시간대 패턴 해석 전용 폴더
  - 노트북/문서: `rep15_error_analysis/`
  - CSV 산출물: `rep15_error_analysis/output/data/`

2차 실험용 통합 피처모음:

- `3조 공유폴더/대표대여소_예측데이터_15개/second_round_data/ddri_prediction_long_train_2023_2024_second_round_feature_collection.csv`
- `3조 공유폴더/대표대여소_예측데이터_15개/second_round_data/ddri_prediction_long_test_2025_second_round_feature_collection.csv`

설명:

- 기본 long-format 데이터에 정적 교통/환경/POI 피처와 공통 파생 후보 피처를 추가한 2차 실험용 정본이다.
- 대기자료는 현재 공식 실험에서 제외했으므로 포함하지 않는다.
- `cheng80` 개인 실험도 이 공용 `second_round_data/`를 사용한다.

## 실행 원칙

- 각 폴더는 해당 군집만 담당한다.
- 결과 해석과 저장은 공통 프로토콜을 그대로 따른다.
- 팀원 결과와 혼동하지 않도록, 이 경로는 `cheng80` 개인 대리 실험 트랙으로만 사용한다.
- 군집별 학습 결과와 별도 목적 분석은 루트에 섞어 두지 않고, 목적별 서브폴더를 만들어 관리한다.
