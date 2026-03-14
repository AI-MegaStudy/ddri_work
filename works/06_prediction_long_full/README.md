# DDRI Full-Station Hourly Prediction Experiments

## 개요

이 폴더는 `161개 공통 스테이션 station-hour` 실험을 별도로 관리하기 위한 작업 공간이다.

- 목적: 실제 서비스 확장용 시간대 수요 예측 실험 관리
- 관리 대상: 노트북, 성능 지표, 시각화, 실험 로그성 산출물
- 원본 데이터 위치: `3조 공유폴더/군집별 데이터_전체 스테이션/full_data/`

즉, 데이터 원본과 실험 산출물을 분리한다.

- 원본 CSV/재생성 노트북: 공유폴더 `full_data`
- Git 관리 실험물: `works/06_prediction_long_full`

## 폴더 운영 원칙

이 폴더에는 원본 데이터가 없다.

- 원본 데이터: 공유폴더 `full_data`
- 노트북: `works/06_prediction_long_full`
- 생성 CSV/이미지: `works/06_prediction_long_full/output/`

## 사용할 원본 데이터

- `3조 공유폴더/군집별 데이터_전체 스테이션/full_data/ddri_prediction_long_train_2023_2024.csv`
- `3조 공유폴더/군집별 데이터_전체 스테이션/full_data/ddri_prediction_long_test_2025.csv`
- `3조 공유폴더/군집별 데이터_전체 스테이션/full_data/build_prediction_long_full_dataset.ipynb`

날씨 원천 관련 참고:

- `2024` 날씨 파일은 `Open-Meteo Archive API` 기준으로 다시 받아 정정한 버전을 사용한다.
- 따라서 현재 원본 학습 CSV에는 과거에 문제였던 `2024-01-01` 날씨 24시간 결측이 없다.

## 관리 원칙

- 대표 대여소 15개 실험은 `works/05_prediction_long`에서만 관리한다.
- 전체 161개 스테이션 실험은 이 폴더에서만 관리한다.
- validation 전략은 `Train=2023`, `Validation=2024`, `Final Train=2023+2024`, `Test=2025`로 고정한다.
- 앱 구현 전까지의 분석과 ML 작업은 모두 노트북에 남긴다.

## 예정 산출물

- 전체 스테이션 `station-hour` 모델링 노트북
- 모델 성능 비교표
- 피처 중요도
- 시간대/스테이션별 오류 분석 차트

## 현재 산출물

- `01_ddri_station_hour_full_baseline.ipynb`
- `02_ddri_station_hour_full_model_comparison.ipynb`
- `output/data/ddri_station_hour_full_model_metrics.csv`
- `output/data/ddri_station_hour_full_lightgbm_feature_importance.csv`
- `output/data/ddri_station_hour_full_station_error_summary.csv`
- `output/data/ddri_station_hour_full_model_comparison_metrics.csv`
- `output/data/ddri_station_hour_full_model_comparison_lightgbm_feature_importance.csv`
- `output/images/ddri_station_hour_full_lightgbm_feature_importance.png`
- `output/images/ddri_station_hour_full_model_comparison_lightgbm_feature_importance.png`

## 1차 baseline 결과

모델:

- `LightGBM_RMSE_Full`

검증 전략:

- Train: `2023`
- Validation: `2024`
- Final Train: `2023 + 2024`
- Test: `2025`

시계열 파생 피처 해석:

- `lag_1h`: 1시간 전 동일 스테이션 대여량
- `lag_24h`: 24시간 전, 즉 하루 전 동일 시각 대여량
- `lag_168h`: 168시간 전, 즉 1주 전 동일 요일·동일 시각 대여량
- `rolling_mean_24h`: 최근 24시간 이동평균
- `rolling_mean_168h`: 최근 168시간, 즉 1주 기준 이동평균
- `rolling_std_24h`: 최근 24시간 이동표준편차

성능:

- validation `RMSE=0.9735`, `MAE=0.6234`, `R²=0.4463`
- test `RMSE=0.8624`, `MAE=0.5594`, `R²=0.4369`

중요 피처 상위:

- `station_id`
- `hour`
- `weekday`
- `lag_168h`
- `lag_1h`
- `temperature`
- `lag_24h`

## 해석

- 전체 161개 스테이션 기준에서도 `station_id`, `hour`, 과거 수요 lag, 날씨가 핵심 설명 변수로 확인되었다.
- 대표 대여소 15개 실험보다 설명력은 다소 낮아졌지만, 전체 서비스 범위로 확장한 baseline으로는 의미 있는 기준선을 확보했다.
- `Poisson 목적함수` 비교 결과도 `RMSE 목적함수`보다 우세하지 않았다.
- 따라서 현재 전체 161개 스테이션 기본 모델은 `LightGBM_RMSE_Full`로 유지한다.
- `CatBoost`는 전체 데이터 전체 반복 학습 시 계산비용이 커서, 필요하면 축소 실험이나 하이퍼파라미터 경량화 조건으로 별도 분리한다.

## objective 비교 결과

노트북:

- `02_ddri_station_hour_full_model_comparison.ipynb`

비교 결과:

- `LightGBM_RMSE_Full`
  - validation `RMSE=0.9735`, `MAE=0.6234`, `R²=0.4463`
  - test `RMSE=0.8624`, `MAE=0.5594`, `R²=0.4369`
- `LightGBM_Poisson_Full`
  - validation `RMSE=0.9827`, `MAE=0.6260`, `R²=0.4359`
  - test `RMSE=0.8704`, `MAE=0.5613`, `R²=0.4262`

해석:

- 전체 161개 기준에서는 `RMSE 목적함수`가 `Poisson 목적함수`보다 검증/테스트 모두 우세했다.
- 따라서 전체 스테이션 트랙의 기본 목적함수는 `RMSE`로 고정한다.
