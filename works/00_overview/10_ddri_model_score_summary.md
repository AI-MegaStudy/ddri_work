# Ddri 모델 점수 요약(Model Score Summary)

작성일: 2026-03-14  
목적: 현재까지 진행한 예측 모델 실험의 점수와 해석을 한눈에 볼 수 있도록 표로 정리한다.

## 1. 요약

현재 예측 파트는 아래 3개 트랙으로 나뉜다.

- 대여소-일 단위(`station-day`) 기준선
- 대표 15개 대여소 시간 단위(`station-hour`) 핵심 검증 트랙
- 전체 161개 스테이션 시간 단위(`station-hour`) 재배치 확장 트랙

현재 결론은 세 트랙 모두 `LightGBM` 계열이 가장 우세하다는 것이다.

## 2. 모델 점수 표(Model Score Table)

| 트랙(Track) | 모델(Model) | 검증 2024 RMSE(Validation 2024 RMSE) | 검증 2024 MAE(Validation 2024 MAE) | 검증 2024 R²(Validation 2024 R²) | 테스트 2025 RMSE(Test 2025 RMSE) | 테스트 2025 MAE(Test 2025 MAE) | 테스트 2025 R²(Test 2025 R²) | 해석(Interpretation) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 대여소-일 단위(`station-day`) | `LinearRegression` | 6.8732 | 4.9723 | 0.7464 | 6.3530 | 4.6139 | 0.7330 | 선형 기준선. 해석은 쉽지만 비선형 패턴 대응은 약함 |
| 대여소-일 단위(`station-day`) | `LightGBM` | 6.0146 | 4.3282 | 0.8058 | 5.3106 | 3.8473 | 0.8134 | 현재 `station-day` 최우수 모델 |
| 대표 15개 시간 단위(`station-hour`) | `LightGBM_RMSE` | 1.0066 | 0.6121 | 0.5703 | 0.8927 | 0.5455 | 0.5608 | 최종 2025 테스트 기준 가장 안정적. 현재 핵심 검증 트랙 기본 모델 |
| 대표 15개 시간 단위(`station-hour`) | `LightGBM_Poisson` | 1.0003 | 0.6074 | 0.5757 | 0.8967 | 0.5402 | 0.5568 | validation은 근소 우세하지만 test는 RMSE 목적함수보다 약간 밀림 |
| 대표 15개 시간 단위(`station-hour`) | `CatBoost_RMSE` | 1.0088 | 0.6139 | 0.5685 | 0.9007 | 0.5488 | 0.5528 | 나쁘지 않지만 `LightGBM_RMSE`보다 약간 뒤처짐 |
| 대표 15개 시간 단위(`station-hour`) | `CatBoost_Poisson` | 1.0081 | 0.6095 | 0.5691 | 0.9049 | 0.5460 | 0.5487 | 현재 실험에서는 우세하지 않음 |
| 전체 161개 시간 단위(`station-hour`) | `LightGBM_RMSE_Full` | 0.9735 | 0.6234 | 0.4463 | 0.8624 | 0.5594 | 0.4369 | 전체 스테이션 확장 트랙 기본 모델 |
| 전체 161개 시간 단위(`station-hour`) | `LightGBM_Poisson_Full` | 0.9827 | 0.6260 | 0.4359 | 0.8704 | 0.5613 | 0.4262 | full-data에서도 RMSE 목적함수보다 약함 |

## 3. 현재 우세 모델(Current Best Model)

| 트랙(Track) | 현재 우세 모델(Current Best Model) | 해석(Interpretation) |
|---|---|---|
| 대여소-일 단위(`station-day`) | `LightGBM` | 일 단위 기준선은 이미 충분히 우세 |
| 대표 15개 시간 단위(`station-hour`) | `LightGBM_RMSE` | 현재 핵심 검증 트랙의 기본 모델 |
| 전체 161개 시간 단위(`station-hour`) | `LightGBM_RMSE_Full` | 재배치 확장용 전체 트랙의 기본 모델 |

## 4. 추가 해석

### 4.1 `station-day`

- `LinearRegression`보다 `LightGBM`이 RMSE, MAE, R² 모두 우세하다.
- 이는 날씨, 과거 수요, 스테이션 고유성의 비선형 효과를 `LightGBM`이 더 잘 반영했기 때문으로 해석할 수 있다.

### 4.2 대표 15개 `station-hour`

- validation만 보면 `LightGBM_Poisson`이 아주 근소하게 좋다.
- 하지만 최종 `2025` 테스트 기준으로는 `LightGBM_RMSE`가 가장 안정적이다.
- 따라서 현재 핵심 검증 트랙의 기본 모델은 `LightGBM_RMSE`로 유지한다.

### 4.3 전체 161개 `station-hour`

- 전체 스테이션 확장에서는 `LightGBM_RMSE_Full`이 `LightGBM_Poisson_Full`보다 validation/test 모두 우세했다.
- 다만 설명력(R²)은 대표 15개보다 낮다. 이는 전체 스테이션으로 범위를 넓히면서 패턴 다양성이 커졌기 때문으로 볼 수 있다.

## 5. 관련 파일(Related Files)

- `works/03_prediction/02_data/ddri_station_day_baseline_model_metrics.csv`
- `works/05_prediction_long/output/data/ddri_station_hour_model_metrics.csv`
- `works/06_prediction_long_full/output/data/ddri_station_hour_full_model_comparison_metrics.csv`
