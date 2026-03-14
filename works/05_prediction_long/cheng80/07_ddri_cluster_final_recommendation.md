# DDRI 군집별 최종 권장안(Final Cluster Recommendation)

작성일: 2026-03-14  
목적: 1차 baseline, 2차 추가 피처 실험, `cluster01` 3차 심화 실험 결과를 바탕으로 현재 시점의 군집별 권장 모델과 적용 수준을 정리한다.

## 1. 결론 요약

현재 결과만으로는 `전 군집 완성형 커스텀 모델`이 확정됐다고 보기는 어렵다.  
다만 `군집별 커스텀 피처/모델 전략이 유효하다`는 근거는 확보되었다.

현재 권장 해석:

- 공통 운영 baseline은 `LightGBM_RMSE`
- 군집별 커스텀 고도화는 `cluster01`부터 우선 적용
- `cluster00`, `cluster02`, `cluster03`, `cluster04`는 2차 피처 보강안까지를 `군집별 커스텀 초안`으로 본다

## 2. 군집별 최종 권장표

| 군집 코드 | 군집명 | 현재 권장 모델 | 대표 확인 점수 | 권장 추가 피처 방향 | 현재 판단 | 적용 권장 수준 |
|---|---|---|---|---|---|---|
| `cluster00` | 업무/상업 혼합형 | `LightGBM_RMSE` | RMSE `0.8085`, MAE `0.5403`, R² `0.3259` | 출퇴근, 상권, 교통 접근성 | 2차에서 아주 작은 개선 확인 | `공통 baseline 유지 + 선택적 피처 보강` |
| `cluster01` | 아침 도착 업무 집중형 | `LightGBM_Poisson` | RMSE `1.3189`, MAE `0.7745`, R² `0.6543` | 출근 피크, 공휴일 다음날, 교통 접근성, 단기 추세 | 2차와 3차에서 가장 분명한 개선 확인 | `군집별 커스텀 모델 우선 적용 후보` |
| `cluster02` | 주거 도착형 | `LightGBM_RMSE` | RMSE `0.8059`, MAE `0.5053`, R² `0.5022` | 야간, 주말, 공휴일 전날, 주거형 입지 | 2차에서 작지만 안정적 개선 | `공통 baseline + 경량 커스텀 피처` |
| `cluster03` | 생활권 혼합형 | `LightGBM_RMSE` | RMSE `0.6882`, MAE `0.4882`, R² `0.1848` | 점심, 상권, 생활편의 POI, 단기 추세 | RMSE는 낮지만 R²가 낮아 구조 보완 필요 | `추가 검토 대상` |
| `cluster04` | 외곽 주거형 | `LightGBM_RMSE` | RMSE `0.7145`, MAE `0.4427`, R² `0.3811` | 야간, 외곽성, 지형, 교통 접근성 | 2차 개선폭이 매우 작음 | `공통 baseline 우선` |

## 3. 군집별 상세 판단

### 3.1 `cluster00` 업무/상업 혼합형

- 1차 test RMSE: `0.8113`
- 1차 test MAE: `0.5439`
- 2차 test RMSE: `0.8085`
- 2차 test MAE: `0.5403`
- 해석:
  - 출퇴근과 상권 피처를 보강했을 때 방향성 있는 개선은 있었다.
  - 다만 개선폭은 아주 작아, 별도 전용 모델을 강하게 주장할 수준은 아니다.
- 현재 권장:
  - `LightGBM_RMSE` 유지
  - `is_commute_hour`, `subway_distance_m`, `bus_stop_count_300m`, `restaurant_count_300m`, `cafe_count_300m`, `rolling_mean_6h` 정도의 보강안을 선택적으로 적용

### 3.2 `cluster01` 아침 도착 업무 집중형

- 1차 test RMSE: `1.3462`
- 1차 test MAE: `0.8042`
- 2차 test RMSE: `1.3324`
- 2차 test MAE: `0.7868`
- 3차 test RMSE: `1.3189`
- 3차 test MAE: `0.7745`
- 해석:
  - 다섯 군집 중 가장 어려운 군집이지만, 동시에 추가 피처와 목적함수 변경에 가장 잘 반응했다.
  - 3차에서 `LightGBM_Poisson`이 최종 우세 모델로 바뀌었다.
- 현재 권장:
  - `LightGBM_Poisson`을 군집별 우선 커스텀 모델 후보로 채택
  - 핵심 추가 피처:
    - `is_commute_hour`
    - `is_after_holiday`
    - `subway_distance_m`
    - `bus_stop_count_300m`
    - `rolling_mean_6h`
    - `rolling_mean_12h`
    - `rolling_std_6h`
    - `rolling_std_12h`
    - `lag_48h`
    - `commute_morning_flag`
    - `commute_evening_flag`

### 3.3 `cluster02` 주거 도착형

- 1차 test RMSE: `0.8088`
- 1차 test MAE: `0.5059`
- 2차 test RMSE: `0.8059`
- 2차 test MAE: `0.5053`
- 해석:
  - 원래도 비교적 안정적이었고, 2차 보강 후에도 작지만 일관된 개선이 있었다.
  - 과도한 복잡화보다 현재 구조 유지가 더 실용적이다.
- 현재 권장:
  - `LightGBM_RMSE` 유지
  - `is_night_hour`, `is_weekend`, `is_holiday_eve`, `heavy_rain_flag`, `station_elevation_m`, `bus_stop_count_300m` 정도의 경량 보강 유지

### 3.4 `cluster03` 생활권 혼합형

- 1차 test RMSE: `0.6901`
- 1차 test MAE: `0.4928`
- 2차 test RMSE: `0.6882`
- 2차 test MAE: `0.4882`
- 해석:
  - RMSE 자체는 낮지만 R²가 낮다.
  - 즉 평균 오차는 작아도 패턴 설명력이 충분하다고 보긴 어렵다.
  - 생활권/상권 관련 정적 피처를 더 정교화할 여지가 있다.
- 현재 권장:
  - 현재는 `LightGBM_RMSE` 유지
  - `restaurant_count_300m`, `cafe_count_300m`, `convenience_store_count_300m`, `is_lunch_hour`, `rolling_mean_6h` 중심으로 유지
  - 후속 보완 후보 군집으로 분류

### 3.5 `cluster04` 외곽 주거형

- 1차 test RMSE: `0.7160`
- 1차 test MAE: `0.4425`
- 2차 test RMSE: `0.7145`
- 2차 test MAE: `0.4427`
- 해석:
  - 방향은 개선이지만 폭이 매우 작다.
  - 현재 지형/외곽성 피처가 큰 차이를 만들었다고 해석하긴 어렵다.
- 현재 권장:
  - `LightGBM_RMSE` 공통 baseline 우선
  - 필요할 때만 `station_elevation_m`, `elevation_diff_nearest_subway_m`, `distance_naturepark_m`, `distance_river_boundary_m`, `bus_stop_count_300m` 보강

## 4. 최종 적용 전략

현재 시점의 가장 현실적인 적용 전략은 아래와 같다.

1. 공통 기본 모델은 `LightGBM_RMSE`로 유지한다.
2. `cluster01`은 별도 군집 커스텀 모델(`LightGBM_Poisson`) 적용 후보로 분리한다.
3. `cluster00`, `cluster02`, `cluster03`, `cluster04`는 2차 피처 보강안을 유지하되, 보고서에서는 `군집별 커스텀 초안`으로 표현한다.
4. 전 군집 완성형 커스텀 모델 체계라고 주장하기보다는, `군집별 최적화 가능성 확인 + cluster01 심화 사례 확보`로 정리한다.

## 5. 예측력은 어떤 점수로 확인하나

현재 군집별 권장안 판단에서는 아래 3개 지표를 함께 본다.

- `RMSE`
  - 최우선 선택 지표
  - 큰 오차를 더 민감하게 반영하므로 모델 선택 기준으로 가장 먼저 본다
- `MAE`
  - 평균적으로 몇 대 차이 나는지 보여주는 직관 지표
  - 운영자/사용자 설명 문구에 쓰기 좋다
- `R²`
  - 패턴 설명력 보조 지표
  - RMSE는 낮지만 구조 설명력이 낮은 군집을 따로 읽을 때 사용한다

현재 해석 원칙:

1. 모델 우열은 먼저 `RMSE`로 본다
2. 평균 체감 오차는 `MAE`로 읽는다
3. 패턴 설명력은 `R²`로 보조 해석한다

## 6. 보고서용 표현 권장안

- 권장 표현:
  - `군집별 커스텀 모델 가능성 확인`
  - `군집별 맞춤 피처 보강의 유효성 검증`
  - `cluster01 심화 최적화 사례 확보`

- 아직 과한 표현:
  - `전 군집 최적 모델 확정`
  - `완성형 군집별 운영 모델 구축 완료`

## 7. 바로 참고할 문서

- `01_ddri_cluster_result_collection.md`
  - 재생성 노트북: `summary_aggregation/11_ddri_cluster_result_collection.ipynb`
  - 집계 CSV: `summary_aggregation/output/data/ddri_cluster_model_metrics_collection_template.csv`
- `04_ddri_second_round_result_summary.md`
  - 재생성 노트북: `summary_aggregation/12_ddri_cluster_second_round_comparison.ipynb`
  - 집계 CSV: `summary_aggregation/output/data/ddri_cluster_second_round_comparison_summary.csv`
- `05_ddri_cluster01_third_round_summary.md`
  - 재생성 노트북: `summary_aggregation/13_ddri_cluster01_third_round_progression.ipynb`
  - 집계 CSV: `summary_aggregation/output/data/cluster01_third_round_progression_summary.csv`
- `06_ddri_cluster_experiment_overall_summary.md`
