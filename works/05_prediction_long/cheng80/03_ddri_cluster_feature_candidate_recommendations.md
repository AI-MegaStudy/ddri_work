# DDRI 군집별 추가 피처 후보 추천(Cluster Feature Candidate Recommendations)

작성일: 2026-03-14  
목적: 대표 대여소 15개 `cluster00 ~ cluster04` 1차 실험 결과를 바탕으로, 현재 기본 피처 외에 어떤 추가 피처를 붙여볼지 군집별로 추천한다.

## 1. 현재 기본 피처

현재 1차 실험에서 이미 사용한 기본 피처는 아래와 같다.

- 시간 피처
  - `hour`
  - `weekday`
  - `month`
  - `holiday`
- 날씨 피처
  - `temperature`
  - `humidity`
  - `precipitation`
  - `wind_speed`
- 식별/공간 피처
  - `station_id`
  - `cluster`
  - `mapped_dong_code`
- 시계열 피처
  - `lag_1h`
  - `lag_24h`
  - `lag_168h`
  - `rolling_mean_24h`
  - `rolling_std_24h`
  - `rolling_mean_168h`
  - `rolling_std_168h`

따라서 이번 문서의 추천은 `날씨와 공휴일 외에`, 그리고 위 기본 피처에 아직 포함되지 않은 후보를 중심으로 본다.

## 2. 추가 피처 후보를 고르는 원칙

### 2.1 바로 실험 가능한 후보

- 현재 프로젝트 안에 이미 원천이 있는 피처
- 2023, 2024, 2025를 모두 맞출 수 있는 피처
- 미래 정보 누수 없이 붙일 수 있는 피처

### 2.2 보류해야 하는 후보

- 2025 데이터가 없어 train/test 스키마를 맞추기 어려운 피처
- 같은 시점 목표값을 직접 암시하는 누수성 피처
- 현재 대표 대여소 실험 범위에 비해 구축 비용이 너무 큰 피처

## 3. 지금 바로 추천하는 공통 추가 피처

## 3.1 시간 구조 강화 피처

### `hour_sin`, `hour_cos`

- 목적: 시간대의 원형(cyclical) 구조 반영
- 이유: `23시`와 `0시`가 멀리 떨어진 숫자로 처리되는 문제를 완화
- 추천 우선순위: 높음

### `is_commute_hour`

- 예: `07~09`, `17~19`
- 목적: 출퇴근 피크 구간을 직접 표시
- 추천 우선순위: 높음

### `is_lunch_hour`

- 예: `11~13`
- 목적: 생활·상권 혼합형의 점심 수요 반영
- 추천 우선순위: 중간

### `is_night_hour`

- 예: `22~05`
- 목적: 외곽 주거형, 주거 도착형의 야간 패턴 보완
- 추천 우선순위: 중간

## 3.2 달력 구조 강화 피처

### `is_weekend`

- 이미 `weekday`는 있지만, 주말 여부를 별도 이진값으로 두면 트리 모델에서 해석이 쉬움
- 추천 우선순위: 높음

### `is_holiday_eve`

- 공휴일 전날 여부
- 귀가/이동 패턴이 바뀔 수 있음
- 추천 우선순위: 중간

### `is_after_holiday`

- 연휴 직후/공휴일 다음날 여부
- 업무형, 아침 집중형에서 특히 유효 가능성
- 추천 우선순위: 중간

### `season`

- 봄/여름/가을/겨울 범주형
- `month`보다 해석이 쉬운 보조 변수
- 추천 우선순위: 중간

## 3.3 날씨 파생 피처

### `is_rainy`

- 강수량이 0보다 큰지 여부
- 약한 비와 강한 비를 먼저 단순하게 분리
- 추천 우선순위: 높음

### `heavy_rain_flag`

- 강수량 일정 기준 이상
- 출근형/생활권형에서 민감할 수 있음
- 추천 우선순위: 중간

### `temperature_bin`

- 한파/저온/보통/고온 범주
- 비선형 온도 반응 보완
- 추천 우선순위: 중간

### `discomfort_proxy`

- 기온 + 습도 기반 단순 체감 더위/불쾌 지표
- 여름철 생활권/외곽형에 유효 가능성
- 추천 우선순위: 낮음~중간

## 3.4 시계열 구조 강화 피처

### `lag_48h`

- 이틀 전 동일 시각
- 연휴나 주말 직후 완충 효과를 볼 수 있음
- 추천 우선순위: 중간

### `rolling_mean_6h`

- 최근 6시간 이동평균
- 단기 추세 반영
- 추천 우선순위: 높음

### `rolling_mean_12h`

- 반나절 추세 반영
- 추천 우선순위: 중간

### `diff_from_lag_24h`

- 현재와 하루 전 동일 시각 차이를 예측 직전 입력으로 쓰는 방식은 누수 위험이 있으므로 직접 차이보다는 과거 기반 차이만 허용해야 함
- 예: `lag_1h - lag_24h`
- 추천 우선순위: 중간

## 3.5 정적 공간/입지 피처

아래는 이미 군집화 파트에서 확보한 정적 피처다. `station_id`만으로 간접 반영하는 대신, 직접 붙이면 군집 간 설명력이 좋아질 수 있다.

### 교통 접근성

- `subway_distance_m`
- `bus_stop_count_300m`

추천 우선순위: 높음

### 환경/지형

- `station_elevation_m`
- `elevation_diff_nearest_subway_m`
- `nearest_park_area_sqm`
- `distance_naturepark_m`
- `distance_river_boundary_m`

추천 우선순위: 중간

### 생활/상권 POI

- `restaurant_count_300m`
- `cafe_count_300m`
- `convenience_store_count_300m`
- `pharmacy_count_300m`
- `food_retail_count_1000m`
- `fitness_count_500m`

추천 우선순위: 중간~높음

주의:

- 이 피처들은 정적이므로 train/test 모두 동일하게 붙일 수 있어야 한다.
- 대표 대여소 15개 기준으로 먼저 붙여보는 실험은 충분히 가치가 있다.

## 4. 군집별 추천 후보

## 4.1 `cluster00` 업무/상업 혼합형

1차 결과:

- validation RMSE `0.8987`
- test RMSE `0.8113`
- 반복 패턴은 잡히지만 설명력은 중간 수준

추천 후보:

- `is_commute_hour`
- `is_after_holiday`
- `subway_distance_m`
- `bus_stop_count_300m`
- `restaurant_count_300m`
- `cafe_count_300m`
- `rolling_mean_6h`

이유:

- 업무/상업 혼합형은 출퇴근과 상권 흐름이 같이 섞여 있을 가능성이 높다.
- 교통 접근성과 점심/업무 수요 관련 상권 피처가 같이 작동할 수 있다.

## 4.2 `cluster01` 아침 도착 업무 집중형

1차 결과:

- validation RMSE `1.5660`
- test RMSE `1.3462`
- 현재 가장 어려운 군집

추천 후보:

- `is_commute_hour`
- `is_after_holiday`
- `is_weekend`
- `subway_distance_m`
- `bus_stop_count_300m`
- `rolling_mean_6h`
- `lag_48h`

이유:

- 이 군집은 출근 피크가 매우 강해 보인다.
- 평일/휴일 전후, 교통 접근성, 직전 단기 추세를 더 직접적으로 알려주는 피처가 필요하다.

우선순위:

- 가장 먼저 2차 실험해야 하는 군집

## 4.3 `cluster02` 주거 도착형

1차 결과:

- validation RMSE `0.8263`
- test RMSE `0.8088`
- test `R²`가 가장 높음

추천 후보:

- `is_night_hour`
- `is_weekend`
- `is_holiday_eve`
- `heavy_rain_flag`
- `station_elevation_m`
- `bus_stop_count_300m`

이유:

- 귀가 패턴과 주말/휴일 이동, 저녁/야간 날씨 영향이 더 중요할 수 있다.
- 이미 비교적 안정적이므로 2차 실험 우선순위는 높지 않다.

## 4.4 `cluster03` 생활권 혼합형

1차 결과:

- validation RMSE `0.7898`
- test RMSE `0.6901`
- 절대 오차는 낮지만 `R²`가 낮음

추천 후보:

- `is_lunch_hour`
- `is_weekend`
- `restaurant_count_300m`
- `cafe_count_300m`
- `convenience_store_count_300m`
- `rolling_mean_6h`
- `temperature_bin`

이유:

- 생활권/상권 혼합형은 점심, 상권 밀집도, 생활편의 접근성과 연관될 가능성이 높다.
- 지금은 RMSE는 낮지만 설명력은 약해서, 입지형 피처 추가 효과를 보기 좋은 군집이다.

우선순위:

- `cluster01` 다음 2순위 후보

## 4.5 `cluster04` 외곽 주거형

1차 결과:

- validation RMSE `0.7852`
- test RMSE `0.7160`
- 비교적 안정적

추천 후보:

- `is_night_hour`
- `is_weekend`
- `station_elevation_m`
- `elevation_diff_nearest_subway_m`
- `distance_naturepark_m`
- `distance_river_boundary_m`
- `bus_stop_count_300m`

이유:

- 외곽형은 교통 접근성 부족, 지형, 생활권 외곽성의 영향을 더 받을 수 있다.
- 환경/지형형 정적 피처가 다른 군집보다 더 잘 듣는지 확인할 가치가 있다.

## 5. 대기자료(미세먼지 등) 사용 판단

확보 자료:

- 경로: `3조 공유폴더/2023-2024_대기자료`
- 확인 컬럼:
  - `PM10`
  - `PM2.5`
  - `오 존`
  - `이산화질소`
  - `일산화탄소`
  - `아황산가스`

판단:

- `2023~2024` 자료는 존재하므로 validation 단계 보조 분석에는 사용할 수 있다.
- `2025` 자료도 `3조 공유폴더/2025_대기자료` 경로에 `1월~10월`까지는 확보되어 있다.
- 하지만 `2025-11 ~ 2025-12`가 비어 있어 test 기간 전체를 덮지 못한다.
- 따라서 현재 공식 `2025 test` 기준 모델에는 넣지 않는 것이 맞다.

권장 사용 방식:

- 공식 1차/2차 비교 모델에는 제외
- 별도 참고 실험으로만 사용
  - 예: `2023 train / 2024 validation` 한정 실험
  - 예: `2025-01 ~ 2025-10` 부분구간 보조 실험
  - 예: 대기질 피처 추가 시 validation 개선폭 확인

즉, 대기자료는 `연관성 탐색용`으로는 좋지만, 현재 프로젝트의 공식 최종 모델 피처로 채택하기는 이르다.

## 6. 2차 실험 추천 순서

1. `cluster01`

- 가장 어려운 군집
- 추천 추가 피처:
  - `is_commute_hour`
  - `is_after_holiday`
  - `subway_distance_m`
  - `bus_stop_count_300m`
  - `rolling_mean_6h`

2. `cluster03`

- 설명력 개선 여지가 큼
- 추천 추가 피처:
  - `is_lunch_hour`
  - `restaurant_count_300m`
  - `cafe_count_300m`
  - `convenience_store_count_300m`
  - `rolling_mean_6h`

3. `cluster00`

- 업무/상업 복합 구조 확인용
- 추천 추가 피처:
  - `is_commute_hour`
  - `restaurant_count_300m`
  - `cafe_count_300m`
  - `subway_distance_m`

## 7. 현재 최종 추천

지금 바로 붙여볼 1차 추가 후보 묶음은 아래가 가장 실무적이다.

### 공통 1차 추가 후보

- `is_weekend`
- `is_commute_hour`
- `is_rainy`
- `rolling_mean_6h`
- `hour_sin`
- `hour_cos`

### 군집별 선택 추가 후보

- `cluster01`: `is_after_holiday`, `subway_distance_m`, `bus_stop_count_300m`
- `cluster03`: `is_lunch_hour`, `restaurant_count_300m`, `cafe_count_300m`
- `cluster04`: `station_elevation_m`, `distance_naturepark_m`

즉, 다음 2차 실험은 `cluster01`부터 시작하고, 그 다음 `cluster03`로 가는 것이 가장 합리적이다.
