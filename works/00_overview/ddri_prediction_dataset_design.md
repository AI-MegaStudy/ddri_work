# Ddri Prediction Dataset Design

작성일: 2026-03-11
목적: 예측용 학습 테이블의 결합 키, 외부 데이터 결합 방식, feature 구조를 고정한다.

## 1. 기준 요약

- target: `rental_count`
- grain: `station-day`
- 학습 구간: 2023-01-01 ~ 2024-12-31
- 테스트 구간: 2025-01-01 ~ 2025-12-31
- 기준 대여소: 2023~2025 공통 대여소 기반

## 2. 기본 학습 테이블 구조

기본 테이블 1행은 아래 의미를 가진다.

- 특정 `station_id`
- 특정 `date`
- 그 날짜의 실제 `rental_count`
- 그 날짜에 대응하는 날씨/공휴일/환경/과거 수요 feature

즉, 최종 학습 테이블의 기본 키는 아래와 같다.

- 결합 키: `station_id + date`

## 3. 내부 데이터 결합 설계

### 3.1 대여 이력 원천 로그

원천 컬럼:
- `대여일시`
- `대여 대여소번호`

처리 방식:
- `대여일시`에서 날짜 추출
- `대여 대여소번호`를 `station_id`로 표준화
- `station_id + date` 기준으로 건수 집계

생성 target:
- `rental_count`: 해당 대여소의 해당 날짜 총 대여 건수

### 3.2 대여소 마스터

결합 키:
- `station_id`

역할:
- 대여소명
- 좌표
- 자치구
- 정적 환경 feature 결합의 기준 테이블

## 4. 외부 데이터 결합 설계

### 4.1 날씨 데이터

원천 파일:
- `gangnam_weather_1year_2023.csv`
- `gangnam_weather_1year_2024.csv`

원천 컬럼:
- `datetime`
- `temperature`
- `humidity`
- `precipitation`
- `wind_speed`

결합 전략:
- 시간 단위 날씨를 `date` 기준 일 단위로 집계
- 대여량 target이 일 단위이므로 날씨도 일 단위 feature로 맞춘다

권장 집계:
- `temperature_mean`
- `temperature_min`
- `temperature_max`
- `humidity_mean`
- `wind_speed_mean`
- `precipitation_sum`
- `is_rainy_day`

결합 키:
- `date`

### 4.2 공휴일 데이터

결합 전략:
- 날짜별 공휴일 여부 테이블 생성

권장 컬럼:
- `is_holiday`
- `is_weekend`
- `day_of_week`
- `is_bridge_holiday` 후보

결합 키:
- `date`

현재 생성 산출물:

- 원천 API 결과:
  - `works/calendar/data/ddri_holiday_api_raw_2023_2025.csv`
- 일 단위 캘린더 테이블:
  - `works/calendar/data/ddri_calendar_daily_2023_2025.csv`

현재 사용 권장 컬럼:

- `date`
- `day_of_week`
- `is_weekend`
- `holiday_name`
- `holiday_count`
- `is_holiday`
- `is_business_holiday`

### 4.3 환경 데이터

환경 데이터는 날짜에 따라 자주 바뀌지 않으므로 `station_id` 기준 정적 feature로 본다.

권장 데이터:
- 공원 거리
- 지하철역 거리
- 버스정류장 수
- 군집 label

결합 키:
- `station_id`

권장 feature:
- `park_distance`
- `subway_distance`
- `bus_stop_count_300m`
- `cluster_label`

### 4.4 생활인구 데이터

현재 상태:
- 행정동 단위 데이터
- 대여소 단위와 바로 결합되지 않음

결합 전 필요 작업:
- 대여소 좌표 → 행정동 매핑
- 날짜/시간 집계 기준 정리

현재 판단:
- 1차 베이스라인에서는 제외 또는 보류
- 2차 고도화 feature로 관리

## 5. feature 그룹 정의

### 5.1 시간 feature

- `year`
- `month`
- `day`
- `day_of_week`
- `is_weekend`
- `is_holiday`

### 5.2 과거 수요 feature

- `rental_count_lag1`
- `rental_count_lag7`
- `rolling_mean_7`
- `rolling_std_7`

설명:
- 일 단위 target이므로 lag와 rolling도 일 단위 기준으로 생성

### 5.3 날씨 feature

- `temperature_mean`
- `temperature_min`
- `temperature_max`
- `humidity_mean`
- `wind_speed_mean`
- `precipitation_sum`
- `is_rainy_day`

### 5.4 정적 환경 feature

- `park_distance`
- `subway_distance`
- `bus_stop_count_300m`
- `cluster_label`

### 5.5 운영 해석용 이동/재고 보조 지표

- `same_station_return_count`
- `same_station_return_ratio`
- `return_count`
- `net_flow`

설명:
- `same_station_return`은 같은 날 같은 대여소에서 빌리고 같은 대여소로 반납한 건수를 의미한다.
- 이 값은 대여소의 하루 말 재고를 크게 바꾸지 않을 수 있지만, 실제 운행 중 자전거 점유는 발생하므로 이상치로 제거하지 않는다.
- 따라서 제거 대상이 아니라 `적정 보유 대수`, `순유출입`, `재고 변동성` 해석용 보조 지표로 관리하는 것이 더 적절하다.

## 6. 군집 label 포함 여부

1차 권장안:
- `cluster_label`을 예측 feature에 포함

이유:
- 군집은 대여소의 전반적 수요 성격을 요약하는 feature 역할을 할 수 있음
- 특히 정적 환경 feature가 부족할 때 보조 설명력을 줄 수 있음

주의:
- 군집 label은 학습 데이터 기준으로 생성된 값을 사용해야 함
- 테스트에도 동일 기준으로 매핑 가능해야 함

## 7. 1차 베이스라인 권장 스키마

핵심 컬럼 예시:

- `station_id`
- `date`
- `rental_count`
- `day_of_week`
- `is_weekend`
- `is_holiday`
- `temperature_mean`
- `precipitation_sum`
- `humidity_mean`
- `wind_speed_mean`
- `rental_count_lag1`
- `rental_count_lag7`
- `rolling_mean_7`
- `park_distance`
- `subway_distance`
- `bus_stop_count_300m`
- `cluster_label`
- `same_station_return_ratio`
- `net_flow`

## 8. 현재 보류 항목

- 생활인구 결합
- 대기질 데이터 결합
- 상업지역/POI 밀도 결합
- `return_count` 보조 target 활용 여부
- `same_station_return` 기반 적정 보유 대수 해석 지표 확장

## 9. 바로 다음 구현 순서

1. 일 단위 `station-day` target 테이블 생성
2. 날씨 일 단위 집계 테이블 생성
3. 공휴일 테이블 생성
4. 환경 feature 결합용 정적 테이블 생성
5. lag/rolling feature 생성
6. 학습용 최종 스키마 CSV 샘플 생성
