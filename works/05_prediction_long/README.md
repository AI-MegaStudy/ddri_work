# DDRI Representative-Station Prediction Dataset

## 개요

이 폴더에는 대표 대여소 15개만 추려서 만든 시간대별 예측용 long-format 데이터셋이 저장되어 있다.

- 학습 데이터: `2023년 + 2024년`
- 테스트 데이터: `2025년`
- 단위 행: `station_id + date + hour`
- 타깃 변수: `rental_count`

생성 파일:

- `data/ddri_prediction_long_train_2023_2024.csv`
- `data/ddri_prediction_long_test_2025.csv`
- `build_prediction_long_dataset.ipynb`
- `03_ddri_station_hour_model_comparison.ipynb`
- `05_ddri_team_cluster_modeling_protocol.md`
- `06_ddri_cluster_modeling_template.ipynb`
- `output/data/ddri_station_hour_model_metrics.csv`
- `output/data/ddri_station_hour_lightgbm_feature_importance.csv`
- `output/images/ddri_station_hour_lightgbm_feature_importance.png`

## 폴더 운영 원칙

혼선을 줄이기 위해 원본 데이터와 생성 산출물을 분리한다.

- `data/`
  - 원본 학습/테스트 long-format CSV만 유지
- `output/data/`
  - 모델 성능표, 오류 요약표, 비교표 등 생성 CSV
- `output/images/`
  - 모델 비교 차트, 오차 차트, 피처 중요도 등 생성 이미지

## 팀 실험 공통 규약

대표 대여소 15개를 팀원별 군집 담당 실험으로 나눠 진행할 때는 아래 문서를 공통 기준으로 사용한다.

- `05_ddri_team_cluster_modeling_protocol.md`
- `06_ddri_cluster_modeling_template.ipynb`

이 문서에는 아래가 고정되어 있다.

- 공통 입력 데이터 경로
- 전처리 규칙
- lag/rolling 피처 생성 규칙
- `2023 train / 2024 validation / 2025 test` 정책
- 필수 모델 비교 순서
- `RMSE`, `MAE`, `R²` 공통 평가 방식

노트북 템플릿 사용 순서:

- `06_ddri_cluster_modeling_template.ipynb`를 복사
- 파일명을 담당 군집에 맞게 변경
- `TARGET_STATION_GROUP` 값을 수정
- 프로토콜 문서 순서대로 실험 수행

## 대표 대여소 구성

이번 데이터셋은 전체 공통 대여소가 아니라, 클러스터 유형별 대표 대여소 15개만 사용했다.

### 업무/상업 혼합형

- `4908` SB타워 앞
- `2328` 르네상스 호텔 사거리 역삼지하보도 7번출구 앞
- `4902` 구역삼세무서 교차로

### 아침 도착 업무 집중형

- `2377` 수서역 5번출구
- `2323` 주식회사 오뚜기 정문 앞
- `2348` 포스코사거리(기업은행)

### 주거 도착형

- `2312` 청담역 13번 출구 앞
- `2354` 청담역 2번출구
- `4917` 일원에코파크 주차장

### 생활권 혼합형

- `2321` 학여울역 사거리
- `2320` 도곡역 대치지구대 방향
- `3616` 역삼중학교 앞(체육관 방향)

### 외곽 주거형

- `3643` 더시그넘하우스 앞
- `2359` 국립국악중,고교 정문 맞은편
- `2392` 구룡산 입구 (구룡산 서울둘레길 입구)

최종 대상 대여소 수:

- `15개`

## 사용한 원천 데이터

### 1. 클러스터 결과 데이터

대여소 메타정보와 클러스터 라벨을 가져오기 위해 아래 파일을 사용했다.

- `works/01_clustering/08_integrated/final/results/second_clustering_results/data/ddri_second_cluster_train_with_labels.csv`
- `works/01_clustering/08_integrated/final/results/second_clustering_results/data/ddri_second_cluster_test_with_labels.csv`

사용 컬럼:

- `station_id`
- `station_name`
- `mapped_dong_code`
- `cluster`

주의:

- 학습 데이터는 학습용 클러스터 파일의 `cluster` 사용
- 테스트 데이터는 테스트용 클러스터 파일의 `cluster` 사용

즉, 같은 대여소라도 연도 구간에 따라 클러스터 라벨이 다를 수 있다.

### 2. 따릉이 이용정보

시간대별 대여 건수를 만들기 위해 월별 이용정보 CSV를 사용했다.

- `3조 공유폴더/2023 강남구 따릉이 이용정보/*.csv`
- `3조 공유폴더/2024 강남구 따릉이 이용정보/*.csv`
- `3조 공유폴더/2025 강남구 따릉이 이용정보/*.csv`

사용 컬럼:

- `대여일시`
- `대여 대여소번호`

### 3. 날씨 데이터

시간 단위 날씨 피처를 붙이기 위해 아래 파일을 사용했다.

- `3조 공유폴더/2023-2024년 강남구 날씨데이터(00시-24시)/gangnam_weather_1year_2023.csv`
- `3조 공유폴더/2023-2024년 강남구 날씨데이터(00시-24시)/gangnam_weather_1year_2024.csv`
- `3조 공유폴더/2024년 강남구 날씨데이터(00시-24시)/gangnam_weather_1year_2025.csv`

사용 컬럼:

- `datetime`
- `temperature`
- `humidity`
- `precipitation`
- `wind_speed`

### 4. 공휴일 정보

`holiday` 컬럼은 2023년, 2024년, 2025년 대한민국 공휴일 날짜를 코드에 고정하여 생성했다.

포함 기준:

- 법정공휴일
- 대체공휴일
- 2024년 4월 10일 국회의원 선거일
- 2024년 10월 1일 임시공휴일

## 최종 데이터셋 구조

최종 컬럼은 아래 15개이다.

- `station_id`
- `station_name`
- `station_group`
- `date`
- `hour`
- `rental_count`
- `cluster`
- `mapped_dong_code`
- `weekday`
- `month`
- `holiday`
- `temperature`
- `humidity`
- `precipitation`
- `wind_speed`

각 컬럼 의미:

- `station_id`: 대여소 ID
- `station_name`: 대여소명
- `station_group`: 대표 유형 그룹명
- `date`: 날짜 문자열 (`YYYY-MM-DD`)
- `hour`: 시간 (`0~23`)
- `rental_count`: 해당 대여소에서 해당 시각에 대여된 건수
- `cluster`: 해당 구간 클러스터 라벨
- `mapped_dong_code`: 매핑된 행정동 코드
- `weekday`: 요일 (`월요일=0`, `일요일=6`)
- `month`: 월 (`1~12`)
- `holiday`: 공휴일 여부 (`1=공휴일`, `0=비공휴일`)
- `temperature`, `humidity`, `precipitation`, `wind_speed`: 시간 단위 날씨 변수

## 전처리 과정

### 1. 대표 대여소 필터링

지정된 15개 `station_id`만 남겼다.

### 2. 시간대별 대여량 집계

따릉이 이용정보에서 다음 방식으로 대여량을 집계했다.

- `대여일시`를 시간 단위로 내림
- `대여 대여소번호`를 `station_id`로 변환
- `station_id + datetime` 기준으로 대여 건수 집계

즉, `rental_count`는 반납이 아니라 대여 시각 기준 집계값이다.

### 3. 전체 시간축 생성

실제 대여가 있었던 시각만 남기지 않고, 전체 시간축을 먼저 만들었다.

- 학습용: `2023-01-01 00:00:00 ~ 2024-12-31 23:00:00`
- 테스트용: `2025-01-01 00:00:00 ~ 2025-12-31 23:00:00`

그 뒤 `station_id x datetime` 전체 조합을 생성하고, 집계되지 않은 시각은 `rental_count = 0`으로 채웠다.

### 4. 파생 시간 변수 생성

`datetime`에서 아래 파생 변수를 만들었다.

- `date`
- `hour`
- `weekday`
- `month`
- `holiday`

### 5. 날씨 데이터 병합

시간 단위 `datetime` 기준으로 날씨 데이터를 병합했다.

병합 키:

- `datetime`

날씨는 모든 대여소에 동일하게 붙는다.

## 윤년 및 결측 처리

2024년은 윤년이므로 전체 시간축을 `366일`, 즉 `8,784시간`으로 생성했다.

중요:

- `2024-02-29`는 윤년으로 인해 정상 포함되는 날짜다.
- 과거에는 `2024-01-01` 24시간이 날씨 원천 파일에서 누락되어 있었지만, 이후 `Open-Meteo Archive API` 기준으로 2024년 파일을 다시 받아 정정했다.

현재 기준 처리 상태:

- 학습용 2024 날씨 파일은 `2024-01-01 00:00:00 ~ 2024-12-31 23:00:00` 전체를 포함한다.
- 따라서 현재 `works/05_prediction_long/data/ddri_prediction_long_train_2023_2024.csv`에는 `2024-01-01` 날씨 결측이 없다.
- 테스트용 날씨도 결측 없이 유지된다.

대여량은 전체 시간축을 생성한 뒤 관측되지 않은 시각을 `0`으로 채웠기 때문에 `rental_count` 결측은 없다.

## 최종 데이터 크기

### 학습 데이터

- 파일: `data/ddri_prediction_long_train_2023_2024.csv`
- 크기: `263,160행 x 15컬럼`

계산:

- 대표 대여소 `15개`
- 2023년 `8,760시간`
- 2024년 `8,784시간`
- 총 `17,544시간`
- `15 x 17,544 = 263,160`

### 테스트 데이터

- 파일: `data/ddri_prediction_long_test_2025.csv`
- 크기: `131,400행 x 15컬럼`

계산:

- 대표 대여소 `15개`
- 2025년 `8,760시간`
- `15 x 8,760 = 131,400`

## 재생성 방법

아래 노트북을 실행하면 데이터셋을 다시 만들 수 있다.

- `build_prediction_long_dataset.ipynb`

노트북은 다음 순서로 동작한다.

1. 대표 대여소 목록 정의
2. 학습/테스트 클러스터 파일에서 메타정보 추출
3. 월별 따릉이 이용정보에서 시간별 대여량 집계
4. 전체 시간축 생성 후 `rental_count` 0 채움
5. 시간 파생 변수 생성
6. 날씨 및 공휴일 정보 병합
7. 학습용 / 테스트용 CSV 저장

## 모델 비교 결과

대표 대여소 15개 `station-hour` 데이터셋 기준으로 아래 모델을 비교했다.

- `LightGBM_RMSE`
- `LightGBM_Poisson`
- `CatBoost_RMSE`
- `CatBoost_Poisson`

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

결과 파일:

- `output/data/ddri_station_hour_model_metrics.csv`
- `output/data/ddri_station_hour_lightgbm_feature_importance.csv`
- `output/images/ddri_station_hour_lightgbm_feature_importance.png`
- `04_ddri_station_hour_evidence_charts.ipynb`
- `output/data/ddri_station_hour_hourly_actual_vs_predicted.csv`
- `output/data/ddri_station_hour_station_group_error_summary.csv`
- `output/data/ddri_station_hour_station_error_summary.csv`
- `output/images/ddri_station_hour_model_comparison_test_rmse.png`
- `output/images/ddri_station_hour_hourly_actual_vs_predicted.png`
- `output/images/ddri_station_hour_station_group_mae.png`
- `output/images/ddri_station_hour_residual_distribution.png`
- `output/images/ddri_station_hour_actual_vs_predicted_scatter.png`

현재 요약:

- validation 기준으로는 `LightGBM_Poisson`이 근소하게 좋음
- `2025` 테스트 기준으로는 `LightGBM_RMSE`가 가장 안정적임
- 현재 1차 기본 후보 모델은 `LightGBM_RMSE`로 본다

## 근거 차트 보강

대표 대여소 실험은 초입 기준선 단계였기 때문에 처음에는 피처 중요도 차트만 있었다.

이후 설명 근거를 보강하기 위해 아래 노트북과 차트를 추가했다.

- 노트북:
  - `04_ddri_station_hour_evidence_charts.ipynb`

### 생성 차트

- 모델별 `2025` 테스트 RMSE 비교
- 시간대별 평균 실제값 vs 예측값
- `station_group`별 MAE 비교
- residual 분포 히스토그램
- 실제값 vs 예측값 scatter

### 생성 표

- 시간대별 실제/예측 평균:
  - `output/data/ddri_station_hour_hourly_actual_vs_predicted.csv`
- 대표 그룹별 오류 요약:
  - `output/data/ddri_station_hour_station_group_error_summary.csv`
- 대표 대여소별 오류 요약:
  - `output/data/ddri_station_hour_station_error_summary.csv`

### 현재 해석

- 대표 그룹 중 `아침 도착 업무 집중형`의 오차가 가장 크다.
- 상위 오류 대여소에는 `2377 수서역 5번출구`, `2348 포스코사거리(기업은행)`, `4917 일원에코파크 주차장`이 포함된다.
- 시간대 평균 기준으로는 출근 시간대와 일부 피크 구간에서 오차가 상대적으로 커진다.
