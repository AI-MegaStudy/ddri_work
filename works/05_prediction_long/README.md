# DDRI Long-Format Prediction Dataset

## 개요

이 폴더에는 강남구 따릉이 대여소의 시간대별 수요 예측을 위한 long-format 데이터셋이 저장되어 있다.

- 학습 데이터: `2023년 + 2024년`
- 테스트 데이터: `2025년`
- 단위 행: `station_id + date + hour`
- 타깃 변수: `rental_count`

생성 파일:

- `data/ddri_prediction_long_train_2023_2024.csv`
- `data/ddri_prediction_long_test_2025.csv`
- `build_prediction_long_dataset.ipynb`

## 사용한 원천 데이터

### 1. 클러스터 결과 데이터

대여소 메타정보와 클러스터 라벨을 가져오기 위해 아래 파일을 사용했다.

- `works/01_clustering/08_integrated/final/results/second_clustering_results/data/ddri_second_cluster_train_with_labels.csv`
- `works/01_clustering/08_integrated/final/results/second_clustering_results/data/ddri_second_cluster_test_with_labels.csv`

사용 컬럼:

- `station_id`
- `mapped_dong_code`
- `cluster`

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

`holiday` 컬럼은 2023년, 2024년, 2025년 대한민국 공휴일 날짜를 코드 내에 직접 고정하여 생성했다.

포함 기준:

- 법정공휴일
- 대체공휴일
- 2024년 4월 10일 국회의원 선거일
- 2024년 10월 1일 임시공휴일

## 공통 대여소 기준

예측 데이터셋은 학습/테스트 모두 동일한 대여소 집합을 사용하도록 구성했다.

기준:

- 학습용 클러스터 결과의 `station_id`
- 테스트용 클러스터 결과의 `station_id`
- 두 집합의 교집합만 사용

최종 공통 대여소 수:

- `161개`

이 기준을 사용한 이유:

- 학습과 테스트의 입력 구조를 동일하게 유지하기 위해서
- 특정 연도에만 존재하는 대여소를 제외해 시계열 예측용 패널 데이터를 안정적으로 만들기 위해서

## 생성한 데이터셋 구조

최종 컬럼은 아래 13개이다.

- `station_id`
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
- `date`: 날짜 문자열 (`YYYY-MM-DD`)
- `hour`: 시간 (`0~23`)
- `rental_count`: 해당 대여소에서 해당 시각에 대여된 건수
- `cluster`: 2차 클러스터링 결과 라벨
- `mapped_dong_code`: 매핑된 행정동 코드
- `weekday`: 요일 (`월요일=0`, `일요일=6`)
- `month`: 월 (`1~12`)
- `holiday`: 공휴일 여부 (`1=공휴일`, `0=비공휴일`)
- `temperature`, `humidity`, `precipitation`, `wind_speed`: 시간 단위 날씨 변수

## 전처리 과정

### 1. 공통 대여소 필터링

학습용과 테스트용 클러스터 결과에서 `station_id` 교집합만 남겼다.

### 2. 시간대별 대여량 집계

따릉이 이용정보에서 다음 방식으로 대여량을 집계했다.

- `대여일시`를 시간 단위로 내림(`floor('h')`)
- `대여 대여소번호`를 `station_id`로 변환
- `station_id + datetime` 기준으로 대여 건수 집계

즉, `rental_count`는 반납이 아니라 대여 시각 기준 집계값이다.

### 3. 전체 시간축 생성

실제 대여가 있었던 시각만 남기지 않고, 전체 시간축을 먼저 만들었다.

- 학습용: `2023-01-01 00:00:00 ~ 2024-12-31 23:00:00`
- 테스트용: `2025-01-01 00:00:00 ~ 2025-12-31 23:00:00`

그 뒤 `station_id x datetime` 전체 조합을 생성하고,
집계되지 않은 시각은 `rental_count = 0`으로 채웠다.

이 방식은 수요가 없었던 시간도 명시적으로 포함하기 때문에 예측 모델 학습에 적합하다.

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

날씨는 모든 대여소에 동일하게 붙는다. 즉, 같은 시간대라면 모든 대여소가 동일한 날씨 값을 가진다.

## 윤년 처리

2024년은 윤년이므로 전체 시간축을 `366일`, 즉 `8,784시간`으로 생성했다.

따라서 아래 시간도 모두 정상 포함된다.

- `2024-02-29 00:00:00 ~ 2024-02-29 23:00:00`

즉, 윤년은 별도 축소 없이 달력 기준 그대로 유지했다.

## 결측 처리

### 날씨 결측

2024년 날씨 원천 파일은 `8,760행`으로 확인되었고, 아래 구간 24시간이 누락되어 있다.

- `2024-01-01 00:00:00 ~ 2024-01-01 23:00:00`

처리 원칙:

- 시간축은 유지
- 날씨 결측은 임의 보간하지 않음
- 해당 구간의 `temperature`, `humidity`, `precipitation`, `wind_speed`는 `NaN` 유지

이렇게 한 이유:

- 원천 데이터 결측을 숨기지 않기 위해서
- 공휴일인 2024년 1월 1일을 임의 값으로 대체하면 왜곡 가능성이 있기 때문

결측 건수:

- 학습용 날씨 결측: `3,864건`
- 계산식: `161개 공통 대여소 x 24시간`
- 테스트용 날씨 결측: `0건`

### 대여량 결측

대여량은 전체 시간축을 생성한 뒤 관측되지 않은 시각을 `0`으로 채웠기 때문에 `rental_count` 결측은 없다.

## 최종 데이터 크기

### 학습 데이터

- 파일: `data/ddri_prediction_long_train_2023_2024.csv`
- 크기: `2,824,584행 x 13컬럼`

계산:

- 공통 대여소 `161개`
- 2023년 `8,760시간`
- 2024년 `8,784시간`
- 총 `17,544시간`
- `161 x 17,544 = 2,824,584`

### 테스트 데이터

- 파일: `data/ddri_prediction_long_test_2025.csv`
- 크기: `1,410,360행 x 13컬럼`

계산:

- 공통 대여소 `161개`
- 2025년 `8,760시간`
- `161 x 8,760 = 1,410,360`

## 재생성 방법

아래 노트북을 실행하면 데이터셋을 다시 만들 수 있다.

- `build_prediction_long_dataset.ipynb`

노트북은 다음 순서로 동작한다.

1. 클러스터 결과 파일에서 공통 대여소 추출
2. 월별 따릉이 이용정보에서 시간별 대여량 집계
3. 전체 시간축 생성 후 `rental_count` 0 채움
4. 시간 파생 변수 생성
5. 날씨 및 공휴일 정보 병합
6. 학습용 / 테스트용 CSV 저장
