# 반납 시간대 기반 지구판단 자료 정리

## 목적

- 원천:
  - `3조 공유폴더/2023 강남구 따릉이 이용정보`
  - `3조 공유폴더/2024 강남구 따릉이 이용정보`
  - `3조 공유폴더/2025 강남구 따릉이 이용정보`
  - `3조 공유폴더/강남구 대여소 정보 (2023~2025)`
- 목표:
  - 각 연도 강남구 이용정보에서 `반납대여소번호` 기준 시간대별 반납 집중 패턴을 집계한다.
  - 대여소 좌표와 매칭해 지도에 분포를 표시한다.
  - 2차 군집화에서 `지구판단` 근거로 쓸 수 있는 피처를 만든다.

## 적용한 전처리

- 필수 컬럼 결측 제거:
  - `대여일시`
  - `반납일시`
  - `대여 대여소번호`
  - `반납대여소번호`
  - `이용시간(분)`
  - `이용거리(M)`
- `이용시간(분) > 0`
- `이용거리(M) > 0`
- `동일 대여소 반납 && 이용시간(분) <= 5` 제거

## 시간대 정의

- `07~10시`
- `11~14시`
- `17~20시`

시간대는 `반납일시.hour` 기준으로 집계했다.

## 생성 파일

### 연도별 근거 자료

- `ddri_return_time_features_2023.csv`
- `ddri_return_time_features_2024.csv`
- `ddri_return_time_features_2025.csv`

주요 컬럼:

- `station_id`
- `station_name`
- `latitude`
- `longitude`
- `total_return_count`
- `return_7_10_count`
- `return_11_14_count`
- `return_17_20_count`
- `arrival_7_10_ratio`
- `arrival_11_14_ratio`
- `arrival_17_20_ratio`
- `district_hypothesis`

### 지도 근거 자료

- `ddri_return_map_2025_7_10.html`
- `ddri_return_map_2025_11_14.html`
- `ddri_return_map_2025_17_20.html`

### 2차 군집화용 피처

- `ddri_second_cluster_return_features_train_2023_2024.csv`
- `ddri_second_cluster_return_features_test_2025.csv`
- `ddri_second_cluster_merged_features_train_2023_2024.csv`
- `ddri_second_cluster_merged_features_test_2025.csv`

설명:

- `train_2023_2024`는 기존 군집화 학습 대여소 `164개` 기준으로 2023년과 2024년 반납 집계를 합산했다.
- `test_2025`는 기존 테스트 대여소 `162개` 기준으로 2025년 반납 집계를 사용했다.
- `merged_features`는 기존 최종 세트에 반납 시간대 피처를 실제로 붙인 버전이다.

## 기존 2차 군집화 세트와 병합하는 방법

기존 세트:

- `works/01_clustering/08_integrated/ddri_final_district_clustering_features_train_2023_2024.csv`
- `works/01_clustering/08_integrated/ddri_final_district_clustering_features_test_2025.csv`

반납 시간대 세트:

- `ddri_second_cluster_return_features_train_2023_2024.csv`
- `ddri_second_cluster_return_features_test_2025.csv`

병합 기준:

- `station_id` inner join
- 기존 세트에서는 아래 컬럼을 유지:
  - `mapped_dong_code`
  - `mapped_dong_name`
  - `morning_net_inflow`
  - `evening_net_inflow`
  - `subway_distance_m`
  - `bus_stop_count_300m`
  - `life_pop_7_10_mean`
  - `life_pop_11_14_mean`
  - `life_pop_17_20_mean`
- 반납 시간대 세트에서는 아래 컬럼을 추가:
  - `total_return_count`
  - `return_7_10_count`
  - `return_11_14_count`
  - `return_17_20_count`
  - `arrival_7_10_ratio`
  - `arrival_11_14_ratio`
  - `arrival_17_20_ratio`
  - `dominant_ratio`
  - `district_hypothesis`

실무 권장:

- 군집화 입력의 주축은 `arrival_*_ratio` 3개와 기존 `subway_distance_m`, `bus_stop_count_300m`으로 둔다.
- `return_*_count`와 `district_hypothesis`는 군집 해석 근거로 주로 사용한다.
- `morning_net_inflow`, `evening_net_inflow`는 보조 해석 피처로 유지한다.

## 이번 라운드 최종 권장 입력 컬럼

실제 군집화 입력으로 우선 추천하는 컬럼:

- `arrival_7_10_ratio`
- `arrival_11_14_ratio`
- `arrival_17_20_ratio`
- `morning_net_inflow`
- `evening_net_inflow`
- `subway_distance_m`
- `bus_stop_count_300m`

이유:

- 반납 시간대 비율 3개가 지구판단의 핵심 축이다.
- 순유입 2개는 주거지/업무지 방향성을 보강한다.
- 교통 접근성 2개는 환승 거점 여부를 보강한다.

해석용으로 함께 들고 갈 컬럼:

- `total_return_count`
- `return_7_10_count`
- `return_11_14_count`
- `return_17_20_count`
- `dominant_ratio`
- `district_hypothesis`
- `mapped_dong_code`
- `mapped_dong_name`
- `life_pop_*`

준비된 파일:

- `ddri_second_cluster_ready_input_train_2023_2024.csv`
- `ddri_second_cluster_ready_input_test_2025.csv`

## 발표 문서에 넣을 지도 3장

주력 추천:

- `ddri_return_map_2025_7_10.html`
- `ddri_return_map_2025_11_14.html`
- `ddri_return_map_2025_17_20.html`

선정 이유:

- 같은 연도에서 시간대만 바꿔 비교해야 해석이 가장 명확하다.
- 2025는 현재 대여소 구성과 가장 가깝고 발표 시점 설명이 직관적이다.
- 세 장을 나란히 놓으면 `아침 도착 집중`, `점심 도착 집중`, `저녁 도착 집중` 패턴 차이를 바로 보여줄 수 있다.

발표 멘트 기준:

- `2025 07~10시 반납 지도`: 출근 도착이 몰리는 업무/상업 후보 지점을 보여준다.
- `2025 11~14시 반납 지도`: 점심 시간대 상권/업무지 인접 거점을 보여준다.
- `2025 17~20시 반납 지도`: 저녁 귀가 흐름이 강한 주거지 후보 지점을 보여준다.

## 지구판단 해석 규칙

- `arrival_7_10_ratio`가 상대적으로 높음:
  - 아침 도착 집중
  - `업무/상업지구 후보`
- `arrival_11_14_ratio`가 상대적으로 높음:
  - 점심 도착 집중
  - `점심 상권/업무지구 후보`
- `arrival_17_20_ratio`가 상대적으로 높음:
  - 저녁 도착 집중
  - `주거지구 후보`

주의:

- 이 라벨은 최종 확정 분류가 아니라 `군집 해석용 가설`이다.
- 실제 발표에서는 지하철 접근성, 버스 접근성, 생활인구 같은 다른 근거와 함께 해석하는 것이 맞다.

## 빠른 관찰

- 2023:
  - `주거지구 후보 131`
  - `업무/상업지구 후보 37`
  - `점심 상권/업무지구 후보 2`
- 2024:
  - `주거지구 후보 130`
  - `업무/상업지구 후보 38`
  - `점심 상권/업무지구 후보 2`
- 2025:
  - `주거지구 후보 124`
  - `업무/상업지구 후보 44`
  - `점심 상권/업무지구 후보 4`

## 결측 처리 규칙

- 생활인구 `life_pop_*` 컬럼에서 초기 결측 3건이 있었다.
- 대상 대여소:
  - `2342`
  - `2404`
  - `2423`
- 원인:
  - 좌표 역변환 결과 행정동명은 `개포3동`
  - 하지만 서울시 생활인구 원천과 행정동코드 매핑표는 명칭 변경 전 체계인 `일원2동(11680740)`을 사용하고 있었다.
- 처리:
  - `개포3동` 3개 대여소는 생활인구 결합 시 `일원2동(11680740)`으로 대응시켰다.
  - 문서 표기는 `일원2동(현 개포3동)`으로 남겼다.
- 이유:
  - `개포3동`은 2022년 12월 말 명칭 변경 이력이 있고,
  - 생활인구 원천 데이터는 여전히 `일원2동` 코드 체계를 유지하는 것으로 확인됐다.

## 재생성 방법

```bash
python3 works/01_clustering/08_integrated/05_build_return_time_district_features.py
```
