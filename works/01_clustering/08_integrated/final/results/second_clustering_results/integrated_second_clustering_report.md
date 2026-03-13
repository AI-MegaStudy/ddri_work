# 통합 군집화 레포트 초안

## 1. 왜 통합 군집화가 필요했는가

1차 군집화는 평균 대여량, 평일/주말 평균, 출퇴근 비율 등 `수요 규모 중심 피처`로 구성되어 있었다.
그 결과 `일반수요형`과 `고수요형`의 분리는 가능했지만, 원래 목표였던 `업무지구`, `주거지구`, `상권`, `복합 거점` 같은 공간적 성격을 설명하는 데는 한계가 있었다.

따라서 이번 통합 군집화는 다음 두 축을 결합했다.

- 반납 시간대 기반 지구판단 피처
- 교통 접근성과 순유입 정보

이렇게 해서 군집을 단순 이용량이 아니라 `도착 수요의 공간적 역할` 기준으로 다시 해석한다.

## 2. 최종 입력 피처

군집화 핵심 입력 7개:

- `arrival_7_10_ratio`
- `arrival_11_14_ratio`
- `arrival_17_20_ratio`
- `morning_net_inflow`
- `evening_net_inflow`
- `subway_distance_m`
- `bus_stop_count_300m`

보조 해석 컬럼:

- `total_return_count`
- `return_7_10_count`
- `return_11_14_count`
- `return_17_20_count`
- `dominant_ratio`
- `district_hypothesis`
- `life_pop_*`

## 3. 군집 수 탐색

탐색 범위는 `k = 5 ~ 7`로 두었다.

| k | inertia | silhouette |
|---|---:|---:|
| 5 | 542.47 | 0.2033 |
| 6 | 498.33 | 0.1795 |
| 7 | 465.91 | 0.1708 |

선택 결과:

- 최종 선택 `k = 5`
- 이유: 탐색 범위 내 silhouette가 가장 높음

주의:

- silhouette 값은 1차보다 낮다.
- 하지만 이번 군집화는 `분리도 최적화`보다 `지구판단 해석력 확보`를 우선한 설계다.

## 4. 군집별 요약

| 군집 | station 수 | 07-10 비율 | 11-14 비율 | 17-20 비율 | 아침 순유입 | 저녁 순유입 | 지하철 거리 | 해석 초안 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 49 | 0.283 | 0.187 | 0.233 | 1162.9 | -792.3 | 472.1m | 업무/상업 혼합형 |
| 1 | 3 | 0.486 | 0.160 | 0.161 | 10421.0 | -4702.0 | 435.7m | 아침 도착 업무 집중형 |
| 2 | 32 | 0.113 | 0.150 | 0.383 | -2016.7 | 1215.0 | 390.5m | 주거 도착형 |
| 3 | 61 | 0.153 | 0.198 | 0.307 | -489.2 | 20.6 | 282.7m | 생활·상권 혼합형 |
| 4 | 19 | 0.157 | 0.159 | 0.312 | -842.9 | 477.4 | 1616.6m | 외곽 주거형 |

## 5. 군집 라벨 후보

### Cluster 0: 업무/상업 혼합형

- 아침 도착 비율이 높고, 점심 비율도 완전히 낮지 않음
- 아침 순유입이 플러스, 저녁 순유입은 마이너스
- 대표 지점:
  - `SB타워 앞`
  - `역삼지하보도 7번출구 앞`
  - `구역삼세무서 교차로`

### Cluster 1: 아침 도착 업무 집중형

- 아침 도착 비율이 압도적으로 높음
- 아침 순유입 규모가 매우 큼
- station 수는 적지만 성격이 매우 분명함
- 대표 지점:
  - `수서역 5번출구`
  - `주식회사 오뚜기 정문 앞`
  - `포스코사거리(기업은행)`

### Cluster 2: 주거 도착형

- 저녁 도착 비율이 가장 높음
- 아침 순유출, 저녁 순유입 구조가 명확함
- 대표 지점:
  - `청담역 13번 출구 앞`
  - `청담역 2번출구`
  - `현대아파트 정문 앞`

### Cluster 3: 생활·상권 혼합형

- 저녁 도착이 우세하지만 점심 비율도 비교적 높음
- 지하철 접근성이 좋고 생활권 중심역 주변이 많음
- 대표 지점:
  - `학여울역 사거리`
  - `도곡역 대치지구대 방향`
  - `대치역 7번출구`

### Cluster 4: 외곽 주거형

- 저녁 도착형이지만 지하철 접근성이 크게 떨어짐
- 외곽/주거 생활권 성격이 상대적으로 강함
- 대표 지점:
  - `더시그넘하우스 앞`
  - `국립국악중,고교 정문 맞은편`
  - `세곡동 사거리`

## 6. 발표에 쓸 핵심 근거

차트:

- `images/ddri_second_kmeans_elbow_silhouette.png`
- `images/ddri_second_kmeans_pca_scatter.png`
- `images/ddri_second_cluster_feature_means.png`
- `images/ddri_second_cluster_profile_heatmap.png`
- `images/ddri_second_cluster_size.png`
- `images/ddri_second_cluster_hypothesis_crosstab.png`

지도:

- `../return_time_district/ddri_return_map_2025_7_10.html`
- `../return_time_district/ddri_return_map_2025_11_14.html`
- `../return_time_district/ddri_return_map_2025_17_20.html`
- `ddri_second_cluster_map.html`

표:

- `data/ddri_second_cluster_summary.csv`
- `data/ddri_second_cluster_representative_stations.csv`
- `data/ddri_second_cluster_hypothesis_crosstab.csv`

## 7. 이후 예측 모델로의 연결

이 군집화의 목적은 군집 자체가 아니라 이후 `station-day 수요 예측`으로 연결되는 것이다.

군집을 통해 다음과 같이 후속 피처를 더 선별할 수 있다.

- 업무/상업 혼합형:
  - 평일/공휴일
  - 아침 생활인구
  - 출근 시간 강수와 기온
- 주거 도착형:
  - 저녁 강수
  - 주말 여부
  - 귀가 시간대 체감기온
- 생활·상권 혼합형:
  - 점심 생활인구
  - 점심 시간 강수/기온
  - 교통 접근성
- 외곽 주거형:
  - 고저차
  - 버스 접근성
  - 장거리 이동 부담 변수

최종 예측 평가는 아래 지표로 진행한다.

- `RMSE`
- `MAE`
- `R²`
