# Ddri Project Report Log

작성일: 2026-03-11
목적: 프로젝트 진행 과정의 의사결정 근거, 참고 자료, 차트 산출물 계획을 누적 기록하여 추후 PPT/보고서 작성에 재사용한다.

## 1. 문서 사용 원칙

- 이 문서는 진행 로그가 아니라 `보고서/PPT에 바로 옮길 수 있는 근거 문서`로 관리한다.
- 각 항목은 가능하면 아래 4가지를 함께 남긴다.
  - 결정 내용
  - 결정 이유
  - 근거 자료
  - 필요한 시각화/표
- 차트 이미지가 생성되면 저장 경로를 이 문서에 연결한다.
- 자료가 부족한 경우 임의 추정으로 채우지 않고 `추가 자료 필요`로 표시한다.

## 2. 프로젝트 기준

### 2.1 분석 범위

- 공간 범위: 강남구 따릉이 대여소
- 학습 기간: 2023-01-01 ~ 2024-12-31
- 테스트 기간: 2025-01-01 ~ 2025-12-31
- 네이밍 규칙: 파일명/폴더명/산출물 prefix는 `ddri`

### 2.2 현재 프로젝트 목표

- 강남구 Station별 따릉이 수요/보유 적정 대수 예측 기반 분석 수행
- 1차 산출물로 대여소 군집화 시각화 자료 제작
- 후속 단계에서 ML 예측 모델 및 Flutter 기반 지도 시각화 웹 설계로 확장

## 3. 주요 의사결정 로그

### Decision 001. 1차 군집화 기준 대여소는 2023~2025 공통 대여소 169개로 고정

#### 결정 내용

- 1차 군집화 및 ML 베이스라인의 기준 대여소 집합은 `2023~2025 공통 대여소 169개`로 사용한다.
- 2025년에만 등장하는 신규 대여소는 1차 분석 대상에서 제외한다.

#### 결정 이유

- 학습 구간과 테스트 구간의 대여소 집합을 동일하게 유지할 수 있다.
- 연도별 신규/폐쇄 대여소를 포함하면 비교 가능한 패널 구성이 깨진다.
- 군집화 해석과 테스트 성능 비교가 단순해진다.
- 신규 대여소는 과거 이용 이력이 없어 별도의 cold-start 문제로 다루는 것이 일반적이다.

#### 내부 확인 근거

- 강남구 대여소 수
  - 2023년: 175개
  - 2024년: 176개
  - 2025년: 179개
- 2023~2025 공통 대여소 수: 169개
- 연도별 단독 대여소 존재
  - 2023 전용: 4개
  - 2024 전용: 1개
  - 2025 전용: 6개

#### 외부 참고 근거

- 패널 분석 일반론에서는 비교 가능성과 해석력 측면에서 balanced panel이 유리하다.
- 자전거/모빌리티 수요 예측에서는 기존 운영 지점 예측과 신규 지점 예측을 분리하는 접근이 일반적이다.

참고 링크:
- UBC COMET panel data overview
  - https://comet.arts.ubc.ca/docs/4_Advanced/advanced_panel_data/advanced_panel_data.html
- Baltagi & Liu, forecasting with unbalanced panel data
  - https://www.maxwell.syr.edu/research/center-for-policy-research/research-publications/working-papers/wp-221-forecasting-with-unbalanced-panel-data
- Applied Sciences 2021, 신규 대여소 장기 수요 예측
  - https://www.mdpi.com/2076-3417/11/15/6748
- Information Fusion 2024, 신규 대여소 planning 수요 문제
  - https://www.sciencedirect.com/science/article/pii/S1566253524000721
- arXiv 2023, system expansion/new stations 문제
  - https://www.emergentmind.com/articles/2303.11977

#### 보고서/PPT에 넣을 수 있는 메시지

- “기존 운영 대여소 예측”과 “신규 대여소 예측”은 다른 문제이므로 분리했다.
- 1차 분석은 비교 가능성을 위해 공통 운영 대여소만 사용했다.
- 신규 대여소 예측은 후속 확장 과제로 남겼다.

#### 필요한 시각화/표

- [x] 연도별 대여소 수 비교 표
- [ ] 공통/신규/제외 대여소 수 요약 표
- [ ] 분석 대상 대여소 선정 흐름도

#### 시각화 저장 경로

- 표 데이터: `works/clustering/data/ddri_station_master_year_summary.csv`
- coverage 요약: `works/clustering/data/ddri_station_coverage_summary.csv`
- coverage 상세: `works/clustering/data/ddri_station_coverage_flags.csv`

---

### Decision 002. 군집화 1차 작업용 베이스라인 노트북을 별도 생성

#### 결정 내용

- 군집화 1차 산출물은 별도 노트북 `ddri_station_clustering_baseline.ipynb`에서 진행한다.
- 노트북 안에는 아래 흐름을 먼저 고정한다.
  - 공통 대여소 마스터 생성
  - 학습/테스트 이용 이력 분리 로드
  - 대여소별 군집화 feature 생성
  - KMeans 후보 탐색
  - Elbow / Silhouette 확인
  - PCA 및 군집 요약

#### 결정 이유

- 분석 흐름을 한 파일에 묶어 재현 가능성을 높일 수 있다.
- 발표/보고서 작성 시 코드, 표, 차트의 근원을 추적하기 쉽다.
- 이후 이미지 저장 경로를 일정하게 관리할 수 있다.

#### 내부 확인 근거

- 생성 파일:
  - `works/clustering/ddri_station_clustering_baseline.ipynb`
- 초기 산출물:
  - `works/clustering/data/ddri_station_master_year_summary.csv`

#### 보고서/PPT에 넣을 수 있는 메시지

- “군집화는 별도 베이스라인 노트북으로 분리하여 재현 가능한 분석 단위로 구성했다.”
- “입력 데이터, 전처리, 군집화, 시각화 흐름을 한 노트북에서 추적 가능하게 만들었다.”

#### 필요한 시각화/표

- [ ] 공통 대여소 선정 과정 표
- [ ] 학습/테스트 데이터 분리 설명 표
- [ ] 군집화 feature 정의 표

#### 시각화 저장 경로

- 노트북: `works/clustering/ddri_station_clustering_baseline.ipynb`
- 표 데이터: `works/clustering/data/ddri_station_master_year_summary.csv`

---

### Decision 003. 1차 KMeans 군집 수는 현재 기준 `k=2`가 가장 유력

#### 결정 내용

- 1차 군집화 베이스라인 탐색에서 `k=2~6`을 비교했다.
- 현재 결과 기준으로 silhouette score가 가장 높은 `k=2`를 우선 채택한다.

#### 결정 이유

- silhouette score가 `k=2`에서 가장 높았다.
- 현재 1차 목적은 복잡한 세분화보다 재현 가능한 baseline 확보에 있다.
- 이후 환경 feature를 결합하면 세분 군집(`k=3` 이상) 재검토 여지는 남겨둔다.

#### 내부 실행 결과

K 탐색 결과:

| k | inertia | silhouette |
|---|---:|---:|
| 2 | 702.846 | 0.426 |
| 3 | 559.837 | 0.272 |
| 4 | 452.351 | 0.287 |
| 5 | 386.765 | 0.255 |
| 6 | 330.316 | 0.280 |

현재 산출물:

- `works/clustering/data/ddri_kmeans_search_metrics.csv`
- `works/clustering/images/ddri_kmeans_elbow_silhouette.png`
- `works/clustering/images/ddri_kmeans_pca_scatter.png`
- `works/clustering/images/ddri_cluster_feature_means.png`
- `works/clustering/data/ddri_cluster_summary.csv`

#### 보고서/PPT에 넣을 수 있는 메시지

- “1차 군집화는 K 후보를 비교한 뒤 silhouette 기준으로 k=2를 채택했다.”
- “초기 baseline에서는 분리도가 가장 좋은 단순 구조를 우선 채택했다.”

#### 필요한 시각화/표

- [x] Elbow / Silhouette 비교 차트
- [x] PCA 군집 산점도
- [x] 군집별 feature 평균표
- [ ] 군집별 명칭 정리 표

#### 시각화 저장 경로

- `works/clustering/images/ddri_kmeans_elbow_silhouette.png`
- `works/clustering/images/ddri_kmeans_pca_scatter.png`
- `works/clustering/images/ddri_cluster_feature_means.png`
- `works/clustering/data/ddri_cluster_summary.csv`

---

### Decision 004. 공통 169개 대여소 기준이라도 실제 feature 생성 대상은 더 줄어든다

#### 결정 내용

- 공통 대여소 마스터는 169개지만, 실제 이용 이력이 없는 대여소는 feature 생성 대상에서 제외된다.
- 현재 결과:
  - 학습 feature 대여소 수: 164
  - 테스트 feature 대여소 수: 162

#### 결정 이유

- 군집화 feature는 실제 대여 이력이 있어야 계산 가능하다.
- 공통 대여소라고 해도 특정 기간에 이용 기록이 전혀 없을 수 있다.

#### 내부 실행 결과

- 공통 마스터: 169개
- 학습 누락: 5개
  - `2316`, `2322`, `2355`, `3603`, `3605`
- 테스트 누락: 7개
  - `2316`, `2322`, `2355`, `2421`, `3603`, `3621`, `4901`
- 학습/테스트 모두 누락: 4개
  - `2316`, `2322`, `2355`, `3603`

관련 파일:

- `works/clustering/data/ddri_station_coverage_summary.csv`
- `works/clustering/data/ddri_station_coverage_flags.csv`

#### 보고서/PPT에 넣을 수 있는 메시지

- “공통 운영 대여소만 남겨도 실제 이용 이력이 없는 대여소는 별도 제외가 필요했다.”
- “분석용 마스터 수와 실제 feature 생성 수는 다를 수 있다.”

#### 필요한 시각화/표

- [x] coverage 요약 표
- [ ] 누락 대여소 처리 기준 표

#### 시각화 저장 경로

- `works/clustering/data/ddri_station_coverage_summary.csv`
- `works/clustering/data/ddri_station_coverage_flags.csv`

---

### Decision 005. 결측치와 이상치는 1차 군집화에서도 반드시 제거

#### 결정 내용

- 1차 군집화 전처리에서 아래 항목을 제거한다.
  - 필수 컬럼 결측치
  - `이용시간(분) <= 0`
  - `이용거리(M) <= 0`
  - 공통 대여소 기준에 포함되지 않는 대여 대여소
  - 강남구 기준 대여소 마스터에 포함되지 않는 반납 대여소

#### 결정 이유

- 군집화는 패턴 기반이므로 비정상 이동이 섞이면 feature 왜곡이 발생한다.
- 사용자 요구사항에 따라 강남구 외부 반납 이동은 이상치로 간주하고 제외한다.
- 결측치와 비정상값을 그대로 두면 평균 대여량, 분산, 시간대 비율 계산이 왜곡될 수 있다.

#### 내부 실행 근거

실행 스크립트:

- `works/clustering/ddri_station_clustering_baseline.py`

전처리 로그:

- `works/clustering/data/ddri_cleaning_log.csv`

집계 결과:

| group | rows_before | rows_after | dropped_nonpositive | dropped_noncommon_rent | dropped_outside_gangnam_return |
|---|---:|---:|---:|---:|---:|
| train_2023 | 1,012,480 | 917,202 | 68,978 | 26,300 | 0 |
| train_2024 | 1,014,918 | 943,381 | 54,057 | 17,480 | 0 |
| test_2025 | 869,397 | 825,111 | 15,022 | 29,264 | 0 |

해석:

- 실제 제거 건의 대부분은 비정상 시간/거리 값과 공통 기준 밖 대여소였다.
- 현재 강남구 대여소 기준으로 유지된 표본에서는 `외부 반납` 제거 건수가 최종 집계상 0건이었다.
- 이는 강남구 이용정보 파일이 이미 상당 부분 지역 기준으로 정리되어 있거나, 공통 대여소 필터를 거치며 함께 제외되었기 때문으로 해석된다.

#### 보고서/PPT에 넣을 수 있는 메시지

- “결측치와 비정상 운행 기록을 제거한 뒤 군집화를 수행했다.”
- “강남구 내부 이동 패턴에 집중하기 위해 외부 반납 이동은 이상치로 정의했다.”
- “실제 1차 분석에서는 비정상 시간/거리 값과 기준 밖 대여소 제거 영향이 더 컸다.”

#### 필요한 시각화/표

- [x] 전처리 전후 row 수 비교 표
- [x] 전처리 제거 사유별 비중 차트
- [ ] 월별 제거 비율 차트

#### 시각화 저장 경로

- `works/clustering/data/ddri_cleaning_log.csv`
- `works/clustering/data/ddri_cleaning_summary_by_group.csv`
- `works/clustering/images/ddri_cleaning_before_after.png`
- `works/clustering/images/ddri_cleaning_drop_reasons.png`

---

### Decision 006. 중복 로그는 존재하지만 규모가 매우 작아 1차 결과에 미치는 영향은 제한적

#### 결정 내용

- 원천 이용 로그의 완전 중복 행 여부를 점검했다.
- 전체 36개 파일, 2,896,795행 중 완전 중복은 21행이었다.

#### 결정 이유

- 중복 로그가 많으면 대여량 집계와 군집화 feature가 왜곡될 수 있어 사전 점검이 필요했다.
- 실제 중복 수가 매우 작아 1차 군집화 결과를 크게 바꾸는 수준은 아니라고 판단했다.

#### 내부 실행 근거

- 요약 파일:
  - `works/clustering/data/ddri_duplicate_check_summary.csv`
- 파일별 상세:
  - `works/clustering/data/ddri_duplicate_check_by_file.csv`

요약:

| files | rows | dup_all | dup_key |
|---:|---:|---:|---:|
| 36 | 2,896,795 | 21 | 21 |

#### 보고서/PPT에 넣을 수 있는 메시지

- “원천 로그 중 완전 중복 행은 존재했지만 전체 규모 대비 매우 적었다.”
- “중복 여부를 확인한 뒤, 1차 분석 결과에 미치는 영향은 제한적이라고 판단했다.”

#### 필요한 시각화/표

- [x] 중복 점검 요약 표
- [ ] 파일별 중복 건수 표

#### 시각화 저장 경로

- `works/clustering/data/ddri_duplicate_check_summary.csv`
- `works/clustering/data/ddri_duplicate_check_by_file.csv`

---

### Decision 007. feature 극단치는 존재하지만 1차 군집화에서는 즉시 제거하지 않고 유지

#### 결정 내용

- 학습용 군집화 feature에 대해 IQR 기준 극단치 후보를 점검했다.
- 일부 feature에서 상단 이상치 후보가 확인되었지만, 1차 군집화에서는 자동 제거하지 않았다.

#### 결정 이유

- 높은 대여량 또는 높은 변동성 대여소는 실제 핵심 수요 거점일 가능성이 높다.
- 군집화 목적상 이런 대여소는 제거 대상 이상치가 아니라 중요한 유형 차이일 수 있다.
- 따라서 1차 단계에서는 제거보다 해석 대상으로 유지하는 것이 더 타당하다.

#### 내부 실행 근거

- 분포 요약:
  - `works/clustering/data/ddri_feature_distribution_summary.csv`
- IQR 요약:
  - `works/clustering/data/ddri_feature_iqr_outlier_summary.csv`

대표 결과:

| feature | outlier_count |
|---|---:|
| avg_rental | 12 |
| weekday_avg | 11 |
| weekend_avg | 13 |
| peak_ratio | 6 |
| night_ratio | 6 |

#### 보고서/PPT에 넣을 수 있는 메시지

- “극단치 후보는 점검했지만, 수요 거점 대여소를 임의 제거하지 않기 위해 1차 군집화에서는 유지했다.”
- “이상치 제거보다 군집 해석 단계에서 핵심 거점 여부를 먼저 확인하는 전략을 선택했다.”

#### 필요한 시각화/표

- [x] feature 분포 요약 표
- [x] IQR 기반 이상치 후보 수 표
- [ ] feature boxplot

#### 시각화 저장 경로

- `works/clustering/data/ddri_feature_distribution_summary.csv`
- `works/clustering/data/ddri_feature_iqr_outlier_summary.csv`

---

### Decision 008. 1차 군집 라벨은 `고수요형`과 `일반수요형`으로 해석

#### 결정 내용

- 현재 `k=2` 군집의 1차 라벨은 아래와 같이 둔다.
  - Cluster 1: `고수요형 대여소`
  - Cluster 0: `일반수요형 대여소`

#### 결정 이유

- 두 군집의 차이는 `peak_ratio`, `night_ratio`보다 `avg_rental`, `weekday_avg`, `weekend_avg`, `weekday_weekend_gap`의 규모 차이가 훨씬 크다.
- 즉, 현재 1차 군집은 출퇴근형/레저형 같은 이용 목적 분리보다 `수요 수준` 분리가 더 강하게 나타난다.
- 따라서 1차 라벨을 `출퇴근형` 등으로 과도 해석하기보다, 수요 규모 중심 라벨이 더 보수적이고 타당하다.

#### 내부 실행 근거

군집 평균 비교:

| cluster | avg_rental | weekday_avg | weekend_avg | peak_ratio | night_ratio |
|---|---:|---:|---:|---:|---:|
| 0 | 11.913 | 12.865 | 9.479 | 0.373 | 0.133 |
| 1 | 31.176 | 33.300 | 25.860 | 0.382 | 0.134 |

해석:

- `peak_ratio`, `night_ratio`는 두 군집 간 차이가 크지 않다.
- 반면 평균 대여량과 평일/주말 대여량은 군집 1이 군집 0보다 훨씬 높다.

참고 대여소 예시:

- 고수요형:
  - `2335` 3호선 매봉역 3번출구앞
  - `2377` 수서역 5번출구
  - `2404` 대모산입구역 4번 출구 앞
  - `2348` 포스코사거리(기업은행)
- 일반수요형:
  - `2308` 압구정파출소 앞
  - `4914` 도심공항타워 앞
  - `2324` 천주교 대치 2동 교회 옆
  - `4923` 수협대치동지점

#### 보고서/PPT에 넣을 수 있는 메시지

- “1차 군집화는 이용 목적보다 수요 규모 차이를 더 강하게 분리했다.”
- “현재 결과만으로는 출퇴근형/레저형보다 고수요형/일반수요형 해석이 더 보수적이고 적절하다.”
- “환경 데이터 결합 후 2차 군집 해석에서 출퇴근형, 공원인접형 등 세부 유형화를 재검토할 수 있다.”

#### 필요한 시각화/표

- [x] 군집별 feature 평균표
- [ ] 군집 라벨 정의 표
- [ ] 군집별 대표 대여소 표

#### 시각화 저장 경로

- `works/clustering/data/ddri_cluster_summary.csv`
- `works/clustering/images/ddri_cluster_feature_means.png`

---

### Decision 008-1. 발표용 군집 지도는 Folium 기반 HTML로 제공

#### 결정 내용

- 군집 결과는 발표용으로 `folium.Map` 기반 HTML 지도로 생성했다.
- 지도에는 군집 색상, 대여소명, 대여소번호, 평균 대여량, 평일/주말 평균 정보를 표시한다.

#### 결정 이유

- 발표 시 직관적으로 공간 분포를 설명하기 가장 좋다.
- Flutter 웹 구현 전 단계에서도 결과물을 바로 공유할 수 있다.
- 발표 일정이 가까우므로 정적 차트보다 인터랙티브 HTML 지도가 효율적이다.

#### 생성 파일

- `works/clustering/ddri_cluster_folium_map.py`
- `works/clustering/maps/ddri_cluster_map_gangnam.html`

#### 보고서/PPT에 넣을 수 있는 메시지

- “군집 결과는 Folium 지도를 통해 공간적으로 시각화하였다.”
- “대여소별 군집과 평균 대여량을 지도에서 직접 확인할 수 있도록 구성하였다.”

#### 필요한 시각화/표

- [x] Folium 군집 지도
- [ ] 발표용 지도 캡처 이미지

#### 시각화 저장 경로

- `works/clustering/maps/ddri_cluster_map_gangnam.html`

---

### Decision 008-2. 군집 해석은 환경 feature로 1단계 보강

#### 결정 내용

- 공원 거리, 지하철 거리, 버스정류장 수를 계산해 군집 해석을 보강했다.
- 군집별 평균 환경 feature와 대표 대여소 목록을 추가 산출물로 생성했다.

#### 결정 이유

- 발표에서는 “군집이 나뉘었다”보다 “왜 이런 군집이 형성됐는가” 설명이 더 중요하다.
- 군집별 수요 차이가 교통 접근성과 공간 특성 차이와 함께 나타나는지 확인할 필요가 있었다.

#### 내부 실행 근거

생성 파일:

- 환경 feature 테이블:
  - `works/clustering/data/ddri_cluster_environment_features.csv`
- 군집별 요약:
  - `works/clustering/data/ddri_cluster_environment_summary.csv`
- 대표 대여소:
  - `works/clustering/data/ddri_cluster_representative_stations.csv`
- 비교 차트:
  - `works/clustering/images/ddri_cluster_environment_comparison.png`

군집 평균 비교:

| 군집 | 공원 거리(m) | 지하철 거리(m) | 300m 버스정류장 수 | 평균 대여량 |
|---|---:|---:|---:|---:|
| 일반수요형 | 1036.82 | 551.64 | 26.98 | 11.91 |
| 고수요형 | 1170.43 | 387.75 | 32.56 | 31.18 |

해석:

- 고수요형은 일반수요형보다 지하철 접근성이 더 좋고, 주변 버스정류장 수가 더 많다.
- 즉, 1차 군집의 수요 차이는 대중교통 접근성과 연결될 가능성이 있다.
- 반면 공원 거리는 고수요형이 더 짧지 않아, 현재 군집은 레저형보다 교통 접근성 차이와 더 가깝게 연결된다.
- 다만 강남의 상업지구 성격이나 출퇴근형 수요를 직접 설명하려면 업무·상업 POI와 시간대 세부 지표가 추가로 필요하다.

#### 보고서/PPT에 넣을 수 있는 메시지

- “고수요형 대여소는 일반수요형보다 지하철과 버스 접근성이 더 좋은 경향을 보였다.”
- “현재 군집은 공원 접근성보다 교통 접근성 차이와 더 강하게 연결된다.”
- “발표 시에는 고수요형을 교통 접근성이 상대적으로 더 좋은 고수요 대여소군으로 설명한다.”
- “상업지구·출퇴근형 해석은 추가 POI와 시간대 지표 보강 후 재검토한다.”

#### 필요한 시각화/표

- [x] 군집별 환경/수요 비교 차트
- [x] 군집별 대표 대여소 표
- [ ] 대표 대여소 지도 캡처

#### 시각화 저장 경로

- `works/clustering/images/ddri_cluster_environment_comparison.png`
- `works/clustering/data/ddri_cluster_environment_summary.csv`
- `works/clustering/data/ddri_cluster_representative_stations.csv`

---

### 발표 패키지 메모

군집화 발표 직전 바로 사용할 수 있는 정리 파일:

- 1페이지 요약:
  - `works/clustering/ddri_clustering_presentation_summary.md`
- 슬라이드 문구/발표 멘트:
  - `works/clustering/ddri_clustering_slide_script.md`
- 슬라이드용 수치표:
  - `works/clustering/data/ddri_clustering_slide_tables.csv`
- 지도:
  - `works/clustering/maps/ddri_cluster_map_gangnam.html`

권장 발표 흐름:

1. 왜 군집화했는가
2. 어떤 feature로 군집화했는가
3. 왜 `k=2`를 선택했는가
4. 일반수요형 / 고수요형 차이
5. 교통 접근성 기반 해석
6. 지도 시각화

### Decision 009. 1차 ML target은 `Station별 하루 대여량 (rental_count)`으로 고정

#### 결정 내용

- 1차 예측 문제는 `Station별 하루 대여량` 예측으로 확정한다.
- 데이터셋 grain은 `station-day`로 고정한다.
- 문제 유형은 회귀로 정의한다.

#### 결정 이유

- 내부 기획 문서 대부분이 `Station별 하루 대여량 예측`을 중심으로 작성되어 있다.
- target 변수도 `rental_count`로 명시되어 있다.
- 웹 조회 기능 역시 날짜 기반 예측 조회 구조와 잘 맞는다.
- 일정과 현재 데이터 구조상 시간 단위보다 일 단위가 더 안정적이다.

#### 내부 문서 근거

- `3조 공유폴더/00.기획문서/ddareungi_demand_project_canvas.md`
  - `Station별 하루 따릉이 대여량을 예측하는 회귀 모델 구축`
  - target 변수: `rental_count`
- `3조 공유폴더/00.기획문서/ddareungi_dataset_links_detailed.md`
  - 대여 이력 데이터는 `예측 타깃 생성, 일별/시간별 집계`에 활용
- `3조 공유폴더/00.기획문서/0311_발표자료/ddareungi_schedule_report.md`
  - `Station별 하루 따릉이 대여량을 예측하는 회귀 모델 구축`

#### `보유 적정 대수`와의 관계

- `보유 적정 대수`는 운영 해석 목적 표현으로 둔다.
- 1차 모델은 먼저 `수요(대여량)`를 예측한다.
- 이후 예측 대여량을 바탕으로 `적정 보유 대수`나 `재배치 필요성`을 해석 지표로 연결한다.

#### 보고서/PPT에 넣을 수 있는 메시지

- “프로젝트의 1차 예측 문제는 Station별 하루 대여량 예측으로 고정했다.”
- “운영 목적의 적정 보유 대수는 수요 예측 결과를 바탕으로 해석하는 2차 지표로 둔다.”
- “현재 데이터 구조와 웹 조회 흐름상 station-day 단위가 가장 일관된 정의다.”

#### 필요한 시각화/표

- [ ] target/grain 정의 표
- [ ] 수요 예측과 운영 해석 관계도

#### 시각화 저장 경로

- 문서: `works/ddri_prediction_target_definition.md`

---

### Decision 010. 1차 예측용 학습 테이블은 `station_id + date` 기준으로 설계

#### 결정 내용

- 최종 학습 테이블의 기본 결합 키는 `station_id + date`로 둔다.
- 원천 대여 로그를 대여소-일 단위로 집계해 target을 만든다.
- 날씨와 공휴일은 `date` 기준으로 결합한다.
- 환경 데이터와 군집 label은 `station_id` 기준 정적 feature로 결합한다.

#### 결정 이유

- 현재 target이 `station-day`이므로 외부 데이터도 일 단위와 정적 단위로 정렬하는 것이 가장 단순하고 안정적이다.
- 날씨는 시간 단위 원천이지만 target이 하루 대여량이므로 일 단위 집계가 자연스럽다.
- 환경 데이터는 대여소 기준 정적 정보로 보는 것이 해석과 운영 측면에서 유리하다.

#### 확정한 결합 기준

- 내부 대여 로그 → `station_id + date`
- 날씨 → `date`
- 공휴일 → `date`
- 환경/군집 → `station_id`

#### 확정한 1차 feature 그룹

- 시간 feature
- 과거 수요 feature
- 날씨 feature
- 정적 환경 feature
- 군집 label

#### 보류 항목

- 생활인구 결합
- 대기질 결합
- 상업지역/POI 결합

#### 보고서/PPT에 넣을 수 있는 메시지

- “예측용 데이터셋은 station-day 단위로 통합 설계하였다.”
- “날씨와 공휴일은 날짜 기준, 환경 정보는 대여소 기준 정적 feature로 결합하였다.”
- “복잡한 외부 데이터는 1차 베이스라인에서 제외하고, 일관된 결합 키를 우선 고정했다.”

#### 필요한 시각화/표

- [ ] 예측용 데이터셋 결합 구조도
- [ ] feature 그룹 표
- [ ] 최종 스키마 표

#### 시각화 저장 경로

- 문서: `works/ddri_prediction_dataset_design.md`

---

### Decision 011. 공휴일 데이터는 한국천문연구원 특일 정보 API로 직접 수집

#### 결정 내용

- 로컬에 보관된 API 문서와 인증키를 사용해 2023~2025 공휴일 데이터를 직접 수집했다.
- 원천 공휴일 테이블과 일 단위 캘린더 테이블을 각각 생성했다.

#### 결정 이유

- 예측용 학습 테이블에서 `is_holiday`, `is_weekend`는 핵심 시간 feature 역할을 한다.
- 로컬에 API 키와 문서가 이미 준비되어 있어 외부 수동 정리보다 API 직접 수집이 더 일관적이다.
- `station-day` 예측 구조상 일 단위 캘린더 테이블이 바로 결합 가능하다.

#### 내부 실행 근거

사용 자료:

- `3조 공유폴더/[일정데이터] 특일 정보 API/API 인증키.txt`
- `3조 공유폴더/[일정데이터] 특일 정보 API/OpenAPI활용가이드_한국천문연구원_천문우주정보__특일_정보제공_서비스_v1.4.docx`

생성 파일:

- 수집 노트북:
  - `works/calendar/ddri_holiday_api_fetch.ipynb`
- 실행 스크립트:
  - `works/calendar/ddri_holiday_api_fetch.py`
- 원천 API 결과:
  - `works/calendar/data/ddri_holiday_api_raw_2023_2025.csv`
- 일 단위 캘린더:
  - `works/calendar/data/ddri_calendar_daily_2023_2025.csv`

수집 결과:

- 공휴일 원천 행 수: 57
- 캘린더 일 수: 1,096

#### 보고서/PPT에 넣을 수 있는 메시지

- “공휴일 데이터는 한국천문연구원 특일 정보 API를 통해 직접 수집하였다.”
- “원천 공휴일 테이블을 수집한 뒤, 예측용 결합을 위해 일 단위 캘린더 테이블로 재구성하였다.”
- “공휴일 여부와 주말 여부를 분리해 날짜 feature로 활용할 수 있도록 설계하였다.”

#### 필요한 시각화/표

- [ ] 연도별 공휴일 수 표
- [ ] 공휴일 캘린더 예시 표
- [ ] 데이터 수집 파이프라인 요약도

#### 시각화 저장 경로

- `works/calendar/data/ddri_holiday_api_raw_2023_2025.csv`
- `works/calendar/data/ddri_calendar_daily_2023_2025.csv`

---

### Decision 012. 예측용 베이스라인 데이터셋은 생성했지만 테스트 최종셋은 2025 날씨 부재로 보류

#### 결정 내용

- 아래 산출물을 생성했다.
  - 학습 target 테이블
  - 테스트 target 테이블
  - 2023~2024 날씨 일 집계 테이블
  - 학습용 베이스라인 데이터셋
- 다만 `2025 날씨 데이터`가 없어서 테스트용 최종 입력 데이터셋은 아직 만들지 않았다.

#### 생성 파일

- `works/prediction/data/ddri_station_day_target_train_2023_2024.csv`
- `works/prediction/data/ddri_station_day_target_test_2025.csv`
- `works/prediction/data/ddri_weather_daily_2023_2024.csv`
- `works/prediction/data/ddri_station_day_train_baseline_dataset.csv`

#### 내부 실행 결과

- 학습 target 행 수: 114,923
- 테스트 target 행 수: 56,610
- 날씨 일 집계 행 수: 730

추가 확인:

- 학습용 베이스라인에서 날씨 결측 143행 존재
- 결측 날짜는 `2024-01-01` 하루
- 이는 2024 날씨 원천 CSV가 `2024-01-02`부터 시작하기 때문

#### 결정 이유

- 없는 날씨 데이터를 임의 생성하거나 보간하면 사용자 원칙에 어긋난다.
- 따라서 현재는 학습용 베이스라인까지만 확정하고, 테스트 최종셋은 보류하는 것이 맞다.

#### 보고서/PPT에 넣을 수 있는 메시지

- “학습용 베이스라인 데이터셋은 구축 완료했으나, 테스트용 최종 입력셋은 2025 날씨 부재로 보류했다.”
- “누락 데이터는 임의 보간하지 않고 별도 요청 후 반영하는 원칙을 유지했다.”

#### 필요한 시각화/표

- [ ] 예측용 데이터셋 구성 표
- [ ] 학습/테스트/날씨 보유 범위 표
- [ ] 누락 데이터 현황 표

#### 시각화 저장 경로

- `works/prediction/data/ddri_station_day_target_train_2023_2024.csv`
- `works/prediction/data/ddri_station_day_target_test_2025.csv`
- `works/prediction/data/ddri_weather_daily_2023_2024.csv`
- `works/prediction/data/ddri_station_day_train_baseline_dataset.csv`

---

### Decision 013. Open-Meteo로 2025 날씨를 보완해 테스트용 베이스라인 데이터셋까지 생성

#### 결정 내용

- 기존 `openmeteo.ipynb` 방식과 동일한 Open-Meteo Archive API를 사용해 아래 데이터를 추가 수집했다.
  - `2024-01-01` 시간별 날씨
  - `2025-01-01 ~ 2025-12-31` 시간별 날씨
- 이를 기반으로 테스트용 베이스라인 데이터셋까지 생성했다.

#### 생성 파일

- 수집 노트북:
  - `works/weather/ddri_openmeteo_fetch.ipynb`
- 실행 스크립트:
  - `works/weather/ddri_openmeteo_fetch.py`
- 추가 시간별 날씨:
  - `works/weather/data/ddri_weather_2024_0101_hourly.csv`
  - `works/weather/data/ddri_weather_2025_hourly.csv`
- 일 단위 날씨:
  - `works/prediction/data/ddri_weather_daily_2023_2024.csv`
  - `works/prediction/data/ddri_weather_daily_2025.csv`
- 최종 베이스라인 데이터셋:
  - `works/prediction/data/ddri_station_day_train_baseline_dataset.csv`
  - `works/prediction/data/ddri_station_day_test_baseline_dataset.csv`

#### 내부 실행 결과

- 2025 시간별 날씨 행 수: 8,760
- 학습 베이스라인 행 수: 114,923
- 테스트 베이스라인 행 수: 56,610

추가 확인:

- 날씨 결측 문제였던 `2024-01-01`은 보완 완료
- 테스트셋에는 `cluster_label` 결측이 1건 존재
  - `station_id = 3605`
  - `2025-12-16`
  - 원인: 학습 구간 군집 라벨이 없는 대여소

#### 보고서/PPT에 넣을 수 있는 메시지

- “기존 Open-Meteo 수집 방식을 재사용해 2025 날씨를 보완하고 테스트용 입력 데이터셋까지 완성했다.”
- “외부 자료가 없을 때는 기존 수집 노트북을 재사용해 동일 규격 데이터를 직접 생성했다.”
- “다만 군집 라벨은 학습 구간 기반이므로 일부 테스트 행에서 결측이 발생할 수 있음을 확인했다.”

#### 필요한 시각화/표

- [ ] 날씨 데이터 수집 범위 표
- [ ] 학습/테스트 베이스라인 데이터셋 구성 표
- [ ] 최종 결측 현황 표

#### 시각화 저장 경로

- `works/weather/data/ddri_weather_2025_hourly.csv`
- `works/prediction/data/ddri_weather_daily_2025.csv`
- `works/prediction/data/ddri_station_day_test_baseline_dataset.csv`

---

### Decision 014. 신규/소멸 스테이션은 메인 모델과 분리해 처리

#### 결정 내용

- 앞으로 프로젝트 전반에서 아래 방침을 기본 원칙으로 적용한다.
  - 기존 운영 스테이션: 메인 분석/군집화/ML 대상
  - 신규 스테이션: cold-start 예외 집합으로 분리
  - 소멸 스테이션: 테스트 대상에서 제외하거나 운영 이슈로만 기록

#### 결정 이유

- 연도별 스테이션 증감은 일반적인 panel 예측에서 구조 변화 요인이다.
- 기존 운영 스테이션과 신규 스테이션은 학습 가능 정보량이 다르므로 같은 문제로 다루기 어렵다.
- 메인 모델에서 공통 스테이션을 기준으로 평가해야 성능 비교와 해석이 안정적이다.
- 신규 스테이션은 환경 기반 추정 또는 별도 cold-start 모델로 다루는 것이 더 타당하다.

#### 현재 프로젝트 적용 방식

- 군집화: 공통 운영 스테이션 중심
- ML 베이스라인: 학습 이력이 존재하는 기존 운영 스테이션 중심
- 테스트셋에서 군집 라벨이 없는 신규/예외 스테이션은 메인 평가셋에서 제외
- 예외 건은 별도 목록으로 기록

#### 현재 확인된 예외

- 테스트 베이스라인에서 `cluster_label` 결측 1건
  - `station_id = 3605`
  - `date = 2025-12-16`
  - `대여소명 = 봉은사역6번출구(현대아이파크타워앞)`

해석:

- 학습 구간 기준 군집 라벨이 없는 스테이션이므로 메인 평가셋에서는 제외하는 것이 원칙에 부합한다.

#### 보고서/PPT에 넣을 수 있는 메시지

- “신규 및 소멸 스테이션은 기존 운영 스테이션과 동일한 예측 문제로 보지 않고 분리했다.”
- “1차 모델은 학습 이력이 존재하는 기존 운영 스테이션을 대상으로 성능을 평가했다.”
- “신규 스테이션은 cold-start 예외 집합으로 관리하고 후속 확장 과제로 분리했다.”

#### 필요한 시각화/표

- [ ] 신규/기존/소멸 스테이션 처리 방침 표
- [ ] 예외 스테이션 목록 표
- [ ] 메인 평가셋 제외 기준 표

#### 시각화 저장 경로

- `works/clustering/ddri_clustering_input_manifest.md`

---

### Decision 015. 동일 대여소 반납은 이상치로 제거하지 않고 운영 해석용 지표로 관리

#### 결정 내용

- `대여 대여소번호 == 반납대여소번호`인 기록은 이상치로 제거하지 않는다.
- 대신 아래 지표를 별도 보조 feature 후보로 관리한다.
  - `same_station_return_count`
  - `same_station_return_ratio`
  - `return_count`
  - `net_flow`

#### 쉬운 설명

- 같은 대여소에서 빌리고 같은 대여소로 반납하면, 하루가 끝났을 때 그 대여소의 자전거 수는 크게 달라지지 않을 수 있다.
- 하지만 그 자전거는 운행 중 실제로 사용되었고, 그 시간 동안은 다른 이용자가 바로 쓸 수 없다.
- 따라서 이런 기록은 `쓸모없는 이동`이나 `이상치`가 아니라, 대여 수요는 있었지만 재고 변화는 작았던 사례로 보는 것이 맞다.

#### 결정 이유

- 군집화와 수요 예측의 target은 `대여량`이므로, self-return도 실제 수요로 포함하는 것이 자연스럽다.
- 적정 보유 대수는 단순히 하루 끝 재고만 보는 문제가 아니라, 시간대별 점유와 순간 부족 가능성까지 함께 봐야 한다.
- self-return을 제거하면 실제 이용 수요를 과소평가할 위험이 있다.
- 반대로 이 값을 별도 지표로 두면 `수요는 있었지만 순재고 변화는 작았던 날`을 해석하는 데 도움이 된다.

#### 현재 코드 적용 상태

- 현재 전처리에서는 self-return 제거 규칙이 없다.
- 즉, 결측치/비정상 시간·거리/기준 밖 대여소만 제거하고 self-return 기록은 유지하고 있다.

관련 코드:

- `works/clustering/ddri_station_clustering_baseline.py`
- `works/prediction/ddri_station_day_dataset_builder.py`

#### 프로젝트 적용 원칙

- 군집화:
  - self-return을 이상치로 제거하지 않음
  - 현재 군집화 feature에는 직접 넣지 않았지만, 향후 운영 해석 보조 지표로 확장 가능
- 예측 데이터셋:
  - self-return 비율과 순유출입(`net_flow`)을 실제 컬럼으로 생성
  - 특히 `적정 보유 대수` 해석 단계에서 중요하게 사용할 수 있음

#### 내부 실행 결과

생성 파일:

- `works/prediction/data/ddri_station_day_flow_metrics_summary.csv`
- `works/prediction/data/ddri_station_day_train_baseline_dataset.csv`
- `works/prediction/data/ddri_station_day_test_baseline_dataset.csv`

요약:

| dataset | rows | rental_count_sum | return_count_sum | same_station_return_count_sum | same_station_return_ratio_mean | net_flow_mean |
|---|---:|---:|---:|---:|---:|---:|
| train_2023_2024 | 114,923 | 1,860,583 | 1,816,593 | 194,456 | 0.092752 | 0.382778 |
| test_2025 | 56,610 | 825,111 | 802,222 | 88,382 | 0.097062 | 0.404328 |

해석:

- self-return은 학습/테스트 모두 평균적으로 약 9%대 비중을 보였다.
- 따라서 이 기록을 이상치로 제거하면 실제 이용 수요를 의미 있게 잃게 된다.
- 반면 `return_count`, `same_station_return_ratio`, `net_flow`를 함께 보면 수요와 재고 변동을 분리해서 해석할 수 있다.

#### 보고서/PPT에 넣을 수 있는 쉬운 메시지

- “같은 대여소에서 빌리고 같은 대여소로 반납한 기록은 이상치로 제거하지 않았다.”
- “이 경우 하루 말 재고 변화는 작을 수 있지만, 실제 이용 수요와 자전거 점유는 발생했기 때문이다.”
- “따라서 self-return은 제거 대상이 아니라, 재고 변동성 해석용 보조 지표로 관리하는 것이 더 적절하다.”
- “실제 station-day 집계 결과 self-return 비율은 평균 약 9%대로 확인되어, 무시하기 어려운 운영 지표였다.”

#### 필요한 시각화/표

- [x] self-return 비율 설명 차트
- [x] 대여량, 반납량, 순유출입 관계 요약 차트
- [ ] 적정 보유 대수 해석용 보조 지표 정의 표

#### 반영 문서

- `works/ddri_prediction_dataset_design.md`
- `works/presentation/ddri_project_temp_presentation.md`
- `works/presentation/ddri_clustering_presentation_a4_landscape.md`

#### 시각화 저장 경로

- `works/prediction/images/ddri_flow_metrics_summary.png`
- `works/prediction/images/ddri_same_station_return_ratio_boxplot.png`

---

## 4. 참고자료 로그

### 4.1 로컬 참고자료

확인된 로컬 참고자료:

- `3조 공유폴더/참고자료/[All about 따릉이 EDA, 6편] 대여소별 따릉이 대여건수 예측.webloc`
- `3조 공유폴더/참고자료/서울 자전거 공유 수요 예측 데이터셋.webloc`
- `3조 공유폴더/참고자료/Must-have 캐글- 2부 6장-자전거 대여 수요 예측 (1) - 컴퓨터를 여행하는 현대인을 위한 안내서.webloc`
- `3조 공유폴더/참고자료/[머신러닝 3조] 서울시 따릉이 자전거 이용 예측 AI모델.webloc`
- `3조 공유폴더/참고자료/서울시공유자전거의수요예측모델개발.pdf`

활용 판단:

- 블로그형 참고자료는 EDA 아이디어, feature 아이디어, 설명용 사례 참고에 유용
- 방법론 의사결정 근거는 학술자료/공식문서 우선 사용

## 5. 차트 및 이미지 산출물 관리

### 5.1 차트 저장 규칙

- 저장 기본 폴더: `works/clustering/images/`
- 파일명 prefix: `ddri_`
- 파일명 예시:
  - `ddri_station_count_by_year.png`
  - `ddri_kmeans_elbow.png`
  - `ddri_kmeans_pca_scatter.png`

### 5.2 현재 예정 차트 목록

- [ ] 연도별 대여소 수 비교 차트
- [ ] 공통 대여소/연도별 전용 대여소 비교 표 또는 차트
- [ ] 대여소별 평균 대여량 분포
- [ ] 평일/주말 평균 대여량 비교
- [x] Elbow plot
- [x] Silhouette 비교 차트
- [x] PCA 군집 산점도
- [x] 군집별 feature 평균 비교 차트
- [ ] 강남구 지도 위 군집 시각화
- [x] 전처리 전후 row 수 비교 차트
- [x] 전처리 제거 사유 차트

### 5.3 차트별 활용 목적

| 차트 | 활용 목적 | 예상 사용 위치 |
|---|---|---|
| 연도별 대여소 수 비교 | 분석 대상 선정 이유 설명 | 발표 초반부 |
| Elbow / Silhouette | 군집 수 선택 근거 설명 | 방법론 파트 |
| PCA 군집 산점도 | 군집 분리 결과 설명 | 결과 파트 |
| 군집별 feature 평균 비교 | 군집 해석 | 결과 파트 |
| 지도 위 군집 시각화 | 직관적 공간 설명 | 결과/PPT 핵심 장표 |

## 6. 전처리 기준 정리

### 6.1 현재 1차 군집화에 실제 적용한 전처리

현재 군집화 베이스라인에서는 아래 규칙을 적용했다.

#### 결측치 처리 기준

- `대여일시` 결측 제거
- `반납일시` 결측 제거
- `대여 대여소번호` 결측 제거
- `반납대여소번호` 결측 제거
- `이용시간(분)` 결측 제거
- `이용거리(M)` 결측 제거

적용 이유:

- 군집화 feature 생성에 직접 필요한 핵심 컬럼이므로, 임의 보간보다 제거가 안전하다.
- 대여 이벤트 단위 원천 로그에서 날짜/대여소/운행 정보가 비어 있으면 행 자체의 신뢰성이 낮다.

#### 이상치 처리 기준

- `이용시간(분) <= 0` 제거
- `이용거리(M) <= 0` 제거
- 공통 대여소 기준에 속하지 않는 대여 대여소 제거
- 강남구 기준 대여소 마스터에 속하지 않는 반납 대여소 제거
- 동일 대여소 대여-반납(self-return)은 제거하지 않고 유지

적용 이유:

- 0분 이하 이용시간, 0m 이하 이용거리는 정상 운행으로 보기 어렵다.
- 이번 군집화 목적은 `강남구 공통 대여소의 내부 이용 패턴` 분석이므로 기준 밖 대여소 이동은 제외한다.
- 외부 지역 반납 이동은 강남구 내부 패턴 해석을 흐릴 수 있으므로 이상치로 간주한다.
- self-return은 하루 말 재고 변화는 작을 수 있어도 실제 대여와 자전거 점유가 발생하므로 이상치보다 운영 해석용 지표로 보는 것이 더 적절하다.

#### 적용 파일 및 근거

- 실행 스크립트:
  - `works/clustering/ddri_station_clustering_baseline.py`
- 전처리 로그:
  - `works/clustering/data/ddri_cleaning_log.csv`

### 6.2 일반적으로 명시해야 하는 전처리 항목

아래 항목은 보고서에서 “검토 대상 전처리”로 명시하는 것이 좋다.

#### 1. 스키마 및 타입 정리

- 날짜형 컬럼 파싱
- 대여소 번호 자료형 통일
- 수치형 컬럼 문자열/쉼표 제거
- 인코딩 이슈 정리

보고서 메시지:
- “원천 파일마다 컬럼 형식이 달라질 수 있으므로 분석 전 자료형을 통일했다.”

#### 2. 중복 제거

- 완전 중복 행 제거
- 동일 이벤트가 중복 적재된 경우 중복 기준 정의

현재 상태:
- 중복 여부 점검 완료
- 전체 2,896,795행 중 완전 중복 21건 확인
- 1차 분석 영향은 제한적이라고 판단

#### 3. 날짜 범위 검증

- 학습 구간: 2023~2024만 포함
- 테스트 구간: 2025만 포함
- 잘못 섞인 날짜 존재 여부 확인

보고서 메시지:
- “학습과 테스트의 시간 누수를 막기 위해 연도 기준으로 데이터 구간을 엄격히 분리했다.”

#### 4. 공간 범위 검증

- 강남구 기준 대여소만 포함
- 기준 밖 대여/반납은 제거 또는 별도 분류

보고서 메시지:
- “공간 범위가 다른 이동 기록은 이번 분석 목적과 맞지 않아 제외했다.”

#### 5. 비정상 운행값 처리

- 음수 또는 0값
- 비현실적으로 큰 시간/거리 값
- 속도 기준으로 비정상 이동 탐지 가능

현재 상태:
- 0 이하 시간/거리만 제거 적용
- feature 기준 IQR 이상치 후보는 점검 완료
- 상한 기반 자동 제거는 아직 미적용

후속 검토 필요:
- IQR 기반 상한
- 분위수 기반 상한
- 속도(`거리/시간`) 기준 비정상값 탐지

#### 6. 결측치 처리 방식 선택

- 제거(drop)
- 대표값 대체
- 시계열 보간
- 도메인 규칙 기반 보정

현재 상태:
- 핵심 로그 컬럼은 제거 방식 적용
- 보간이나 대체는 아직 적용하지 않음

#### 7. 대여소 마스터 정합성 처리

- 연도별 대여소 증감 반영
- 공통 대여소만 사용할지, 신규 대여소를 별도 처리할지 결정

현재 상태:
- 1차 군집화는 2023~2025 공통 대여소 기준으로 확정

#### 8. 분석용 feature 생성 전 집계 기준 통일

- 시간 단위 분석인지 일 단위 분석인지 확정
- 평일/주말 구분 기준 고정
- 출퇴근/야간 시간대 정의 고정

현재 상태:
- 군집화에서는 대여소별 일 단위 집계 기반 feature 사용
- 출퇴근 시간: `07~09`, `18~20`
- 야간 시간: `22~05`

#### 9. 스케일링

- KMeans는 거리 기반 알고리즘이라 feature scaling 필요
- StandardScaler 또는 MinMaxScaler 적용

현재 상태:
- StandardScaler 적용 완료

#### 10. 데이터 누락 대여소 처리

- 공통 대여소라도 실제 기간 내 이용 이력이 없을 수 있음
- feature 생성 불가 대여소는 별도 표로 관리 필요

현재 상태:
- coverage 요약 파일 생성 완료
  - `works/clustering/data/ddri_station_coverage_summary.csv`
  - `works/clustering/data/ddri_station_coverage_flags.csv`

### 6.3 보고서에 그대로 넣을 수 있는 전처리 요약 문장

예시 문장:

“본 연구에서는 강남구 공통 대여소를 기준으로 데이터를 정합화한 뒤, 결측 행 제거, 비정상 운행값 제거, 기준 외 대여소/반납대여소 제거를 수행하였다. 이후 날짜형/수치형 컬럼의 자료형을 통일하고, 대여소별 일 단위 집계를 통해 군집화 feature를 생성하였다. KMeans 적용 전에는 feature 스케일 차이를 줄이기 위해 StandardScaler를 적용하였다.”

### 6.4 아직 남아 있는 전처리 검토 항목

- 중복 로그 21건을 실제 제거할지 여부
- 극단치 상한을 2차 분석에서 적용할지 여부
- 속도 기반 이상치 탐지 여부
- 외부 반납 이동을 완전 제거할지, 별도 파생 feature로 보존할지 여부

## 7. 추가 자료 필요 메모

- 현재 없음
- 이후 작업 중 필수 기준이나 데이터가 부족하면 여기에 누적 기록
