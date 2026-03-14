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

- 표 데이터: `works/01_clustering/06_data/ddri_station_master_year_summary.csv`
- coverage 요약: `works/01_clustering/06_data/ddri_station_coverage_summary.csv`
- coverage 상세: `works/01_clustering/06_data/ddri_station_coverage_flags.csv`

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
  - `works/01_clustering/01_baseline/ddri_station_clustering_baseline.ipynb`
- 초기 산출물:
  - `works/01_clustering/06_data/ddri_station_master_year_summary.csv`

#### 보고서/PPT에 넣을 수 있는 메시지

- “군집화는 별도 베이스라인 노트북으로 분리하여 재현 가능한 분석 단위로 구성했다.”
- “입력 데이터, 전처리, 군집화, 시각화 흐름을 한 노트북에서 추적 가능하게 만들었다.”

#### 필요한 시각화/표

- [ ] 공통 대여소 선정 과정 표
- [ ] 학습/테스트 데이터 분리 설명 표
- [ ] 군집화 feature 정의 표

#### 시각화 저장 경로

- 노트북: `works/01_clustering/01_baseline/ddri_station_clustering_baseline.ipynb`
- 표 데이터: `works/01_clustering/06_data/ddri_station_master_year_summary.csv`

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

- `works/01_clustering/06_data/ddri_kmeans_search_metrics.csv`
- `works/01_clustering/07_images/ddri_kmeans_elbow_silhouette.png`
- `works/01_clustering/07_images/ddri_kmeans_pca_scatter.png`
- `works/01_clustering/07_images/ddri_cluster_feature_means.png`
- `works/01_clustering/06_data/ddri_cluster_summary.csv`

#### 보고서/PPT에 넣을 수 있는 메시지

- “1차 군집화는 K 후보를 비교한 뒤 silhouette 기준으로 k=2를 채택했다.”
- “초기 baseline에서는 분리도가 가장 좋은 단순 구조를 우선 채택했다.”

#### 필요한 시각화/표

- [x] Elbow / Silhouette 비교 차트
- [x] PCA 군집 산점도
- [x] 군집별 feature 평균표
- [ ] 군집별 명칭 정리 표

#### 시각화 저장 경로

- `works/01_clustering/07_images/ddri_kmeans_elbow_silhouette.png`
- `works/01_clustering/07_images/ddri_kmeans_pca_scatter.png`
- `works/01_clustering/07_images/ddri_cluster_feature_means.png`
- `works/01_clustering/06_data/ddri_cluster_summary.csv`

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

- `works/01_clustering/06_data/ddri_station_coverage_summary.csv`
- `works/01_clustering/06_data/ddri_station_coverage_flags.csv`

#### 보고서/PPT에 넣을 수 있는 메시지

- “공통 운영 대여소만 남겨도 실제 이용 이력이 없는 대여소는 별도 제외가 필요했다.”
- “분석용 마스터 수와 실제 feature 생성 수는 다를 수 있다.”

#### 필요한 시각화/표

- [x] coverage 요약 표
- [ ] 누락 대여소 처리 기준 표

#### 시각화 저장 경로

- `works/01_clustering/06_data/ddri_station_coverage_summary.csv`
- `works/01_clustering/06_data/ddri_station_coverage_flags.csv`

---

### Decision 005. 결측치와 이상치는 1차 군집화에서도 반드시 제거

#### 결정 내용

- 1차 군집화 전처리에서 아래 항목을 제거한다.
  - 필수 컬럼 결측치
  - `이용시간(분) <= 0`
  - `이용거리(M) <= 0`
  - `동일 대여소 반납`이면서 `이용시간(분) <= 5`
  - 공통 대여소 기준에 포함되지 않는 대여 대여소
  - 강남구 기준 대여소 마스터에 포함되지 않는 반납 대여소

#### 결정 이유

- 군집화는 패턴 기반이므로 비정상 이동이 섞이면 feature 왜곡이 발생한다.
- 사용자 요구사항에 따라 강남구 외부 반납 이동은 이상치로 간주하고 제외한다.
- 결측치와 비정상값을 그대로 두면 평균 대여량, 분산, 시간대 비율 계산이 왜곡될 수 있다.

#### 내부 실행 근거

실행 스크립트:

- `works/01_clustering/01_baseline/ddri_station_clustering_baseline.py`

전처리 로그:

- `works/01_clustering/02_preprocessing/data/ddri_cleaning_log.csv`

집계 결과:

| group | rows_before | rows_after | dropped_nonpositive | dropped_short_same_station_return | dropped_noncommon_rent | dropped_outside_gangnam_return |
|---|---:|---:|---:|---:|---:|---:|
| train_2023 | 1,012,480 | 907,642 | 68,978 | 9,767 | 26,093 | 0 |
| train_2024 | 1,014,918 | 936,407 | 54,057 | 7,069 | 17,385 | 0 |
| test_2025 | 869,397 | 819,889 | 15,022 | 5,558 | 28,928 | 0 |

해석:

- 실제 제거 건의 대부분은 비정상 시간/거리 값과 공통 기준 밖 대여소였다.
- 추가로 `동일 대여소 반납 + 5분 이하` 로그를 별도 제거해, 미사용 또는 즉시 반납에 가까운 사례를 정리했다.
- 현재 강남구 대여소 기준으로 유지된 표본에서는 `외부 반납` 제거 건수가 최종 집계상 0건이었다.
- 이는 강남구 이용정보 파일이 이미 상당 부분 지역 기준으로 정리되어 있거나, 공통 대여소 필터를 거치며 함께 제외되었기 때문으로 해석된다.

#### 보고서/PPT에 넣을 수 있는 메시지

- “결측치와 비정상 운행 기록을 제거한 뒤 군집화를 수행했다.”
- “동일 대여소로 5분 이내 반납된 로그는 미사용 가능성이 있어 전처리 단계에서 제외했다.”
- “강남구 내부 이동 패턴에 집중하기 위해 외부 반납 이동은 이상치로 정의했다.”
- “실제 1차 분석에서는 비정상 시간/거리 값과 기준 밖 대여소 제거 영향이 더 컸다.”

#### 필요한 시각화/표

- [x] 전처리 전후 row 수 비교 표
- [x] 전처리 제거 사유별 비중 차트
- [ ] 월별 제거 비율 차트

#### 시각화 저장 경로

- `works/01_clustering/02_preprocessing/data/ddri_cleaning_log.csv`
- `works/01_clustering/02_preprocessing/data/ddri_cleaning_summary_by_group.csv`
- `works/01_clustering/02_preprocessing/images/ddri_cleaning_before_after.png`
- `works/01_clustering/02_preprocessing/images/ddri_cleaning_drop_reasons.png`

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
  - `works/01_clustering/02_preprocessing/data/ddri_duplicate_check_summary.csv`
- 파일별 상세:
  - `works/01_clustering/02_preprocessing/data/ddri_duplicate_check_by_file.csv`

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

- `works/01_clustering/02_preprocessing/data/ddri_duplicate_check_summary.csv`
- `works/01_clustering/02_preprocessing/data/ddri_duplicate_check_by_file.csv`

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
  - `works/01_clustering/02_preprocessing/data/ddri_feature_distribution_summary.csv`
- IQR 요약:
  - `works/01_clustering/02_preprocessing/data/ddri_feature_iqr_outlier_summary.csv`

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

- `works/01_clustering/02_preprocessing/data/ddri_feature_distribution_summary.csv`
- `works/01_clustering/02_preprocessing/data/ddri_feature_iqr_outlier_summary.csv`

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

- `works/01_clustering/06_data/ddri_cluster_summary.csv`
- `works/01_clustering/07_images/ddri_cluster_feature_means.png`

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

- `works/01_clustering/04_maps/ddri_cluster_folium_map.py`
- `works/01_clustering/04_maps/ddri_cluster_map_gangnam.html`

#### 보고서/PPT에 넣을 수 있는 메시지

- “군집 결과는 Folium 지도를 통해 공간적으로 시각화하였다.”
- “대여소별 군집과 평균 대여량을 지도에서 직접 확인할 수 있도록 구성하였다.”

#### 필요한 시각화/표

- [x] Folium 군집 지도
- [ ] 발표용 지도 캡처 이미지

#### 시각화 저장 경로

- `works/01_clustering/04_maps/ddri_cluster_map_gangnam.html`

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
  - `works/01_clustering/03_environment/data/ddri_cluster_environment_features.csv`
- 군집별 요약:
  - `works/01_clustering/03_environment/data/ddri_cluster_environment_summary.csv`
- 대표 대여소:
  - `works/01_clustering/03_environment/data/ddri_cluster_representative_stations.csv`
- 비교 차트:
  - `works/01_clustering/03_environment/images/ddri_cluster_environment_comparison.png`

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

- `works/01_clustering/03_environment/images/ddri_cluster_environment_comparison.png`
- `works/01_clustering/03_environment/data/ddri_cluster_environment_summary.csv`
- `works/01_clustering/03_environment/data/ddri_cluster_representative_stations.csv`

---

### 발표 패키지 메모

군집화 발표 직전 바로 사용할 수 있는 정리 파일:

- 1페이지 요약:
  - `works/01_clustering/05_presentation/01_ddri_clustering_presentation_summary.md`
- 슬라이드 문구/발표 멘트:
  - `works/01_clustering/05_presentation/02_ddri_clustering_slide_script.md`
- 슬라이드용 수치표:
  - `works/01_clustering/06_data/ddri_clustering_slide_tables.csv`
- 지도:
  - `works/01_clustering/04_maps/ddri_cluster_map_gangnam.html`

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

- 문서: `works/03_ddri_prediction_target_definition.md`

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

- 문서: `works/04_ddri_prediction_dataset_design.md`

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
  - `works/02_data_collection/01_calendar/ddri_holiday_api_fetch.ipynb`
- 실행 스크립트:
  - `works/02_data_collection/01_calendar/ddri_holiday_api_fetch.py`
- 원천 API 결과:
  - `works/02_data_collection/01_calendar/data/ddri_holiday_api_raw_2023_2025.csv`
- 일 단위 캘린더:
  - `works/02_data_collection/01_calendar/data/ddri_calendar_daily_2023_2025.csv`

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

- `works/02_data_collection/01_calendar/data/ddri_holiday_api_raw_2023_2025.csv`
- `works/02_data_collection/01_calendar/data/ddri_calendar_daily_2023_2025.csv`

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

- `works/03_prediction/02_data/ddri_station_day_target_train_2023_2024.csv`
- `works/03_prediction/02_data/ddri_station_day_target_test_2025.csv`
- `works/03_prediction/02_data/ddri_weather_daily_2023_2024.csv`
- `works/03_prediction/02_data/ddri_station_day_train_baseline_dataset.csv`

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

- `works/03_prediction/02_data/ddri_station_day_target_train_2023_2024.csv`
- `works/03_prediction/02_data/ddri_station_day_target_test_2025.csv`
- `works/03_prediction/02_data/ddri_weather_daily_2023_2024.csv`
- `works/03_prediction/02_data/ddri_station_day_train_baseline_dataset.csv`

---

### Decision 013. Open-Meteo로 2025 날씨를 보완해 테스트용 베이스라인 데이터셋까지 생성

#### 결정 내용

- 기존 `openmeteo.ipynb` 방식과 동일한 Open-Meteo Archive API를 사용해 아래 데이터를 추가 수집했다.
  - `2024-01-01` 시간별 날씨
  - `2025-01-01 ~ 2025-12-31` 시간별 날씨
- 이를 기반으로 테스트용 베이스라인 데이터셋까지 생성했다.

#### 생성 파일

- 수집 노트북:
  - `works/02_data_collection/02_weather/ddri_openmeteo_fetch.ipynb`
- 실행 스크립트:
  - `works/02_data_collection/02_weather/ddri_openmeteo_fetch.py`
- 추가 시간별 날씨:
  - `works/02_data_collection/02_weather/data/ddri_weather_2024_0101_hourly.csv`
  - `works/02_data_collection/02_weather/data/ddri_weather_2025_hourly.csv`
- 일 단위 날씨:
  - `works/03_prediction/02_data/ddri_weather_daily_2023_2024.csv`
  - `works/03_prediction/02_data/ddri_weather_daily_2025.csv`
- 최종 베이스라인 데이터셋:
  - `works/03_prediction/02_data/ddri_station_day_train_baseline_dataset.csv`
  - `works/03_prediction/02_data/ddri_station_day_test_baseline_dataset.csv`

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

- `works/02_data_collection/02_weather/data/ddri_weather_2025_hourly.csv`
- `works/03_prediction/02_data/ddri_weather_daily_2025.csv`
- `works/03_prediction/02_data/ddri_station_day_test_baseline_dataset.csv`

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

- `works/01_clustering/01_baseline/ddri_clustering_input_manifest.md`

---

### Decision 015. 동일 대여소 반납은 5분 이하 즉시 반납만 제거하고 나머지는 운영 해석용 지표로 관리

#### 결정 내용

- `대여 대여소번호 == 반납대여소번호`인 기록 중 `이용시간(분) <= 5`인 경우는 전처리에서 제거한다.
- 그 외 self-return 기록은 유지한다.
- 대신 아래 지표를 별도 보조 feature 후보로 관리한다.
  - `same_station_return_count`
  - `same_station_return_ratio`
  - `return_count`
  - `net_flow`

#### 쉬운 설명

- 같은 대여소에서 빌리고 같은 대여소로 반납하면, 하루가 끝났을 때 그 대여소의 자전거 수는 크게 달라지지 않을 수 있다.
- 하지만 그 자전거는 운행 중 실제로 사용되었고, 그 시간 동안은 다른 이용자가 바로 쓸 수 없다.
- 다만 `5분 이하`의 즉시 반납은 실제 미사용 또는 점검성 이동일 가능성이 있어 제거하는 편이 더 보수적이다.
- 따라서 짧은 즉시 반납만 걸러내고, 그 외 기록은 대여 수요는 있었지만 재고 변화는 작았던 사례로 보는 것이 맞다.

#### 결정 이유

- 군집화와 수요 예측의 target은 `대여량`이므로, 일정 시간 이상 실제 사용된 self-return은 수요로 포함하는 것이 자연스럽다.
- 적정 보유 대수는 단순히 하루 끝 재고만 보는 문제가 아니라, 시간대별 점유와 순간 부족 가능성까지 함께 봐야 한다.
- self-return을 제거하면 실제 이용 수요를 과소평가할 위험이 있다.
- 반대로 이 값을 별도 지표로 두면 `수요는 있었지만 순재고 변화는 작았던 날`을 해석하는 데 도움이 된다.

#### 현재 코드 적용 상태

- 현재 전처리에서는 `동일 대여소 반납 + 5분 이하`만 제거한다.
- 즉, 결측치/비정상 시간·거리/기준 밖 대여소 제거에 더해 짧은 즉시 반납은 제외하고, 그 외 self-return 기록은 유지하고 있다.

관련 코드:

- `works/01_clustering/01_baseline/ddri_station_clustering_baseline.py`
- `works/03_prediction/04_scripts/ddri_station_day_dataset_builder.py`

#### 프로젝트 적용 원칙

- 군집화:
  - `동일 대여소 반납 + 5분 이하`는 제거
  - 그 외 self-return은 이상치로 제거하지 않음
  - 현재 군집화 feature에는 직접 넣지 않았지만, 향후 운영 해석 보조 지표로 확장 가능
- 예측 데이터셋:
  - self-return 비율과 순유출입(`net_flow`)을 실제 컬럼으로 생성
  - 특히 `적정 보유 대수` 해석 단계에서 중요하게 사용할 수 있음

#### 내부 실행 결과

생성 파일:

- `works/03_prediction/02_data/ddri_station_day_flow_metrics_summary.csv`
- `works/03_prediction/02_data/ddri_station_day_train_baseline_dataset.csv`
- `works/03_prediction/02_data/ddri_station_day_test_baseline_dataset.csv`

요약:

| dataset | rows | rental_count_sum | return_count_sum | same_station_return_count_sum | same_station_return_ratio_mean | net_flow_mean |
|---|---:|---:|---:|---:|---:|---:|
| train_2023_2024 | 114,883 | 1,844,049 | 1,799,990 | 177,958 | 0.084884 | 0.383512 |
| test_2025 | 56,586 | 819,889 | 796,958 | 83,168 | 0.091156 | 0.405242 |

해석:

- `동일 대여소 반납 + 5분 이하` 제거 이후에도 self-return은 학습 약 8.49%, 테스트 약 9.12% 수준을 보였다.
- 즉, 짧은 즉시 반납을 제외하더라도 self-return 전체를 이상치로 보기에는 비중이 작지 않다.
- 반면 `return_count`, `same_station_return_ratio`, `net_flow`를 함께 보면 수요와 재고 변동을 분리해서 해석할 수 있다.

#### 보고서/PPT에 넣을 수 있는 쉬운 메시지

- “동일 대여소로 5분 이내 반납된 로그만 전처리에서 제거했다.”
- “그 외 self-return은 하루 말 재고 변화는 작을 수 있지만, 실제 이용 수요와 자전거 점유는 발생했기 때문에 유지했다.”
- “따라서 짧은 즉시 반납만 걸러내고, 나머지 self-return은 재고 변동성 해석용 보조 지표로 관리하는 것이 더 적절하다.”

#### 필요한 시각화/표

- [x] self-return 비율 설명 차트
- [x] 대여량, 반납량, 순유출입 관계 요약 차트
- [ ] 적정 보유 대수 해석용 보조 지표 정의 표

#### 반영 문서

- `works/04_ddri_prediction_dataset_design.md`
- `works/04_presentation/02_project/01_ddri_project_temp_presentation.md`
- `works/04_presentation/01_clustering/01_ddri_clustering_presentation_a4_landscape.md`

#### 시각화 저장 경로

- `works/03_prediction/03_images/ddri_flow_metrics_summary.png`
- `works/03_prediction/03_images/ddri_same_station_return_ratio_boxplot.png`

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

- 저장 기본 폴더: `works/01_clustering/07_images/`
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
- `동일 대여소 반납`이면서 `이용시간(분) <= 5` 제거
- 공통 대여소 기준에 속하지 않는 대여 대여소 제거
- 강남구 기준 대여소 마스터에 속하지 않는 반납 대여소 제거
- 그 외 동일 대여소 대여-반납(self-return)은 유지

적용 이유:

- 0분 이하 이용시간, 0m 이하 이용거리는 정상 운행으로 보기 어렵다.
- 동일 대여소로 5분 이내 반납된 로그는 실제 미사용 또는 즉시 반납 가능성이 있어 전처리에서 제외하는 것이 더 보수적이다.
- 이번 군집화 목적은 `강남구 공통 대여소의 내부 이용 패턴` 분석이므로 기준 밖 대여소 이동은 제외한다.
- 외부 지역 반납 이동은 강남구 내부 패턴 해석을 흐릴 수 있으므로 이상치로 간주한다.
- 나머지 self-return은 하루 말 재고 변화는 작을 수 있어도 실제 대여와 자전거 점유가 발생하므로 이상치보다 운영 해석용 지표로 보는 것이 더 적절하다.

#### 적용 파일 및 근거

- 실행 스크립트:
  - `works/01_clustering/01_baseline/ddri_station_clustering_baseline.py`
- 전처리 로그:
  - `works/01_clustering/02_preprocessing/data/ddri_cleaning_log.csv`

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
  - `works/01_clustering/06_data/ddri_station_coverage_summary.csv`
  - `works/01_clustering/06_data/ddri_station_coverage_flags.csv`

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

---

## 8. 문서화/근거 차트 정책

### Decision 015. 대표 노트북은 설명형 실험 문서로 유지하고, 근거 차트는 선정 이유 설명용으로 확장

#### 결정 내용

- 대표 노트북은 실행 코드만 있는 형태가 아니라, 충분한 마크다운 설명과 코드 주석을 포함한 `설명형 실험 문서`로 관리한다.
- 전처리, 군집화, 환경 해석, 예측 데이터셋 단계별로 근거 차트를 최대한 확장하되, `왜 이 기준을 선택했는지`를 설명하는 차트를 우선한다.
- 이 기준의 정본은 아래 문서로 고정한다.
  - `works/00_overview/07_ddri_notebook_and_evidence_chart_policy.md`

#### 결정 이유

- 현재 프로젝트는 개인 메모가 아니라 팀 단위 재현, 발표, 인수인계가 가능한 구조가 필요하다.
- 코드만 있는 노트북은 만든 사람은 이해하지만, 읽는 사람은 흐름을 따라가기 어렵다.
- 발표/보고서에서도 결과 차트만으로는 부족하고, 데이터 선정 및 방법 선택의 근거 차트가 함께 있어야 설명력이 올라간다.

#### 보고서/PPT에 넣을 수 있는 메시지

- “이 프로젝트는 재실행용 스크립트와 설명형 노트북을 분리해 재현성과 열람성을 함께 확보했다.”
- “차트는 결과 제시용뿐 아니라 데이터 선정과 전처리 기준의 근거를 설명하는 용도로 확장했다.”

#### 필요한 시각화/표

- [x] 전처리 전후 비교 차트
- [x] 제거 사유별 차트
- [x] self-return / 순유출입 차트
- [ ] 군집 입력 특성 상관관계 히트맵
- [ ] 군집 프로파일 히트맵
- [ ] 휴일/주말 비교 차트
- [ ] 예측 입력 상관관계 히트맵

#### 관련 정책 문서

- `works/00_overview/07_ddri_notebook_and_evidence_chart_policy.md`

### Decision 016. 군집화/예측 baseline에 상관관계, 프로파일, 시계열 근거 차트를 추가

#### 결정 내용

- 기존의 결과 소개 차트 외에, 선정 이유를 설명하는 근거 차트를 추가 생성했다.
- 군집화 파트에는 아래 차트를 추가했다.
  - 입력 특성 상관관계 히트맵
  - 군집 프로파일 히트맵
  - 군집별 대여소 수 차트
  - 월별 대여 건수 추이
  - 요일-시간대 히트맵
- 예측 데이터셋 파트에는 아래 차트를 추가했다.
  - 예측 입력/운영 지표 상관관계 히트맵
  - 공휴일/비공휴일, 평일/주말 평균 대여량 비교
  - 월별 평균 대여량 추이

#### 결정 이유

- 기존 차트만으로는 `왜 이 feature를 쓰는지`, `왜 k=2를 유지하는지`, `왜 운영 보조 지표를 남겼는지` 설명이 부족했다.
- heatmap, 분포, 시계열 추이 차트를 함께 두면 발표와 보고서에서 해석의 근거를 더 직접적으로 제시할 수 있다.

#### 시각화 저장 경로

- `works/01_clustering/07_images/ddri_feature_correlation_heatmap.png`
- `works/01_clustering/07_images/ddri_cluster_profile_heatmap.png`
- `works/01_clustering/07_images/ddri_cluster_size.png`
- `works/01_clustering/07_images/ddri_monthly_rental_trend.png`
- `works/01_clustering/07_images/ddri_weekday_hour_heatmap.png`
- `works/03_prediction/03_images/ddri_prediction_feature_correlation_heatmap.png`
- `works/03_prediction/03_images/ddri_holiday_weekend_rental_comparison.png`
- `works/03_prediction/03_images/ddri_monthly_avg_rental_trend.png`

---

### Decision 017. 군집화는 1차/2차를 분리 제시하지 않고, 지구판단 중심 통합 군집화 결과로 정리

#### 결정 내용

- 기존 이용량 중심 baseline 결과 위에 반납 시간대, 순유입, 교통 접근성, 생활인구를 결합한 `통합 군집화`를 최종 결과로 사용한다.
- 보고서와 발표에서는 `1차 한계 -> 2차 보완` 구조보다, `기초 패턴 분석 + 지구판단 피처 고도화 + 최종 통합 결과` 흐름으로 설명한다.
- 최종 군집화의 메인 입력은 아래 7개로 확정했다.
  - `arrival_7_10_ratio` (07~10시 반납 비율)
  - `arrival_11_14_ratio` (11~14시 반납 비율)
  - `arrival_17_20_ratio` (17~20시 반납 비율)
  - `morning_net_inflow` (아침 순유입)
  - `evening_net_inflow` (저녁 순유입)
  - `subway_distance_m` (최근접 지하철 거리)
  - `bus_stop_count_300m` (300m 내 버스정류장 수)

#### 결정 이유

- 이용량 규모만으로는 업무지구/주거지구/상권의 공간적 역할 해석이 약했다.
- 반납 시간대와 순유입은 `도착 지구 역할`을 직접적으로 보여준다.
- 교통 접근성은 역세권/생활권/외곽형 해석을 보강한다.

#### 내부 확인 근거

- 최종 통합 피처:
  - `works/01_clustering/08_integrated/final/features/ddri_final_district_clustering_features_train_2023_2024.csv`
  - `works/01_clustering/08_integrated/final/features/ddri_final_district_clustering_features_test_2025.csv`
- 통합 군집화 입력 생성:
  - `works/01_clustering/08_integrated/intermediate/return_time_district/ddri_second_cluster_ready_input_train_2023_2024.csv`
  - `works/01_clustering/08_integrated/intermediate/return_time_district/ddri_second_cluster_ready_input_test_2025.csv`

#### 지표 원천

- `arrival_*_ratio`, `net_inflow`: `서울 열린데이터광장 공공자전거 이용정보`
- `subway_distance_m`: `서울 열린데이터광장 서울시 역사마스터 정보`
- `bus_stop_count_300m`: `서울 열린데이터광장 서울시 버스정류소 위치정보`
- `life_pop_*`: `서울 생활인구(내국인) 데이터` + `서울시 행정동코드 매핑표`

#### 보고서/PPT에 넣을 수 있는 메시지

- “군집화는 이용량 규모 분류에 머물지 않고, 반납 시간대와 순유입을 결합해 대여소의 지구 역할을 읽는 통합 구조로 확장했다.”
- “최종 입력은 아침·점심·저녁 도착 비율과 순유입, 교통 접근성을 중심으로 구성했다.”

#### 필요한 시각화/표

- [x] 2025 반납 시간대 지도 3장
- [x] k 탐색 차트
- [x] PCA scatter
- [x] 군집 프로파일 heatmap
- [x] 사분면 해석 차트
- [x] 군집 정적 지도

#### 시각화 저장 경로

- `works/01_clustering/08_integrated/intermediate/return_time_district/ddri_return_map_2025_7_10.png`
- `works/01_clustering/08_integrated/intermediate/return_time_district/ddri_return_map_2025_11_14.png`
- `works/01_clustering/08_integrated/intermediate/return_time_district/ddri_return_map_2025_17_20.png`
- `works/01_clustering/08_integrated/final/results/second_clustering_results/images/ddri_second_kmeans_elbow_silhouette.png`
- `works/01_clustering/08_integrated/final/results/second_clustering_results/images/ddri_second_kmeans_pca_scatter.png`
- `works/01_clustering/08_integrated/final/results/second_clustering_results/images/ddri_second_cluster_profile_heatmap.png`
- `works/01_clustering/08_integrated/final/results/second_clustering_results/images/ddri_second_cluster_quadrant_views.png`
- `works/01_clustering/08_integrated/final/results/second_clustering_results/images/ddri_second_cluster_static_map.png`

---

### Decision 018. 기본 통합 군집화는 k=5를 메인 결과로 채택

#### 결정 내용

- 통합 군집화는 `k=5 ~ 7` 범위에서 탐색했고, 최종 메인 결과는 `k=5`를 사용한다.
- 현재 해석 초안은 아래와 같다.
  - Cluster 0: 업무/상업 혼합형
  - Cluster 1: 초강한 아침 도착 업무 거점형
  - Cluster 2: 주거 도착형
  - Cluster 3: 생활·상권 혼합형
  - Cluster 4: 외곽 주거형

#### 결정 이유

- 사용자 요구상 5개 이상 군집을 우선 검토했다.
- `k=5`가 검토 범위 내에서 가장 높은 silhouette를 보였다.
- 발표와 이후 예측 연결성을 고려할 때, 5개 군집은 지구 역할 구분과 군집 규모의 균형이 가장 좋았다.

#### 내부 확인 근거

- `k=5`: `0.2033`
- `k=6`: `0.1795`
- `k=7`: `0.1708`
- 결과 저장:
  - `works/01_clustering/08_integrated/final/results/second_clustering_results/data/ddri_second_kmeans_search_metrics.csv`
  - `works/01_clustering/08_integrated/final/results/second_clustering_results/data/ddri_second_cluster_summary.csv`
  - `works/01_clustering/08_integrated/final/results/second_clustering_results/data/ddri_second_cluster_train_with_labels.csv`

#### 보고서/PPT에 넣을 수 있는 메시지

- “5개 군집은 업무/상업, 강한 아침 도착 허브, 주거 도착형, 생활·상권 혼합형, 외곽 주거형을 구분하는 데 가장 균형적이었다.”

#### 필요한 시각화/표

- [x] k별 inertia/silhouette 표
- [x] 군집별 station 수
- [x] 군집별 대표 대여소 표
- [x] 군집 가설 cross-tab

#### 시각화/표 저장 경로

- `works/01_clustering/08_integrated/final/results/second_clustering_results/data/ddri_second_cluster_representative_stations.csv`
- `works/01_clustering/08_integrated/final/results/second_clustering_results/data/ddri_second_cluster_hypothesis_crosstab.csv`
- `works/01_clustering/08_integrated/final/results/second_clustering_results/images/ddri_second_cluster_size.png`
- `works/01_clustering/08_integrated/final/results/second_clustering_results/images/ddri_second_cluster_hypothesis_crosstab.png`

---

### Decision 019. 표고·녹지·하천경계 기반 환경 보강 실험은 보조 근거로 유지

#### 결정 내용

- 환경 보강 피처를 추가한 별도 군집화 실험을 수행했다.
- 추가한 피처는 아래와 같다.
  - `station_elevation_m` (대여소 표고)
  - `elevation_diff_nearest_subway_m` (최근접 지하철 대비 고도차)
  - `nearest_park_area_sqm` (최근접 공원 면적)
  - `distance_naturepark_m` (도시자연공원구역 거리)
  - `distance_river_boundary_m` (최근접 하천경계 거리)
- 현재 단계에서는 환경 보강 버전을 메인 군집화로 채택하지 않고, `외곽형/녹지형 해석 보강` 근거로 사용한다.

#### 결정 이유

- 환경 피처는 외곽 주거형과 대형 녹지 인접형의 해석을 강화했다.
- 그러나 전체 군집 분리도는 기본 통합 군집화보다 개선되지 않았다.
- 따라서 발표 메인 구조는 반납 시간대 기반 통합 군집화를 유지하고, 환경 보강은 `고도화 실험 결과`로 제시하는 것이 안전하다.

#### 내부 확인 근거

- 환경 보강 silhouette
  - `k=5`: `0.1569`
  - `k=6`: `0.1577`
  - `k=7`: `0.1456`
- 결과 저장:
  - `works/01_clustering/08_integrated/intermediate/enriched_second_clustering_results/data/ddri_enriched_kmeans_search_metrics.csv`
  - `works/01_clustering/08_integrated/intermediate/enriched_second_clustering_results/enriched_vs_base_comparison.md`

#### 지표 원천

- `station_elevation_m`, `elevation_diff_*`: `Open-Meteo Elevation API`
- `nearest_park_area_sqm`: `강남구 공원 정보 공공데이터`
- `distance_naturepark_m`: `서울 열린데이터광장 도시자연공원구역`
  - https://data.seoul.go.kr/dataList/OA-21135/S/1/datasetView.do
- `distance_river_boundary_m`: `브이월드/국토정보지리원 연속수치지형도 하천경계 데이터`
  - https://www.vworld.kr/dtmk/dtmk_ntads_s002.do?svcCde=MK&dsId=30248

#### 보고서/PPT에 넣을 수 있는 메시지

- “표고·녹지·하천경계 접근성은 외곽 군집의 공간 해석을 강화했지만, 메인 군집 구조를 대체할 정도의 분리도 개선은 보이지 않았다.”

#### 필요한 시각화/표

- [x] 환경 보강 feature 평균 차트
- [x] 환경 보강 k 탐색 차트
- [x] 기본 결과와의 비교 메모

#### 시각화/표 저장 경로

- `works/01_clustering/08_integrated/intermediate/enriched_second_clustering_results/images/ddri_enriched_cluster_feature_means.png`
- `works/01_clustering/08_integrated/intermediate/enriched_second_clustering_results/images/ddri_enriched_kmeans_elbow_silhouette.png`
- `works/01_clustering/08_integrated/intermediate/enriched_second_clustering_results/enriched_vs_base_comparison.md`

---

### Decision 020. 군집화 작업물은 08_integrated를 기준으로 final / intermediate / pipeline / source_data 구조로 재정리

#### 결정 내용

- 개인 작업 폴더였던 `cheng80/`의 산출물을 `works/01_clustering/08_integrated` 아래로 이관했다.
- `works/01_clustering`의 활성 구조는 아래 기준으로 단순화했다.
  - `08_integrated`: 현재 최신 통합 군집화 작업 루트
  - `archive_1st`: 기존 1차 군집화 결과 보관본
- `08_integrated` 내부는 다시 아래 4개 역할로 나눴다.
  - `pipeline/`: 스크립트, 노트북, 작업 메모
  - `source_data/`: 공통 기준표
  - `final/`: 최종 입력과 최종 결과
  - `intermediate/`: 반납 시간대, 환경 보강, 비교 실험 중간 산출물

#### 결정 이유

- 기존 구조에서는 최신 통합 군집화 산출물이 개인 작업 폴더와 공식 폴더에 나뉘어 있어 추적이 어려웠다.
- `final`과 `intermediate`를 구분하면 작업자가 최종본과 중간 산출물을 혼동하지 않는다.
- `pipeline`을 따로 두면 재실행 경로가 명확해진다.

#### 내부 확인 근거

- 구조 정리 후 핵심 실행 검증 완료:
  - `05_build_return_time_district_features.py`
  - `06_build_second_clustering_ready_inputs.py`
  - `07_run_integrated_second_clustering.py`
  - `08_build_environment_enrichment_features.py`
  - `09_analyze_environment_enrichment.py`
  - `10_build_enriched_clustering_inputs.py`
  - `11_run_enriched_second_clustering.py`
  - `13_build_presentation_quadrant_charts.py`
- `py_compile` 검증 완료

#### 관련 문서

- `works/README.md`
- `works/01_clustering/README.md`
- `works/01_clustering/08_integrated/README.md`

#### 보고서/PPT에 넣을 수 있는 메시지

- “통합 군집화 작업물은 최종 결과, 중간 산출물, 재생성 파이프라인을 분리해 재현성과 열람성을 함께 확보했다.”

---

### Decision 021. 서울시 사업장 인허가 원천으로 POI 후보 피처를 가공했지만, 이번 단계에서는 메인 군집화 피처로 채택하지 않는다

#### 배경

- 상업지구와 생활상권 해석을 더 직접적으로 보강하려면 `오프라인 상권/생활편의 POI`가 필요하다고 판단했다.
- 기존 통합 군집화는 반납 시간대, 순유입, 교통 접근성 중심이라 `상권 밀집도`를 직접 설명하는 축은 약했다.

#### 공식 원천

- `지방행정 인허가 데이터개방`
- https://www.localdata.go.kr/devcenter/dataDown.do?menuNo=20001

#### 가공 기준

- 강남구 주소만 사용
- `영업/정상` 상태만 사용
- 좌표 존재 사업장만 사용
- 공통 대여소 169개 기준 반경 카운트형 피처 생성
- 실제 적용 세트는 편차 완화를 위해 `log1p` 변환 사용

#### 실제 사용 피처

- `log1p_restaurant_count_300m` : log1p(300m 내 일반음식점 수)
- `log1p_cafe_count_300m` : log1p(300m 내 커피숍 수)
- `log1p_convenience_store_count_300m` : log1p(300m 내 편의점 수)
- `log1p_pharmacy_count_300m` : log1p(300m 내 약국 수)
- `log1p_food_retail_count_1000m` : log1p(1000m 내 식품판매업(기타) 수)
- `log1p_fitness_count_500m` : log1p(500m 내 체력단련장 수)
- `log1p_cinema_count_1000m` : log1p(1000m 내 영화상영관 수)

#### 제외한 피처와 이유

- `golf_practice_count_1000m`
  - 골프연습장은 따릉이 목적지 직접성이 낮아 제외
- `bakery_count_300m`
  - 카페 축과 중복이 커 제외
- `hospital_count_500m`
  - 후보로 실험했으나 silhouette를 더 낮춰 제외
- `통신판매업`
  - 온라인 판매 성격이 강해 오프라인 상권 해석과 맞지 않아 제외

#### 실험 결과

- 메인 통합 군집화 최고 silhouette: `0.2033`
- POI 보강 군집화 최고 silhouette: `0.1576`
- 선택 `k = 5`

#### 판단

- `restaurant`와 `food_retail`을 포함한 세트는 상권 해석을 보강하는 데는 의미가 있었다.
- 그러나 현재는 메인 7개 피처보다 분리도가 낮아, 메인 군집화를 교체할 만큼 강하지 않았다.
- 따라서 POI 피처는 이번 발표에서는 `후반 실험 결과`로만 제시하고, 메인 군집화 피처에는 채택하지 않는다.
- 이후 예측 모델 단계에서 군집별 보조 변수 후보로 재검토할 수 있다.

---

### Decision 022. station-day 베이스라인 모델은 연 단위 time-based validation과 누수 제거 기준으로 먼저 비교한다

#### 배경

- 기존 station-day 베이스라인 데이터셋에는 아래 운영 지표가 포함되어 있었다.
  - `return_count`
  - `same_station_return_count`
  - `same_station_return_ratio`
  - `net_flow`
- 그러나 이 값들은 모두 같은 날짜의 실제 운영 결과이므로, 예측 시점에는 알 수 없는 정보다.
- 이를 그대로 feature에 넣으면 타깃 누수가 발생해 비정상적으로 높은 성능이 나올 수 있다.

#### 결정 내용

- station-day 1차 베이스라인 모델에서는 같은 날 운영 지표를 feature에서 제외한다.
- 검증 전략은 계절성과 날씨 영향을 반영하기 위해 아래처럼 연 단위로 고정한다.
  - Train: `2023`
  - Validation: `2024`
  - Test: `2025`
- 대신 아래 조합으로 먼저 비교한다.
  - 시간 feature
  - 공휴일 feature
  - 일 단위 날씨 feature
  - 정적 스테이션 정보
  - `cluster_label`
  - 과거 수요 기반 `lag/rolling` feature

#### 모델 비교 구성

- 튜닝 단계:
  - Train: `2023-01-01 ~ 2023-12-31`
  - Validation: `2024-01-01 ~ 2024-12-31`
- 최종 평가 단계:
  - Final Train: `2023-01-01 ~ 2024-12-31`
  - Test: `2025-01-01 ~ 2025-12-31`
- 비교 모델:
  - `LinearRegression`
  - `LightGBMRegressor`

#### 생성 feature

- `rental_count_lag1`
- `rental_count_lag7`
- `rolling_mean_7`
- `rolling_std_7`

#### 생성 파일

- 모델링 노트북:
  - `works/03_prediction/04_scripts/02_ddri_station_day_baseline_modeling.ipynb`
- 성능 비교표:
  - `works/03_prediction/02_data/ddri_station_day_baseline_model_metrics.csv`
- LightGBM 중요도:
  - `works/03_prediction/02_data/ddri_station_day_lightgbm_feature_importance.csv`
  - `works/03_prediction/03_images/ddri_station_day_lightgbm_feature_importance.png`

#### 내부 실행 결과

- `LinearRegression`
  - validation RMSE: `6.8732`
  - validation MAE: `4.9723`
  - validation R²: `0.7464`
  - test RMSE: `6.3530`
  - test MAE: `4.6139`
  - test R²: `0.7330`

- `LightGBM`
  - validation RMSE: `6.0146`
  - validation MAE: `4.3282`
  - validation R²: `0.8058`
  - test RMSE: `5.3106`
  - test MAE: `3.8473`
  - test R²: `0.8134`

#### 추가 확인

- 누수 피처를 포함한 초기 실험에서는 비정상적으로 거의 완벽한 성능이 나왔다.
- 이는 `return_count`와 `net_flow` 등이 같은 날짜의 실제 정보를 포함해 타깃을 역산할 수 있었기 때문이다.
- 따라서 이후 보고서와 모델 비교에서는 해당 운영 지표를 baseline 직접 feature로 쓰지 않는다.
- 추가로 분기 validation은 계절 대표성이 약하다는 참고 전략 문서를 반영해, 연 단위 validation으로 수정했다.

#### 현재 해석

- 1차 station-day 베이스라인에서는 `LightGBM`이 `LinearRegression`보다 일관되게 우세하다.
- 중요 피처 상위에는 `station_id`, `temperature_max`, `rolling_mean_7`, `humidity_mean`, `precipitation_sum`, `rental_count_lag7`, `rental_count_lag1`이 포함되었다.
- 즉, 스테이션 고유성 + 날씨 + 과거 수요가 현재 baseline 예측력의 핵심 축으로 보인다.

#### 보고서/PPT에 넣을 수 있는 메시지

- “같은 날 운영 지표는 예측 시점에 알 수 없는 정보이므로 baseline feature에서 제외하고, lag 기반 시계열 feature로 대체했다.”
- “연 단위 시계열 검증 기준에서 LightGBM이 LinearRegression보다 더 낮은 오차와 높은 설명력을 보였다.”
- “현재 baseline은 날씨, 공휴일, 정적 스테이션 정보, 과거 수요의 조합만으로도 2025 테스트셋에서 안정적인 예측력을 확인했다.”

#### 필요한 시각화/표

- [x] 모델별 성능 비교표
- [x] LightGBM feature importance 차트
- [ ] validation / test 예측 오차 분포 차트
- [ ] 상위 오류 스테이션 사례 표

---

### Decision 023. 대표 대여소 station-hour 추천 모델은 `LightGBM`과 `CatBoost`, 그리고 `RMSE/Poisson objective`를 비교해 판단한다

#### 배경

- `works/05_prediction_long` 데이터는 시간 단위 카운트 예측 문제다.
- 실제 학습 데이터에서 `rental_count = 0` 비율이 높고, `hour`, `station_group`, `cluster`, `holiday`, `weather` 등 범주형·비선형 효과가 함께 존재한다.
- 추천 문서에서는 `CatBoost`, `LightGBM`, `XGBoost`가 후보로 제안되었고, objective는 `Poisson` 또는 `Tweedie` 검토가 권장되었다.

#### 결정 내용

- 1차 비교는 아래 4개 조합으로 제한한다.
  - `LightGBM_RMSE`
  - `LightGBM_Poisson`
  - `CatBoost_RMSE`
  - `CatBoost_Poisson`
- 검증 전략은 계절성을 반영해 연 단위로 고정한다.
  - Train: `2023`
  - Validation: `2024`
  - Final Train: `2023 + 2024`
  - Test: `2025`

#### 사용 데이터

- 학습:
  - `works/05_prediction_long/data/ddri_prediction_long_train_2023_2024.csv`
- 테스트:
  - `works/05_prediction_long/data/ddri_prediction_long_test_2025.csv`

#### 생성 feature

- 기본 입력:
  - `station_id`
  - `station_group`
  - `cluster`
  - `mapped_dong_code`
  - `hour`
  - `weekday`
  - `month`
  - `holiday`
  - `temperature`
  - `humidity`
  - `precipitation`
  - `wind_speed`
- 시계열 파생:
  - `lag_1h`
  - `lag_24h`
  - `lag_168h`
  - `rolling_mean_24h`
  - `rolling_mean_168h`
  - `rolling_std_24h`

추가 설명:

- `lag_168h`는 원천 데이터에 원래 있던 컬럼이 아니라, `rental_count`에서 만든 시계열 파생 피처다.
- 의미는 `168시간 전`, 즉 `1주 전 동일 요일·동일 시각 대여량`이다.

#### 생성 파일

- 모델 비교 노트북:
  - `works/05_prediction_long/03_ddri_station_hour_model_comparison.ipynb`
- 성능 비교표:
  - `works/05_prediction_long/data/ddri_station_hour_model_metrics.csv`
- LightGBM 중요도:
  - `works/05_prediction_long/data/ddri_station_hour_lightgbm_feature_importance.csv`
  - `works/05_prediction_long/images/ddri_station_hour_lightgbm_feature_importance.png`

#### 내부 실행 결과

- `LightGBM_RMSE`
  - validation RMSE: `1.0066`
  - validation MAE: `0.6121`
  - validation R²: `0.5703`
  - test RMSE: `0.8927`
  - test MAE: `0.5455`
  - test R²: `0.5608`

- `LightGBM_Poisson`
  - validation RMSE: `1.0003`
  - validation MAE: `0.6074`
  - validation R²: `0.5757`
  - test RMSE: `0.8967`
  - test MAE: `0.5402`
  - test R²: `0.5568`

- `CatBoost_RMSE`
  - validation RMSE: `1.0088`
  - validation MAE: `0.6139`
  - validation R²: `0.5685`
  - test RMSE: `0.9007`
  - test MAE: `0.5488`
  - test R²: `0.5528`

- `CatBoost_Poisson`
  - validation RMSE: `1.0081`
  - validation MAE: `0.6095`
  - validation R²: `0.5691`
  - test RMSE: `0.9049`
  - test MAE: `0.5460`
  - test R²: `0.5487`

#### 현재 해석

- validation만 보면 `LightGBM_Poisson`이 근소 우세하다.
- 하지만 최종 `2025` 테스트 기준으로는 `LightGBM_RMSE`가 가장 안정적이다.
- 이번 대표 대여소 15개 station-hour 실험에서는 `CatBoost`가 기대만큼 우세하지 않았다.
- 따라서 현재 1차 기본 후보는 `LightGBM_RMSE`로 두고, `Poisson objective`는 보조 후보로 유지한다.

#### 중요 피처

LightGBM 기준 상위 피처:

- `hour`
- `lag_168h`
- `lag_1h`
- `station_id`
- `temperature`
- `lag_24h`
- `rolling_mean_24h`
- `rolling_mean_168h`

즉, 시간대 정보와 과거 반복 패턴, 그리고 날씨가 핵심 설명 변수로 보인다.

#### 보고서/PPT에 넣을 수 있는 메시지

- “대표 대여소 station-hour 예측에서는 추천 모델 후보를 직접 비교한 결과 LightGBM 계열이 가장 안정적이었다.”
- “Poisson objective는 validation에서 유리했지만, 최종 2025 테스트에서는 RMSE objective가 더 좋은 종합 성능을 보였다.”
- “시간대 예측에서는 `hour`, `1시간/24시간/1주 lag`, 날씨가 핵심 설명 변수로 확인되었다.”

#### 필요한 시각화/표

- [x] 모델별 성능 비교표
- [x] LightGBM feature importance 차트
- [ ] station_group별 성능 비교표
- [ ] 시간대별 오차 분포 차트

---

### Decision 024. 전체 161개 스테이션 `station-hour` 실험은 대표 대여소 15개 실험과 별도 경로에서 관리한다

#### 배경

- `works/05_prediction_long`는 대표 대여소 15개를 이용한 설명용/보조 실험 경로다.
- 반면 전체 공통 스테이션 `161개` `station-hour` 데이터는 실제 서비스 확장용 핵심 실험 트랙이다.
- 두 실험은 목적, 데이터 크기, 결과 해석이 다르므로 같은 폴더에서 관리하면 혼선이 생긴다.

#### 결정 내용

- 전체 스테이션 `station-hour` 원본 데이터는 계속 아래 공유폴더에 둔다.
  - `3조 공유폴더/군집별 데이터_전체 스테이션/full_data/`
- 전체 스테이션 `station-hour` 실험 노트북, 성능표, 차트는 아래 별도 경로에서 관리한다.
  - `works/06_prediction_long_full/`
- 따라서 앞으로 `works/05_prediction_long`에는 대표 대여소 15개 실험만 남긴다.

#### 생성 파일

- 실험 관리 README:
  - `works/06_prediction_long_full/README.md`

#### 현재 해석

- 대표 대여소 실험은 유형별 패턴 설명용으로 유지한다.
- 전체 161개 스테이션 실험은 운영자 재배치 지원과 일반 사용자 잔여 자전거 예측 기능으로 직접 연결되는 본 실험 트랙으로 취급한다.
- 다음 ML 작업은 `works/06_prediction_long_full` 기준으로 진행한다.

---

### Decision 025. 전체 161개 스테이션 `station-hour` 1차 baseline은 `LightGBM_RMSE` 단일 모델로 먼저 확인한다

#### 배경

- 전체 공통 스테이션 `station-hour` 데이터는 `2,824,584행` 학습셋과 `1,410,360행` 테스트셋을 가지는 본 서비스용 대형 데이터다.
- 대표 대여소 15개 실험에서 `LightGBM_RMSE`가 가장 안정적인 결과를 보였으므로, 전체 스테이션 확장에서도 같은 모델로 먼저 기준선을 확인하는 것이 효율적이다.
- 또한 대표 대여소 노트북에서 `rolling_mean/std`가 스테이션 경계를 명확히 분리하지 못하는 계산 방식이 있었기 때문에, 전체 스테이션 baseline에서는 `station_id`별 rolling으로 수정해 적용한다.

#### 결정 내용

- 전체 161개 스테이션 baseline 1차 모델은 `LightGBM_RMSE_Full`로 고정한다.
- 검증 전략은 그대로 유지한다.
  - Train: `2023`
  - Validation: `2024`
  - Final Train: `2023 + 2024`
  - Test: `2025`
- 입력 피처는 정적 스테이션 정보 + 시간 변수 + 날씨 + lag/rolling 피처를 사용한다.

#### 사용 데이터

- 원본 데이터:
  - `3조 공유폴더/군집별 데이터_전체 스테이션/full_data/ddri_prediction_long_train_2023_2024.csv`
  - `3조 공유폴더/군집별 데이터_전체 스테이션/full_data/ddri_prediction_long_test_2025.csv`

#### 생성 feature

- 기본 입력:
  - `station_id`
  - `cluster`
  - `mapped_dong_code`
  - `hour`
  - `weekday`
  - `month`
  - `holiday`
  - `temperature`
  - `humidity`
  - `precipitation`
  - `wind_speed`
- 시계열 파생:
  - `lag_1h`
  - `lag_24h`
  - `lag_168h`
  - `rolling_mean_24h`
  - `rolling_mean_168h`
  - `rolling_std_24h`

추가 설명:

- `lag_168h`는 원천 CSV에 있던 필드가 아니라, 현재 시점 기준 `1주 전 동일 시각 대여량`을 뜻하는 파생 피처다.

#### 생성 파일

- baseline 노트북:
  - `works/06_prediction_long_full/01_ddri_station_hour_full_baseline.ipynb`
- 성능 비교표:
  - `works/06_prediction_long_full/data/ddri_station_hour_full_model_metrics.csv`
- LightGBM 중요도:
  - `works/06_prediction_long_full/data/ddri_station_hour_full_lightgbm_feature_importance.csv`
  - `works/06_prediction_long_full/images/ddri_station_hour_full_lightgbm_feature_importance.png`
- 스테이션별 오류 요약:
  - `works/06_prediction_long_full/data/ddri_station_hour_full_station_error_summary.csv`

#### 내부 실행 결과

- `LightGBM_RMSE_Full`
  - validation RMSE: `0.9735`
  - validation MAE: `0.6234`
  - validation R²: `0.4463`
  - test RMSE: `0.8624`
  - test MAE: `0.5594`
  - test R²: `0.4369`

#### 중요 피처

상위 피처:

- `station_id`
- `hour`
- `weekday`
- `lag_168h`
- `lag_1h`
- `temperature`
- `lag_24h`
- `rolling_mean_168h`

즉, 전체 스테이션 범위에서도 스테이션 고유성, 시간대, 반복 수요 패턴, 날씨가 핵심 축으로 유지된다.

#### 현재 해석

- 대표 대여소 15개 실험보다 성능은 다소 낮아졌지만, 전체 서비스 범위를 포괄한 본실험 baseline으로는 합리적인 출발점이다.
- 2025 테스트 RMSE가 validation보다 낮게 나온 것은 전체 스테이션 공통 패턴이 2025 테스트에서 더 안정적으로 맞아떨어진 일부 영향으로 해석할 수 있다.
- 다음 단계는 `Poisson objective`, `CatBoost`, 그리고 스테이션별 상위 오류 사례 분석이다.

#### 보고서/PPT에 넣을 수 있는 메시지

- “대표 대여소 설명용 실험과 별도로, 전체 161개 공통 스테이션을 대상으로 한 서비스용 station-hour baseline을 구축했다.”
- “전체 스테이션 범위에서도 station_id, hour, 과거 수요 lag, 날씨가 핵심 설명 변수로 유지되었다.”
- “전체 서비스 범위 확장 시 설명력은 일부 낮아졌지만, 운영자/사용자 기능 확장을 위한 실사용 기준선을 확보했다.”

---

### Decision 026. 전체 161개 스테이션 `station-hour` objective 비교에서는 `LightGBM_RMSE_Full`을 기본 모델로 유지한다

#### 배경

- baseline 실험에서 `LightGBM_RMSE_Full`이 전체 161개 스테이션 기준선을 확보했다.
- 다음 판단은 count target 특성을 고려한 `Poisson objective`가 실제로 더 유리한지 확인하는 것이었다.
- 초기 `CatBoost`까지 포함한 전체 비교는 계산 시간이 과도하게 길어져, 본 트랙에서는 먼저 LightGBM objective 비교로 범위를 좁혔다.

#### 결정 내용

- 전체 161개 스테이션 본실험의 objective 비교는 아래 2개로 우선 확정한다.
  - `LightGBM_RMSE_Full`
  - `LightGBM_Poisson_Full`
- `CatBoost`는 전체 데이터 전체 반복학습보다 계산비용이 커서, 필요 시 축소 실험 또는 경량 설정으로 분리한다.

#### 생성 파일

- 비교 노트북:
  - `works/06_prediction_long_full/02_ddri_station_hour_full_model_comparison.ipynb`
- 비교표:
  - `works/06_prediction_long_full/data/ddri_station_hour_full_model_comparison_metrics.csv`
- LightGBM 중요도:
  - `works/06_prediction_long_full/data/ddri_station_hour_full_model_comparison_lightgbm_feature_importance.csv`
  - `works/06_prediction_long_full/images/ddri_station_hour_full_model_comparison_lightgbm_feature_importance.png`

#### 내부 실행 결과

- `LightGBM_RMSE_Full`
  - validation RMSE: `0.9735`
  - validation MAE: `0.6234`
  - validation R²: `0.4463`
  - test RMSE: `0.8624`
  - test MAE: `0.5594`
  - test R²: `0.4369`

- `LightGBM_Poisson_Full`
  - validation RMSE: `0.9827`
  - validation MAE: `0.6260`
  - validation R²: `0.4359`
  - test RMSE: `0.8704`
  - test MAE: `0.5613`
  - test R²: `0.4262`

#### 현재 해석

- 전체 161개 스테이션 기준에서는 `Poisson objective`가 `RMSE objective`를 넘지 못했다.
- 따라서 full-data 트랙의 기본 objective는 `RMSE`로 유지한다.
- 이후 고도화는 objective 변경보다 스테이션별 오류 분석, 추가 피처, 서비스 후처리 연결이 우선이다.

#### 보고서/PPT에 넣을 수 있는 메시지

- “전체 161개 스테이션 full-data 실험에서는 count-aware Poisson objective보다 기본 RMSE objective가 더 안정적인 결과를 보였다.”
- “full-data 트랙에서는 모델 종류를 무리하게 늘리기보다, LightGBM 기준선 위에서 오류 분석과 서비스 연결성을 먼저 확보하는 것이 더 효율적이다.”

---

### Decision 027. 대표 대여소 단계는 군집별 최적화 자체가 아니라, 군집별 본실험 전의 탐색/설명 단계로 해석한다

#### 배경

- 군집화를 수행한 이유는 최종적으로 군집별 수요 구조 차이를 해석하고, 필요하면 군집별로 다른 피처 조합과 모델 전략을 적용하기 위함이다.
- 하지만 현재까지는 대표 대여소 15개 실험과 전체 161개 공통 baseline 실험이 먼저 진행되었다.
- 이때 대표 대여소 단계가 군집별 최적화의 최종단계인지, 아니면 그 전 단계인지 해석을 분명히 할 필요가 생겼다.

#### 결정 내용

- 대표 대여소 15개 `station-hour` 실험은 군집별 본격 최적화의 최종단계가 아니다.
- 이 단계는 아래 목적을 가진다.
  - 군집별 시간대 패턴 설명
  - 시간대 예측 구조의 빠른 탐색
  - 전체 스테이션 실험 전 모델/피처 전략 감 잡기
- 전체 161개 `station-hour` 실험은 서비스용 공통 baseline 단계로 해석한다.
- 이후 군집화를 예측모델까지 완성하려면, 다음 단계는 `군집별 subset 기반 모델링`이어야 한다.

#### 문서 반영

- 전략 문서:
  - `works/00_overview/09_ddri_cluster_specific_modeling_strategy.md`

#### 현재 해석

- 대표 대여소 단계는 설명용/탐색용 중간 단계다.
- 전체 161개 baseline은 서비스 범위 기준선 확보다.
- 진짜 군집별 모델링은 아직 시작 전이며, 다음 ML 고도화 단계에서 수행해야 한다.

#### 보고서/PPT에 넣을 수 있는 메시지

- “대표 대여소 실험은 군집별 최적 모델을 확정하기 위한 마지막 단계가 아니라, 군집별 패턴 설명과 시간대 예측 구조 탐색을 위한 중간 단계였다.”
- “서비스 적용을 위해 먼저 전체 161개 스테이션 공통 baseline을 확보했고, 이후 군집별 subset 실험으로 고도화할 계획이다.”

---

### Decision 028. 대표 대여소 `station-hour` 실험에도 군집화 파트처럼 설명용 근거 차트를 누적한다

#### 배경

- 대표 대여소 `station-hour` 실험은 모델 비교표와 피처 중요도 차트까지만 생성되어, 군집화 파트에 비해 설명 근거가 부족했다.
- 실제 보고서와 발표에서는 “왜 이 모델을 기준선으로 삼는가”, “어느 그룹에서 오차가 큰가”, “시간대 패턴을 얼마나 따라가는가”를 보여주는 차트가 필요하다.

#### 결정 내용

- 대표 대여소 `station-hour` 실험에도 별도 evidence notebook을 추가한다.
- 현재 우세 모델인 `LightGBM_RMSE` 기준으로 아래 근거 자료를 만든다.
  - 모델별 `2025` 테스트 RMSE 비교 차트
  - 시간대별 평균 실제값 vs 예측값 차트
  - `station_group`별 MAE 차트
  - residual 분포 차트
  - 실제값 vs 예측값 scatter
  - 그룹/스테이션별 오류 요약표

#### 생성 파일

- 근거 차트 노트북:
  - `works/05_prediction_long/04_ddri_station_hour_evidence_charts.ipynb`
- 요약표:
  - `works/05_prediction_long/data/ddri_station_hour_hourly_actual_vs_predicted.csv`
  - `works/05_prediction_long/data/ddri_station_hour_station_group_error_summary.csv`
  - `works/05_prediction_long/data/ddri_station_hour_station_error_summary.csv`
- 차트:
  - `works/05_prediction_long/images/ddri_station_hour_model_comparison_test_rmse.png`
  - `works/05_prediction_long/images/ddri_station_hour_hourly_actual_vs_predicted.png`
  - `works/05_prediction_long/images/ddri_station_hour_station_group_mae.png`
  - `works/05_prediction_long/images/ddri_station_hour_residual_distribution.png`
  - `works/05_prediction_long/images/ddri_station_hour_actual_vs_predicted_scatter.png`

#### 내부 해석

- 대표 그룹 중 `아침 도착 업무 집중형`의 MAE가 가장 높다.
- 상위 오류 대여소에는 `2377`, `2348`, `4917`이 포함된다.
- 시간대 평균 패턴은 전반적으로 따라가지만, 출근 피크 구간과 일부 고수요 구간에서 오차가 상대적으로 커진다.

#### 보고서/PPT에 넣을 수 있는 메시지

- “대표 대여소 실험도 군집화 파트처럼 설명 근거 차트를 보강해, 모델 선정과 오류 특성을 시각적으로 설명할 수 있도록 정리했다.”
- “대표 그룹별 오차를 보면 아침 도착 업무 집중형에서 상대적으로 어려운 패턴이 확인되며, 이는 이후 군집별 subset 실험 필요성을 뒷받침한다.”

---

### Decision 029. `works/05_prediction_base`는 미사용 잔여 폴더로 판단하고 제거한다

#### 배경

- `works/05_prediction_base`는 초기에 예측 기초 데이터셋을 따로 두려던 흔적이지만, 현재 실사용 예측 트랙은 아래 세 경로로 정리되었다.
  - `works/03_prediction`
  - `works/05_prediction_long`
  - `works/06_prediction_long_full`
- 실제로 `works/05_prediction_base` 아래에는 비어 있는 `data/`만 남아 있었고, 현재 파이프라인에서 참조되지 않는다.

#### 결정 내용

- `works/05_prediction_base` 폴더를 삭제한다.
- 이후 예측 관련 active 경로는 아래처럼 해석한다.
  - `works/03_prediction`: station-day baseline
  - `works/05_prediction_long`: 핵심 15개 검증용 station-hour
  - `works/06_prediction_long_full`: 161개 재배치 확장용 station-hour

#### 현재 해석

- `works/05_prediction_base`는 현 시점에서는 불필요한 잔여 폴더였다.
- 작업 구조를 단순화하기 위해 제거하는 것이 맞다.

---

### Decision 030. `works/04_presentation/02_project` 임시 발표 자료는 제거하고, 발표 폴더에는 군집화 자료만 유지한다

#### 배경

- `works/04_presentation/02_project`에는 과거 임시 프로젝트 발표 자료가 남아 있었지만, 현재 프로젝트 상태와 정확히 동기화되어 있지 않았다.
- 현재는 군집화 파트만 별도 발표 자료로 정리되어 있고, 프로젝트 전체 발표는 아직 최종 분석/ML/서비스 설계가 끝나지 않은 상태다.
- 이런 상황에서 임시 발표 자료를 유지하면 나중에 실제 최종 발표 자료와 혼동될 가능성이 높다.

#### 결정 내용

- `works/04_presentation/02_project` 폴더를 제거한다.
- 현재 `works/04_presentation`에는 `01_clustering`만 유지한다.
- 전체 프로젝트 발표 자료와 최종 분석 레포트는 추후 최종 단계에서 새로 작성한다.

#### 현재 해석

- 지금 발표 폴더는 “현재 유효한 발표 자료”만 남기는 것이 맞다.
- 군집화 발표 자료는 유지하고, 프로젝트 전체 발표는 나중에 최종 결과 기준으로 다시 만드는 편이 더 안전하다.

---

### Decision 031. `works/05_prediction_long`는 원본 데이터와 생성 산출물을 분리해 관리한다

#### 배경

- `works/05_prediction_long/data` 안에는 원본 train/test long-format CSV와 모델 성능표, 오류 요약표가 함께 섞여 있었다.
- 이렇게 두면 나중에 원본 데이터와 실험 생성물을 구분하기 어렵고, 폴더 해석도 불명확해진다.

#### 결정 내용

- `data/`에는 아래 원본 데이터만 유지한다.
  - `ddri_prediction_long_train_2023_2024.csv`
  - `ddri_prediction_long_test_2025.csv`
- 생성 산출물은 아래로 이동한다.
  - `output/data/`
  - `output/images/`
- 관련 노트북도 새 저장 경로를 따르도록 수정한다.

#### 현재 해석

- `works/05_prediction_long/data`는 이제 원본 데이터 전용 경로다.
- 실험 결과 표와 차트는 `output/` 아래에서 확인하는 것이 맞다.

---

### Decision 032. `works/06_prediction_long_full`는 원본 데이터가 외부에 있으므로 `output/`만 생성한다

#### 배경

- `works/06_prediction_long_full`는 대표 대여소 실험과 달리, 원본 train/test CSV를 직접 보관하지 않는다.
- 실제 원본 데이터는 아래 공유폴더에 존재한다.
  - `3조 공유폴더/군집별 데이터_전체 스테이션/full_data/`
- 따라서 이 폴더 안에서 `data/`, `images/`를 따로 둘 필요가 없고, 생성 산출물만 `output/`으로 모으는 편이 더 명확하다.

#### 결정 내용

- `works/06_prediction_long_full` 아래 생성물은 모두 `output/`으로 저장한다.
  - `output/data/`
  - `output/images/`
- 관련 노트북 저장 경로도 이 구조를 따르도록 수정한다.

#### 현재 해석

- `works/06_prediction_long_full`는 원본 데이터 폴더가 아니라, 전체 161개 스테이션 실험 노트북과 생성물만 관리하는 경로다.
- 따라서 이 폴더에서는 `output/`만 보면 된다.

---

### Decision 033. 차트와 문서 용어 표기는 `한글(영문)` 형식으로 통일한다

#### 배경

- 예측 파트 산출물에서 차트 제목, 축, 피처명, 문서 용어가 영어로만 표기된 경우가 있었다.
- 반대로 한글만 쓰면 원래 용어가 무엇을 가리키는지 헷갈릴 수 있다.

#### 결정 내용

- 앞으로 차트와 문서의 핵심 용어는 `한글(영문)` 형식으로 표기한다.
- 적용 대상:
  - 차트 제목
  - 축 라벨
  - 범례
  - 피처명
  - README, overview, report log 용어

#### 현재 해석

- 한글만 또는 영어만 쓰는 방식보다, `한글(영문)`이 발표와 재현성 양쪽에 더 유리하다.

---

### Decision 034. 현재까지의 예측 모델 점수는 별도 요약 문서로 정리한다

#### 배경

- `station-day`, 대표 15개 `station-hour`, 전체 161개 `station-hour` 점수가 여러 파일에 흩어져 있어 한 번에 보기 어려웠다.
- 현재 단계에서는 모델 성능을 한눈에 비교할 수 있는 요약표가 필요하다.

#### 결정 내용

- 아래 문서를 추가한다.
  - `works/00_overview/10_ddri_model_score_summary.md`
- 이 문서에는 트랙별 모델 점수, 현재 우세 모델, 짧은 해석을 함께 정리한다.

#### 현재 해석

- 현재까지의 공통 결론은 세 트랙 모두 `LightGBM` 계열이 가장 우세하다는 것이다.

---

### Decision 035. 팀원 공통 군집별 모델링 프로토콜을 별도 문서로 고정한다

#### 배경

- 대표 대여소 15개를 `5개 군집 x 각 3개 대여소`로 나누어 팀원별로 병렬 실험할 계획이 생겼다.
- 이때 팀원마다 전처리, validation/test 분할, 파생 피처, 평가 지표 기록 방식이 달라지면 결과를 공정하게 비교하기 어렵다.

#### 결정 내용

- 아래 문서를 추가한다.
  - `works/05_prediction_long/05_ddri_team_cluster_modeling_protocol.md`
- 이 문서에 아래를 고정한다.
  - 공통 입력 데이터 경로
  - 군집별 대표 대여소 구성
  - 공통 전처리 규칙
  - lag/rolling 피처 생성 규칙
  - `2023 train / 2024 validation / 2025 test` 분할 정책
  - 필수 모델 비교 순서
  - `RMSE`, `MAE`, `R²` 필수 기록 규칙
  - 노트북 문서화 규칙
  - 팀원별 진행 순서와 제출 체크리스트

#### 현재 해석

- 이제 군집별 실험은 개인 취향이 아니라 팀 공통 규약에 따라 진행하는 것이 맞다.
- 이후 결과 취합과 최종 분석 레포트 작성도 이 문서를 기준으로 통일할 수 있다.

---

### Decision 036. 팀원 공통 프로토콜에 맞춘 군집별 노트북 템플릿을 대표 대여소 실험 경로에 추가한다

#### 배경

- 프로토콜 문서만으로는 팀원이 바로 실험을 시작할 때 셀 구조나 코드 뼈대를 새로 잡아야 한다.
- 같은 규약을 따르더라도 노트북 구조가 제각각이면 결과 취합과 검토가 번거로워진다.

#### 결정 내용

- 아래 노트북 템플릿을 추가한다.
  - `works/05_prediction_long/06_ddri_cluster_modeling_template.ipynb`
- 이 템플릿에는 아래를 포함한다.
  - 입력 데이터 로드
  - 담당 군집 필터링
  - 공통 lag/rolling 피처 생성
  - `2023 / 2024 / 2025` 시간 분할
  - `LinearRegression`, `LightGBM_RMSE` 기본 실험
  - `RMSE`, `MAE`, `R²` 계산
  - validation 결과 기반 우세 모델 선택과 `2025` test 평가
  - 결과 저장 셀과 해석/체크리스트 마크다운

#### 현재 해석

- 이제 팀원은 템플릿을 복사해 군집명만 바꾸면 같은 구조로 실험을 시작할 수 있다.
- 문서와 코드 뼈대가 동시에 갖춰져 있어 팀 공통 비교가 쉬워진다.

---

### Decision 037. 2024 날씨 원천 파일은 Open-Meteo로 다시 받아 정정하고, 대표/전체 학습 데이터와 관련 실험을 재생성한다

#### 배경

- 대표 대여소 학습 데이터와 전체 스테이션 학습 데이터에는 `2024-01-01` 24시간의 날씨 값이 비어 있었다.
- 이 결측은 윤년과 무관하며, 기존 `gangnam_weather_1year_2024.csv`가 `2024-01-02`부터 시작하는 잘못된 원천 파일이었기 때문이다.

#### 결정 내용

- `Open-Meteo Archive API`로 `2024-01-01 ~ 2024-12-31` 시간별 날씨를 다시 받는다.
- 아래 원천 파일을 정정한다.
  - `3조 공유폴더/2023-2024년 강남구 날씨데이터(00시-24시)/gangnam_weather_1year_2024.csv`
- 정정 후 행 수와 범위를 확인한다.
  - `8784행`
  - `2024-01-01 00:00:00 ~ 2024-12-31 23:00:00`
- 대표 대여소 학습 데이터도 다시 생성해 `2024-01-01` 날씨 결측을 제거한다.
- 전체 스테이션 `full_data` 학습 데이터도 다시 생성해 같은 결측을 제거한다.
- `works/06_prediction_long_full` 실험 노트북도 새 원본 기준으로 다시 실행한다.

#### 현재 해석

- `2024-02-29`는 윤년으로 인해 정상 포함되는 날짜이고, `2024-01-01` 결측은 원천 파일 오류였다.
- 현재 `works/05_prediction_long/data/ddri_prediction_long_train_2023_2024.csv` 기준으로는 `2024-01-01` 날씨 결측이 제거되었다.
- 현재 `3조 공유폴더/군집별 데이터_전체 스테이션/full_data/ddri_prediction_long_train_2023_2024.csv` 기준으로도 `2024-01-01` 날씨 결측이 제거되었다.
- `works/06_prediction_long_full`의 baseline/objective 비교 산출물도 새 원본 기준으로 다시 생성되었고, 점수 자체는 기존과 동일하게 유지되었다.
