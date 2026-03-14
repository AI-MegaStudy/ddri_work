# DDRI 대표 15개 상위 5개 오류 스테이션 피처 연결 및 subset 우선순위 요약

작성일: 2026-03-14  
목적: 대표 15개 상위 오류 5개 스테이션을 군집별 보강 피처 후보와 연결하고, 다음 군집별 subset 실험 우선순위를 확정한다.

## 1. 결론 요약

다음 subset 실험 우선순위는 아래 순서가 적절하다.

1. `cluster01` 아침 도착 업무 집중형
2. `cluster02` 주거 도착형
3. `cluster00` 업무/상업 혼합형
4. `cluster04` 외곽 주거형

`cluster03`은 대표 15개 상위 오류 5개에 직접 포함되지 않아 이번 우선순위에서는 뒤로 둔다.

## 2. 스테이션별 연결 결과

### `2377` 수서역 5번출구

- 군집: `cluster01`
- 핵심 시간대: `16~20시`
- 패턴: 저녁 피크 변동형
- 우선 피처:
  - `is_commute_hour`
  - `commute_evening_flag`
  - `rolling_mean_6h`
  - `rolling_mean_12h`
  - `bus_stop_count_300m`

### `2348` 포스코사거리(기업은행)

- 군집: `cluster01`
- 핵심 시간대: `16~19시`, `11시`
- 패턴: 저녁 피크 과대예측형
- 우선 피처:
  - `is_commute_hour`
  - `commute_evening_flag`
  - `is_after_holiday`
  - `rolling_std_6h`
  - `subway_distance_m`

### `4917` 일원에코파크 주차장

- 군집: `cluster02`
- 핵심 시간대: `05~08시`, `17시`
- 패턴: 아침 과소예측형
- 우선 피처:
  - `is_night_hour`
  - `is_weekend`
  - `is_holiday_eve`
  - `bus_stop_count_300m`

### `2328` 르네상스 호텔 사거리 역삼지하보도 7번출구 앞

- 군집: `cluster00`
- 핵심 시간대: `08시`, `11시`, `16~18시`
- 패턴: 업무권 혼합 시간대형
- 우선 피처:
  - `is_commute_hour`
  - `restaurant_count_300m`
  - `cafe_count_300m`
  - `rolling_mean_6h`

### `2359` 국립국악중,고교 정문 맞은편

- 군집: `cluster04`
- 핵심 시간대: `08시`, `16~18시`, `20시`
- 패턴: 외곽 생활 이동형
- 우선 피처:
  - `station_elevation_m`
  - `distance_naturepark_m`
  - `distance_river_boundary_m`
  - `bus_stop_count_300m`

## 3. 군집별 subset 우선순위 해석

### 1순위 `cluster01`

- 대표 상위 오류 1, 2위가 모두 포함된다.
- 기존 2차와 3차에서 실제 개선이 가장 분명했다.
- 따라서 다음 subset 실험은 `cluster01`을 먼저 심화하는 것이 가장 합리적이다.

### 2순위 `cluster02`

- 대표 상위 오류 3위 `4917`이 포함된다.
- 아침 시간대 과소예측이 뚜렷해, 주거형 아침 이동 보강 피처를 검증할 가치가 있다.

### 3순위 `cluster00`

- 대표 상위 오류 4위 `2328`이 포함된다.
- 다만 기존 개선폭이 작아 `cluster02` 뒤로 두는 편이 낫다.

### 4순위 `cluster04`

- 대표 상위 오류 5위 `2359`가 포함된다.
- 전체 개선폭이 작아 우선순위는 가장 낮다.

## 4. 현재 판단

대표 15개 오류 해석 기준의 다음 직접 실험은 `cluster01 -> cluster02` 순으로 좁혀졌다.  
즉 구조 정리를 시작하기 전에, 예측 파트의 핵심 서사는 사실상 아래처럼 고정됐다.

- `cluster01`은 심화 커스텀 우선 대상
- `cluster02`는 경량 subset 재검증 후보
- `cluster00`, `cluster04`는 후속 검토

## 5. 참고 파일

- `output/data/ddri_rep15_top5_feature_linkage_table.csv`
- `output/data/ddri_cluster_subset_priority_table.csv`
