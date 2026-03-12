# Second Clustering Feature

작성일: 2026-03-12  
목적: 2차 군집화 후보 추가 피처 CSV의 생성 기준과 컬럼 의미를 설명한다.

## 1. 생성 목적

이 문서는 아래 6개 추가 피처를 별도 CSV로 생성한 이유와 의미를 정리한다.

- `same_station_return_ratio`
- `net_flow_mean`
- `abs_net_flow_mean`
- `morning_peak_ratio`
- `evening_peak_ratio`
- `lunch_ratio`

현재 단계에서는 이 피처들을 기존 군집화 feature 테이블에 바로 합치지 않는다.

이유:

- 팀원들과 함께 지표 해석 가능성을 먼저 검토해야 한다.
- 기존 1차 군집화 결과와 혼동되지 않도록 별도 산출물로 관리하는 편이 안전하다.
- 이후 합칠 경우 `station_id` 기준으로 명확하게 join할 수 있다.

## 2. 현재 위치

이 문서는 초기 2차 후보 피처 검토 메모다.

현재는 별도 추가 피처 CSV를 유지하지 않고, 아래 통합 파이프라인과 최종 피처 테이블에 흡수해 관리한다.

- 통합 생성 노트북: [12_ddri_integrated_clustering_report_builder.ipynb](/Users/cheng80/Desktop/ddri_work/works/01_clustering/08_integrated/12_ddri_integrated_clustering_report_builder.ipynb)
- 최종 통합 피처: [ddri_final_district_clustering_features_train_2023_2024.csv](/Users/cheng80/Desktop/ddri_work/works/01_clustering/08_integrated/ddri_final_district_clustering_features_train_2023_2024.csv)
- 반납 시간대 기반 중간 산출물: [return_time_district_notes.md](/Users/cheng80/Desktop/ddri_work/works/01_clustering/08_integrated/return_time_district/return_time_district_notes.md)

## 3. 전처리 기준

추가 피처 생성에는 baseline 군집화와 동일한 cleaning 기준을 사용한다.

- 필수 컬럼 결측 제거
- `이용시간(분) <= 0` 제거
- `이용거리(M) <= 0` 제거
- `동일 대여소 반납 + 5분 이하` 제거
- 공통 대여소 기준 밖 대여 제거
- 해당 연도 강남구 기준 밖 반납 제거

즉, 2차 피처 후보도 1차 군집화와 동일한 표본 기준 위에서 계산된다.

## 4. 컬럼 설명

| 컬럼명 | 한글명 | 설명 |
|---|---|---|
| `station_id` | 대여소 ID | 대여소 식별자 |
| `same_station_return_ratio` | 동일 대여소 반납 비율 | 전체 대여 중 같은 날 같은 대여소로 반납된 비율 |
| `net_flow_mean` | 평균 순유출입 | `대여량 - 반납량`의 일 평균 |
| `abs_net_flow_mean` | 평균 절대 순유출입 | `abs(대여량 - 반납량)`의 일 평균 |
| `morning_peak_ratio` | 아침 출근 시간대 비율 | 전체 대여 중 `07~09시` 비율 |
| `evening_peak_ratio` | 저녁 퇴근 시간대 비율 | 전체 대여 중 `18~20시` 비율 |
| `lunch_ratio` | 점심 시간대 비율 | 전체 대여 중 `11~13시` 비율 |

## 5. 계산 방식 요약

### `same_station_return_ratio`

```text
sum(same_station_return_count) / sum(rental_count)
```

### `net_flow_mean`

```text
mean(rental_count - return_count)
```

### `abs_net_flow_mean`

```text
mean(abs(rental_count - return_count))
```

### `morning_peak_ratio`

```text
count(hour in [7, 8, 9]) / count(all rentals)
```

### `evening_peak_ratio`

```text
count(hour in [18, 19, 20]) / count(all rentals)
```

### `lunch_ratio`

```text
count(hour in [11, 12, 13]) / count(all rentals)
```

## 6. 생성 결과

- 학습용 row 수: `164`
- 테스트용 row 수: `162`

이 row 수는 현재 baseline 군집화 feature 테이블의 station 수와 같은 기준으로 해석하면 된다.

## 7. 값이 `0`인 경우 해석

이 추가 피처 테이블에서는 일부 값이 `0`으로 나올 수 있다.

이 값은 대부분 오류가 아니라, 해당 조건의 이벤트가 실제로 없음을 뜻한다.

예시:

- `same_station_return_ratio = 0`
  - 같은 날 같은 대여소로 반납된 기록이 한 번도 없다는 뜻
- `morning_peak_ratio = 0`
  - `07~09시` 대여가 한 번도 없다는 뜻
- `evening_peak_ratio = 0`
  - `18~20시` 대여가 한 번도 없다는 뜻
- `lunch_ratio = 0`
  - `11~13시` 대여가 한 번도 없다는 뜻
- `net_flow_mean = 0`
  - 기간 평균 순유출입이 거의 균형이라는 뜻

또한 비율 계산 과정에서는, 특정 조건의 건수가 없는 경우를 결측치로 두지 않고 `0`으로 채운다.

즉:

- `0`은 “데이터 없음”이 아니라 “해당 패턴이 관측되지 않음”으로 해석한다.
- 이 처리 방식은 군집화 단계에서 결측치보다 해석 가능한 숫자형 입력을 유지하기 위한 것이다.

## 8. 검토 포인트

팀 검토 시 아래를 우선 보면 된다.

- `same_station_return_ratio`가 실제로 제자리 회전형 대여소를 잘 설명하는지
- `net_flow_mean`이 순유출형 / 순유입형 대여소를 분리하는 데 의미가 있는지
- `abs_net_flow_mean`이 재고 변동 강도를 설명하는 데 도움이 되는지
- `morning_peak_ratio`, `evening_peak_ratio`, `lunch_ratio`가 출근형 / 퇴근형 / 점심권 이동형을 나누는 데 유효한지
- 기존 1차 군집화의 수요 규모 중심 분리를 더 세분화할 수 있는지

## 9. 이 6개 피처로 붙일 수 있는 군집 라벨 후보

이 피처들은 기존 1차 군집화의 `수요 규모 중심 라벨`과 달리, 대여소의 `운영 성격`과 `시간대 역할`을 더 잘 설명하는 데 쓰일 수 있다.

가능한 라벨 후보는 아래와 같다.

### 9.1 출발 거점형

판단 기준 예시:

- `net_flow_mean`가 양수
- `morning_peak_ratio`가 높음

해석:

- 아침 시간대에 자전거가 많이 빠져나가는 대여소
- 주거지 인근 출발 거점일 가능성

### 9.2 도착 거점형

판단 기준 예시:

- `net_flow_mean`가 음수
- `evening_peak_ratio`가 높음

해석:

- 저녁 시간대에 자전거가 더 많이 들어오는 대여소
- 퇴근 도착지 또는 저녁 집결형 거점일 가능성

### 9.3 점심 이동형

판단 기준 예시:

- `lunch_ratio`가 상대적으로 높음
- `morning_peak_ratio`, `evening_peak_ratio`보다 점심 비중이 두드러짐

해석:

- 오피스 밀집 지역이나 상업지구 인근의 점심권 이동 수요 가능성
- 단, 위치 정보나 POI 없이 단정하지는 않는다

### 9.4 제자리 회전형

판단 기준 예시:

- `same_station_return_ratio`가 높음
- `net_flow_mean` 절댓값은 크지 않음

해석:

- 같은 대여소에서 빌리고 같은 대여소로 돌아오는 비중이 높은 대여소
- 산책형, 짧은 왕복 이동형, 내부 순환형 패턴 가능성

### 9.5 재고 변동형

판단 기준 예시:

- `abs_net_flow_mean`가 높음

해석:

- 순유출/순유입 방향과 무관하게 하루 재고 변동 폭이 큰 대여소
- 운영상 재배치 민감도가 높은 후보

### 9.6 시간대 균형형

판단 기준 예시:

- `morning_peak_ratio`, `evening_peak_ratio`, `lunch_ratio`가 모두 특정 방향으로 치우치지 않음
- `net_flow_mean`도 0 부근

해석:

- 특정 시간대 쏠림보다 생활권 전반의 고른 이용 패턴을 보이는 대여소

## 10. 라벨 해석 시 주의점

이 라벨들은 현재 단계에서 `후보 라벨`이다.

즉:

- `lunch_ratio`가 높다고 바로 오피스형으로 단정하지 않는다.
- `same_station_return_ratio`가 높다고 바로 관광형으로 단정하지 않는다.
- `net_flow_mean` 하나만으로 주거지/업무지구를 확정하지 않는다.

따라서 실제 라벨링은 아래를 함께 보고 결정하는 것이 좋다.

- 기존 1차 군집화 결과
- 지하철/버스/공원 접근성
- 지도 위치
- 대표 대여소 목록

## 11. 다음 단계

현재는 검토용 별도 CSV만 생성한 상태다.

이후 팀 합의가 되면:

1. 기존 군집화 feature 테이블과 `station_id` 기준으로 병합
2. 추가 피처 포함 버전으로 K 탐색 재실행
3. `k=2` 유지 여부 또는 `k=3` 이상 재검토
