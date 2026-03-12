# Return Station Pattern Clustering Note

`반납일시 + 반납대여소번호 + 시간대 패턴 + 반경 300m 환경`을 같이 봐서, 각 반납 대여소를 `수요 규모`가 아니라 `반납 패턴 유형`으로 묶는 메모다.

## 목표

- `여기는 평일 저녁 반납이 많고 지하철/버스 접근성이 좋아서 퇴근/환승형 후보`
- `여기는 주말 낮 반납 비중이 높고 공원이 가까워서 주말/공원형 후보`

같은 식의 해석이 가능하도록 군집화 기준을 바꿨다.

## 입력 데이터

- `3조 공유폴더/2023 강남구 따릉이 이용정보/2023_강남구_따릉이_이용정보_통합.csv`
- `3조 공유폴더/강남구 대여소 정보 (2023~2025)/2023_강남구_대여소.csv`
- `3조 공유폴더/[교통데이터] 지하철 정보/서울시 역사마스터 정보/서울시 역사마스터 정보.csv`
- `3조 공유폴더/서울시 버스정류소 위치정보/2023년/2023년각월1일기준_서울시버스정류소위치정보.csv`
- `3조 공유폴더/서울시 강남구 공원 정보.csv`

## 군집화 단위

- 한 행은 `반납 대여소 1개`
- key는 `반납대여소번호 -> station_id`

## 군집화에 직접 쓰는 feature

시간대 패턴:

- `weekday_morning_ratio`
- `weekday_evening_ratio`
- `weekend_day_ratio`
- `night_ratio`
- `weekday_return_ratio`
- `morning_evening_gap`

반경 300m 환경:

- `nearest_subway_m`
- `bus_stop_count_300m`
- `nearest_park_m`
- `park_count_300m`

중요:

- `total_returns`, `avg_daily_returns` 같은 규모 변수는 요약/설명용으로는 남기지만, 군집을 가르는 핵심 기준으로는 쓰지 않는다.
- 즉 이번 버전은 `많이/적게 반납되는 곳`보다 `언제 반납되고 주변이 어떤 곳인지`를 보려는 목적이다.

## 해석 방식

군집이 만들어진 뒤 군집 평균을 보고 아래 라벨을 붙인다.

- `교통거점형`
  - 평일 저녁 반납 비중이 높음
  - 평일 비중이 높음
  - 지하철이 가깝고 300m 내 버스정류소 수가 많음
- `업무·학교 도착형`
  - 평일 아침 반납 비중이 더 높음
  - 평일 중심 도착 패턴이 강함
- `생활도착형`
  - 저녁 또는 생활시간대 반납 비중이 높음
  - 생활권 안에서 도착 수요가 모이는 패턴

## 산출물

CSV:

- `ksm_note/outputs/data/ddri_return_station_cluster_features.csv`
- `ksm_note/outputs/data/ddri_return_k_search_metrics.csv`
- `ksm_note/outputs/data/ddri_return_cluster_summary.csv`

이미지:

- `ksm_note/outputs/images/ddri_return_cluster_elbow_silhouette.png`
- `ksm_note/outputs/images/ddri_return_cluster_pca.png`
- `ksm_note/outputs/images/ddri_return_cluster_feature_means.png`

## 실행 방법

```bash
python ksm_note/ddri_return_station_clustering.py
```

## 현재 확인 포인트

- `ddri_return_station_cluster_features.csv`의 `cluster_profile`, `station_note` 컬럼을 보면 대여소별 해석 문장이 붙는다.
- PPT에는 `군집별 평균 차트 + 대표 대여소 3~5개 note` 조합으로 가져가면 설명이 훨씬 쉽다.

현재 실행 결과:

- `k=3`
- `업무·학교 도착형`: 52개 대여소
- `생활도착형`: 114개 대여소
- `교통거점형`: 4개 대여소
