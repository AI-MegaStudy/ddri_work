# Ddri 2차 군집화 추가 지표 후보 정리

작성일: 2026-03-12  
목적: 1차 군집화에 없는 추가 지표 후보를 정리하고, 각 지표의 산출 방식과 고저차 데이터 확보 경로를 명확히 한다.

## 1. 범위

이 문서는 아래 6개 추가 지표를 대상으로 한다.

- `same_station_return_ratio` (동일 대여소 반납 비율)
- `net_flow_mean` (평균 순유출입)
- `abs_net_flow_mean` (평균 절대 순유출입)
- `morning_peak_ratio` (아침 출근 시간대 비율)
- `evening_peak_ratio` (저녁 퇴근 시간대 비율)
- `lunch_ratio` (점심 시간대 비율)

또한 고저차 관련 지표를 만들기 위한 데이터 확보 방안도 함께 정리한다.

## 2. 공통 원천 데이터

### 2.1 원천 파일

- `3조 공유폴더/2023 강남구 따릉이 이용정보/*.csv`
- `3조 공유폴더/2024 강남구 따릉이 이용정보/*.csv`
- `3조 공유폴더/2025 강남구 따릉이 이용정보/*.csv`

### 2.2 주요 원천 컬럼

- `대여일시`
- `대여 대여소번호`
- `반납일시`
- `반납대여소번호`
- `이용시간(분)`
- `이용거리(M)`

### 2.3 공통 전처리 기준

현재 프로젝트의 baseline 전처리 기준을 그대로 따른다.

- `대여일시`, `반납일시`, `대여 대여소번호`, `반납대여소번호`, `이용시간(분)`, `이용거리(M)` 결측 제거
- `이용시간(분) > 0`
- `이용거리(M) > 0`
- 대여 대여소는 공통 대여소 마스터에 포함된 경우만 유지
- 반납 대여소는 해당 연도 강남구 유효 대여소 마스터에 포함된 경우만 유지

관련 코드:

- [ddri_station_clustering_baseline.py](/Users/cheng80/Desktop/ddri_work/works/01_clustering/01_baseline/ddri_station_clustering_baseline.py)
- [ddri_station_day_dataset_builder.py](/Users/cheng80/Desktop/ddri_work/works/03_prediction/04_scripts/ddri_station_day_dataset_builder.py)

## 3. 추가 지표 정의

### 3.1 요약 표

| 컬럼명 | 한글명 | 최종 grain | 원천 컬럼 | 산출 개념 |
|---|---|---|---|---|
| `same_station_return_ratio` | 동일 대여소 반납 비율 | `station` | `대여일시`, `대여 대여소번호`, `반납일시`, `반납대여소번호` | 같은 대여소에서 빌리고 같은 날 같은 대여소로 반납한 비율 |
| `net_flow_mean` | 평균 순유출입 | `station` | `대여일시`, `대여 대여소번호`, `반납일시`, `반납대여소번호` | 일 단위 대여량에서 일 단위 반납량을 뺀 값의 평균 |
| `abs_net_flow_mean` | 평균 절대 순유출입 | `station` | `대여일시`, `대여 대여소번호`, `반납일시`, `반납대여소번호` | 일 단위 순유출입 절댓값의 평균 |
| `morning_peak_ratio` | 아침 출근 시간대 비율 | `station` | `대여일시`, `대여 대여소번호` | 전체 대여 중 아침 시간대 대여 비중 |
| `evening_peak_ratio` | 저녁 퇴근 시간대 비율 | `station` | `대여일시`, `대여 대여소번호` | 전체 대여 중 저녁 시간대 대여 비중 |
| `lunch_ratio` | 점심 시간대 비율 | `station` | `대여일시`, `대여 대여소번호` | 전체 대여 중 점심 시간대 대여 비중 |

### 3.2 이벤트 단위 중간 컬럼

아래 중간 컬럼을 먼저 만든다.

- `station_id` = `대여 대여소번호`
- `return_station_id` = `반납대여소번호`
- `date` = `대여일시`의 날짜 부분
- `return_date` = `반납일시`의 날짜 부분
- `hour` = `대여일시`의 시(hour)

## 4. 지표별 산출 방식

### 4.1 `same_station_return_ratio` | 동일 대여소 반납 비율

정의:

- 특정 대여소에서 발생한 전체 대여 중,
- 같은 날 같은 대여소로 다시 반납된 건의 비율

이벤트 단위 조건:

- `station_id == return_station_id`
- `date == return_date`

먼저 일 단위 지표를 만든다.

- `rental_count(station_id, date)` = 해당 날짜 해당 대여소 출발 대여 건수
- `same_station_return_count(station_id, date)` = 위 조건을 만족하는 건수

그다음 군집화용 `station` 단위로 집계한다.

권장 식:

```text
same_station_return_ratio
= sum(same_station_return_count by station_id)
  / sum(rental_count by station_id)
```

설명:

- 일별 평균 비율보다 기간 전체 합 비율을 권장한다.
- 이유는 소규모 대여소에서 일별 비율이 과도하게 흔들리는 문제를 줄일 수 있기 때문이다.

### 4.2 `net_flow_mean` | 평균 순유출입

정의:

- 대여소 기준으로 하루 동안 얼마나 자전거가 빠져나갔는지 또는 들어왔는지의 평균

먼저 일 단위 지표를 만든다.

- `rental_count(station_id, date)` = 해당 날짜 해당 대여소 출발 대여 건수
- `return_count(station_id, date)` = 해당 날짜 해당 대여소 반납 건수
- `net_flow(station_id, date) = rental_count - return_count`

그다음 군집화용 `station` 단위로 집계한다.

```text
net_flow_mean
= mean(net_flow by station_id over all dates)
```

해석:

- 양수: 그 대여소에서 평균적으로 더 많이 빠져나감
- 음수: 그 대여소로 평균적으로 더 많이 들어옴

### 4.3 `abs_net_flow_mean` | 평균 절대 순유출입

정의:

- 방향은 무시하고, 하루 기준 재고 변동이 얼마나 큰지 보는 지표

먼저 일 단위 지표를 만든다.

```text
abs_net_flow(station_id, date) = abs(net_flow(station_id, date))
```

그다음 군집화용 `station` 단위로 집계한다.

```text
abs_net_flow_mean
= mean(abs_net_flow by station_id over all dates)
```

해석:

- `net_flow_mean`은 방향성을 본다.
- `abs_net_flow_mean`은 변동 강도를 본다.

### 4.4 `morning_peak_ratio` | 아침 출근 시간대 비율

정의:

- 전체 대여 중 아침 출근 시간대에 발생한 대여 비율

권장 시간대:

- `07:00`, `08:00`, `09:00`

산출 식:

```text
morning_peak_ratio
= count(hour in [7, 8, 9] by station_id)
  / count(all rentals by station_id)
```

해석:

- 출근형 대여소일수록 값이 높을 가능성이 있다.

### 4.5 `evening_peak_ratio` | 저녁 퇴근 시간대 비율

정의:

- 전체 대여 중 저녁 퇴근 시간대에 발생한 대여 비율

권장 시간대:

- `18:00`, `19:00`, `20:00`

산출 식:

```text
evening_peak_ratio
= count(hour in [18, 19, 20] by station_id)
  / count(all rentals by station_id)
```

해석:

- 퇴근형 또는 저녁 이동 집중 대여소 분리에 도움이 된다.

### 4.6 `lunch_ratio` | 점심 시간대 비율

정의:

- 전체 대여 중 점심 시간대에 발생한 대여 비율

권장 시간대:

- `11:00`, `12:00`, `13:00`

산출 식:

```text
lunch_ratio
= count(hour in [11, 12, 13] by station_id)
  / count(all rentals by station_id)
```

해석:

- 오피스 밀집 지역, 상업지구, 점심권 단거리 이동 수요를 포착하는 데 유용할 수 있다.

## 5. 권장 산출 순서

1. 원천 CSV를 baseline 전처리 기준으로 정제한다.
2. 이벤트 단위에서 `station_id`, `return_station_id`, `date`, `return_date`, `hour`를 만든다.
3. `station-day` 기준으로 `rental_count`, `return_count`, `same_station_return_count`, `net_flow`를 만든다.
4. 이를 다시 `station` 기준으로 요약해 `same_station_return_ratio`, `net_flow_mean`, `abs_net_flow_mean`을 만든다.
5. 이벤트 단위에서 `hour`를 이용해 `morning_peak_ratio`, `evening_peak_ratio`, `lunch_ratio`를 만든다.

## 6. 고저차 지표 후보

고저차는 따릉이 이용 패턴에 영향을 줄 수 있으므로 2차 군집화 후보로 타당하다.

추천 후보:

- `station_elevation_m` (대여소 표고)
- `slope_mean_300m` (반경 300m 평균 경사도)
- `slope_max_300m` (반경 300m 최대 경사도)
- `elevation_diff_nearest_subway` (가장 가까운 지하철역과의 표고 차)
- `elevation_diff_nearest_park` (가장 가까운 공원과의 표고 차)

## 7. 고저차 데이터 확보 방안

### 7.1 1순위: 서울시 경사도 / 등고선 파일 데이터

가장 실무적으로 좋은 시작점은 서울시 파일 데이터를 직접 쓰는 방식이다.

장점:

- 서울시 공식 파일 데이터
- 경사도 SHP와 등고선 SHP를 바로 내려받을 수 있음
- 강남구 대여소라는 현재 프로젝트 범위와 맞음
- API 호출 비용 없이 오프라인 재현 가능

활용 방식:

- `서울시 경사도.zip`을 받아 GIS 또는 Python에서 읽는다.
- 대여소 좌표를 경사도 격자 또는 폴리곤과 spatial join 한다.
- 반경 버퍼 100m, 300m 기준 평균/최대 경사도를 계산할 수 있다.
- `서울시 등고선.zip`을 함께 쓰면 표고 차 근사도 가능하다.

출처:

- 서울시 경사도 데이터셋
  - https://data.seoul.go.kr/dataList/OA-22241/F/1/datasetView.do

확인한 내용:

- 서울 열린데이터광장에는 `서울시 등고선.zip`과 `서울시 경사도.zip`이 파일 형태로 제공된다.
- 데이터 설명에는 경사도를 추출할 수 있는 표고점, 등고선 SHP 파일 제공이라고 적혀 있다.

### 7.2 2순위: Open-Meteo Elevation API

간단히 대여소별 표고값만 붙이고 싶으면 가장 빠른 선택지다.

장점:

- HTTP 호출만으로 사용 가능
- 여러 좌표를 한 번에 요청 가능
- 90m 해상도 DEM 기반
- 구현 난도가 낮음

주의:

- 점 표고값은 쉽게 가져올 수 있지만, 주변 경사도 계산은 별도 로직이 더 필요하다.
- 아주 세밀한 도시 미세 지형 해석에는 한계가 있다.

활용 방식:

- 각 대여소의 `lat`, `lon`을 API에 전달
- `station_elevation_m` 생성
- 필요하면 주변 격자 샘플을 추가로 조회해 간이 slope를 계산

출처:

- Open-Meteo Elevation API
  - https://open-meteo.com/en/docs/elevation-api

확인한 내용:

- `/v1/elevation` 엔드포인트 사용
- 위경도 배열 입력 가능
- Copernicus DEM 2021 GLO-90 기반

### 7.3 3순위: Google Maps Elevation API

정식 상용 API로 안정성이 높고, 경로 기반 표고 샘플링도 지원한다.

장점:

- 단일 좌표 표고 조회 가능
- path 기반 샘플링 지원
- 문서화와 SDK 지원이 잘 되어 있음

주의:

- API 키와 과금 관리가 필요함
- 공개 결과 사용 시 정책과 attribution 확인 필요

활용 방식:

- 대여소 좌표별 표고 조회
- 대여소-인근 지하철역 경로의 고도 변화 샘플링
- `elevation_diff_nearest_subway` 같은 지표 생성

출처:

- Google Elevation API overview
  - https://developers.google.com/maps/documentation/elevation/overview
- Google Elevation API policies
  - https://developers.google.com/maps/documentation/elevation/policies

### 7.4 4순위: OpenTopography Global Datasets API

직접 DEM 원천을 받아 분석하고 싶을 때 적합하다.

장점:

- COP30, COP90, NASADEM, ALOS World 3D 등 여러 DEM 접근 가능
- DEM 원천을 받아 로컬에서 경사도와 표고 차를 재현성 있게 계산 가능

주의:

- API 사용과 후처리 난도가 높다.
- 프로젝트 초기 탐색보다는 고도화 단계에 더 적합하다.

출처:

- OpenTopography for Developers
  - https://opentopography.org/developers

### 7.5 5순위: Copernicus DEM 직접 다운로드

장기적으로는 가장 재현성이 높은 방법 중 하나다.

장점:

- 공식 DEM 원천을 직접 확보 가능
- 로컬 분석 파이프라인에 편입하기 좋음

주의:

- 다운로드와 인증 과정이 다소 무겁다.
- 서울시 단일 프로젝트에는 과할 수 있다.

출처:

- Copernicus DEM documentation
  - https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM.html
- Copernicus bulk download FAQ
  - https://documentation.dataspace.copernicus.eu/FAQ.html

## 8. 고저차 데이터 확보 추천안

현재 프로젝트 기준 추천 순서는 아래와 같다.

1. 서울시 경사도 파일을 우선 사용한다.
2. 빠른 검증이 필요하면 Open-Meteo Elevation API로 대여소 표고를 먼저 붙인다.
3. 고저차가 실제로 군집 분리에 의미가 보이면, 서울시 경사도 파일 또는 OpenTopography/Copernicus DEM 기반으로 정교화한다.

실무 권장안:

- `station_elevation_m`: Open-Meteo 또는 Google Elevation API로 빠르게 생성
- `slope_mean_300m`, `slope_max_300m`: 서울시 경사도 SHP로 생성
- `elevation_diff_nearest_subway`: 대여소 표고와 최근접 지하철역 표고 차로 생성

## 9. 다음 구현 메모

코드 구현 시에는 아래 순서가 안전하다.

1. 기존 cleaning 로직을 재사용하는 station-level feature builder 작성
2. 운영 지표 3종 생성
   - `same_station_return_ratio`
   - `net_flow_mean`
   - `abs_net_flow_mean`
3. 시간대 비율 3종 생성
   - `morning_peak_ratio`
   - `evening_peak_ratio`
   - `lunch_ratio`
4. 서울시 경사도 파일 확보 후 spatial join 기반 slope feature 추가
