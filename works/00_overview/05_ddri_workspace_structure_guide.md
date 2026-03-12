# Ddri 작업물 구조 가이드

작성일: 2026-03-12  
목적: 계획 문서 순서대로 작업물을 따라가기 쉽도록 `works` 폴더의 권장 구조와 저장 원칙을 정의한다.

## 1. 왜 구조를 바꾸는가

현재 작업물은 기능별로는 구분되어 있지만, 처음 보는 사람이 `어떤 문서를 먼저 열고`, `어떤 실험이 어떤 산출물을 만들었는지` 따라가기는 어렵다.

현재는 아래 원칙으로 구조를 정리했다.

- 계획 문서 순서에 맞춰 상위 폴더를 번호로 구분
- 각 파트 안에서는 `baseline → 전처리 → 환경 해석 → 시각화 → 발표자료` 흐름으로 다시 분류
- 이미지도 한 폴더에 몰지 않고 목적별 파트 안에서 함께 관리
- 실사용 경로를 번호 체계 기준으로 맞추고, 기능별 혼합 구조를 해소

## 2. 권장 상위 구조

### `works/00_overview`

- 전체 프로젝트를 이해하기 위한 문서
- 예:
  - 마스터 플랜
  - 프로젝트 레포트 로그
  - 데이터/산출물 인벤토리
  - target 정의 문서
  - 데이터셋 설계 문서

### `works/01_clustering`

- 현재 군집화 최신본은 `08_integrated` 기준으로 관리
- 기존 1차 군집화 자료는 `archive_1st`로 보관

세부 구조:
- `08_integrated/pipeline`
  - 통합 군집화 생성 스크립트, 노트북, 작업 메모
- `08_integrated/source_data`
  - 통합 군집화 공통 기준표
- `08_integrated/final`
  - 최종 입력, 최종 결과, 최종 차트, 최종 지도
- `08_integrated/intermediate`
  - 반납 시간대/환경 보강/비교 실험 중간 산출물
- `archive_1st`
  - baseline, 전처리, 환경, 지도, 구 발표 자료 보관본

### `works/02_data_collection`

- 외부 데이터 수집 전용

세부 구조:
- `01_calendar`
  - 공휴일 API 수집 노트북, 스크립트, 결과
- `02_weather`
  - Open-Meteo 수집 노트북, 스크립트, 결과

### `works/03_prediction`

- 예측용 데이터셋과 모델링 준비 파트

세부 구조:
- `01_dataset_design`
  - 예측용 스키마, feature 설계 문서
- `02_data`
  - 학습/테스트 CSV, flow metric 요약
- `03_images`
  - 예측 관련 설명 차트
- `04_scripts`
  - 데이터셋 빌더, 차트 생성 스크립트

### `works/04_presentation`

- 최종 발표용 문서만 모아두는 폴더

세부 구조:
- `01_clustering`
  - 군집화 발표 자료
- `02_project`
  - 전체 프로젝트 발표 자료

## 3. 저장 원칙

### 문서

- 사람이 읽는 문서는 `목적별 폴더` 안에 저장
- 파일명 앞에는 필요하면 세부 순서를 붙임
  - 예: `01_problem_definition.md`
  - 예: `02_preprocessing_rules.md`

### 코드

- 노트북과 실행 스크립트는 같은 파트 안에서 함께 관리
- 재생성 기준이 되는 실행 스크립트는 파트 폴더의 앞쪽에 둔다

### 데이터

- CSV는 해당 파트 안의 `data` 또는 `06_data`에 저장
- 중간 산출물과 최종 산출물을 혼합하지 않도록 목적별 분리

### 이미지

- 차트는 차트를 만든 파트의 이미지 폴더에 저장
- 발표 전용 캡처본은 `presentation` 또는 `maps`와 분리 가능

## 4. 현재 권장 열람 순서

프로젝트 전체 흐름을 따라가려면 아래 순서가 가장 좋다.

1. `works/00_overview`
2. `works/01_clustering`
3. `works/02_data_collection`
4. `works/03_prediction`
5. `works/04_presentation`

## 5. 현재 적용 상태

현재는 실제 파일도 모두 번호 체계 폴더 기준으로 이동한 상태다.

- 전체 기준 문서: `works/00_overview`
- 군집화 최신본: `works/01_clustering/08_integrated`
- 군집화 보관본: `works/01_clustering/archive_1st`
- 외부 데이터 수집: `works/02_data_collection`
- 예측 데이터셋 및 스크립트: `works/03_prediction`
- 발표 자료: `works/04_presentation`

## 6. 다음 정리 과제

1. `03_prediction`에 통합 군집화 결과 결합 전략 반영
2. 예측 파트의 active/archive 정리 여부 검토
3. 발표 PDF를 최신 경로 기준으로 재생성
4. 문서 내부 파일 경로를 최종 점검
