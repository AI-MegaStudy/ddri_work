# DDRI 군집별 결과 취합 문서(Result Collection)

목적: `cluster00 ~ cluster04` 실험 결과를 한 자리에서 모으고, 군집별 성능 차이와 다음 판단을 빠르게 정리한다.

## 1. 사용 파일

- 결과 취합 CSV:
  - `summary_aggregation/output/data/ddri_cluster_model_metrics_collection_template.csv`
- 군집별 실행 노트북:
  - `cluster00/01_cluster_modeling.ipynb`
  - `cluster01/01_cluster_modeling.ipynb`
  - `cluster02/01_cluster_modeling.ipynb`
  - `cluster03/01_cluster_modeling.ipynb`
  - `cluster04/01_cluster_modeling.ipynb`
- 취합 노트북:
  - `summary_aggregation/11_ddri_cluster_result_collection.ipynb`

## 2. 취합 절차

1. 각 군집 노트북 실행
2. `validation_2024`에서 `LinearRegression`, `LightGBM_RMSE` 점수 기록
3. validation 기준 우세 모델 선택
4. `test_2025_refit` 점수 기록
5. `best_model_flag`를 우세 모델 test 행에 `1`로 표시
6. `notes`에 특이사항 기재

## 3. 군집 코드 대응표

| 군집 코드(Cluster Code) | 군집명(Station Group) |
|---|---|
| `cluster00` | 업무/상업 혼합형 |
| `cluster01` | 아침 도착 업무 집중형 |
| `cluster02` | 주거 도착형 |
| `cluster03` | 생활권 혼합형 |
| `cluster04` | 외곽 주거형 |

## 4. 1차 요약표 작성 규칙

최종 발표/보고용 1차 요약표는 아래 5개 행으로 만든다.

| 군집 코드 | 군집명 | 우세 모델 | Validation RMSE | Test RMSE | 해석 한 줄 |
|---|---|---|---:|---:|---|
| `cluster00` | 업무/상업 혼합형 | `LightGBM_RMSE` | 0.8987 | 0.8113 | 업무형 반복 패턴은 비교적 잘 잡히지만 설명력은 중간 수준이다. |
| `cluster01` | 아침 도착 업무 집중형 | `LightGBM_RMSE` | 1.5660 | 1.3462 | 가장 어려운 군집으로 출근 피크 영향이 커 추가 실험 우선순위가 높다. |
| `cluster02` | 주거 도착형 | `LightGBM_RMSE` | 0.8263 | 0.8088 | 주거형 패턴은 안정적이며 test R²가 가장 높다. |
| `cluster03` | 생활권 혼합형 | `LightGBM_RMSE` | 0.7898 | 0.6901 | 오차 절대값은 낮지만 설명력은 낮아 추가 피처 검토 여지가 있다. |
| `cluster04` | 외곽 주거형 | `LightGBM_RMSE` | 0.7852 | 0.7160 | 외곽 주거형은 비교적 안정적으로 학습되며 test 성능도 양호하다. |

### 4.1 수치를 직관적으로 읽는 기준

- `Test RMSE 0.65 ~ 0.75`
  - 직관적 해석: 이 군집 안에서는 비교적 잘 맞는 편
- `Test RMSE 0.75 ~ 0.90`
  - 직관적 해석: 보통 수준, 추가 개선 여지 있음
- `Test RMSE 0.90 이상`
  - 직관적 해석: 상대적으로 어려운 군집
- `Test RMSE 1.20 이상`
  - 직관적 해석: 현재 기준 가장 어려운 축에 해당

- `Test R² 0.50 이상`
  - 직관적 해석: 패턴 설명력이 비교적 안정적
- `Test R² 0.30 ~ 0.50`
  - 직관적 해석: 일정 수준 설명하지만 아직 부족함
- `Test R² 0.20 ~ 0.30`
  - 직관적 해석: 설명력이 낮은 편
- `Test R² 0.20 미만`
  - 직관적 해석: 평균 오차는 작아도 패턴 설명은 약함

주의:

- 이 기준은 대표 대여소 15개 군집 실험 안에서 비교하기 위한 내부 해석 기준이다.
- 절대적 통계 기준이 아니라 “어느 군집이 상대적으로 어렵고, 어느 군집이 더 안정적인가”를 읽기 위한 실무용 보조 기준이다.

### 4.2 군집별 직관 해석

- `cluster00` 업무/상업 혼합형
  - `Test RMSE 0.8113`, `Test R² 0.3212`
  - 해석: 아주 나쁜 수준은 아니지만, 아직 설명력은 중간 이하라 더 좋아질 여지가 있다.

- `cluster01` 아침 도착 업무 집중형
  - `Test RMSE 1.3462`, `Test R² 0.6398`
  - 해석: 다섯 군집 중 오차가 가장 크다. 다만 설명력 자체는 높은 편이라, 강한 출근 피크를 어느 정도는 잡고 있지만 절대 오차가 크게 남는 군집이다.

- `cluster02` 주거 도착형
  - `Test RMSE 0.8088`, `Test R² 0.4987`
  - 해석: 안정적인 편이다. 다섯 군집 중 설명력이 높은 축에 속한다.

- `cluster03` 생활권 혼합형
  - `Test RMSE 0.6901`, `Test R² 0.1802`
  - 해석: 오차 절대값은 가장 낮은 축이지만, 설명력은 매우 낮다. 즉 평균적으로는 맞는 듯 보여도 패턴 구조를 충분히 설명하지 못할 수 있다.

- `cluster04` 외곽 주거형
  - `Test RMSE 0.7160`, `Test R² 0.3785`
  - 해석: 비교적 안정적이다. 아주 강한 군집은 아니지만 무난하게 예측되는 축이다.

## 5. 비교 해석 질문

취합 후 아래 질문에 답해야 한다.

- 어느 군집이 가장 예측이 어려운가
- 어느 군집에서 `LightGBM_RMSE` 우세 폭이 큰가
- 어느 군집은 선형 기준선과 차이가 작나
- 어느 군집이 시간대 패턴, 날씨, `lag_168h` 영향이 더 큰가
- 다음 2차 실험이 가장 필요한 군집은 어디인가

## 6. 기록 형식

- 용어는 `한글(영문)` 형식 사용
- 점수는 소수점 넷째 자리까지 유지
- 해석 문장은 짧게 쓴다
- 추정성 해석이면 `추정`, `가능성`, `해석상` 같은 표현을 사용한다
