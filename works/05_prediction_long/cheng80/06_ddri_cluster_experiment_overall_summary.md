# DDRI 군집별 실험 종합 정리(Overall Cluster Experiment Summary)

작성일: 2026-03-14  
목적: `cluster00 ~ cluster04`의 1차 baseline, 2차 추가 피처 실험, `cluster01` 3차 심화 실험까지의 흐름을 한 문서에서 정리한다.

## 1. 실험 흐름 요약

이번 군집별 실험은 아래 순서로 진행되었다.

1. 1차 실험
   - 기본 피처만 사용한 군집별 baseline 비교
   - 공통 비교 모델: `LinearRegression`, `LightGBM_RMSE`

2. 2차 실험
   - 군집 특성에 맞는 추가 피처를 군집별로 선택해 보강
   - 1차 대비 개선 여부 확인

3. 3차 실험
   - 전 군집이 아니라 `cluster01(아침 도착 업무 집중형)`만 심화
   - 목적함수(`LightGBM_Poisson`)까지 확장 비교

## 2. 1차 결과 핵심

| 군집 코드 | 군집명 | 1차 우세 모델 | Test RMSE | Test MAE | Test R² | 1차 해석 |
|---|---|---|---:|---:|---:|---|
| `cluster00` | 업무/상업 혼합형 | `LightGBM_RMSE` | 0.8113 | 0.5439 | 0.3212 | 중간 수준, 추가 개선 여지 있음 |
| `cluster01` | 아침 도착 업무 집중형 | `LightGBM_RMSE` | 1.3462 | 0.8042 | 0.6398 | 가장 어려운 군집 |
| `cluster02` | 주거 도착형 | `LightGBM_RMSE` | 0.8088 | 0.5059 | 0.4987 | 비교적 안정적 |
| `cluster03` | 생활권 혼합형 | `LightGBM_RMSE` | 0.6901 | 0.4928 | 0.1802 | 오차는 낮지만 설명력은 낮음 |
| `cluster04` | 외곽 주거형 | `LightGBM_RMSE` | 0.7160 | 0.4425 | 0.3785 | 무난하게 예측되는 군집 |

1차 결론:

- 다섯 군집 모두 `LightGBM_RMSE`가 우세했다.
- 가장 어려운 군집은 `cluster01`
- 가장 구조적 보완 여지가 컸던 군집은 `cluster03`

## 3. 2차 실험 핵심

2차 실험은 `3조 공유폴더/대표대여소_예측데이터_15개/second_round_data/`의 통합 피처모음 CSV를 사용했다.

핵심 방향:

- `cluster00`: 출퇴근 + 상권 보강
- `cluster01`: 출근 피크 + 교통 접근성 + 단기 추세 보강
- `cluster02`: 야간/주말/주거형 보강
- `cluster03`: 점심/생활권/상권 보강
- `cluster04`: 외곽성/지형/교통 접근성 보강

## 4. 1차 vs 2차 비교

| 군집 코드 | 군집명 | 1차 Test RMSE | 2차 Test RMSE | RMSE 변화 | 1차 Test MAE | 2차 Test MAE | MAE 변화 | 2차 해석 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `cluster00` | 업무/상업 혼합형 | 0.8113 | 0.8085 | -0.0028 | 0.5439 | 0.5403 | -0.0036 | 아주 작은 개선 |
| `cluster01` | 아침 도착 업무 집중형 | 1.3462 | 1.3324 | -0.0138 | 0.8042 | 0.7868 | -0.0173 | 가장 의미 있는 개선 |
| `cluster02` | 주거 도착형 | 0.8088 | 0.8059 | -0.0028 | 0.5059 | 0.5053 | -0.0006 | 작지만 안정적인 개선 |
| `cluster03` | 생활권 혼합형 | 0.6901 | 0.6882 | -0.0019 | 0.4928 | 0.4882 | -0.0046 | 미세 개선 |
| `cluster04` | 외곽 주거형 | 0.7160 | 0.7145 | -0.0015 | 0.4425 | 0.4427 | +0.0002 | 거의 변화 없는 수준의 개선 |

2차 결론:

- 5개 군집 모두 test RMSE가 소폭 개선되었다.
- 가장 큰 개선은 `cluster01`에서 나타났다.
- 따라서 군집별 추가 피처 전략은 완전히 실패한 것이 아니라, 작지만 일관된 개선을 만든 것으로 본다.

## 5. 3차 실험 핵심

3차 실험은 `cluster01`만 수행했다.

추가한 것:

- 출근형 세분 피처
- 단기 추세 피처 보강
- `LightGBM_Poisson` 목적함수 비교

결과:

| 라운드 | 최종 모델 | Test RMSE | Test MAE | Test R² | 해석 |
|---|---|---:|---:|---:|---|
| 1차 | `LightGBM_RMSE` | 1.3462 | 0.8042 | 0.6398 | baseline |
| 2차 | `LightGBM_RMSE` | 1.3324 | 0.7868 | 0.6471 | 소폭 개선 |
| 3차 | `LightGBM_Poisson` | 1.3189 | 0.7745 | 0.6543 | 가장 좋은 결과 |

3차 결론:

- `cluster01`은 추가 피처와 목적함수 변경에 실제로 반응했다.
- 전 군집 3차 실험까지 확장할 필요는 없지만, `cluster01`은 심화 사례로 충분히 의미 있다.

## 6. 현재 최종 판단

현재 결과를 이렇게 정리한다.

- 공통 공식 결과:
  - 1차 baseline
  - 2차 군집별 추가 피처 실험

- 심화 사례:
  - `cluster01` 3차 실험

즉, 최종 보고서/발표에서는 아래 구조가 가장 자연스럽다.

1. 군집별 baseline 차이 소개
2. 군집별 추가 피처 2차 실험
3. `cluster01` 심화 최적화 사례 제시

## 7. 예측력은 어떤 점수로 확인하나

현재 프로젝트에서는 예측력을 아래 3개 지표로 함께 본다.

- `RMSE`
  - 기본 대표 지표
  - 큰 오차에 더 민감해서, 모델 선택의 1차 기준으로 사용한다
- `MAE`
  - 평균적으로 몇 대 정도 빗나가는지 직관적으로 보여준다
  - “평균 오차 크기”를 설명할 때 가장 읽기 쉽다
- `R²`
  - 패턴 설명력을 보여준다
  - 특히 `cluster03`처럼 RMSE는 낮지만 구조 설명력이 낮은 경우를 구분할 때 유용하다

현재 해석 원칙:

1. 최우선 비교는 `RMSE`
2. 실제 체감 오차 해석은 `MAE`
3. 패턴 설명력 보조 해석은 `R²`

## 8. 바로 참고할 파일

- 1차:
  - `01_ddri_cluster_result_collection.md`
  - 재생성 노트북: `summary_aggregation/11_ddri_cluster_result_collection.ipynb`
  - 집계 CSV: `summary_aggregation/output/data/ddri_cluster_model_metrics_collection_template.csv`
- 2차:
  - `04_ddri_second_round_result_summary.md`
  - 재생성 노트북: `summary_aggregation/12_ddri_cluster_second_round_comparison.ipynb`
  - 집계 CSV: `summary_aggregation/output/data/ddri_cluster_second_round_comparison_summary.csv`
- 3차:
  - `05_ddri_cluster01_third_round_summary.md`
  - 재생성 노트북: `summary_aggregation/13_ddri_cluster01_third_round_progression.ipynb`
  - 집계 CSV: `summary_aggregation/output/data/cluster01_third_round_progression_summary.csv`

## 9. 한 줄 결론

군집별 실험은 1차에서 구조를 확인했고, 2차에서 전 군집 소폭 개선을 만들었으며, 3차에서는 `cluster01`에서 추가 최적화 가능성까지 확인했다.
