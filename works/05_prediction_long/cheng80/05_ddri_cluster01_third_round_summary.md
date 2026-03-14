# DDRI cluster01 3차 심화 실험 요약(Third-Round Deep Dive Summary)

## 요약

- 대상 군집: `cluster01(아침 도착 업무 집중형)`
- 목적: 가장 어려운 군집에 대해 2차보다 한 단계 더 세분화한 피처와 `LightGBM_Poisson` 목적함수를 검증
- 결론: 3차 실험이 1차, 2차보다 모두 더 좋았고, 현재 `cluster01`의 최고 성능은 `LightGBM_Poisson`이다.

## 라운드별 비교표

| 라운드 | 최종 모델 | Test RMSE | Test MAE | Test R² | 1차 대비 해석 |
|---|---|---:|---:|---:|---|
| first | `LightGBM_RMSE` | 1.3462 | 0.8042 | 0.6398 | 기준선 |
| second | `LightGBM_RMSE` | 1.3324 | 0.7868 | 0.6471 | 소폭 개선 |
| third | `LightGBM_Poisson` | 1.3189 | 0.7745 | 0.6543 | 이번 심화 실험 중 가장 좋은 결과 |

## 추가 해석

- 2차 대비 3차에서 RMSE는 -0.0135 개선되었다.
- 1차 대비 3차에서 RMSE는 -0.0273, MAE는 -0.0296, R²는 +0.0145 변했다.
- 즉 `cluster01`은 추가 피처 보강과 목적함수 변경에 실제로 반응하는 군집으로 볼 수 있다.
- 전 군집 3차 실험까지 확장할 필요는 없지만, 보고서에서는 `심화 사례`로 충분히 쓸 만하다.

## 현재 판단

- 전체 프로젝트 기준으로는 2차 실험까지를 공통 결과로 둔다.
- `cluster01`만 3차 심화 사례로 별도 제시한다.
- 이 결과는 “가장 어려운 군집은 추가 최적화 여지가 크다”는 메시지를 뒷받침한다.

## 관련 파일

- `summary_aggregation/output/data/cluster01_third_round_progression_summary.csv`
- `summary_aggregation/13_ddri_cluster01_third_round_progression.ipynb`
