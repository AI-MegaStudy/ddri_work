# rep15_error_analysis

목적: 대표 15개 스테이션 기준 오류 우선순위와 핵심 스테이션 시간대 오차 패턴을 별도 분석하는 작업 폴더다.

## 구성

- `10_ddri_rep15_station_error_analysis.ipynb`
  - 대표 15개 오류 우선순위, 군집별 평균 오류, `2377`/`2348` 시간대 패턴을 재생성하는 대표 노트북
- `11_ddri_rep15_top5_station_error_analysis.ipynb`
  - 대표 15개 상위 5개 오류 스테이션의 시간대 패턴을 확장 재생성하는 노트북
- `12_ddri_rep15_top5_feature_linkage_and_subset_priority.ipynb`
  - 상위 오류 5개 스테이션을 군집별 보강 피처와 연결하고 subset 우선순위를 정리하는 노트북
- `08_ddri_rep15_station_error_priority_summary.md`
  - 대표 15개 오류 우선순위 해석 요약
- `09_ddri_rep15_top2_station_hourly_error_summary.md`
  - `2377`, `2348` 시간대 오차 패턴 요약
- `11_ddri_rep15_top5_station_hourly_error_summary.md`
  - 상위 5개 오류 스테이션 시간대 패턴 확장 요약
- `12_ddri_rep15_top5_feature_linkage_summary.md`
  - 상위 오류 5개와 군집별 subset 우선순위 연결 요약
- `output/data/`
  - 이 분석에서 생성한 CSV 산출물 저장 경로

## 출력 위치

- `output/data/ddri_rep15_station_error_priority_table.csv`
- `output/data/ddri_rep15_station_error_cluster_summary.csv`
- `output/data/ddri_rep15_top2_station_hourly_error_patterns.csv`
- `output/data/ddri_rep15_top2_station_error_summary.csv`
- `output/data/ddri_rep15_top2_station_peak_error_hours.csv`
- `output/data/ddri_rep15_top5_station_hourly_error_patterns.csv`
- `output/data/ddri_rep15_top5_station_error_summary.csv`
- `output/data/ddri_rep15_top5_station_peak_error_hours.csv`
- `output/data/ddri_rep15_top5_feature_linkage_table.csv`
- `output/data/ddri_cluster_subset_priority_table.csv`
