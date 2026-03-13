# 08 Integrated

`01_clustering`의 현재 최신 통합 군집화 작업 루트다.

## Structure

```text
08_integrated/
├── pipeline/
│   └── 생성 스크립트, 노트북, 작업 메모
├── source_data/
│   └── 공통 기준표
├── final/
│   ├── features/
│   │   └── 최종 입력 피처 CSV
│   └── results/
│       └── 최종 통합 군집화 결과, 차트, 지도, 리포트
└── intermediate/
    ├── return_time_district/
    │   └── 반납 시간대 기반 지구판단 중간 산출물
    ├── environment_enrichment/
    │   └── 표고·공원·하천경계 등 환경 보강 중간 산출물
    ├── poi_features/
    │   └── 서울시 사업장 CSV 기반 POI 후보 피처
    ├── poi_enriched_clustering/
    │   └── POI 보강 군집화 입력 테이블
    ├── poi_enriched_second_clustering_results/
    │   └── POI 보강 군집화 비교 결과
    ├── enriched_second_clustering/
    │   └── 환경 보강 입력 테이블
    └── enriched_second_clustering_results/
        └── 환경 보강 비교 실험 결과
```

## What To Open

- 최종 결과만 볼 때:
  - [final/results/second_clustering_results/integrated_second_clustering_report.md](/Users/cheng80/Desktop/ddri_work/works/01_clustering/08_integrated/final/results/second_clustering_results/integrated_second_clustering_report.md)
- 최종 입력 피처를 볼 때:
  - [final/features/ddri_final_district_clustering_features_train_2023_2024.csv](/Users/cheng80/Desktop/ddri_work/works/01_clustering/08_integrated/final/features/ddri_final_district_clustering_features_train_2023_2024.csv)
- 처음부터 다시 생성할 때:
  - [pipeline/12_ddri_integrated_clustering_report_builder.ipynb](/Users/cheng80/Desktop/ddri_work/works/01_clustering/08_integrated/pipeline/12_ddri_integrated_clustering_report_builder.ipynb)
  - [pipeline/05_build_return_time_district_features.py](/Users/cheng80/Desktop/ddri_work/works/01_clustering/08_integrated/pipeline/05_build_return_time_district_features.py)
  - [pipeline/07_run_integrated_second_clustering.py](/Users/cheng80/Desktop/ddri_work/works/01_clustering/08_integrated/pipeline/07_run_integrated_second_clustering.py)

## Notes

- `final/`은 문서와 발표에서 직접 참조하는 결과물이다.
- `intermediate/`는 재생성 과정의 중간 산출물이다.
- `2023/2024` 반납 시간대 HTML 지도는 현재 파이프라인이 함께 생성하므로 남겨 둔다.
