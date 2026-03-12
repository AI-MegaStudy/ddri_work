# works

`works/`는 프로젝트 실작업 폴더다.  
작업자가 현재 기준 구조를 빠르게 파악할 수 있도록 핵심 폴더를 트리로 정리한다.

## Tree

```text
works/
├── 00_overview/
│   └── 전체 기준 문서, 계획, 로그, 설계 문서
├── 01_clustering/
│   ├── 05_presentation/
│   │   └── 군집화 요약 문서, 발표 스크립트
│   ├── 08_integrated/
│   │   ├── pipeline/
│   │   │   └── 통합 군집화 생성 스크립트, 노트북, 작업 메모
│   │   ├── final/
│   │   │   ├── features/
│   │   │   └── results/
│   │   │       └── 최종 군집화 결과 CSV, 차트, 지도, 리포트
│   │   ├── intermediate/
│   │   │   └── 반납 시간대·환경 보강·비교 실험 중간 산출물
│   │   ├── source_data/
│   │   │   └── 2차 통합 군집화에 계속 사용하는 공통 기준표
│   │   └── 통합 군집화 작업 루트
│   ├── archive_1st/
│   │   └── 기존 1차 군집화 결과 보관본
│   └── README.md
├── 02_data_collection/
│   ├── 01_calendar/
│   ├── 02_weather/
│   └── 수집 원천과 정리본
├── 03_prediction/
│   ├── 01_dataset_design/
│   ├── 02_data/
│   ├── 03_images/
│   ├── 04_scripts/
│   └── 대여소별 수요 예측용 데이터셋/스크립트/이미지
├── 04_presentation/
│   ├── 01_clustering/
│   │   └── 군집화 A4 문서, 스피치 노트, 정적 지도 생성 스크립트
│   ├── 02_project/
│   │   └── 전체 프로젝트 발표 문서
│   ├── README.md
│   └── 공통 발표 CSS
└── lsy/
    └── 개인 작업/참고 자료
```

## Quick Guide

- 군집화 최신본부터 볼 때:
  - [01_clustering/08_integrated](/Users/cheng80/Desktop/ddri_work/works/01_clustering/08_integrated)
  - [04_presentation/01_clustering](/Users/cheng80/Desktop/ddri_work/works/04_presentation/01_clustering)
- 군집화 구버전 비교가 필요할 때:
  - [01_clustering/archive_1st](/Users/cheng80/Desktop/ddri_work/works/01_clustering/archive_1st)
- 예측 파트로 넘어갈 때:
  - [03_prediction](/Users/cheng80/Desktop/ddri_work/works/03_prediction)
- 전체 기준 문서를 먼저 볼 때:
  - [00_overview](/Users/cheng80/Desktop/ddri_work/works/00_overview)

## Notes

- 현재 군집화 공식 산출물은 `01_clustering/08_integrated` 기준으로 관리한다.
- `archive_1st`는 삭제 대상이 아니라, 1차 군집화 비교와 추적을 위한 보관 영역이다.
- `02_data_collection`과 `03_prediction`은 아직 별도 정리 축을 유지한다.
