# Station Admin Dong Mapping

강남구 따릉이 대여소가 어느 `행정동`에 속하는지 1차로 추정하는 스크립트다.

중요:

- 현재 워크스페이스에는 `서울시 행정동 경계(geojson/shp)` 파일이 없다.
- 그래서 이번 결과는 `대여소명 + 주소` 안에 들어 있는 동 이름을 이용한 `키워드 기반 1차 매핑`이다.
- `논현1동/논현2동`, `역삼1동/역삼2동`, `개포1동/개포2동`처럼 번호가 갈리는 동은 주소에 동명이 직접 없으면 `ambiguous` 또는 `unmatched`로 남을 수 있다.

## 입력

- `3조 공유폴더/강남구 대여소 정보 (2023~2025)/2023_강남구_대여소.csv`
- `3조 공유폴더/서울특별시행정동별 서울생활인구(내국인) 2025년/서울시_행정동코드_매핑표.csv`

## 실행

```bash
python ksm_note/ddri_station_admin_dong_mapping.py
```

## 출력

- `ksm_note/outputs/admin_dong/ddri_station_admin_dong_mapping_candidates.csv`
- `ksm_note/outputs/admin_dong/ddri_station_admin_dong_mapping_matched.csv`
- `ksm_note/outputs/admin_dong/ddri_station_admin_dong_mapping_ambiguous.csv`
- `ksm_note/outputs/admin_dong/ddri_station_admin_dong_mapping_unmatched.csv`
- `ksm_note/outputs/admin_dong/ddri_station_admin_dong_mapping_summary.csv`

## 더 정확하게 하려면

정확 매핑이 필요하면 `대여소 위도/경도`와 `서울시 행정동 경계`를 이용한 spatial join으로 바꿔야 한다.
