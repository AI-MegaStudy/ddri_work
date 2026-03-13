# POI 후보 피처

공식 원천은 `지방행정 인허가 데이터개방`이다.

- 원천 사이트: https://www.localdata.go.kr/devcenter/dataDown.do?menuNo=20001
- 로컬 저장 폴더: `3조 공유폴더/서울시 상원정보6110000_CSV`

위 원천 CSV에서 강남구 영업중 사업장을 추출해, 공통 대여소 169개 기준 반경 카운트형 POI 피처로 가공한 중간 산출물이다.

적용 원칙:

- 온라인 성격 업종은 후보 단계부터 제외
- 실제 대여소 주변 `오프라인 상권/생활편의/여가`로 해석 가능한 업종만 검토

## 생성 파일

- `ddri_station_poi_candidate_features.csv`
  - 대여소별 POI 후보 피처
- `ddri_station_poi_candidate_feature_summary.csv`
  - 피처별 평균, 중앙값, 최대값, 0이 아닌 대여소 비율
- `ddri_station_poi_candidate_source_summary.csv`
  - 피처별 공식 원천, 사용 파일, 실제 사용된 강남구 영업중 사업장 수

## 현재 생성한 후보 피처

- `restaurant_count_300m` : 300m 내 일반음식점 수
- `cafe_count_300m` : 300m 내 커피숍 수
- `convenience_store_count_300m` : 300m 내 편의점 수
- `bakery_count_300m` : 300m 내 제과점 수
- `pharmacy_count_300m` : 300m 내 약국 수
- `hospital_count_500m` : 500m 내 병원 수
- `food_retail_count_1000m` : 1000m 내 식품판매업(기타) 수
- `fitness_count_500m` : 500m 내 체력단련장 수
- `cinema_count_1000m` : 1000m 내 영화상영관 수
- `golf_practice_count_1000m` : 1000m 내 골프연습장 수

## 해석 메모

- `restaurant_count_300m`는 직접적인 상권 밀집도 축이므로, log 변환 또는 반경 조정 후에는 다시 검토 가치가 있다.
- `cafe_count_300m`, `convenience_store_count_300m`, `bakery_count_300m`, `pharmacy_count_300m`는 생활/상권 접근성 보조 피처로 쓸 수 있다.
- `hospital_count_500m`는 후보로 검토할 수 있지만, 현재 군집화 실험에서는 분리도를 낮춰 최종 적용 세트에서는 제외했다.
- `food_retail_count_1000m`는 마트·백화점 식품관·오프라인 판매 거점을 일부 포함하므로, 상업지구 해석 보강용 후보로 적합하다.
- `fitness_count_500m`는 생활권/업무권 혼합 성격을 볼 때 유용할 가능성이 있다.
- `hospital_count_500m`, `cinema_count_1000m`는 상대적으로 희소해 해석 보조용으로 적합하다.
- `golf_practice_count_1000m`는 강남구 전역에서 밀도가 높아 분리력이 낮을 수 있다.
- `golf_practice_count_1000m`는 따릉이 이용 목적지로서 직접성이 낮다고 판단해, 실제 군집화 적용 피처에서는 제외한다.

## 권장 후속 처리

- 군집화 실험 전 `restaurant_count_300m`는 `log1p` 변환 또는 반경 축소 검토
- 상관이 높은 생활권 피처는 `cafe`, `bakery`, `convenience_store` 중 일부만 선택
- `통신판매업`, `건강기능식품일반판매업`은 온라인/다단계/유통 성격이 섞여 있어 오프라인 상권 피처에서는 제외
- 메인 군집화에 바로 합치기보다 보조 환경 실험 세트로 먼저 검토
