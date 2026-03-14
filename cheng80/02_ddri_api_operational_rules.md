# DDRI API 운영 및 예외 처리 규칙

작성일: 2026-03-14  
대상: `cheng80/01_ddri_api_verification.ipynb`에서 확인한 `OA-15493`, `OA-21235`, `tbCycleStationInfo` 검증 결과

## 1. 목적

실시간 따릉이 API를 서비스 로직에 연결할 때, 어떤 API를 기준 소스로 쓰고 누락/비정상 스테이션을 어떻게 처리할지 규칙으로 고정한다.

## 2. 기준 API

- 실시간 재고 기준: `OA-15493 bikeList`
- 대여소 마스터/매핑 기준: `tbCycleStationInfo`
- 주소/좌표 보조 기준: `OA-21235 bikeStationMaster`

현재 해석:

- 실시간 자전거 수는 `bikeList`를 정본으로 사용한다.
- 스테이션 존재 여부와 `numeric station_id ↔ ST-xxxx` 연결은 `tbCycleStationInfo`를 우선 사용한다.
- 주소/좌표 재확인이나 누락 대조는 `bikeStationMaster`를 보조로 사용한다.

## 3. 운영 상태 판정 규칙

- `bikeList`에 존재하고 좌표가 정상값이면: `운영 중`
- `bikeList`에는 없지만 `tbCycleStationInfo` 또는 `bikeStationMaster`에는 존재하면: `실시간 비노출 / 비활성 후보`
- `bikeStationMaster` 좌표가 `(0, 0)`이면: `비정상 좌표 / 운영 중지 후보`
- 위 두 경우는 서비스에서 정상 운영 대여소와 같은 방식으로 취급하지 않는다.

## 4. 현재 확인된 예외 스테이션

공식 실시간 `bikeList` 기준 미매칭 스테이션:

- `2314` 청담나들목입구
- `2323` 주식회사 오뚜기 정문 앞
- `3628` 역삼1동 주민센터

현재 해석:

- 위 3개는 강남구 연도별 대여소 정보에는 존재한다.
- `tbCycleStationInfo`와 `bikeStationMaster`에도 존재한다.
- 하지만 현재 시점 `bikeList` 실시간 응답에는 없다.
- 따라서 서비스 기준으로는 `실시간 비노출 / 비활성 후보`로 분류한다.

## 5. 서비스 적용 규칙

- 일반 사용자 화면:
  - `bikeList` 미매칭 또는 `(0, 0)` 좌표 대여소는 `실시간 정보 없음`으로 표기
  - 필요하면 예측 결과 자체도 숨기거나 예외 메시지로 대체
- 운영자 화면:
  - 위 대여소를 `예외 대여소 목록`으로 분리
  - 재배치 판단 대상 기본 목록에서는 제외 가능
- 배치/후처리:
  - `numeric station_id ↔ stationId(ST-xxxx)` 매핑 테이블을 별도 CSV로 고정
  - 실시간 수집 시 `bikeList` 전체 페이지 조회 후 강남구/대상 스테이션만 로컬 필터링

현재 생성된 재사용 파일:

- `cheng80/api_output/ddri_full161_station_api_mapping_table.csv`
  - 161개 서비스 대상 스테이션의 상세 매핑 테이블
  - `station_id`, `cluster`, 강남구 기준 이름/주소, `resolved_api_station_id`, 실시간 매칭 여부, 예외 상태 포함
- `cheng80/api_output/ddri_station_id_api_lookup.csv`
  - 호출 최소화용 경량 lookup
  - `station_id -> resolved_api_station_id -> operational_status` 중심

API 키 관련 현재 판단:

- 현재 로컬 검증 기준으로는 `SEOUL_BIKE_API_KEY`, `SEOUL_RTD_API_KEY`, `OA-21235` 키 모두 `bikeList`, `tbCycleStationInfo`의 소규모 호출(`1~5`)에 응답했다.
- 다만 이 결과는 `호환성 1차 확인` 수준으로만 해석한다.
- 실제 서비스 기준 키는 계속 `OA-15493` 실시간 따릉이 API 키를 우선 사용한다.

## 6. 남은 확인 사항

- `stationId` 선택 파라미터가 다른 호출 형식에서 실제 지원되는지 추가 확인
- 미매칭 3개가 일시적 누락인지 장기 비노출인지 날짜를 달리해 재확인
- 좌표 `(0, 0)` 대여소 목록을 정식 예외 테이블로 분리
