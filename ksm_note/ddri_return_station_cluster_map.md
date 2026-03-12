# Return Station Cluster Map

군집 결과 CSV를 읽어서 각 대여소를 지도에 표시한다.

- 점: 대여소 중심점
- 원: 해당 대여소 반경 `300m`
- 색: `cluster_profile`
- 추가 레이어: 지하철역, 버스정류소, 공원 마커
- 팝업: 300m 내 지하철/버스/공원 이름 목록

실행:

```bash
python ksm_note/ddri_return_station_cluster_map.py
```

출력:

- `ksm_note/outputs/maps/ddri_return_station_cluster_300m_map.html`
