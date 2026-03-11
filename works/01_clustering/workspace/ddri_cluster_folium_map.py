from pathlib import Path

import folium
import pandas as pd
from branca.element import Template, MacroElement


BASE_DIR = Path("/Users/cheng80/Desktop/ddri_work")
DATA_DIR = BASE_DIR / "works" / "clustering" / "data"
OUTPUT_DIR = BASE_DIR / "works" / "clustering" / "maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def add_legend(fmap):
    template = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 220px;
        z-index:9999;
        background-color: white;
        border: 2px solid #444;
        border-radius: 8px;
        padding: 12px 14px;
        font-size: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    ">
      <div style="font-weight: 700; margin-bottom: 8px;">군집 범례</div>
      <div style="margin-bottom: 6px;">
        <span style="display:inline-block;width:12px;height:12px;background:#1f77b4;border-radius:50%;margin-right:8px;"></span>
        일반수요형
      </div>
      <div>
        <span style="display:inline-block;width:12px;height:12px;background:#d62728;border-radius:50%;margin-right:8px;"></span>
        고수요형
      </div>
    </div>
    {% endmacro %}
    """
    macro = MacroElement()
    macro._template = Template(template)
    fmap.get_root().add_child(macro)


def main():
    env_path = DATA_DIR / "ddri_cluster_environment_features.csv"
    if env_path.exists():
        merged = pd.read_csv(env_path)
    else:
        labels = pd.read_csv(DATA_DIR / "ddri_station_cluster_features_train_with_labels.csv")
        master = pd.read_csv(DATA_DIR / "ddri_common_station_master.csv")
        merged = labels.merge(master, left_on="station_id", right_on="대여소번호", how="left")
        merged["cluster_name"] = merged["cluster"].map({0: "일반수요형", 1: "고수요형"})

    merged["color"] = merged["cluster"].map({0: "#1f77b4", 1: "#d62728"})

    lat_col = "station_lat" if "station_lat" in merged.columns else "위도"
    lon_col = "station_lon" if "station_lon" in merged.columns else "경도"

    center_lat = merged[lat_col].mean()
    center_lon = merged[lon_col].mean()

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="CartoDB positron",
    )

    for _, row in merged.iterrows():
        popup_html = f"""
        <div style="font-size:13px; line-height:1.5;">
          <b>대여소명</b>: {row['대여소명']}<br>
          <b>대여소번호</b>: {row['station_id']}<br>
          <b>군집</b>: {row['cluster_name']}<br>
          <b>평균 대여량</b>: {row['avg_rental']:.2f}<br>
          <b>평일 평균</b>: {row['weekday_avg']:.2f}<br>
          <b>주말 평균</b>: {row['weekend_avg']:.2f}<br>
          <b>가장 가까운 공원</b>: {row.get('nearest_park_name', '-') }<br>
          <b>공원 거리</b>: {row.get('park_distance_m', float('nan')):.1f}m<br>
          <b>가장 가까운 지하철역</b>: {row.get('nearest_subway_name', '-') }<br>
          <b>지하철 거리</b>: {row.get('subway_distance_m', float('nan')):.1f}m<br>
          <b>300m 버스정류장 수</b>: {int(row.get('bus_stop_count_300m', 0))}
        </div>
        """
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=6 if row["cluster"] == 0 else 8,
            color=row["color"],
            fill=True,
            fill_color=row["color"],
            fill_opacity=0.8,
            weight=1,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{row['대여소명']} / {row['cluster_name']}",
        ).add_to(fmap)

    title_html = """
    <h3 style="
        position: fixed;
        top: 18px;
        left: 50px;
        z-index:9999;
        background-color: rgba(255,255,255,0.9);
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #bbb;
        font-size: 18px;
    ">
      강남구 따릉이 대여소 군집 지도
    </h3>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))
    add_legend(fmap)

    output_path = OUTPUT_DIR / "ddri_cluster_map_gangnam.html"
    fmap.save(str(output_path))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
