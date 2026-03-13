from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree


BASE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BASE_DIR.parents[2]
STATION_PATH = BASE_DIR / "source_data" / "ddri_common_station_master.csv"
POI_ROOT = ROOT_DIR / "3조 공유폴더" / "서울시 상원정보6110000_CSV"
OUTPUT_DIR = BASE_DIR / "intermediate" / "poi_features"
POI_SOURCE_NAME = "지방행정 인허가 데이터개방"
POI_SOURCE_URL = "https://www.localdata.go.kr/devcenter/dataDown.do?menuNo=20001"


@dataclass(frozen=True)
class PoiSpec:
    feature_name: str
    feature_name_ko: str
    filename: str
    radius_m: int
    address_keyword: str = "강남구"
    business_status_prefix: str = "영업"
    service_name: str | None = None
    category_column: str | None = None
    category_values: tuple[str, ...] | None = None


POI_SPECS: tuple[PoiSpec, ...] = (
    PoiSpec(
        feature_name="restaurant_count_300m",
        feature_name_ko="300m 내 일반음식점 수",
        filename="6110000_서울특별시_07_24_04_P_일반음식점.csv",
        radius_m=300,
        service_name="일반음식점",
    ),
    PoiSpec(
        feature_name="cafe_count_300m",
        feature_name_ko="300m 내 커피숍 수",
        filename="6110000_서울특별시_07_24_05_P_휴게음식점.csv",
        radius_m=300,
        service_name="휴게음식점",
        category_column="업태구분명",
        category_values=("커피숍",),
    ),
    PoiSpec(
        feature_name="convenience_store_count_300m",
        feature_name_ko="300m 내 편의점 수",
        filename="6110000_서울특별시_07_24_05_P_휴게음식점.csv",
        radius_m=300,
        service_name="휴게음식점",
        category_column="업태구분명",
        category_values=("편의점",),
    ),
    PoiSpec(
        feature_name="bakery_count_300m",
        feature_name_ko="300m 내 제과점 수",
        filename="6110000_서울특별시_07_22_18_P_제과점영업.csv",
        radius_m=300,
        service_name="제과점영업",
    ),
    PoiSpec(
        feature_name="pharmacy_count_300m",
        feature_name_ko="300m 내 약국 수",
        filename="6110000_서울특별시_01_01_06_P_약국.csv",
        radius_m=300,
        service_name="약국",
    ),
    PoiSpec(
        feature_name="food_retail_count_1000m",
        feature_name_ko="1000m 내 식품판매업(기타) 수",
        filename="6110000_서울특별시_07_22_13_P_식품판매업기타.csv",
        radius_m=1000,
        service_name="식품판매업(기타)",
    ),
    PoiSpec(
        feature_name="fitness_count_500m",
        feature_name_ko="500m 내 체력단련장 수",
        filename="6110000_서울특별시_10_42_01_P_체력단련장업.csv",
        radius_m=500,
        service_name="체력단련장업",
    ),
    PoiSpec(
        feature_name="hospital_count_500m",
        feature_name_ko="500m 내 병원 수",
        filename="6110000_서울특별시_01_01_01_P_병원.csv",
        radius_m=500,
        service_name="병원",
    ),
    PoiSpec(
        feature_name="cinema_count_1000m",
        feature_name_ko="1000m 내 영화상영관 수",
        filename="6110000_서울특별시_03_13_02_P_영화상영관.csv",
        radius_m=1000,
        service_name="영화상영관",
    ),
    PoiSpec(
        feature_name="golf_practice_count_1000m",
        feature_name_ko="1000m 내 골프연습장 수",
        filename="6110000_서울특별시_10_31_01_P_골프연습장업.csv",
        radius_m=1000,
        service_name="골프연습장업",
    ),
)


def load_station_master() -> pd.DataFrame:
    stations = pd.read_csv(STATION_PATH).rename(columns={"대여소번호": "station_id", "대여소명": "station_name"})
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5174", always_xy=True)
    xs, ys = transformer.transform(stations["경도"].to_numpy(), stations["위도"].to_numpy())
    stations["x_5174"] = xs
    stations["y_5174"] = ys
    return stations


def load_poi_rows(spec: PoiSpec) -> pd.DataFrame:
    path = POI_ROOT / spec.filename
    with path.open("r", encoding="cp949", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        x_col = next(k for k in fieldnames if "좌표정보x" in k)
        y_col = next(k for k in fieldnames if "좌표정보y" in k)
        rows: list[dict[str, object]] = []
        for row in reader:
            address = row.get("도로명전체주소") or row.get("소재지전체주소") or ""
            if spec.address_keyword not in address:
                continue
            if not (row.get("영업상태명") or "").startswith(spec.business_status_prefix):
                continue
            x_val = row.get(x_col)
            y_val = row.get(y_col)
            if not x_val or not y_val:
                continue
            if spec.service_name and row.get("개방서비스명") != spec.service_name:
                continue
            if spec.category_column and spec.category_values:
                if (row.get(spec.category_column) or "") not in spec.category_values:
                    continue
            rows.append(
                {
                    "x_5174": float(x_val),
                    "y_5174": float(y_val),
                    "business_name": row.get("사업장명", ""),
                    "service_name": row.get("개방서비스명", ""),
                    "category_name": row.get(spec.category_column or "업태구분명", "") if spec.category_column else row.get("업태구분명", ""),
                }
            )
    return pd.DataFrame(rows)


def count_pois_within_radius(stations: pd.DataFrame, poi_df: pd.DataFrame, radius_m: int) -> np.ndarray:
    if poi_df.empty:
        return np.zeros(len(stations), dtype=int)
    tree = cKDTree(poi_df[["x_5174", "y_5174"]].to_numpy())
    station_points = stations[["x_5174", "y_5174"]].to_numpy()
    matched = tree.query_ball_point(station_points, r=radius_m)
    return np.array([len(idx_list) for idx_list in matched], dtype=int)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stations = load_station_master()
    features = stations[["station_id", "station_name", "자치구", "주소", "위도", "경도"]].copy()

    source_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for spec in POI_SPECS:
        poi_df = load_poi_rows(spec)
        features[spec.feature_name] = count_pois_within_radius(stations, poi_df, spec.radius_m)
        nonzero_rate = float((features[spec.feature_name] > 0).mean())
        source_rows.append(
            {
                "feature_name": spec.feature_name,
                "feature_name_ko": spec.feature_name_ko,
                "source_name": POI_SOURCE_NAME,
                "source_url": POI_SOURCE_URL,
                "source_file": spec.filename,
                "radius_m": spec.radius_m,
                "poi_rows_used": int(len(poi_df)),
            }
        )
        summary_rows.append(
            {
                "feature_name": spec.feature_name,
                "feature_name_ko": spec.feature_name_ko,
                "radius_m": spec.radius_m,
                "mean": round(float(features[spec.feature_name].mean()), 3),
                "median": round(float(features[spec.feature_name].median()), 3),
                "max": int(features[spec.feature_name].max()),
                "nonzero_station_count": int((features[spec.feature_name] > 0).sum()),
                "nonzero_station_ratio": round(nonzero_rate, 4),
            }
        )

    features.to_csv(OUTPUT_DIR / "ddri_station_poi_candidate_features.csv", index=False)
    pd.DataFrame(source_rows).to_csv(OUTPUT_DIR / "ddri_station_poi_candidate_source_summary.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "ddri_station_poi_candidate_feature_summary.csv", index=False)


if __name__ == "__main__":
    main()
