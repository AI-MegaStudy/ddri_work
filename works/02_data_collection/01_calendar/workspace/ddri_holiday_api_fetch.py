from pathlib import Path
from urllib.parse import quote
import re

import pandas as pd
import requests


BASE_DIR = Path("/Users/cheng80/Desktop/ddri_work")
API_DIR = BASE_DIR / "3조 공유폴더" / "[일정데이터] 특일 정보 API"
OUTPUT_DIR = BASE_DIR / "works" / "calendar" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"


def read_service_key() -> str:
    text = (API_DIR / "API 인증키.txt").read_text().strip()
    match = re.search(r"일반 인증키\s*:\s*(\S+)", text)
    if not match:
        raise ValueError("API 인증키 파일에서 일반 인증키를 찾지 못했습니다.")
    return match.group(1)


def fetch_month(year: int, month: int, service_key: str) -> list[dict]:
    params = {
        "solYear": f"{year:04d}",
        "solMonth": f"{month:02d}",
        "ServiceKey": service_key,
        "_type": "json",
        "numOfRows": "100",
    }
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    body = data.get("response", {}).get("body", {})
    items = body.get("items", {})
    if isinstance(items, str):
        return []
    item = items.get("item", [])
    if isinstance(item, str):
        return []
    if isinstance(item, dict):
        item = [item]
    return item


def build_holiday_table(years: list[int]) -> pd.DataFrame:
    service_key = read_service_key()
    rows = []
    for year in years:
        for month in range(1, 13):
            items = fetch_month(year, month, service_key)
            for item in items:
                rows.append(
                    {
                        "date": pd.to_datetime(str(item["locdate"]), format="%Y%m%d"),
                        "locdate": str(item["locdate"]),
                        "date_name": item.get("dateName"),
                        "date_kind": item.get("dateKind"),
                        "is_holiday_api": item.get("isHoliday"),
                        "seq": item.get("seq"),
                        "source_year": year,
                        "source_month": month,
                    }
                )

    holiday_df = pd.DataFrame(rows).sort_values(["date", "seq"]).reset_index(drop=True)
    return holiday_df


def build_daily_calendar(years: list[int], holiday_df: pd.DataFrame) -> pd.DataFrame:
    all_dates = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D")
    calendar_df = pd.DataFrame({"date": all_dates})
    calendar_df["year"] = calendar_df["date"].dt.year
    calendar_df["month"] = calendar_df["date"].dt.month
    calendar_df["day"] = calendar_df["date"].dt.day
    calendar_df["day_of_week"] = calendar_df["date"].dt.dayofweek
    calendar_df["is_weekend"] = (calendar_df["day_of_week"] >= 5).astype(int)

    holiday_daily = (
        holiday_df.groupby("date")
        .agg(
            holiday_name=("date_name", lambda x: " | ".join(sorted(set(str(v) for v in x)))),
            holiday_count=("date_name", "size"),
            is_holiday=("is_holiday_api", lambda x: int(any(str(v) == "Y" for v in x))),
        )
        .reset_index()
    )
    calendar_df = calendar_df.merge(holiday_daily, on="date", how="left")
    calendar_df["holiday_count"] = calendar_df["holiday_count"].fillna(0).astype(int)
    calendar_df["is_holiday"] = calendar_df["is_holiday"].fillna(0).astype(int)
    calendar_df["holiday_name"] = calendar_df["holiday_name"].fillna("")
    calendar_df["is_business_holiday"] = ((calendar_df["is_weekend"] == 1) | (calendar_df["is_holiday"] == 1)).astype(int)
    return calendar_df


def main():
    years = [2023, 2024, 2025]
    holiday_df = build_holiday_table(years)
    calendar_df = build_daily_calendar(years, holiday_df)

    holiday_path = OUTPUT_DIR / "ddri_holiday_api_raw_2023_2025.csv"
    calendar_path = OUTPUT_DIR / "ddri_calendar_daily_2023_2025.csv"
    holiday_df.to_csv(holiday_path, index=False)
    calendar_df.to_csv(calendar_path, index=False)

    print(f"saved: {holiday_path}")
    print(f"saved: {calendar_path}")
    print(f"holiday_rows={len(holiday_df)}")
    print(f"calendar_rows={len(calendar_df)}")


if __name__ == "__main__":
    main()
