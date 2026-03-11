from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path("/Users/cheng80/Desktop/ddri_work")
OUTPUT_DIR = BASE_DIR / "works" / "weather" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LATITUDE = 37.514557
LONGITUDE = 127.0495556
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_open_meteo_history(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": "Asia/Seoul",
    }
    response = requests.get(BASE_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    hourly = data["hourly"]
    df = pd.DataFrame(
        {
            "datetime": hourly["time"],
            "temperature": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "precipitation": hourly["precipitation"],
            "wind_speed": hourly["wind_speed_10m"],
        }
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def save_range(start_date: str, end_date: str, output_name: str):
    df = fetch_open_meteo_history(start_date, end_date)
    output_path = OUTPUT_DIR / output_name
    df.to_csv(output_path, index=False)
    print(f"saved: {output_path}")
    print(f"rows={len(df)} range={start_date}~{end_date}")


def main():
    save_range("2024-01-01", "2024-01-01", "ddri_weather_2024_0101_hourly.csv")
    save_range("2025-01-01", "2025-12-31", "ddri_weather_2025_hourly.csv")


if __name__ == "__main__":
    main()
