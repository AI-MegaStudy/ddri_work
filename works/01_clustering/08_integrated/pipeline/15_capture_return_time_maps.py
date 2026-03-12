from pathlib import Path
import time

from selenium import webdriver
from selenium.webdriver.safari.options import Options


ROOT = Path("/Users/cheng80/Desktop/ddri_work")
BASE_DIR = ROOT / "works" / "01_clustering" / "08_integrated" / "intermediate" / "return_time_district"

TARGETS = [
    ("ddri_return_map_2025_7_10.html", "ddri_return_map_2025_7_10_safari.png"),
    ("ddri_return_map_2025_11_14.html", "ddri_return_map_2025_11_14_safari.png"),
    ("ddri_return_map_2025_17_20.html", "ddri_return_map_2025_17_20_safari.png"),
]


def main() -> None:
    opts = Options()
    driver = webdriver.Safari(options=opts)
    driver.set_window_size(1600, 1200)

    try:
        for html_name, png_name in TARGETS:
            html_path = (BASE_DIR / html_name).resolve()
            png_path = BASE_DIR / png_name
            driver.get(html_path.as_uri())
            time.sleep(2.5)
            driver.save_screenshot(str(png_path))
            print(f"saved: {png_path}")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
