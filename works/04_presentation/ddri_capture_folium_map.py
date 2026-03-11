from pathlib import Path
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.safari.options import Options as SafariOptions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAP_HTML_PATH = PROJECT_ROOT / "works" / "01_clustering" / "04_maps" / "ddri_cluster_map_gangnam.html"
OUTPUT_IMAGE_PATH = PROJECT_ROOT / "works" / "04_presentation" / "01_clustering" / "ddri_cluster_map_gangnam.png"


def capture_folium_map() -> None:
    driver = None
    try:
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--window-size=1800,1100")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--hide-scrollbars")
        driver = webdriver.Chrome(options=chrome_options)
    except Exception:
        safari_options = SafariOptions()
        driver = webdriver.Safari(options=safari_options)

    try:
        driver.set_window_size(1800, 1100)
        driver.get(MAP_HTML_PATH.resolve().as_uri())
        # Give Leaflet tiles and markers enough time to render.
        time.sleep(4)
        OUTPUT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        driver.save_screenshot(str(OUTPUT_IMAGE_PATH))
    finally:
        if driver is not None:
            driver.quit()


if __name__ == "__main__":
    capture_folium_map()
