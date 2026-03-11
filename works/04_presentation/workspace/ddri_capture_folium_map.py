from pathlib import Path
import time

from selenium import webdriver
from selenium.webdriver.safari.options import Options as SafariOptions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAP_HTML_PATH = PROJECT_ROOT / "works" / "clustering" / "maps" / "ddri_cluster_map_gangnam.html"
OUTPUT_IMAGE_PATH = PROJECT_ROOT / "works" / "presentation" / "ddri_cluster_map_gangnam.png"


def capture_folium_map() -> None:
    options = SafariOptions()
    driver = webdriver.Safari(options=options)
    try:
        driver.set_window_size(1800, 1100)
        driver.get(MAP_HTML_PATH.resolve().as_uri())
        # Give Leaflet tiles and markers enough time to render.
        time.sleep(4)
        OUTPUT_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        driver.save_screenshot(str(OUTPUT_IMAGE_PATH))
    finally:
        driver.quit()


if __name__ == "__main__":
    capture_folium_map()
