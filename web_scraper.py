# Import packages

import PIL
import selenium
from selenium import webdriver
import requests
import shutil
import hashlib
import os
import io
import time
from PIL import Image

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# A common browser User-Agent
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


# Create the scraper class
class GoogleScraper:
    """Downloades images from google based on the query.
    webdriver - Selenium webdriver
    max_num_of_images - Maximum number of images that we want to download
    """

    def __init__(self, webdriver: webdriver, max_num_of_images: int):
        self.wd = webdriver
        self.max_num_of_images = max_num_of_images
        self.wait = WebDriverWait(self.wd, 10)

    def _scroll_to_the_end(self):
        self.wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

    def _build_query(self, query: str):
        return f"https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={query}&oq={query}&gs_l=img"

    def _get_info(self, query: str):
        image_urls = set()

        self.wd.get(self._build_query(query))
        self._scroll_to_the_end()
        time.sleep(2)

        # scroll back to start
        self.wd.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)

        # thumbnails = self.wd.find_elements(By.CSS_SELECTOR, "img.YQ4gaf")
        thumbnail_boxes = self.wd.find_elements(By.CSS_SELECTOR, "div.q1MG4e.mNsIhb")
        print(f"Found {len(thumbnail_boxes)} thumbnail boxes")
        total = min(len(thumbnail_boxes), self.max_num_of_images)

        for i in range(total):
            try:
                time.sleep(1)
                self.wait.until(EC.element_to_be_clickable(thumbnail_boxes[i]))
                print("[INFO] element ready to click, clicking...")
                # self.wd.execute_script(
                #     "arguments[0].scrollIntoView();", thumbnail_boxes[i]
                # )
                thumbnail_boxes[i].click()
            except:
                try:
                    self.wd.execute_script("window.scrollBy(0, 100);")
                    time.sleep(2)
                    thumbnail_boxes[i].click()

                except Exception as e:
                    print(f"[{i}] ERROR clicking (2nd attempt):", e)
                    continue

            time.sleep(5)
            img_box_2 = self.wd.find_elements(By.CSS_SELECTOR, "div.p7sI2")[1]
            link = img_box_2.find_elements(By.TAG_NAME, "img")[0].get_attribute("src")
            if link and link.startswith("http"):
                image_urls.add(link)

        return image_urls

    def download_image(self, folder_path: str, url: str):
        try:
            # 1. Make the request with headers and a timeout
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)

            # 2. Check for HTTP errors (e.g., 404, 403, 500)
            response.raise_for_status()

            # 3. Check the content type to ensure it's an image
            content_type = response.headers.get("content-type", "").lower()
            if not content_type.startswith("image/"):
                print(
                    f"ERROR: URL did not return an image. Content-Type: {content_type}"
                )
                return False

            # 4. Get the raw image content
            image_content = response.content

        except requests.exceptions.RequestException as e:
            print(f"ERROR: Could not download {url} - {e}")
            return False  # Stop execution if download fails

        # --- Only proceed to save if download was successful ---

        try:
            # 5. Use BytesIO to open the image content with Pillow
            image_file = io.BytesIO(image_content)
            image = Image.open(image_file).convert("RGB")

            # 6. Create the file path
            # Note: Hashing content is good for de-duplication
            file_hash = hashlib.sha1(image_content).hexdigest()[:10]
            file_path = os.path.join(folder_path, f"{file_hash}.jpg")

            # 7. Save the image as JPEG
            with open(file_path, "wb") as f:
                image.save(f, "JPEG", quality=85)

            print(f"SUCCESS: saved {url} - as {file_path}")
            return True

        # 8. Catch the specific Pillow error
        except PIL.UnidentifiedImageError:
            print(
                f"ERROR: Could not identify image from {url}. The file may be corrupt or not a valid image."
            )
            return False
        except Exception as e:
            print(f"ERROR: Could not process or save {url} - {e}")
            return False

    def scrape_images(self, query: str, folder_path="path"):
        folder = folder_path

        if not os.path.exists(folder):
            os.makedirs(folder)

        image_info = self._get_info(query)
        image_counts = len(image_info)
        print(f"Found {image_counts} images for the query '{query}'")

        if image_counts == 0:
            print(f"No images found for the query '{query}'")
            return 0
        print(f"Downloading images...")

        saved = 0
        for image in image_info:
            if self.download_image(folder, image):
                saved += 1
        print(f"Downloaded {saved} images out of {image_counts}")
        return saved


# images = self.wd.find_elements(By.CSS_SELECTOR, "img.sFlh5c.FyHeAf.iPVvYb")
# time.sleep(1)

# for image in images:
#     if image.get_attribute("src") and "http" in image.get_attribute("src"):
#         image_urls.add(image.get_attribute("src"))
