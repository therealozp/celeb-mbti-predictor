import os
import time
from pathlib import Path

import pandas as pd
import requests
from web_scraper import GoogleScraper
from selenium import webdriver

import random

DATA_CSV = "data/balanced_mbti.csv"  # your 3k entries
OUTPUT_DIR = Path("images")  # will contain images/{four_letter}/...
LOG_PATH = "fetch_log.csv"
MAX_IMAGES_PER_PERSON = 3


def slugify(name: str) -> str:
    name = name.strip().lower()
    out = []
    for ch in name:
        if ch.isalnum():
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    slug = "".join(out)
    slug = "_".join([p for p in slug.split("_") if p])
    return slug or "unknown"


# --- load data and existing log ---
if __name__ == "__main__":
    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH, encoding="latin1")
        count_col = next(
            (c for c in ("images_fetched", "images_fetched") if c in log_df.columns),
            None,
        )
        if count_col is not None:
            fetched = pd.to_numeric(log_df[count_col], errors="coerce").fillna(0)
            mask = fetched > 0
            done = set(zip(log_df.loc[mask, "id"], log_df.loc[mask, "four_letter"]))
        else:
            done = set(zip(log_df["id"], log_df["four_letter"]))
    else:
        done = set()
        with open(LOG_PATH, "w") as f:
            f.write("id,name,four_letter,images_fetched\n")  # header

    # open log in append mode
    log_file = open(LOG_PATH, "a", buffering=1)  # line-buffered

    wd = webdriver.Chrome()
    wd.maximize_window()
    wd.get("https://google.com")
    scraper = GoogleScraper(webdriver=wd, max_num_of_images=MAX_IMAGES_PER_PERSON)

    df = pd.read_csv(DATA_CSV)

    for idx, row in df.iterrows():
        id = row["id"]
        name = str(row["name"]).strip()
        mbti = str(row["four_letter"]).strip().upper()
        key = (id, mbti)

        if key in done:
            continue  # already processed in a previous run

        wd.execute_script("window.open('about:blank', '_blank');")
        wd.switch_to.window(wd.window_handles[-1])
        print(f"\n[INFO] {idx+1}/{len(df)}: {name} ({mbti})")
        slug = slugify(name)
        query = f"{name} portrait photograph"

        dest_path = OUTPUT_DIR / mbti
        num_images_fetched = scraper.scrape_images(query, folder_path=dest_path)

        # write-ahead log: record that this celeb was processed
        if num_images_fetched is None:
            num_images_fetched = 0

        if num_images_fetched == 0:
            print(f"[WARN] No images fetched for {name} ({mbti}).")
        else:
            try:
                log_file.write(f'{id},"{name}",{mbti},{num_images_fetched}\n')
            except Exception:
                safe_line = f"{id},some_celeb,{mbti},{num_images_fetched}\n"
                log_file.write(safe_line)
            log_file.flush()
            done.add(key)

        print(f"[INFO] Completed fetching images for {name} ({mbti}).")

        random_delay = random.random() * 5 + 1
        time.sleep(random_delay)  # optional politeness delay
        wd.close()
        wd.switch_to.window(wd.window_handles[0])

    wd.quit()
    log_file.close()
    print("\n[DONE] Pass complete.")
