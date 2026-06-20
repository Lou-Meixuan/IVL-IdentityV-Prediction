"""
IVL Bilibili match result scraper

Features:
- Scrape IVL match result posts from Bilibili
- Download two images per match
  - Image 1: score
  - Image 2: MVP
- OCR to identify MVP player
- Output CSV

Optimizations:
- Extract real image URLs directly from HTML
- OCR only the MVP region
- Auto-insert underscore in player ID
- OCR error correction
- Filter out part-1-of-2 posts
- Each match gets its own folder
"""

from playwright.sync_api import sync_playwright
import requests
import json
import os
import re
import pandas as pd


# =====================================
# Config
# =====================================

SAVE_DIR = "result_images"
CSV_DIR  = "csv_output"
JSON_DUMP = "all_bilibili_dynamic.json"

BILIBILI_UID = "105022844"

SCROLL_COUNT = 100

# Set to a specific year (e.g. 2025) to only collect that year's matches.
# Set to None to collect all years.
TARGET_YEAR = 2025

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

all_items: list[dict] = []


# =====================================
# Download image
# =====================================

def download_image(url: str, filepath: str) -> bool:

    try:

        resp = requests.get(url, timeout=15)

        resp.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(resp.content)

        print(f"  -> {filepath}")

        return True

    except Exception as e:

        print(f"  x Download failed {url} ({e})")

        return False


# =====================================
# Detect season
# =====================================

# Maps Chinese season keywords to English folder names
# More specific patterns must come first (Finals before Regular)
_SEASON_EN = [
    ("夏季赛总决赛", "Summer_Finals"),
    ("秋季赛总决赛", "Autumn_Finals"),
    ("夏季赛",       "Summer"),
    ("秋季赛",       "Autumn"),
]


def detect_season(text: str) -> str | None:

    year_m = re.search(r"(\d{4})IVL", text)
    if not year_m:
        return None
    year = year_m.group(1)

    for cn, en in _SEASON_EN:
        if cn in text:
            return f"{year}IVL_{en}"

    return None



# =====================================
# Extract real image URLs from HTML
# =====================================

def extract_image_urls_from_html(page, debug=False) -> list[str]:

    html = page.content()

    matches = re.findall(
        r'src="(//i\d\.hdslb\.com/[^"]+)"',
        html
    )

    if debug:
        print(f"\n[HTML] raw matches: {len(matches)}")

    img_urls = []

    for m in matches:

        url = "https:" + m

        if "new_dyn" not in url:
            continue

        url = url.split("@")[0]

        if any(x in url for x in [
            "face",
            "garb",
            "archive",
        ]):
            continue

        if url not in img_urls:

            img_urls.append(url)

            if debug:
                print(f"  + {url}")

    return img_urls[:2]


# =====================================
# Team name cleaning
# =====================================

_TEAM_NOISE = re.compile(
    r"赛果公示|常规赛|季后赛|淘汰赛|积分赛|小组赛|决赛|半决赛"
)


def _clean_team(raw: str) -> str:
    # remove match-type noise words
    raw = _TEAM_NOISE.sub("", raw).strip()
    # remove any remaining Chinese characters
    raw = re.sub(r'[一-鿿]+', '', raw)
    return raw.strip()


def extract_teams(text: str):

    m = re.search(
        r"([\w一-鿿.]+)\s*[Vv][Ss]\s*([\w一-鿿.]+)",
        text
    )

    if m:

        return (
            _clean_team(m.group(1)),
            _clean_team(m.group(2))
        )

    return None, None


def extract_winner(text: str):

    m = re.search(
        r"#([\w一-鿿.]+)获胜",
        text
    )

    return _clean_team(m.group(1)) if m else None


# =====================================
# Calculate BO match score
# =====================================

def calc_match_score(text: str):

    raw_scores = re.findall(
        r"(\d+)\s*[：:]\s*(\d+)",
        text
    )

    wins_a = 0
    wins_b = 0

    game_scores = []

    for a_str, b_str in raw_scores:

        a = int(a_str)
        b = int(b_str)

        game_scores.append(f"{a}:{b}")

        if a > b:
            wins_a += 1
        elif b > a:
            wins_b += 1

    match_score = (
        f"{wins_a}-{wins_b}"
        if (wins_a or wins_b)
        else "N/A"
    )

    return match_score, wins_a, wins_b, game_scores


# =====================================
# Safe folder name (strip forbidden chars)
# =====================================

def safe_folder_name(s: str):

    return re.sub(
        r'[\\/*?:"<>|]',
        "_",
        s
    )


# =====================================
# Playwright: scrape Bilibili dynamics
# =====================================

print("=" * 50)
print("Scraping Bilibili dynamics...")
print("=" * 50)

with sync_playwright() as p:

    browser = p.chromium.launch(
        headless=False
    )

    page = browser.new_page()

    def handle_response(response):

        if "opus/feed/space" in response.url:

            print(f"\n[API] {response.url}")

            try:

                data = response.json()

                items = data["data"]["items"]

                all_items.extend(items)

                print(f"  +{len(items)} items (total: {len(all_items)})")

            except Exception as e:

                print(f"  parse failed: {e}")

    page.on("response", handle_response)

    page.goto(
        f"https://space.bilibili.com/{BILIBILI_UID}/upload/opus"
    )

    for i in range(SCROLL_COUNT):

        page.mouse.wheel(0, 8000)

        page.wait_for_timeout(2000)

        if (i + 1) % 10 == 0:
            print(f"  scrolled {i+1}/{SCROLL_COUNT}")

    browser.close()


# =====================================
# Save raw JSON
# =====================================

with open(JSON_DUMP, "w", encoding="utf-8") as f:

    json.dump(
        all_items,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"\nRaw JSON saved: {JSON_DUMP}")


# =====================================
# Second browser for post pages
# =====================================

playwright2 = sync_playwright().start()

browser2 = playwright2.chromium.launch(
    headless=False
)

page2 = browser2.new_page()


# =====================================
# Parse match results
# =====================================

results: list[dict] = []

for item in all_items:

    try:

        text = item.get("content", "")

        opus_id = item.get("opus_id", "unknown")

        season = detect_season(text)

        if not season:
            continue

        if TARGET_YEAR and not season.startswith(str(TARGET_YEAR)):
            continue

        if "赛果公示" not in text:
            continue

        if "（1/2）" in text:
            continue

        if "(1/2)" in text:
            continue

        print("\n" + "=" * 50)
        print(f"[match] {season}")

        team_a, team_b = extract_teams(text)

        winner = extract_winner(text)

        match_score, wins_a, wins_b, game_scores = calc_match_score(text)

        # Folder structure:
        # result_images/
        #   └── 2025IVL_Autumn_Finals/
        #         └── DOU5_vs_FPX.ZQ_1145xxx/

        season_dir = os.path.join(
            SAVE_DIR,
            safe_folder_name(season)
        )

        os.makedirs(season_dir, exist_ok=True)

        match_folder = safe_folder_name(
            f"{team_a}_vs_{team_b}_{opus_id}"
        )

        match_dir = os.path.join(
            season_dir,
            match_folder
        )

        os.makedirs(match_dir, exist_ok=True)

        opus_url = f"https://www.bilibili.com/opus/{opus_id}"

        print(f"  opening post: {opus_url}")

        page2.goto(
            opus_url,
            wait_until="domcontentloaded"
        )

        page2.wait_for_timeout(3000)

        img_urls = extract_image_urls_from_html(
            page2,
            debug=True
        )

        tmp_paths = []

        for idx, url in enumerate(img_urls):

            ext = (
                url.rsplit(".", 1)[-1]
                .split("?")[0]
                .lower()
            )

            if ext not in (
                "jpg",
                "jpeg",
                "png",
                "webp",
                "gif"
            ):
                ext = "jpg"

            tmp_path = os.path.join(
                match_dir,
                f"_tmp_{idx+1}.{ext}"
            )

            if download_image(url, tmp_path):
                tmp_paths.append(tmp_path)

        print(
            f"  downloaded: {len(tmp_paths)} image(s)"
        )

        # =====================================
        # Rename images
        # =====================================

        mvp = None  # will be filled later by clip_label.py

        final_paths = []

        for idx, tmp_path in enumerate(tmp_paths):

            ext = tmp_path.rsplit(".", 1)[-1]

            new_name = (
                f"score_{match_score}.{ext}"
                if idx == 0
                else f"mvp.{ext}"
            )

            new_path = os.path.join(
                match_dir,
                new_name
            )

            os.rename(tmp_path, new_path)

            final_paths.append(new_path)

        print(
            f"  {team_a} vs {team_b} | "
            f"{match_score} | MVP={mvp}"
        )

        results.append({
            "match_id": opus_id,
            "season": season,
            "team_a": team_a,
            "team_b": team_b,
            "winner": winner,
            "match_score": match_score,
            "game_scores": " | ".join(game_scores),
            "mvp": mvp,
            "images": "|".join(final_paths),
        })

    except Exception as e:

        print(f"  x error: {e}")

        continue


# =====================================
# Output CSV
# =====================================

OUTPUT_COLS = [
    "match_id",
    "team_a",
    "team_b",
    "winner",
    "match_score",
    "game_scores",
    "mvp"
]

df = pd.DataFrame(results)

print(f"\n{'='*50}")
print(f"Total dynamics scraped: {len(all_items)}")
print(f"Match results found: {len(results)}")

if df.empty:

    print("\nNo match results found.")

else:

    df = df.sort_values([
        "season",
        "match_id"
    ]).reset_index(drop=True)

    print("\nFINAL RESULTS")
    print(df[OUTPUT_COLS].to_string(index=False))

    # Split into one CSV per season and save to csv_output/
    for season_name, group in df.groupby("season"):
        filename = f"{season_name}.csv"
        filepath = os.path.join(CSV_DIR, filename)
        group[OUTPUT_COLS].to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"\nCSV saved: {filepath} ({len(group)} matches)")

    print(f"\nImages saved to: {SAVE_DIR}/")


browser2.close()

playwright2.stop()

print("\nDONE")
