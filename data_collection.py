"""
IVL Bilibili 赛果抓取脚本（OCR.Space API 版）

功能：
- 抓取 IVL 赛果公示
- 每场比赛下载两张图片
- 第一张：比分图
- 第二张：MVP 图
- OCR 自动识别 MVP
- 输出 CSV

优化：
✓ HTML 直接提取真实图片
✓ OCR.Space API
✓ 只 OCR MVP 区域
✓ 自动补 _
✓ OCR 自动纠错
✓ 过滤 (1/2) 分篇帖子
✓ 每场比赛独立文件夹
"""

from playwright.sync_api import sync_playwright
import requests
import json
import os
import re
import pandas as pd
from datetime import datetime
import cv2
import unicodedata


# =====================================
# 配置
# =====================================

SAVE_DIR = "result_images"
OUTPUT_CSV = "2025ivl_总决赛结果.csv"
JSON_DUMP = "all_bilibili_dynamic.json"

BILIBILI_UID = "105022844"

SCROLL_COUNT = 75

OCR_API_KEY = "helloworld"

os.makedirs(SAVE_DIR, exist_ok=True)

all_items: list[dict] = []


# =====================================
# MVP OCR 配置
# =====================================

TEAM_PREFIXES = [
    "Gr",
    "WBG",
    "ACT",
    "DOU5",
    "FPX.ZQ",
    "TE",
    "GW",
    "MRC",
    "GG",
    "Wolves",
]

PLAYER_ID_RE = re.compile(
    r"^[A-Za-z0-9.]{1,10}_[A-Za-z0-9]{2,16}$"
)


# =====================================
# 裁切 MVP 区域
# =====================================

def crop_mvp_region(image_path: str):

    img = cv2.imread(image_path)

    if img is None:
        return None

    h, w = img.shape[:2]

    crop = img[
        int(h * 0.14):int(h * 0.36),
        int(w * 0.08):int(w * 0.60)
    ]

    # 放大
    crop = cv2.resize(
        crop,
        None,
        fx=6,
        fy=6,
        interpolation=cv2.INTER_CUBIC
    )

    # 灰度
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 锐化
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )

    gray = cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        kernel
    )

    # 二值化
    _, thresh = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY
    )

    cv2.imwrite("debug_crop.png", thresh)

    return thresh

# =====================================
# 新增：OCR 后处理
# =====================================

def clean_ocr_text(text: str):

    # Unicode 转 ASCII
    text = unicodedata.normalize(
        "NFKD",
        text
    ).encode(
        "ascii",
        "ignore"
    ).decode()

    text = text.strip()

    # 去空格换行
    text = text.replace(" ", "")
    text = text.replace("\n", "")

    # 去垃圾符号
    text = text.replace("-", "")
    text = text.replace("—", "")
    text = text.replace("~", "")
    text = text.replace("{", "")
    text = text.replace("}", "")
    text = text.replace("|", "")
    text = text.replace("'", "")
    text = text.replace("`", "")

    # 只保留：
    # 英文 数字 . _
    text = re.sub(
        r"[^A-Za-z0-9._]",
        "",
        text
    )

    # 自动补 _
    if "_" not in text:

        for team in TEAM_PREFIXES:

            if text.upper().startswith(
                team.upper()
            ):

                remain = text[len(team):]

                if len(remain) >= 2:

                    text = (
                        team + "_" + remain
                    )

                    break

    return text

# =====================================
# OCR.Space API
# =====================================

def ocr_mvp_from_image(image_path: str) -> str | None:

    try:

        crop = crop_mvp_region(image_path)

        if crop is None:
            return None

        temp_path = "temp_mvp.png"

        cv2.imwrite(temp_path, crop)

        url = "https://api.ocr.space/parse/image"

        with open(temp_path, "rb") as f:

            response = requests.post(
                url,
                files={
                    "temp_mvp.png": f
                },
                data={
                    "apikey": OCR_API_KEY,
                    "language": "eng",
                    "isOverlayRequired": False,
                    "OCREngine": 2,
                    "scale": True,
                }
            )

        result = response.json()

        parsed = result.get("ParsedResults")

        if not parsed:
            return None

        text = parsed[0]["ParsedText"]

        if not text:
            return None

        text = clean_ocr_text(text)

        # 自动补 _
        if "_" not in text:

            for team in TEAM_PREFIXES:

                if text.lower().startswith(team.lower()):

                    remain = text[len(team):]

                    if len(remain) >= 2:

                        text = team + "_" + remain
                        break

        print(f"    OCR API RESULT: {text}")

        return text

    except Exception as e:

        print(f"  ✗ OCR失败: {e}")

        return None


# =====================================
# 下载图片
# =====================================

def download_image(url: str, filepath: str) -> bool:

    try:

        resp = requests.get(url, timeout=15)

        resp.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(resp.content)

        print(f"  ↓ {filepath}")

        return True

    except Exception as e:

        print(f"  ✗ 下载失败 {url} ({e})")

        return False


# =====================================
# 检测赛季
# =====================================

def detect_season(text: str) -> str | None:

    for pattern in [
        r"\d{4}IVL夏季赛总决赛",
        r"\d{4}IVL秋季赛总决赛"
    ]:

        m = re.search(pattern, text)

        if m:
            return m.group(0)

    return None


# =====================================
# 提取日期
# =====================================

def extract_pub_date(item: dict) -> str | None:

    paths = [
        lambda x: x["modules"]["module_author"]["pub_ts"],
        lambda x: x["pub_ts"],
        lambda x: x["timestamp"],
    ]

    for fn in paths:

        try:

            ts = fn(item)

            return datetime.fromtimestamp(
                int(ts)
            ).strftime("%Y-%m-%d")

        except Exception:
            continue

    return None


# =====================================
# HTML 提取真实图片
# =====================================

def extract_image_urls_from_html(page, debug=False) -> list[str]:

    html = page.content()

    matches = re.findall(
        r'src="(//i\d\.hdslb\.com/[^"]+)"',
        html
    )

    if debug:
        print(f"\n[HTML] 原始匹配数量: {len(matches)}")

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
# 队名处理
# =====================================

_TEAM_NOISE = re.compile(
    r"赛果公示|常规赛|季后赛|淘汰赛|积分赛|小组赛|决赛|半决赛"
)


def _clean_team(raw: str) -> str:

    return _TEAM_NOISE.sub("", raw).strip()


def extract_teams(text: str):

    m = re.search(
        r"([\w\u4e00-\u9fff.]+)\s*[Vv][Ss]\s*([\w\u4e00-\u9fff.]+)",
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
        r"#([\w\u4e00-\u9fff.]+)获胜",
        text
    )

    return _clean_team(m.group(1)) if m else None


# =====================================
# 计算 BO 分数
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
# 文件夹安全名
# =====================================

def safe_folder_name(s: str):

    return re.sub(
        r'[\\/*?:"<>|]',
        "_",
        s
    )


# =====================================
# Playwright 抓取动态
# =====================================

print("=" * 50)
print("开始抓取 Bilibili 动态……")
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

                print(f"  +{len(items)} 条（累计 {len(all_items)} 条）")

            except Exception as e:

                print(f"  解析失败: {e}")

    page.on("response", handle_response)

    page.goto(
        f"https://space.bilibili.com/{BILIBILI_UID}/upload/opus"
    )

    for i in range(SCROLL_COUNT):

        page.mouse.wheel(0, 8000)

        page.wait_for_timeout(2000)

        if (i + 1) % 10 == 0:
            print(f"  已滚动 {i+1}/{SCROLL_COUNT} 次")

    browser.close()


# =====================================
# 保存 JSON
# =====================================

with open(JSON_DUMP, "w", encoding="utf-8") as f:

    json.dump(
        all_items,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"\n原始 JSON 已保存：{JSON_DUMP}")


# =====================================
# 第二浏览器
# =====================================

playwright2 = sync_playwright().start()

browser2 = playwright2.chromium.launch(
    headless=False
)

page2 = browser2.new_page()


# =====================================
# 解析赛果
# =====================================

results: list[dict] = []

for item in all_items:

    try:

        text = item.get("content", "")

        opus_id = item.get("opus_id", "unknown")

        season = detect_season(text)

        if not season:
            continue

        if "赛果公示" not in text:
            continue

        if "总决赛" not in text:
            continue

        if "（1/2）" in text:
            continue

        if "(1/2)" in text:
            continue

        print("\n" + "=" * 50)
        print(f"[命中] {season}")

        team_a, team_b = extract_teams(text)

        winner = extract_winner(text)

        match_score, wins_a, wins_b, game_scores = calc_match_score(text)

        date = extract_pub_date(item)

        # =====================================
        # 新文件夹结构
        # result_images/
        #   └── 2025IVL秋季赛/
        #         └── DOU5_vs_FPX.ZQ_1145xxx/
        # =====================================

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

        print(f"  打开帖子页面：{opus_url}")

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
            f"  共下载图片：{len(tmp_paths)} 张"
        )

        # =====================================
        # OCR MVP
        # =====================================

        mvp = None

        if len(tmp_paths) >= 2:

            print("  OCR 第二张图……")

            mvp = ocr_mvp_from_image(
                tmp_paths[1]
            )

            if mvp:
                print(f"  ✓ MVP={mvp}")

        elif len(tmp_paths) == 1:

            print("  只有一张图，OCR……")

            mvp = ocr_mvp_from_image(
                tmp_paths[0]
            )

        # =====================================
        # 重命名图片
        # =====================================

        final_paths = []

        for idx, tmp_path in enumerate(tmp_paths):

            ext = tmp_path.rsplit(".", 1)[-1]

            new_name = (
                f"score_{match_score}.{ext}"
                if idx == 0
                else f"mvp_{safe_folder_name(mvp or 'unknown')}.{ext}"
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
            "date": date,
            "winner": winner,
            "match_score": match_score,
            "game_scores": " | ".join(game_scores),
            "mvp": mvp,
            "images": "|".join(final_paths),
        })

    except Exception as e:

        print(f"  ✗ 处理出错：{e}")

        continue


# =====================================
# 输出 CSV
# =====================================

OUTPUT_COLS = [
    "match_id",
    "team_a",
    "team_b",
    "date",
    "winner",
    "match_score",
    "game_scores",
    "mvp"
]

df = pd.DataFrame(results)

print(f"\n{'='*50}")

print(f"共抓取动态：{len(all_items)} 条")

print(f"赛果公示命中：{len(results)} 条")

if df.empty:

    print("\n⚠️ 未匹配到任何赛果公示")

else:

    df = df.sort_values([
        "season",
        "date",
        "match_id"
    ]).reset_index(drop=True)

    print("\nFINAL RESULTS")

    print(
        df[OUTPUT_COLS].to_string(index=False)
    )

    df[OUTPUT_COLS].to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nCSV 已保存：{OUTPUT_CSV}")

    print(f"图片已保存至：{SAVE_DIR}/")


browser2.close()

playwright2.stop()

print("\nDONE ✓")