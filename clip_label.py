"""
OCR-based MVP image grouping and labeling

Why OCR instead of CLIP clustering:
  The ticket/name-tag region already has the player name printed clearly.
  OCR reads the text directly; images with the same name are grouped together.
  This avoids visual-similarity mistakes (e.g. two players from the same team
  being merged into one cluster).

=====================================
Setup
=====================================
    pip install opencv-python pytesseract Pillow pandas
    # also install Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki

=====================================
Usage
=====================================
    # Step 1: OCR all images, group by name, generate contact sheet + CSV
    python clip_label.py --build --year 2025

    # Step 2: open the contact sheet PNG, fix wrong names in the CSV
    #         (each row = one OCR group; edit player_name column)

    # Step 3: apply
    python clip_label.py --apply --year 2025            # preview
    python clip_label.py --apply --year 2025 --confirm  # actually rename + update CSV

MVP image formats by year:
  2025: light-colored ticket banner with player name (fixed crop)
  2026: full-screen player photo with large name text in top-left corner (fixed crop)
"""

import os
import cv2
import csv
import json
import re
import numpy as np
import argparse
import pandas as pd
from collections import defaultdict, Counter

# ============================
# Config
# ============================

SAVE_DIR    = "result_images"
CLUSTER_DIR = "cluster_output"   # all cluster files go here

os.makedirs(CLUSTER_DIR, exist_ok=True)

# Output paths — updated at runtime when --year is passed
OUTPUT_SHEET   = os.path.join(CLUSTER_DIR, "cluster_sheet_clip.png")
OUTPUT_CSV     = os.path.join(CLUSTER_DIR, "cluster_labels_clip.csv")
OUTPUT_MAPPING = os.path.join(CLUSTER_DIR, "cluster_mapping_clip.json")


def set_year(year: int | None):
    """Update output paths to include year suffix, e.g. cluster_output/cluster_labels_clip_2026.csv"""
    global OUTPUT_SHEET, OUTPUT_CSV, OUTPUT_MAPPING
    if year:
        OUTPUT_SHEET   = os.path.join(CLUSTER_DIR, f"cluster_sheet_clip_{year}.png")
        OUTPUT_CSV     = os.path.join(CLUSTER_DIR, f"cluster_labels_clip_{year}.csv")
        OUTPUT_MAPPING = os.path.join(CLUSTER_DIR, f"cluster_mapping_clip_{year}.json")


TICKET_W, TICKET_H = 480, 120

TEAM_PREFIXES    = ["FPX.ZQ", "Wolves", "DOU5", "WBG", "ACT", "MRC", "GW", "GG", "TE", "Gr"]
_WHITELIST       = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._"
_TEAM_FIXES      = {"DOUS": "DOU5", "DOU$": "DOU5"}
_FIRST_CHAR_ALTS = {"6": "G", "N": "W", "J": "W", "0": "G"}


# ============================
# Ticket crop
# ============================

def get_ticket(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]

    # Detect year from folder path and apply fixed crop accordingly.
    # Fixed crops give consistent cell sizes in the contact sheet.
    if "2026" in path:
        # 2026 UI: full-screen player photo, name in top-left
        #   PLAYER_NAME <- y: 13-21%, x: 11-40%
        region = img[int(h * 0.13):int(h * 0.21), int(w * 0.11):int(w * 0.40)]
    elif "2024" in path:
        # 2024 UI: name overlaid on image, no ticket banner
        #   coordinates from reference image (1916x1075):
        #   x: 120-500, y: 280-350  ->  x: 6.3-26.1%, y: 26.0-32.6%
        region = img[int(h * 0.260):int(h * 0.326), int(w * 0.063):int(w * 0.261)]
    elif "2023" in path:
        # 2023 UI: name in right-side card panel
        #   coordinates from reference image (1317x740):
        #   x: 970-1160, y: 140-170  ->  x: 71.0-88.1%, y: 18.9-23.0%
        region = img[int(h * 0.189):int(h * 0.230), int(w * 0.710):int(w * 0.881)]
    else:
        # 2025 UI: ticket banner, player name area
        #   coordinates from reference image (2800x1576):
        #   x: 418-1150, y: 309-530  ->  x: 14.9-41.1%, y: 19.6-33.6%
        region = img[int(h * 0.196):int(h * 0.336), int(w * 0.149):int(w * 0.411)]

    if region.size > 0:
        return cv2.resize(region, (TICKET_W, TICKET_H))
    return None


# ============================
# OCR
# ============================

def _team_of(text: str) -> str | None:
    up = text.upper()
    for t in TEAM_PREFIXES:
        if up.startswith(t.upper()):
            return t
    return None


def ocr_ticket(ticket: np.ndarray, invert: bool = False) -> str:
    try:
        import pytesseract
        big = cv2.resize(ticket, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        if invert:
            # White text on dark/gradient background: fixed threshold works better than OTSU
            _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
        else:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cfg = f"--oem 1 --psm 6 -c tessedit_char_whitelist={_WHITELIST}"
        raw = pytesseract.image_to_string(thresh, config=cfg)
        text = re.sub(r"[^A-Za-z0-9._]", "", raw.strip())
        up = text.upper()
        for wrong, right in _TEAM_FIXES.items():
            if up.startswith(wrong):
                text = right + text[len(wrong):]
                break
        if not _team_of(text) and text:
            alt = _FIRST_CHAR_ALTS.get(text[0])
            if alt:
                text = alt + text[1:]
        return text
    except Exception:
        return ""


def _normalize_for_grouping(text: str) -> str:
    """Strip underscores, dots, and case differences for grouping key.
    e.g. 'Gr_AK' and 'GrAK' both map to 'grak' -> same group."""
    return re.sub(r'[^a-z0-9]', '', text.lower())


# ============================
# Collect MVP images
# ============================

def collect_paths(save_dir: str, year: int | None = None) -> list[str]:
    # Only collect unlabeled mvp.png files.
    # Already-labeled files (mvp_*.png) are skipped.
    # If year is set, only collect from folders whose season starts with that year.
    paths = []
    for season in sorted(os.listdir(save_dir)):
        if year and not season.startswith(str(year)):
            continue
        sp = os.path.join(save_dir, season)
        if not os.path.isdir(sp):
            continue
        for match_dir in sorted(os.listdir(sp)):
            mp = os.path.join(sp, match_dir)
            if not os.path.isdir(mp):
                continue
            for fname in sorted(os.listdir(mp)):
                if fname == "mvp.png":
                    paths.append(os.path.join(mp, fname))
    return paths


# ============================
# Build
# ============================

def build(save_dir: str, year: int | None = None):
    paths = collect_paths(save_dir, year=year)
    print(f"Found {len(paths)} unlabeled MVP images")

    # Crop name tag and run OCR on each image
    tickets     = []
    valid_paths = []
    ocr_results = []

    for i, p in enumerate(paths):
        t = get_ticket(p)
        if t is None:
            print(f"  Warning: no ticket region in {p}")
            continue
        if "2023" in p:
            # 2023 has mixed backgrounds — try both and pick whichever gives a result
            text = ocr_ticket(t, invert=False) or ocr_ticket(t, invert=True)
        else:
            text = ocr_ticket(t, invert="2024" in p)
        tickets.append(t)
        valid_paths.append(p)
        ocr_results.append(text)
        folder = os.path.basename(os.path.dirname(p))
        print(f"  [{i+1}/{len(paths)}] {folder}: '{text}'")

    print(f"\nOCR done: {len(tickets)} images")

    if not tickets:
        print("No unlabeled mvp.png files found. Nothing to do.")
        print("(Already-labeled mvp_*.png files are skipped.)")
        return

    # Group images by normalized OCR text
    # Normalization removes underscores/dots and lowercases, so minor OCR
    # differences ('Gr_AK' vs 'GrAK') land in the same group.
    group_map: dict[str, list[int]] = defaultdict(list)  # norm_key -> [indices]
    key_to_texts: dict[str, list[str]] = defaultdict(list)  # norm_key -> [raw texts]

    for i, text in enumerate(ocr_results):
        key = _normalize_for_grouping(text) if text else f"__empty_{i}__"
        group_map[key].append(i)
        key_to_texts[key].append(text)

    # Pick best display name per group: most common non-empty OCR result
    def best_display(texts: list[str]) -> str:
        non_empty = [t for t in texts if t]
        if not non_empty:
            return ""
        return Counter(non_empty).most_common(1)[0][0]

    # Sort groups by size descending
    ordered_keys = sorted(group_map.keys(), key=lambda k: -len(group_map[k]))
    key_to_gid   = {k: gid for gid, k in enumerate(ordered_keys)}

    print(f"Groups: {len(ordered_keys)}")
    for k in ordered_keys:
        gid = key_to_gid[k]
        display = best_display(key_to_texts[k])
        print(f"  #{gid:3d} n={len(group_map[k]):3d}  '{display}'")

    # Contact sheet: one representative (first image) per group
    COLS   = 4
    LABEL_H = 40
    rows   = (len(ordered_keys) + COLS - 1) // COLS
    sheet  = np.ones((rows * (TICKET_H + LABEL_H), COLS * TICKET_W, 3), dtype=np.uint8) * 230

    for idx, key in enumerate(ordered_keys):
        gid    = key_to_gid[key]
        idxs   = group_map[key]
        r, c   = divmod(idx, COLS)
        rep    = tickets[idxs[0]]
        y0, x0 = r * (TICKET_H + LABEL_H), c * TICKET_W
        sheet[y0:y0+TICKET_H, x0:x0+TICKET_W] = rep
        cv2.rectangle(sheet, (x0, y0), (x0+TICKET_W-1, y0+TICKET_H-1), (150, 150, 150), 1)
        display = best_display(key_to_texts[key])
        label   = f"#{gid} n={len(idxs)}  {display}"
        cv2.putText(sheet, label, (x0+4, y0+TICKET_H+28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 180), 2)

    cv2.imwrite(OUTPUT_SHEET, sheet)
    print(f"\nContact sheet: {OUTPUT_SHEET}")

    # CSV (same format as before — compatible with apply_labels)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "n_images", "player_name"])
        for key in ordered_keys:
            gid     = key_to_gid[key]
            display = best_display(key_to_texts[key])
            w.writerow([gid, len(group_map[key]), display])
    print(f"CSV: {OUTPUT_CSV}  (fix any wrong names, then run --apply --confirm)")

    # Mapping (same format as before — compatible with apply_labels)
    mapping = {
        str(key_to_gid[key]): [valid_paths[i] for i in group_map[key]]
        for key in ordered_keys
    }
    with open(OUTPUT_MAPPING, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Mapping: {OUTPUT_MAPPING}")


# ============================
# Apply
# ============================

def apply_labels(confirm: bool = False):
    if not os.path.exists(OUTPUT_CSV):
        print(f"Cannot find {OUTPUT_CSV}. Run --build first.")
        return
    if not os.path.exists(OUTPUT_MAPPING):
        print(f"Cannot find {OUTPUT_MAPPING}. Run --build first.")
        return

    labels: dict[str, str] = {}
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["player_name"].strip()
            if name:
                labels[row["cluster_id"]] = name

    with open(OUTPUT_MAPPING, encoding="utf-8") as f:
        mapping: dict[str, list[str]] = json.load(f)

    renamed, skipped, unchanged = 0, 0, 0
    for cid, file_paths in mapping.items():
        player = labels.get(cid)
        if not player:
            print(f"  Group {cid}: no label, skipping {len(file_paths)} file(s)")
            skipped += len(file_paths)
            continue
        safe = player.replace("/", "_").replace("\\", "_")
        for old_path in file_paths:
            new_path = os.path.join(os.path.dirname(old_path),
                                    f"mvp_{safe}{os.path.splitext(old_path)[1]}")
            if old_path == new_path:
                unchanged += 1
                continue
            if not os.path.exists(old_path):
                unchanged += 1
                continue
            print(f"  [{'rename' if confirm else 'preview'}] "
                  f"{os.path.basename(old_path)} -> mvp_{safe}.png")
            if confirm:
                os.rename(old_path, new_path)
            renamed += 1

    mode = "Done" if confirm else "Preview"
    print(f"\n{mode}: rename={renamed}, unchanged={unchanged}, unlabeled={skipped}")
    if not confirm and renamed > 0:
        print("Add --confirm to actually rename.")

    if confirm:
        _update_csv_mvp(SAVE_DIR)


# ============================
# Update CSV with MVP names from renamed files
# ============================

def _update_csv_mvp(save_dir: str):
    """
    Scan all match folders in save_dir, read MVP name from mvp_*.png filename,
    then update the mvp column in any CSV files in csv_output/ that have
    match_id and mvp columns.
    """
    import glob

    # Build {match_id: mvp_name} from renamed image files
    mvp_map: dict[str, str] = {}
    for season in sorted(os.listdir(save_dir)):
        sp = os.path.join(save_dir, season)
        if not os.path.isdir(sp):
            continue
        for match_dir in sorted(os.listdir(sp)):
            mp = os.path.join(sp, match_dir)
            if not os.path.isdir(mp):
                continue
            m = re.search(r'(\d{15,})', match_dir)
            if not m:
                continue
            match_id = m.group(1)
            for fname in os.listdir(mp):
                if fname.startswith("mvp_") and fname.endswith(".png"):
                    mvp_name = fname[4:-4]
                    mvp_map[match_id] = mvp_name
                    break

    if not mvp_map:
        print("  No labeled MVP files found, skipping CSV update.")
        return

    updated_files = 0
    for csv_path in glob.glob(os.path.join("csv_output", "*.csv")):
        try:
            df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
            if "match_id" not in df.columns or "mvp" not in df.columns:
                continue
            before = df["mvp"].copy()
            df["mvp"] = df["match_id"].map(mvp_map).fillna(df["mvp"])
            changed = (df["mvp"] != before).sum()
            if changed > 0:
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                print(f"  Updated {changed} MVP entries in {csv_path}")
                updated_files += 1
        except Exception as e:
            print(f"  Warning: could not update {csv_path}: {e}")

    if updated_files == 0:
        print("  No CSV files updated.")


# ============================
# Reset: undo a previous --apply --confirm
# ============================

def reset(save_dir: str, year: int | None = None):
    """
    Rename mvp_PLAYERNAME.png back to mvp.png and clear the mvp column in csv_output/.
    Use this to redo labeling from scratch for a given year.
    """
    import glob

    renamed = 0
    for season in sorted(os.listdir(save_dir)):
        if year and not season.startswith(str(year)):
            continue
        sp = os.path.join(save_dir, season)
        if not os.path.isdir(sp):
            continue
        for match_dir in sorted(os.listdir(sp)):
            mp = os.path.join(sp, match_dir)
            if not os.path.isdir(mp):
                continue
            for fname in os.listdir(mp):
                if fname.startswith("mvp_") and fname.endswith(".png"):
                    old = os.path.join(mp, fname)
                    new = os.path.join(mp, "mvp.png")
                    os.rename(old, new)
                    print(f"  reset: {fname} -> mvp.png")
                    renamed += 1

    print(f"\nRenamed {renamed} file(s) back to mvp.png")

    # Clear mvp column in CSVs
    updated = 0
    for csv_path in glob.glob(os.path.join("csv_output", "*.csv")):
        if year and not os.path.basename(csv_path).startswith(str(year)):
            continue
        try:
            df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
            if "mvp" not in df.columns:
                continue
            df["mvp"] = None
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"  Cleared mvp column in {csv_path}")
            updated += 1
        except Exception as e:
            print(f"  Warning: could not update {csv_path}: {e}")

    print(f"Cleared mvp column in {updated} CSV file(s)")
    print("\nNow run: python clip_label.py --build --year", year or "")


# ============================
# CLI
# ============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR-based MVP image grouping and labeling")
    parser.add_argument("--build",   action="store_true",
                        help="OCR all images, group by name, generate contact sheet + CSV")
    parser.add_argument("--apply",   action="store_true",
                        help="Rename files based on filled CSV")
    parser.add_argument("--confirm", action="store_true",
                        help="Used with --apply: actually rename and update CSV")
    parser.add_argument("--year",    type=int, default=None,
                        help="Filter by year and use year-suffixed output files (e.g. --year 2025)")
    parser.add_argument("--reset",   action="store_true",
                        help="Undo a previous --apply --confirm: rename mvp_*.png back to mvp.png and clear CSV mvp column")
    args = parser.parse_args()

    set_year(args.year)

    if args.build:
        build(SAVE_DIR, year=args.year)
    elif args.apply:
        apply_labels(confirm=args.confirm)
    elif args.reset:
        reset(SAVE_DIR, year=args.year)
    else:
        parser.print_help()
