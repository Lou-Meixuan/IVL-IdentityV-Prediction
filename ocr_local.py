"""
Local MVP OCR module (replaces OCR.Space API)

Dependencies:
    pip install opencv-python pytesseract --break-system-packages
    # macOS: brew install tesseract
    # Ubuntu: sudo apt install tesseract-ocr

Usage:
    from ocr_local import ocr_mvp_from_image
    mvp = ocr_mvp_from_image("path/to/mvp.png")

    # Drop-in replacement for the ocr_mvp_from_image function in data_collection.py
"""

import cv2
import pytesseract
import numpy as np
import re
from difflib import SequenceMatcher


# =====================================
# Known player list
# Maintain this list — OCR results are snapped to the closest name here.
# Format: TEAM_PlayerID
# Add new players directly to this list.
# =====================================

KNOWN_PLAYERS: list[str] = [
    # ACT
    "ACT_zz9",
    "ACT_Gougou",
    # DOU5
    "DOU5_Chen1",
    "DOU5_mhxm",
    "DOU5_sChen",   # unconfirmed — may be the same person as Chen1
    # FPX.ZQ
    "FPX.ZQ_Yuan",
    "FPX.ZQ_stian",
    "FPX.ZQ_lin",
    "FPX.ZQ_xbuon",
    "FPX.ZQ_DongX",
    # GG
    "GG_18",
    "GG_Dixdd",
    "GG_xawm",
    # GW
    "GW_Lancelot",
    "GW_Littlie",
    # Gr
    "Gr_Jin",
    "Gr_Red",
    "Gr_heart",
    # MRC
    "MRC_BoiLu",
    "MRC_HuaC",
    "MRC_Nanako",
    # TE
    "TE_Eppei",
    "TE_ppei",
    "TE_ppns",
    "TE_suda",
    "TE_Drop",
    # WBG
    "WBG_Guoker",
    "WBG_PPersica",
    "WBG_nle",
    # Wolves
    "Wolves_ChoAi",
    "Wolves_L8h.",
    "Wolves_XWx.",
    "Wolves_487",
]

# =====================================
# Known team prefixes (longest first to avoid Gr matching GW)
# =====================================

TEAM_PREFIXES = [
    "FPX.ZQ",
    "Wolves",
    "DOU5",
    "WBG",
    "ACT",
    "MRC",
    "GW",
    "GG",
    "TE",
    "Gr",
]

# Tesseract character whitelist
_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._"

# Minimum similarity to accept a snap (below this, return the basic-fixed result)
_SNAP_THRESHOLD = 0.55


# =====================================
# Ticket region detection
# =====================================

def _find_ticket_gray(img: np.ndarray) -> np.ndarray | None:
    """Detect the light-colored ticket region in an MVP image and return a grayscale crop."""
    h, w = img.shape[:2]
    region = img[int(h * 0.08):int(h * 0.42), int(w * 0.04):int(w * 0.70)]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    for lo_v, hi_s in [(160, 60), (130, 80), (100, 110)]:
        mask = cv2.inRange(hsv, (0, 0, lo_v), (180, hi_s, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = [
            c for c in contours
            if cv2.contourArea(c) > 5000
            and cv2.boundingRect(c)[2] / max(cv2.boundingRect(c)[3], 1) > 1.5
        ]
        if not candidates:
            continue

        c = max(candidates, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(c)
        # Trim left 10% (team logo) and right 8% (barcode)
        ticket = region[
            y : y + ch,
            x + int(cw * 0.10) : x + int(cw * 0.92)
        ]
        if ticket.size == 0:
            continue

        ticket = cv2.resize(ticket, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(ticket, cv2.COLOR_BGR2GRAY)

    return None


# =====================================
# Raw OCR
# =====================================

def _raw_ocr(gray: np.ndarray) -> str:
    """Binarize with OTSU and run Tesseract PSM 6 on the ticket image."""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cfg = f"--oem 1 --psm 6 -c tessedit_char_whitelist={_WHITELIST}"
    raw = pytesseract.image_to_string(thresh, config=cfg)
    return raw.strip().replace("\n", "").replace(" ", "")


# =====================================
# Fuzzy matching
# =====================================

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _snap_to_known(raw: str) -> tuple[str, float]:
    """Snap an OCR result to the closest name in KNOWN_PLAYERS. Returns (best_match, score)."""
    best, best_score = raw, 0.0
    for known in KNOWN_PLAYERS:
        score = _similarity(raw, known)
        if score > best_score:
            best_score = score
            best = known
    return best, best_score


def _team_of(text: str) -> str | None:
    """Return the matching team prefix for a string (case-insensitive)."""
    up = text.upper()
    for t in TEAM_PREFIXES:
        if up.startswith(t.upper()):
            return t
    return None


def _snap_within_team(raw: str) -> tuple[str, float]:
    """
    Identify the team from the OCR result, then fuzzy-match only within that team's players.
    Falls back to global match if no team is identified.
    """
    team = _team_of(raw)
    if team:
        pool = [p for p in KNOWN_PLAYERS if _team_of(p) == team]
        if pool:
            best = max(pool, key=lambda p: _similarity(raw, p))
            return best, _similarity(raw, best)
    return _snap_to_known(raw)


# =====================================
# Basic post-processing (independent of KNOWN_PLAYERS)
# =====================================

# Whole-string substitutions for known OCR mistakes in team names
_TEAM_FIXES = {
    "DOUS": "DOU5",
    "DOU$": "DOU5",
    "DOUB": "DOU5",
}

# First-character confusions: key -> possible correct characters
_FIRST_CHAR_ALTS: dict[str, list[str]] = {
    "6": ["G"],
    "N": ["W"],
    "J": ["W", "D"],
    "0": ["G"],
}


def _basic_fix(text: str) -> str:
    """
    Coarse correction: strip non-allowed characters, apply whole-string fixes,
    and try first-character substitutions to match a known team prefix.
    """
    text = re.sub(r"[^A-Za-z0-9._]", "", text)

    # Whole-string fix (e.g. DOUS -> DOU5)
    up = text.upper()
    for wrong, right in _TEAM_FIXES.items():
        if up.startswith(wrong):
            text = right + text[len(wrong):]
            break

    # If already matches a team prefix, return as-is
    if _team_of(text):
        return text

    # Try first-character substitution
    if text:
        for alt in _FIRST_CHAR_ALTS.get(text[0], []):
            cand = alt + text[1:]
            if _team_of(cand):
                return cand

    return text


# =====================================
# Main OCR function
# =====================================

def ocr_mvp_from_image(image_path: str) -> str | None:
    """
    Recognize the player ID from an MVP screenshot.

    Pipeline:
      1. Detect ticket region
      2. Tesseract OCR
      3. Basic character correction (e.g. DOUS -> DOU5)
      4. Fuzzy snap to KNOWN_PLAYERS

    Args:
        image_path: Path to the MVP image

    Returns:
        Recognized player ID (e.g. "ACT_zz9"), or None on failure.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  x Cannot read image: {image_path}")
        return None

    gray = _find_ticket_gray(img)
    if gray is None:
        print(f"  x No ticket region found: {image_path}")
        return None

    raw = _raw_ocr(gray)
    if not raw:
        print(f"  x OCR returned empty: {image_path}")
        return None

    fixed = _basic_fix(raw)

    if KNOWN_PLAYERS:
        snapped, score = _snap_within_team(fixed)
        if score >= _SNAP_THRESHOLD:
            print(f"  ok OCR raw=[{raw}] -> snap=[{snapped}] ({score:.2f})")
            return snapped
        else:
            print(f"  warn OCR raw=[{raw}] -> low snap score ({score:.2f}), returning basic fix")
            return fixed
    else:
        print(f"  ok OCR: {fixed}")
        return fixed


# =====================================
# Batch re-OCR (fix existing image folders)
# =====================================

def reocr_all(save_dir: str = "result_images", dry_run: bool = False):
    """
    Walk all mvp_*.png files under save_dir, re-recognize and rename them.

    Args:
        save_dir: Root directory (result_images)
        dry_run:  If True, only print what would happen without actually renaming
    """
    import os

    renamed, failed, skipped = 0, 0, 0

    for season in sorted(os.listdir(save_dir)):
        season_path = os.path.join(save_dir, season)
        if not os.path.isdir(season_path):
            continue
        for match_dir in sorted(os.listdir(season_path)):
            match_path = os.path.join(season_path, match_dir)
            if not os.path.isdir(match_path):
                continue
            for fname in os.listdir(match_path):
                if not (fname.startswith("mvp_") and fname.endswith(".png")):
                    continue
                old_path = os.path.join(match_path, fname)
                print(f"\n[{match_dir}] {fname}")
                mvp = ocr_mvp_from_image(old_path)
                if not mvp:
                    failed += 1
                    continue
                new_name = f"mvp_{mvp}.png"
                new_path = os.path.join(match_path, new_name)
                if old_path == new_path:
                    skipped += 1
                    print(f"  = no change")
                    continue
                if dry_run:
                    print(f"  -> (dry run) {fname} -> {new_name}")
                else:
                    os.rename(old_path, new_path)
                    print(f"  -> renamed: {fname} -> {new_name}")
                renamed += 1

    print(f"\nDone: renamed {renamed}, unchanged {skipped}, failed {failed}")


# =====================================
# CLI entry point
# =====================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("=== Preview mode (no actual renaming) ===")
        print("To actually rename: python ocr_local.py --apply\n")
        reocr_all(dry_run=True)
    elif sys.argv[1] == "--apply":
        reocr_all(dry_run=False)
    else:
        result = ocr_mvp_from_image(sys.argv[1])
        print(f"Result: {result}")
