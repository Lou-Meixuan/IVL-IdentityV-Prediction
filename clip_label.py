"""
CLIP embedding + nearest-neighbor MVP labeling

Why this works better than pixel clustering:
  CLIP encodes images into a semantic embedding space.
  Two ticket images showing "ACT_zz9" will have very similar
  embeddings regardless of background texture or lighting.

=====================================
Setup
=====================================
    pip install torch torchvision transformers Pillow opencv-python scikit-learn pytesseract

=====================================
Usage
=====================================
    # Step 1: build embeddings, cluster, generate contact sheet + CSV
    python clip_label.py --build

    # Step 2: open cluster_sheet_clip.png, fix wrong names in cluster_labels_clip.csv

    # Step 3: apply
    python clip_label.py --apply            # preview
    python clip_label.py --apply --confirm  # actually rename
"""

import os
import cv2
import csv
import json
import re
import numpy as np
import argparse
import pandas as pd
from collections import defaultdict
from difflib import SequenceMatcher

# ============================
# Config
# ============================

SAVE_DIR       = "result_images"
OUTPUT_SHEET   = "cluster_sheet_clip.png"
OUTPUT_CSV     = "cluster_labels_clip.csv"
OUTPUT_MAPPING = "cluster_mapping_clip.json"

TICKET_W, TICKET_H = 480, 120
N_CLUSTERS = 40          # adjust if you have more/fewer unique players
CLIP_MODEL = "openai/clip-vit-base-patch32"

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
    region = img[int(h * 0.08):int(h * 0.42), int(w * 0.04):int(w * 0.70)]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 160), (180, 60, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = [c for c in contours
             if cv2.contourArea(c) > 5000
             and cv2.boundingRect(c)[2] / max(cv2.boundingRect(c)[3], 1) > 1.5]
    if not cands:
        return None
    c = max(cands, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(c)
    ticket = region[y:y+ch, x+int(cw*0.10):x+int(cw*0.90)]
    return cv2.resize(ticket, (TICKET_W, TICKET_H)) if ticket.size > 0 else None


# ============================
# CLIP embedding
# ============================

def load_clip():
    from transformers import CLIPProcessor, CLIPModel
    import torch
    print(f"Loading CLIP model ({CLIP_MODEL})...")
    model = CLIPModel.from_pretrained(CLIP_MODEL)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model.eval()
    return model, processor


def embed_tickets(tickets: list[np.ndarray], model, processor) -> np.ndarray:
    import torch
    from PIL import Image

    embeddings = []
    batch_size = 32

    for i in range(0, len(tickets), batch_size):
        batch = tickets[i:i + batch_size]
        pil_imgs = [Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)) for t in batch]
        inputs = processor(images=pil_imgs, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            # get_image_features returns a tensor directly in some versions,
            # but a ModelOutput object in others — handle both
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            elif hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state[:, 0]
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        embeddings.append(feats.cpu().numpy())
        print(f"  Embedded {min(i + batch_size, len(tickets))}/{len(tickets)}")

    return np.vstack(embeddings)


# ============================
# OCR suggestion
# ============================

def _team_of(text: str) -> str | None:
    up = text.upper()
    for t in TEAM_PREFIXES:
        if up.startswith(t.upper()):
            return t
    return None


def ocr_ticket(ticket: np.ndarray) -> str:
    try:
        import pytesseract
        big = cv2.resize(ticket, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
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


# ============================
# Collect MVP images
# ============================

def collect_paths(save_dir: str) -> list[str]:
    # Only collect unlabeled mvp.png files.
    # Already-labeled files (mvp_*.png) are skipped — they don't need re-clustering.
    paths = []
    for season in sorted(os.listdir(save_dir)):
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

def build(save_dir: str, n_clusters: int):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    paths = collect_paths(save_dir)
    print(f"Found {len(paths)} MVP images")

    # Crop tickets
    tickets, valid_paths = [], []
    for p in paths:
        t = get_ticket(p)
        if t is not None:
            tickets.append(t)
            valid_paths.append(p)
        else:
            print(f"  Warning: no ticket in {p}")

    # CLIP embeddings
    model, processor = load_clip()
    print(f"\nEmbedding {len(tickets)} ticket images with CLIP...")
    embeddings = embed_tickets(tickets, model, processor)
    print(f"Embeddings shape: {embeddings.shape}")

    # Reduce dims then cluster
    pca_dim = min(50, embeddings.shape[1])
    reduced = PCA(n_components=pca_dim, random_state=42).fit_transform(embeddings)

    print(f"\nClustering into {n_clusters} groups...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_ids = km.fit_predict(reduced)

    cluster_map: dict[int, list[int]] = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        cluster_map[int(cid)].append(i)

    # Sort by size
    ordered = sorted(cluster_map.keys(), key=lambda c: -len(cluster_map[c]))

    # OCR suggestions
    print("\nRunning OCR on cluster representatives...")
    suggestions: dict[int, str] = {}
    for cid in ordered:
        # Use the image closest to cluster centroid as rep
        idxs = cluster_map[cid]
        center = reduced[idxs].mean(axis=0)
        dists = [np.linalg.norm(reduced[i] - center) for i in idxs]
        best_idx = idxs[int(np.argmin(dists))]
        suggestions[cid] = ocr_ticket(tickets[best_idx])
        print(f"  Cluster {cid:3d} (n={len(idxs):3d}): {suggestions[cid]}")

    # Contact sheet
    COLS = 4
    LABEL_H = 40
    rows = (len(ordered) + COLS - 1) // COLS
    sheet = np.ones((rows * (TICKET_H + LABEL_H), COLS * TICKET_W, 3), dtype=np.uint8) * 230

    for idx, cid in enumerate(ordered):
        r, c = divmod(idx, COLS)
        idxs = cluster_map[cid]
        center = reduced[idxs].mean(axis=0)
        dists = [np.linalg.norm(reduced[i] - center) for i in idxs]
        rep = tickets[idxs[int(np.argmin(dists))]]
        y0, x0 = r * (TICKET_H + LABEL_H), c * TICKET_W
        sheet[y0:y0+TICKET_H, x0:x0+TICKET_W] = rep
        cv2.rectangle(sheet, (x0, y0), (x0+TICKET_W-1, y0+TICKET_H-1), (150, 150, 150), 1)
        label = f"#{cid} n={len(idxs)}  {suggestions.get(cid, '')}"
        cv2.putText(sheet, label, (x0+4, y0+TICKET_H+28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 180), 2)

    cv2.imwrite(OUTPUT_SHEET, sheet)
    print(f"\nContact sheet: {OUTPUT_SHEET}")

    # CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "n_images", "player_name"])
        for cid in ordered:
            w.writerow([cid, len(cluster_map[cid]), suggestions.get(cid, "")])
    print(f"CSV: {OUTPUT_CSV}  (fix wrong names)")

    # Mapping
    mapping = {str(cid): [valid_paths[i] for i in cluster_map[cid]] for cid in cluster_map}
    with open(OUTPUT_MAPPING, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Mapping: {OUTPUT_MAPPING}")

    print("\nNext: fix names in cluster_labels_clip.csv, then:")
    print("  python clip_label.py --apply --confirm")


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
            print(f"  Cluster {cid}: no label, skipping {len(file_paths)} file(s)")
            skipped += len(file_paths)
            continue
        safe = player.replace("/", "_").replace("\\", "_")
        for old_path in file_paths:
            new_path = os.path.join(os.path.dirname(old_path),
                                    f"mvp_{safe}{os.path.splitext(old_path)[1]}")
            if old_path == new_path:
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

    # After confirming renames, write correct MVP names back to any CSV in cwd
    if confirm:
        _update_csv_mvp(SAVE_DIR)


# ============================
# Update CSV with MVP names from renamed files
# ============================

def _update_csv_mvp(save_dir: str):
    """
    Scan all match folders in save_dir, read MVP name from mvp_*.png filename,
    then update the mvp column in any CSV files in the current directory
    that have match_id and mvp columns.
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
            # Extract match_id: last long numeric token in folder name
            m = re.search(r'(\d{15,})', match_dir)
            if not m:
                continue
            match_id = m.group(1)
            # Find mvp_*.png (skip plain mvp.png — not yet labeled)
            for fname in os.listdir(mp):
                if fname.startswith("mvp_") and fname.endswith(".png"):
                    mvp_name = fname[4:-4]  # strip "mvp_" prefix and ".png"
                    mvp_map[match_id] = mvp_name
                    break

    if not mvp_map:
        print("  No labeled MVP files found, skipping CSV update.")
        return

    # Update all CSV files in csv_output/ that have match_id + mvp columns
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
# Nearest-neighbor inference (for new images after gallery is built)
# ============================

def predict_single(image_path: str, gallery_csv: str, gallery_mapping: str):
    """
    Given a new MVP image, find the most similar labeled cluster.
    Useful after --build + labeling for classifying new incoming images.
    """
    import torch
    from PIL import Image

    ticket = get_ticket(image_path)
    if ticket is None:
        print("No ticket region found.")
        return None

    model, processor = load_clip()
    pil = Image.fromarray(cv2.cvtColor(ticket, cv2.COLOR_BGR2RGB))
    inputs = processor(images=[pil], return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    emb = emb.cpu().numpy()[0]

    # Load gallery embeddings (re-embed on-the-fly for simplicity)
    labels: dict[str, str] = {}
    with open(gallery_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["player_name"].strip()
            if name:
                labels[row["cluster_id"]] = name

    with open(gallery_mapping, encoding="utf-8") as f:
        mapping: dict[str, list[str]] = json.load(f)

    best_cid, best_sim = None, -1.0
    for cid, file_paths in mapping.items():
        if cid not in labels:
            continue
        for fp in file_paths[:3]:   # use up to 3 examples per cluster
            t = get_ticket(fp)
            if t is None:
                continue
            pil2 = Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
            inp2 = processor(images=[pil2], return_tensors="pt")
            with torch.no_grad():
                e2 = model.get_image_features(**inp2)
                e2 = e2 / e2.norm(dim=-1, keepdim=True)
            sim = float((emb * e2.cpu().numpy()[0]).sum())
            if sim > best_sim:
                best_sim = sim
                best_cid = cid

    result = labels.get(best_cid, "unknown")
    print(f"Prediction: {result}  (similarity={best_sim:.3f}, cluster={best_cid})")
    return result


# ============================
# CLI
# ============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP-based MVP image labeling")
    parser.add_argument("--build",      action="store_true",
                        help="Embed all images with CLIP, cluster, generate sheet + CSV")
    parser.add_argument("--apply",      action="store_true",
                        help="Rename files based on filled CSV")
    parser.add_argument("--confirm",    action="store_true",
                        help="Used with --apply: actually rename")
    parser.add_argument("--predict",    type=str, default=None,
                        help="Path to a single new MVP image to classify")
    parser.add_argument("--n-clusters", type=int, default=N_CLUSTERS,
                        help=f"Number of clusters (default: {N_CLUSTERS})")
    args = parser.parse_args()

    if args.build:
        build(SAVE_DIR, args.n_clusters)
    elif args.apply:
        apply_labels(confirm=args.confirm)
    elif args.predict:
        predict_single(args.predict, OUTPUT_CSV, OUTPUT_MAPPING)
    else:
        parser.print_help()
