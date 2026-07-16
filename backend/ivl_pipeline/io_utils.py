"""
io_utils.py — small load/save helpers so the scripts under scripts/ can
stay thin wrappers ("read files -> call functions -> write files").

Conventional directory layout (relative to the project root):
    data/raw/          the original IVL.xlsx (plus external inputs like the official schedule)
    data/interim/       the clean match table produced by 02_clean
    data/processed/      the feature matrix + current state snapshot produced by 04_feature
    data/models/          the model bundle produced by 05_train
    data/predictions/       the rank prediction output produced by 06_predict
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_joblib(obj, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    joblib.dump(obj, p)


def load_joblib(path: str | Path):
    return joblib.load(path)


def save_json(obj: dict, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False, encoding="utf-8-sig")  # utf-8-sig so non-ASCII text opens correctly in Excel