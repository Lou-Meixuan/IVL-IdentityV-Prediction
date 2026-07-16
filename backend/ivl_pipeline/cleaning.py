"""
cleaning.py — data cleaning: splitting composite fields / team name
normalization / merging multiple seasons.

Corresponds to steps 01_load_data + 02_clean in the overall architecture.
This chat's focus is 04_feature -> 05_train -> 06_predict, but feature
engineering needs a clean match-level table to work from, so this module
adds a lightweight load+clean layer that produces the unified matches table
used by the later steps.
"""
from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Team name normalization: the same team can appear with different
# casing/formatting across sheets and seasons (e.g. Gr / GR are the same
# team; FPX.ZQ's dot needs to be kept but casing needs to be unified).
# ---------------------------------------------------------------------------
_TEAM_ALIAS = {
    "gr": "GR",
    "fpx.zq": "FPX.ZQ",
}


def normalize_team_name(name: str) -> str:
    """Normalize a team name's casing so the same team isn't counted as two."""
    if pd.isna(name):
        return name
    key = str(name).strip().lower()
    return _TEAM_ALIAS.get(key, str(name).strip())


# ---------------------------------------------------------------------------
# Composite field parsing
# ---------------------------------------------------------------------------
def parse_score_pair(value: str) -> tuple[float, float]:
    """Parse an 'a-b' / 'a:b' style string into a (a, b) numeric tuple.
    Returns (nan, nan) if parsing fails."""
    if pd.isna(value):
        return (np.nan, np.nan)
    m = re.match(r"\s*(\d+)\s*[-:]\s*(\d+)\s*$", str(value))
    if not m:
        return (np.nan, np.nan)
    return (float(m.group(1)), float(m.group(2)))


def parse_game_scores(value: str) -> list[tuple[float, float]]:
    """Parse '4:4 | 5:3 | 3:5' into a list of per-game (a, b) tuples."""
    if pd.isna(value):
        return []
    games = []
    for part in str(value).split("|"):
        a, b = parse_score_pair(part)
        if not (np.isnan(a) or np.isnan(b)):
            games.append((a, b))
    return games


def enrich_match_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed numeric columns to df, returning a new copy (original untouched)."""
    df = df.copy()

    ms = df["match_score"].apply(parse_score_pair)
    df["match_score_a"] = ms.apply(lambda t: t[0])
    df["match_score_b"] = ms.apply(lambda t: t[1])

    ts = df["total_small_score"].apply(parse_score_pair)
    df["small_score_a"] = ts.apply(lambda t: t[0])
    df["small_score_b"] = ts.apply(lambda t: t[1])
    df["small_score_diff"] = df["small_score_a"] - df["small_score_b"]

    df["n_games"] = df["game_scores"].apply(lambda v: len(parse_game_scores(v)))

    return df


def fill_missing_winner(df: pd.DataFrame) -> pd.DataFrame:
    """When winner is missing, infer it from match_score_a/b (higher score wins)."""
    df = df.copy()
    missing = df["winner"].isna()
    if missing.any():
        a_wins = df.loc[missing, "match_score_a"] > df.loc[missing, "match_score_b"]
        df.loc[missing & a_wins.reindex(df.index, fill_value=False), "winner"] = df["team_a"]
        b_wins = df.loc[missing, "match_score_b"] > df.loc[missing, "match_score_a"]
        df.loc[missing & b_wins.reindex(df.index, fill_value=False), "winner"] = df["team_b"]
    return df


# ---------------------------------------------------------------------------
# Season/stage parsing, used to sort matches chronologically (prevents
# "future data leakage" during feature engineering).
# ---------------------------------------------------------------------------
_SEASON_ORDER = {
    "2023_Summer": 1,
    "2023_Autumn": 2,
    "2023_Autumn_Finals": 3,
    "2024_Summer": 4,
    "2024_Autumn": 5,
    "2025_Summer": 6,
    "2025_Summer_Finals": 7,
    "2025_Autumn": 8,
    "2025_Autumn_Finals": 9,
    "2026_Summer": 10,
}


def season_sort_key(season: str) -> int:
    """Give a season a comparable integer order; unknown season names sort last."""
    return _SEASON_ORDER.get(season, 999)


def load_raw_sheets(xlsx_path: str) -> dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(xlsx_path)
    return {name: xl.parse(name) for name in xl.sheet_names}


def _clean_common(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["team_a"] = df["team_a"].apply(normalize_team_name)
    df["team_b"] = df["team_b"].apply(normalize_team_name)
    df["winner"] = df["winner"].apply(normalize_team_name)
    df = enrich_match_fields(df)
    df = fill_missing_winner(df)
    df["winner_missing"] = df["winner"].isna()
    return df


def build_league_matches(xlsx_path: str) -> pd.DataFrame:
    """
    Only processes match_overview (domestic league regular season + regular
    playoffs, 597 matches).

    This is the primary timeline for feature engineering / Elo / training /
    season simulation. International matches (COA series) are NOT merged
    into this timeline — because the real interleaving order between them
    and the league schedule is unknown, and forcing them together would
    corrupt Elo's chronological updates. They're handled separately via
    build_international_matches() and only used as an out-of-sample
    validation set.

    Adds columns: season_order (cross-season ordering), match_order (global
    chronological index — feature engineering strictly follows this order
    for expanding/rolling calculations to prevent using future data).
    """
    sheets = load_raw_sheets(xlsx_path)
    league = sheets["match_overview"].copy()
    league["season_stage"] = "regular"
    league["is_international"] = False

    league = _clean_common(league)

    league["season_order"] = league["season"].apply(season_sort_key)
    # Within a season, use the original row order as an approximation of
    # chronological order (the data source has no explicit timestamp).
    league = league.sort_values(["season_order"], kind="stable").reset_index(drop=True)

    before = len(league)
    league = league.drop_duplicates(subset=["match_id", "team_a", "team_b"]).reset_index(drop=True)
    league.attrs["n_duplicates_dropped"] = before - len(league)

    league["match_order"] = np.arange(len(league))
    return league


def build_international_matches(xlsx_path: str) -> pd.DataFrame:
    """
    Processes the four COA_global_finals events (52 matches each, 208
    total), used only for cross-season/cross-event model validation
    (out-of-sample evaluation) — not part of the training feature timeline.

    Within each event, the original row order (match_id like COA6_001,
    002...) is treated as chronological order.
    """
    sheets = load_raw_sheets(xlsx_path)
    frames = []
    for tag in ["COA6", "COA7", "COA8", "COA9"]:
        key = f"{tag}_global_finals"
        if key not in sheets:
            continue
        f = sheets[key].copy()
        f["event"] = tag
        f["is_international"] = True
        f["event_order"] = np.arange(len(f))
        frames.append(f)

    intl = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    intl = _clean_common(intl)
    return intl