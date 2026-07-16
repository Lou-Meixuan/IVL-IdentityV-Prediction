"""
features.py — feature engineering: builds single-match prediction features
using only information known "before the match".

Core anti-leakage principle: any statistic (win rate, score margin, H2H)
used for match i's features may only use matches with match_order < i,
implemented via groupby + shift(1)/expanding — this match's own result
(winner / total_small_score, etc.) never leaks into its own features.

The output X consists of "difference features" (team_a's perspective minus
team_b's), and each match is mirror-augmented (a second sample is generated
with team_a/team_b swapped, features negated, and the label flipped) — this
both prevents the model from learning a spurious "team_a slot" bias and
effectively doubles the training set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .elo import EloRatingSystem

ROLLING_WINDOWS = (5, 10)


def _long_format(league_matches: pd.DataFrame) -> pd.DataFrame:
    """Splits each match into two "team perspective" records, to make
    per-team rolling stats easier to compute."""
    a = league_matches[["match_order", "season", "team_a", "team_b", "winner", "small_score_diff"]].copy()
    a = a.rename(columns={"team_a": "team", "team_b": "opponent"})
    a["is_win"] = (a["winner"] == a["team"]).astype(int)
    a["margin"] = a["small_score_diff"]

    b = league_matches[["match_order", "season", "team_a", "team_b", "winner", "small_score_diff"]].copy()
    b = b.rename(columns={"team_b": "team", "team_a": "opponent"})
    b["is_win"] = (b["winner"] == b["team"]).astype(int)
    b["margin"] = -b["small_score_diff"]

    long_df = pd.concat([a, b], ignore_index=True).sort_values(["team", "match_order"]).reset_index(drop=True)
    return long_df


def _team_rolling_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """Groups by team and computes rolling/cumulative stats as of "the
    previous match" (shift(1) guarantees the current match is excluded)."""
    long_df = long_df.sort_values(["team", "match_order"]).copy()
    g = long_df.groupby("team", group_keys=False)

    long_df["games_played"] = g.cumcount()  # matches played before this one (0-based, naturally excludes current match)
    long_df["winrate_expanding"] = g["is_win"].apply(lambda s: s.shift(1).expanding().mean())
    long_df["margin_expanding"] = g["margin"].apply(lambda s: s.shift(1).expanding().mean())

    for w in ROLLING_WINDOWS:
        long_df[f"winrate_last{w}"] = g["is_win"].apply(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())
        long_df[f"margin_last{w}"] = g["margin"].apply(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())

    # Fill with neutral values (50% win rate / 0 margin) when there's no
    # history yet, to avoid NaNs reaching the model.
    long_df["winrate_expanding"] = long_df["winrate_expanding"].fillna(0.5)
    long_df["margin_expanding"] = long_df["margin_expanding"].fillna(0.0)
    for w in ROLLING_WINDOWS:
        long_df[f"winrate_last{w}"] = long_df[f"winrate_last{w}"].fillna(0.5)
        long_df[f"margin_last{w}"] = long_df[f"margin_last{w}"].fillna(0.0)

    return long_df


def _h2h_stats(league_matches: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, computes team_a's historical head-to-head win rate
    against team_b (strictly using matches earlier in match_order).
    Returns columns: match_order, h2h_winrate_a, h2h_matches
    """
    df = league_matches.sort_values("match_order").reset_index(drop=True)
    records = []
    history: dict[tuple[str, str], list[int]] = {}  # (teamX, teamY) sorted pair -> list of winner flags (1 if teamX won)

    for _, row in df.iterrows():
        a, b = row["team_a"], row["team_b"]
        pair = tuple(sorted([a, b]))
        low, high = pair
        prior = history.get(pair, [])
        if prior:
            low_winrate = float(np.mean(prior))
        else:
            low_winrate = 0.5
        # Convert to team_a's perspective
        if a == low:
            h2h_winrate_a = low_winrate
        else:
            h2h_winrate_a = 1 - low_winrate

        records.append(
            {
                "match_order": row["match_order"],
                "h2h_winrate_a": h2h_winrate_a,
                "h2h_matches": len(prior),
            }
        )

        # Update history with this match's result (used by later matches, not this one)
        winner = row["winner"]
        history.setdefault(pair, [])
        history[pair].append(1 if winner == low else 0)

    return pd.DataFrame(records)


FEATURE_COLUMNS = [
    "elo_diff",
    "winrate_exp_diff",
    "winrate_last5_diff",
    "winrate_last10_diff",
    "margin_exp_diff",
    "margin_last5_diff",
    "margin_last10_diff",
    "games_played_diff",
    "h2h_winrate_a",
    "h2h_matches",
]


def build_feature_matrix(
    league_matches: pd.DataFrame,
    elo_system: EloRatingSystem | None = None,
    symmetric_augment: bool = True,
) -> tuple[pd.DataFrame, EloRatingSystem]:
    """
    Input: the output of build_league_matches() (a clean match table).
    Output: (feature_df, fitted_elo_system)
        feature_df has one "prediction sample" per row, containing
        FEATURE_COLUMNS + y_win + y_margin + metadata columns (match_order,
        season, team_a, team_b — the mirrored rows have team_a/team_b
        swapped).
        fitted_elo_system is the Elo state after rolling through the full
        history; season simulation continues rolling forward from this
        state.
    """
    if elo_system is None:
        elo_system = EloRatingSystem()

    with_elo = elo_system.fit_transform(league_matches)

    long_df = _long_format(with_elo)
    long_df = _team_rolling_stats(long_df)

    h2h = _h2h_stats(with_elo)

    # Merge the per-team rolling stats back onto both the team_a and team_b sides
    stat_cols = ["games_played", "winrate_expanding", "margin_expanding"] + [
        f"winrate_last{w}" for w in ROLLING_WINDOWS
    ] + [f"margin_last{w}" for w in ROLLING_WINDOWS]

    a_stats = long_df.merge(
        with_elo[["match_order", "team_a"]], left_on=["match_order", "team"], right_on=["match_order", "team_a"]
    )[["match_order"] + stat_cols].add_suffix("_a").rename(columns={"match_order_a": "match_order"})

    b_stats = long_df.merge(
        with_elo[["match_order", "team_b"]], left_on=["match_order", "team"], right_on=["match_order", "team_b"]
    )[["match_order"] + stat_cols].add_suffix("_b").rename(columns={"match_order_b": "match_order"})

    feat = with_elo.merge(a_stats, on="match_order").merge(b_stats, on="match_order").merge(h2h, on="match_order")

    feat["winrate_exp_diff"] = feat["winrate_expanding_a"] - feat["winrate_expanding_b"]
    feat["margin_exp_diff"] = feat["margin_expanding_a"] - feat["margin_expanding_b"]
    feat["games_played_diff"] = feat["games_played_a"] - feat["games_played_b"]
    for w in ROLLING_WINDOWS:
        feat[f"winrate_last{w}_diff"] = feat[f"winrate_last{w}_a"] - feat[f"winrate_last{w}_b"]
        feat[f"margin_last{w}_diff"] = feat[f"margin_last{w}_a"] - feat[f"margin_last{w}_b"]
    feat["elo_diff"] = feat["elo_diff_pre"]

    feat["y_win"] = (feat["winner"] == feat["team_a"]).astype(int)
    feat["y_margin"] = feat["small_score_diff"]

    keep = ["match_order", "match_id", "season", "team_a", "team_b"] + FEATURE_COLUMNS + ["y_win", "y_margin"]
    feat = feat[keep].reset_index(drop=True)

    if not symmetric_augment:
        return feat, elo_system

    mirror = feat.copy()
    mirror["team_a"], mirror["team_b"] = feat["team_b"], feat["team_a"]
    for col in ["elo_diff", "winrate_exp_diff", "winrate_last5_diff", "winrate_last10_diff",
                "margin_exp_diff", "margin_last5_diff", "margin_last10_diff", "games_played_diff"]:
        mirror[col] = -feat[col]
    mirror["h2h_winrate_a"] = 1 - feat["h2h_winrate_a"]
    mirror["y_win"] = 1 - feat["y_win"]
    mirror["y_margin"] = -feat["y_margin"]
    mirror["_mirrored"] = True
    feat["_mirrored"] = False

    augmented = pd.concat([feat, mirror], ignore_index=True).sort_values(
        ["match_order", "_mirrored"]
    ).reset_index(drop=True)
    return augmented, elo_system


def latest_team_form(league_matches: pd.DataFrame, max_window: int = max(ROLLING_WINDOWS)) -> dict[str, dict]:
    """
    Each team's current state snapshot as of "the last match played"
    (inclusive — this is a real, already-happened match, not a future match
    to be predicted). Used as the starting point for season Monte Carlo
    simulation.

    Returns dict[team] = {
        games_played, sum_wins, winrate_expanding, sum_margin, margin_expanding,
        recent_wins: [...], recent_margins: [...]   # last up to max_window matches, for rolling-window metrics
    }
    """
    long_df = _long_format(league_matches).sort_values(["team", "match_order"])
    form = {}
    for team, g in long_df.groupby("team"):
        wins = g["is_win"].tolist()
        margins = g["margin"].tolist()
        n = len(wins)
        form[team] = {
            "games_played": n,
            "sum_wins": int(sum(wins)),
            "winrate_expanding": (sum(wins) / n) if n else 0.5,
            "sum_margin": float(sum(margins)),
            "margin_expanding": (sum(margins) / n) if n else 0.0,
            "recent_wins": wins[-max_window:],
            "recent_margins": margins[-max_window:],
        }
    return form


def team_form_to_feature_dict(form_entry: dict) -> dict:
    """Converts a latest_team_form() snapshot into the fields expected by
    build_prematch_features_for_pair()."""
    wins = form_entry.get("recent_wins", [])
    margins = form_entry.get("recent_margins", [])
    out = {
        "games_played": form_entry.get("games_played", 0),
        "winrate_expanding": form_entry.get("winrate_expanding", 0.5),
        "margin_expanding": form_entry.get("margin_expanding", 0.0),
    }
    for w in ROLLING_WINDOWS:
        recent_w = wins[-w:]
        recent_m = margins[-w:]
        out[f"winrate_last{w}"] = (sum(recent_w) / len(recent_w)) if recent_w else 0.5
        out[f"margin_last{w}"] = (sum(recent_m) / len(recent_m)) if recent_m else 0.0
    return out


def update_team_form(form_entry: dict, win: bool, margin: float, max_window: int = max(ROLLING_WINDOWS)) -> dict:
    """During simulation, incrementally updates a team's state snapshot
    after one match ends (no need to recompute the whole history)."""
    n = form_entry.get("games_played", 0)
    sum_wins = form_entry.get("sum_wins", 0) + int(win)
    sum_margin = form_entry.get("sum_margin", 0.0) + margin
    n_new = n + 1
    recent_wins = (form_entry.get("recent_wins", []) + [int(win)])[-max_window:]
    recent_margins = (form_entry.get("recent_margins", []) + [margin])[-max_window:]
    return {
        "games_played": n_new,
        "sum_wins": sum_wins,
        "winrate_expanding": sum_wins / n_new,
        "sum_margin": sum_margin,
        "margin_expanding": sum_margin / n_new,
        "recent_wins": recent_wins,
        "recent_margins": recent_margins,
    }


def latest_h2h_lookup(league_matches: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """Full-history H2H record as of the last match, keyed by sorted(team pair)."""
    df = league_matches.sort_values("match_order")
    history: dict[tuple[str, str], list[int]] = {}
    for _, row in df.iterrows():
        a, b, winner = row["team_a"], row["team_b"], row["winner"]
        pair = tuple(sorted([a, b]))
        low = pair[0]
        history.setdefault(pair, [])
        history[pair].append(1 if winner == low else 0)
    return {pair: {"low_winrate": sum(v) / len(v), "matches": len(v)} for pair, v in history.items()}


def build_prematch_features_for_pair(
    team_a: str,
    team_b: str,
    elo_system: EloRatingSystem,
    team_form: dict[str, dict],
    h2h_lookup: dict[tuple[str, str], dict],
) -> pd.DataFrame:
    """
    For season simulation use: given the current Elo state + each team's
    latest rolling state snapshot + H2H records, assembles the feature row
    for a single match (doesn't depend on a DataFrame timeline — used
    inside the simulation loop, called repeatedly).

    team_form[team] needs to contain the "that team's perspective" stats
    used by FEATURE_COLUMNS aside from elo_diff/h2h_*: games_played,
    winrate_expanding, winrate_last5, winrate_last10, margin_expanding,
    margin_last5, margin_last10.
    h2h_lookup[(sorted pair)] contains {"low_winrate":..., "matches": ...}.
    """
    elo_a = elo_system.rating_of(team_a)
    elo_b = elo_system.rating_of(team_b)

    fa = team_form.get(team_a, {})
    fb = team_form.get(team_b, {})

    def g(d, k, default):
        return d.get(k, default)

    pair = tuple(sorted([team_a, team_b]))
    h2h = h2h_lookup.get(pair, {"low_winrate": 0.5, "matches": 0})
    h2h_winrate_a = h2h["low_winrate"] if team_a == pair[0] else 1 - h2h["low_winrate"]

    row = {
        "elo_diff": elo_a - elo_b,
        "winrate_exp_diff": g(fa, "winrate_expanding", 0.5) - g(fb, "winrate_expanding", 0.5),
        "winrate_last5_diff": g(fa, "winrate_last5", 0.5) - g(fb, "winrate_last5", 0.5),
        "winrate_last10_diff": g(fa, "winrate_last10", 0.5) - g(fb, "winrate_last10", 0.5),
        "margin_exp_diff": g(fa, "margin_expanding", 0.0) - g(fb, "margin_expanding", 0.0),
        "margin_last5_diff": g(fa, "margin_last5", 0.0) - g(fb, "margin_last5", 0.0),
        "margin_last10_diff": g(fa, "margin_last10", 0.0) - g(fb, "margin_last10", 0.0),
        "games_played_diff": g(fa, "games_played", 0) - g(fb, "games_played", 0),
        "h2h_winrate_a": h2h_winrate_a,
        "h2h_matches": h2h["matches"],
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]