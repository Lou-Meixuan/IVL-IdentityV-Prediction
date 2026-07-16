"""
standings.py — aggregates per-match results into a per-team standings table
(wins/losses/win rate/score margin/rank).

Two uses:
1. Historical season review / validation: the standings table required by
   key decision #2.
2. Starting point for Monte Carlo simulation: if a season is halfway
   through, compute the standings from the "already played" portion first,
   then accumulate the simulated remaining matches on top of that instead
   of simulating the whole season from scratch.
"""
from __future__ import annotations

import pandas as pd


def compute_standings(matches: pd.DataFrame, season: str | None = None) -> pd.DataFrame:
    """
    matches needs team_a, team_b, winner, small_score_a, small_score_b
    columns (the output of build_league_matches can be used directly).
    When season is given, only that season's matches are counted.

    Returns a standard standings table sorted by (wins desc, score margin
    desc), with a rank column.
    """
    df = matches if season is None else matches[matches["season"] == season]

    rows = {}

    def _get(team):
        if team not in rows:
            rows[team] = {
                "team": team,
                "matches_played": 0,
                "wins": 0,
                "losses": 0,
                "small_score_for": 0.0,
                "small_score_against": 0.0,
            }
        return rows[team]

    for _, m in df.iterrows():
        a, b, winner = m["team_a"], m["team_b"], m["winner"]
        sa, sb = m.get("small_score_a", 0.0), m.get("small_score_b", 0.0)
        sa = 0.0 if pd.isna(sa) else sa
        sb = 0.0 if pd.isna(sb) else sb

        ra, rb = _get(a), _get(b)
        ra["matches_played"] += 1
        rb["matches_played"] += 1
        ra["small_score_for"] += sa
        ra["small_score_against"] += sb
        rb["small_score_for"] += sb
        rb["small_score_against"] += sa

        if winner == a:
            ra["wins"] += 1
            rb["losses"] += 1
        elif winner == b:
            rb["wins"] += 1
            ra["losses"] += 1
        # Matches where winner is still missing (a handful of rows the raw
        # data couldn't be parsed for) aren't counted toward wins/losses,
        # only toward matches played and score totals.

    standings = pd.DataFrame(rows.values())
    if standings.empty:
        return standings

    standings["win_rate"] = standings["wins"] / standings["matches_played"].replace(0, pd.NA)
    standings["net_margin"] = standings["small_score_for"] - standings["small_score_against"]
    standings["avg_net_margin"] = standings["net_margin"] / standings["matches_played"].replace(0, pd.NA)

    standings = standings.sort_values(
        ["wins", "net_margin"], ascending=[False, False]
    ).reset_index(drop=True)
    standings["rank"] = standings.index + 1

    cols = [
        "rank", "team", "matches_played", "wins", "losses", "win_rate",
        "small_score_for", "small_score_against", "net_margin", "avg_net_margin",
    ]
    return standings[cols]