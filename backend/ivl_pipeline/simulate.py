"""
simulate.py — Monte Carlo season simulation: given the standings from the
"already played" portion of a season plus the remaining schedule, repeatedly
simulates the outcomes of the remaining matches and accumulates each team's
final rank distribution.

simulate_season() only cares about a `fixtures` list, and doesn't care
whether it's a real schedule or an inferred one. Two ways to obtain fixtures:
    1. remaining_fixtures_from_schedule()  —— use this when an official schedule is available (recommended, default path in this file)
    2. infer_remaining_fixtures()          —— fallback when no official schedule exists, based on a "double round-robin" assumption
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from .cleaning import normalize_team_name
from .elo import EloRatingSystem
from .features import FEATURE_COLUMNS, team_form_to_feature_dict, update_team_form
from .models import MatchPredictor


def infer_remaining_fixtures(
    league_matches: pd.DataFrame, season: str, matches_per_pair: int = 2
) -> list[tuple[str, str]]:
    """
    Fallback used when there's no official schedule: assumes each pair of
    teams plays matches_per_pair matches this season (default: one double
    round-robin meeting each), and works out how many matches are still
    missing per pair from what's already been played.

    Prefer remaining_fixtures_from_schedule() over this whenever an
    official schedule is available.
    """
    season_matches = league_matches[league_matches["season"] == season]
    teams = sorted(set(season_matches["team_a"]) | set(season_matches["team_b"]))

    played_count: dict[tuple[str, str], int] = {}
    for _, row in season_matches.iterrows():
        pair = tuple(sorted([row["team_a"], row["team_b"]]))
        played_count[pair] = played_count.get(pair, 0) + 1

    fixtures = []
    for t1, t2 in combinations(teams, 2):
        pair = tuple(sorted([t1, t2]))
        remaining = matches_per_pair - played_count.get(pair, 0)
        for i in range(max(0, remaining)):
            if i % 2 == 0:
                fixtures.append((t1, t2))
            else:
                fixtures.append((t2, t1))
    return fixtures


def load_official_schedule(schedule_csv_path: str) -> pd.DataFrame:
    """
    Reads the official schedule CSV. Required columns: week, date, team_a,
    team_b (team names don't need to be pre-normalized, this handles it).
    data/raw/2026_Summer_schedule.csv was manually transcribed from the
    schedule graphic released by the event organizer (WEEK1-9 regular
    season, 90 matches, double round-robin — internal consistency verified
    by checking each team appears exactly 18 times). WEEK10 is a seeded
    play-in bracket based on final regular season rank ("5th place vs 6th
    place" and so on), not fixed teams, so it isn't included in this table
    and needs to be added once the regular season finishes.
    """
    df = pd.read_csv(schedule_csv_path)
    df["team_a"] = df["team_a"].apply(normalize_team_name)
    df["team_b"] = df["team_b"].apply(normalize_team_name)
    return df


def remaining_fixtures_from_schedule(
    schedule: pd.DataFrame, league_matches: pd.DataFrame, season: str
) -> list[tuple[str, str]]:
    """
    Uses the official schedule plus the real recorded results to work out
    which matches haven't been played yet.

    Approach: treat this as a "multiset difference" over unordered team
    pairs — for however many times two teams are scheduled to meet vs how
    many times they've actually played, walk through the schedule in
    chronological order and mark off "already played" occurrences as
    consumed; whatever's left unconsumed is what's still to be played
    (this keeps things close to the real chronology). This is more accurate
    than the double round-robin guess, since the pairings come from the
    real official schedule instead of an assumption.
    """
    sched = schedule.copy()
    if "week" in sched.columns:
        sort_cols = [c for c in ["week", "date"] if c in sched.columns]
        sched = sched.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    played = league_matches[league_matches["season"] == season]
    played_count: dict[tuple[str, str], int] = {}
    for _, row in played.iterrows():
        pair = tuple(sorted([row["team_a"], row["team_b"]]))
        played_count[pair] = played_count.get(pair, 0) + 1

    consumed: dict[tuple[str, str], int] = {}
    remaining: list[tuple[str, str]] = []
    for _, row in sched.iterrows():
        pair = tuple(sorted([row["team_a"], row["team_b"]]))
        used = consumed.get(pair, 0)
        if used < played_count.get(pair, 0):
            consumed[pair] = used + 1
            continue  # this match has already happened in the real data, skip it
        remaining.append((row["team_a"], row["team_b"]))

    return remaining


def standings_to_state(standings_df: pd.DataFrame) -> dict[str, dict]:
    """Converts standings.compute_standings()'s output into the starting state dict used by simulation."""
    state = {}
    for _, row in standings_df.iterrows():
        state[row["team"]] = {
            "wins": int(row["wins"]),
            "losses": int(row["losses"]),
            "net_margin": float(row["net_margin"]),
        }
    return state


@dataclass
class SimulationResult:
    rank_table: pd.DataFrame  # team x summary stats (expected_rank, champion_prob, ...)
    rank_prob_matrix: pd.DataFrame  # team x rank -> probability
    raw_final_wins: pd.DataFrame  # n_sims x n_teams, final win count per simulation (for debugging/auditing)
    n_sims: int


def simulate_season(
    fixtures: list[tuple[str, str]],
    model: MatchPredictor,
    elo_system: EloRatingSystem,
    team_form: dict[str, dict],
    h2h_lookup: dict[tuple[str, str], dict],
    current_standings: dict[str, dict] | None = None,
    margin_noise_std: float = 6.0,
    n_sims: int = 5000,
    top_n_for_prob: int = 4,
    random_state: int | None = 42,
) -> SimulationResult:
    """
    fixtures: [(team_a, team_b), ...] the remaining schedule, in match order
              (order affects how Elo/form dynamically evolve — use the real
              schedule's chronological order).
    model:    the trained MatchPredictor produced by 05_train.py (a win
              classifier + a margin regressor).
    elo_system / team_form / h2h_lookup: the "current state" snapshot
              produced by 04_feature.py (an EloRatingSystem instance,
              latest_team_form(), latest_h2h_lookup()).
    current_standings: the starting point from already-played matches
              (the output of standings_to_state()); when None, every team
              starts at 0 wins / 0 losses (e.g. simulating a brand new
              season).
    margin_noise_std: the standard deviation of the Gaussian noise added on
              top of the predicted score margin during simulation (models
              the randomness of a single match). Recommended to set this
              from the margin regressor's residual std on the
              training/validation split — 05_train.py computes this and
              stores it in the model bundle (bundle["margin_noise_std"]).

    Returns each team's rank distribution statistics.
    """
    all_teams = sorted(set([t for f in fixtures for t in f]) | set((current_standings or {}).keys()))
    rng = np.random.default_rng(random_state)

    base_standings = current_standings or {t: {"wins": 0, "losses": 0, "net_margin": 0.0} for t in all_teams}

    # Each simulation run keeps its own state, but the loop order is
    # "fixture outer / simulation inner" so that the same fixture across all
    # n_sims runs can be batched into a single matrix and predicted in one
    # model call, instead of calling the model once per simulation per
    # fixture — the latter is too slow for thousands of simulations given
    # sklearn's per-call overhead. State updates themselves are plain
    # Python/numpy arithmetic (no model calls), so they're still done per
    # simulation and stay cheap.
    sim_elo = [copy.deepcopy(elo_system) for _ in range(n_sims)]
    sim_form = [
        {t: dict(v) for t, v in team_form.items()} for _ in range(n_sims)
    ]
    for s in range(n_sims):
        for t in all_teams:
            sim_form[s].setdefault(t, {"games_played": 0, "sum_wins": 0, "winrate_expanding": 0.5,
                                        "sum_margin": 0.0, "margin_expanding": 0.0,
                                        "recent_wins": [], "recent_margins": []})
    sim_h2h = [{k: dict(v) for k, v in h2h_lookup.items()} for _ in range(n_sims)]
    sim_standings = [
        {t: dict(base_standings.get(t, {"wins": 0, "losses": 0, "net_margin": 0.0})) for t in all_teams}
        for _ in range(n_sims)
    ]

    for team_a, team_b in fixtures:
        rows = []
        for s in range(n_sims):
            fa = team_form_to_feature_dict(sim_form[s][team_a])
            fb = team_form_to_feature_dict(sim_form[s][team_b])
            elo_a = sim_elo[s].rating_of(team_a)
            elo_b = sim_elo[s].rating_of(team_b)
            pair = tuple(sorted([team_a, team_b]))
            h = sim_h2h[s].get(pair, {"low_winrate": 0.5, "matches": 0})
            h2h_winrate_a = h["low_winrate"] if team_a == pair[0] else 1 - h["low_winrate"]
            rows.append(
                {
                    "elo_diff": elo_a - elo_b,
                    "winrate_exp_diff": fa["winrate_expanding"] - fb["winrate_expanding"],
                    "winrate_last5_diff": fa["winrate_last5"] - fb["winrate_last5"],
                    "winrate_last10_diff": fa["winrate_last10"] - fb["winrate_last10"],
                    "margin_exp_diff": fa["margin_expanding"] - fb["margin_expanding"],
                    "margin_last5_diff": fa["margin_last5"] - fb["margin_last5"],
                    "margin_last10_diff": fa["margin_last10"] - fb["margin_last10"],
                    "games_played_diff": fa["games_played"] - fb["games_played"],
                    "h2h_winrate_a": h2h_winrate_a,
                    "h2h_matches": h["matches"],
                }
            )
        X_batch = pd.DataFrame(rows)[FEATURE_COLUMNS]

        # Predict this fixture across all n_sims "parallel universes" at once
        # (vectorized, avoiding per-call model overhead).
        p_a_win_arr = model.predict_proba(X_batch)
        pred_margin_arr = model.predict_margin(X_batch)

        a_wins_arr = rng.random(n_sims) < p_a_win_arr
        margin_arr = pred_margin_arr + rng.normal(0, margin_noise_std, n_sims)
        # Align the margin's sign with the win/loss outcome, to avoid the
        # rare-but-confusing case where a simulated "win" comes with a
        # negative margin, which would mess up rank tiebreaking.
        flip_pos = a_wins_arr & (margin_arr < 0)
        flip_neg = (~a_wins_arr) & (margin_arr > 0)
        margin_arr = np.where(flip_pos, np.abs(margin_arr), margin_arr)
        margin_arr = np.where(flip_neg, -np.abs(margin_arr), margin_arr)

        for s in range(n_sims):
            a_wins = bool(a_wins_arr[s])
            margin = float(margin_arr[s])

            sim_standings[s][team_a]["net_margin"] += margin
            sim_standings[s][team_b]["net_margin"] -= margin
            if a_wins:
                sim_standings[s][team_a]["wins"] += 1
                sim_standings[s][team_b]["losses"] += 1
            else:
                sim_standings[s][team_b]["wins"] += 1
                sim_standings[s][team_a]["losses"] += 1

            sim_elo[s].update_one(team_a, team_b, a_wins, margin)
            sim_form[s][team_a] = update_team_form(sim_form[s][team_a], a_wins, margin)
            sim_form[s][team_b] = update_team_form(sim_form[s][team_b], not a_wins, -margin)

            pair = tuple(sorted([team_a, team_b]))
            low = pair[0]
            low_won = a_wins if team_a == low else (not a_wins)
            h = sim_h2h[s].get(pair, {"low_winrate": 0.5, "matches": 0})
            new_matches = h["matches"] + 1
            new_low_wins = h["low_winrate"] * h["matches"] + int(low_won)
            sim_h2h[s][pair] = {"low_winrate": new_low_wins / new_matches, "matches": new_matches}

    final_wins = np.zeros((n_sims, len(all_teams)))
    final_ranks = np.zeros((n_sims, len(all_teams)), dtype=int)

    for s in range(n_sims):
        final_table = pd.DataFrame(
            [{"team": t, "wins": sim_standings[s][t]["wins"], "net_margin": sim_standings[s][t]["net_margin"]}
             for t in all_teams]
        ).sort_values(["wins", "net_margin"], ascending=[False, False]).reset_index(drop=True)
        final_table["rank"] = final_table.index + 1

        rank_lookup = dict(zip(final_table["team"], final_table["rank"]))
        wins_lookup = dict(zip(final_table["team"], final_table["wins"]))
        for idx, t in enumerate(all_teams):
            final_ranks[s, idx] = rank_lookup[t]
            final_wins[s, idx] = wins_lookup[t]

    rank_df = pd.DataFrame(final_ranks, columns=all_teams)
    wins_df = pd.DataFrame(final_wins, columns=all_teams)

    n_teams = len(all_teams)
    summary_rows = []
    prob_rows = []
    for t in all_teams:
        ranks = rank_df[t]
        summary_rows.append(
            {
                "team": t,
                "expected_rank": ranks.mean(),
                "median_rank": ranks.median(),
                "champion_prob": (ranks == 1).mean(),
                f"top{top_n_for_prob}_prob": (ranks <= top_n_for_prob).mean(),
                "expected_final_wins": wins_df[t].mean(),
            }
        )
        for r in range(1, n_teams + 1):
            prob_rows.append({"team": t, "rank": r, "probability": (ranks == r).mean()})

    rank_table = pd.DataFrame(summary_rows).sort_values("expected_rank").reset_index(drop=True)
    rank_prob_matrix = pd.DataFrame(prob_rows).pivot(index="team", columns="rank", values="probability")
    rank_prob_matrix = rank_prob_matrix.loc[rank_table["team"]]

    return SimulationResult(
        rank_table=rank_table, rank_prob_matrix=rank_prob_matrix, raw_final_wins=wins_df, n_sims=n_sims
    )