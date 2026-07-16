#!/usr/bin/env python3
"""
06_predict.py — Monte Carlo season simulation + rank prediction output (thin wrapper).

Reads:
    data/models/match_predictor.joblib      model bundle produced by 05_train
    data/processed/state_snapshot.joblib     current Elo/form/H2H state produced by 04_feature
    data/interim/league_matches.parquet      cleaned league match table produced by 04_feature
    data/raw/2026_Summer_schedule.csv        official schedule (manually transcribed from the event
                                               organizer's schedule graphic, 90 matches across WEEK1-9,
                                               verified via "each team appears exactly 18 times")

Writes:
    data/predictions/{season}_rank_prediction.csv        expected rank / champion probability / playoff probability per team
    data/predictions/{season}_rank_prob_matrix.csv        full rank probability distribution, team x rank

Schedule source: by default this prefers data/raw/{season}_schedule.csv (the
real official schedule). remaining_fixtures_from_schedule() subtracts the
matches already present in league_matches from the schedule, leaving the
matches still to be simulated — this is more accurate than guessing a
"double round-robin", since the pairings themselves are real. If no official
schedule file is found, it falls back to infer_remaining_fixtures()'s
double round-robin guess.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ivl_pipeline.io_utils import load_joblib, load_parquet, save_csv
from ivl_pipeline.simulate import (
    infer_remaining_fixtures,
    load_official_schedule,
    remaining_fixtures_from_schedule,
    simulate_season,
    standings_to_state,
)
from ivl_pipeline.standings import compute_standings


def run(
    season: str,
    interim_dir: str,
    processed_dir: str,
    models_dir: str,
    predictions_dir: str,
    raw_dir: str = "data/raw",
    matches_per_pair: int = 2,
    n_sims: int = 5000,
    fixtures: list[tuple[str, str]] | None = None,
) -> dict:
    league = load_parquet(Path(interim_dir) / "league_matches.parquet")
    state_snapshot = load_joblib(Path(processed_dir) / "state_snapshot.joblib")
    bundle = load_joblib(Path(models_dir) / "match_predictor.joblib")

    standings = compute_standings(league, season=season)
    current_state = standings_to_state(standings)

    fixture_source = "explicit_arg"
    if fixtures is None:
        schedule_path = Path(raw_dir) / f"{season}_schedule.csv"
        if schedule_path.exists():
            schedule = load_official_schedule(str(schedule_path))
            fixtures = remaining_fixtures_from_schedule(schedule, league, season)
            fixture_source = f"official_schedule:{schedule_path.name}"
        else:
            fixtures = infer_remaining_fixtures(league, season, matches_per_pair=matches_per_pair)
            fixture_source = "inferred_double_round_robin"

    result = simulate_season(
        fixtures=fixtures,
        model=bundle["model"],
        elo_system=state_snapshot["elo_system"],
        team_form=state_snapshot["team_form"],
        h2h_lookup=state_snapshot["h2h_lookup"],
        current_standings=current_state,
        margin_noise_std=bundle["margin_noise_std"],
        n_sims=n_sims,
    )

    save_csv(result.rank_table, Path(predictions_dir) / f"{season}_rank_prediction.csv")
    save_csv(
        result.rank_prob_matrix.reset_index(), Path(predictions_dir) / f"{season}_rank_prob_matrix.csv"
    )
    save_csv(standings, Path(predictions_dir) / f"{season}_current_standings.csv")

    return {
        "season": season,
        "fixture_source": fixture_source,
        "n_remaining_fixtures": len(fixtures),
        "n_sims": n_sims,
        "model_used": bundle["chosen_model_name"],
        "top3": result.rank_table.head(3)[["team", "expected_rank", "champion_prob"]].to_dict("records"),
    }


def main():
    parser = argparse.ArgumentParser(description="06: Monte Carlo season simulation + rank prediction")
    parser.add_argument("--season", default="2026_Summer")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="data/models")
    parser.add_argument("--predictions-dir", default="data/predictions")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--matches-per-pair", type=int, default=2)
    parser.add_argument("--n-sims", type=int, default=5000)
    args = parser.parse_args()

    summary = run(
        season=args.season,
        interim_dir=args.interim_dir,
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        predictions_dir=args.predictions_dir,
        raw_dir=args.raw_dir,
        matches_per_pair=args.matches_per_pair,
        n_sims=args.n_sims,
    )
    print("[06_predict] done:", summary)


if __name__ == "__main__":
    main()