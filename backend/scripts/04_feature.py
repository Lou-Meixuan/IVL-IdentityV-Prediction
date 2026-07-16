#!/usr/bin/env python3
"""
04_feature.py — feature engineering step (thin wrapper; business logic lives in ivl_pipeline).

Reads: data/raw/IVL.xlsx
Writes:
    data/interim/league_matches.parquet         cleaned league match table (597 matches, with parsed numeric columns)
    data/interim/international_matches.parquet  cleaned international match table (208 matches, for cross-season validation, not used in training)
    data/processed/train_features.parquet        training feature matrix (with mirror augmentation, ~1194 rows)
    data/processed/state_snapshot.joblib          current Elo/form/H2H state snapshot (needed by 05_train's evaluation and 06_predict's simulation)

When migrating to Airflow (stage 2), this file's run() function is called
directly by a PythonOperator — no logic changes needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ivl_pipeline.cleaning import build_international_matches, build_league_matches
from ivl_pipeline.features import build_feature_matrix, latest_h2h_lookup, latest_team_form
from ivl_pipeline.io_utils import save_joblib, save_parquet


def run(raw_xlsx: str, interim_dir: str, processed_dir: str) -> dict:
    league = build_league_matches(raw_xlsx)
    intl = build_international_matches(raw_xlsx)

    save_parquet(league, Path(interim_dir) / "league_matches.parquet")
    save_parquet(intl, Path(interim_dir) / "international_matches.parquet")

    feature_df, elo_system = build_feature_matrix(league)
    save_parquet(feature_df, Path(processed_dir) / "train_features.parquet")

    team_form = latest_team_form(league)
    h2h_lookup = latest_h2h_lookup(league)
    state_snapshot = {"elo_system": elo_system, "team_form": team_form, "h2h_lookup": h2h_lookup}
    save_joblib(state_snapshot, Path(processed_dir) / "state_snapshot.joblib")

    summary = {
        "n_league_matches": len(league),
        "n_international_matches": len(intl),
        "n_feature_rows": len(feature_df),
        "n_duplicates_dropped": league.attrs.get("n_duplicates_dropped", 0),
        "seasons": sorted(league["season"].unique().tolist(), key=lambda s: league[league["season"] == s]["season_order"].iloc[0]),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="04: feature engineering")
    parser.add_argument("--raw-xlsx", default="data/raw/IVL.xlsx")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    summary = run(args.raw_xlsx, args.interim_dir, args.processed_dir)
    print("[04_feature] done:", summary)


if __name__ == "__main__":
    main()