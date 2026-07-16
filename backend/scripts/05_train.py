#!/usr/bin/env python3
"""
05_train.py — single-match prediction model training (thin wrapper).

Reads: data/processed/train_features.parquet
Writes:
    data/models/match_predictor.joblib   model bundle:
        {model, feature_columns, margin_noise_std, chosen_model_name, candidate_metrics}
    data/models/train_report.json         validation metrics for both candidates (logistic regression / GBDT),
                                            for reviewing why one was chosen

Training/serving are separated: this step only produces model files.
FastAPI (stage 3) only loads them, never retrains.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ivl_pipeline.features import FEATURE_COLUMNS
from ivl_pipeline.io_utils import load_parquet, save_joblib, save_json
from ivl_pipeline.models import train_and_select


def run(processed_dir: str, models_dir: str, val_fraction: float = 0.15) -> dict:
    feature_df = load_parquet(Path(processed_dir) / "train_features.parquet")

    model, report = train_and_select(feature_df, FEATURE_COLUMNS, val_fraction=val_fraction)

    chosen_metrics = report.candidate_metrics[report.chosen_model_name]
    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "chosen_model_name": report.chosen_model_name,
        "margin_noise_std": chosen_metrics["margin_residual_std"],
        "candidate_metrics": report.candidate_metrics,
    }
    save_joblib(bundle, Path(models_dir) / "match_predictor.joblib")

    save_json(
        {
            "chosen_model_name": report.chosen_model_name,
            "candidate_metrics": report.candidate_metrics,
            "n_train_matches": len(report.train_match_orders),
            "n_val_matches": len(report.val_match_orders),
        },
        Path(models_dir) / "train_report.json",
    )

    return {"chosen_model_name": report.chosen_model_name, "metrics": report.candidate_metrics}


def main():
    parser = argparse.ArgumentParser(description="05: single-match prediction model training")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="data/models")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()

    summary = run(args.processed_dir, args.models_dir, args.val_fraction)
    print("[05_train] done:", summary)


if __name__ == "__main__":
    main()