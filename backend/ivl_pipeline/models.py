"""
models.py — pluggable model layer for single-match prediction.

Unified interface MatchPredictor:
    fit(X, y_win, y_margin)
    predict_proba(X) -> team_a's win probability
    predict_margin(X) -> team_a's expected score margin

Two implementations:
    LogisticRegressionMatchModel  —— default/baseline, interpretable, robust on small samples
    GBDTMatchModel                —— sklearn HistGradientBoosting, captures nonlinearity/interactions

train_and_select() trains both, splits out a chronological validation set
to compare metrics, and automatically saves the better one as the
production model (recording both candidates' metrics for review).

Note: each match in the feature matrix has an "original" row and a
"mirrored" row (see features.py). Train/validation splits must be done on
whole match_order groups, never on random rows — otherwise a match's
mirrored row could end up in both the training and validation sets,
causing leakage and inflated metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class MatchPredictor(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, y_win: pd.Series, y_margin: pd.Series) -> "MatchPredictor": ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
    def predict_margin(self, X: pd.DataFrame) -> np.ndarray: ...


class LogisticRegressionMatchModel:
    name = "logistic_regression"

    def __init__(self):
        self.win_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
        self.margin_reg = make_pipeline(StandardScaler(), Ridge(alpha=5.0))

    def fit(self, X, y_win, y_margin):
        self.win_clf.fit(X, y_win)
        self.margin_reg.fit(X, y_margin)
        return self

    def predict_proba(self, X):
        return self.win_clf.predict_proba(X)[:, 1]

    def predict_margin(self, X):
        return self.margin_reg.predict(X)


class GBDTMatchModel:
    name = "gbdt"

    def __init__(self):
        self.win_clf = HistGradientBoostingClassifier(
            max_depth=3, max_iter=150, learning_rate=0.05, l2_regularization=1.0, random_state=42
        )
        self.margin_reg = HistGradientBoostingRegressor(
            max_depth=3, max_iter=150, learning_rate=0.05, l2_regularization=1.0, random_state=42
        )

    def fit(self, X, y_win, y_margin):
        self.win_clf.fit(X, y_win)
        self.margin_reg.fit(X, y_margin)
        return self

    def predict_proba(self, X):
        return self.win_clf.predict_proba(X)[:, 1]

    def predict_margin(self, X):
        return self.margin_reg.predict(X)


MODEL_REGISTRY = {
    "logistic_regression": LogisticRegressionMatchModel,
    "gbdt": GBDTMatchModel,
}


@dataclass
class TrainReport:
    candidate_metrics: dict = field(default_factory=dict)  # {model_name: {metric: value}}
    chosen_model_name: str = ""
    val_match_orders: list = field(default_factory=list)
    train_match_orders: list = field(default_factory=list)


def time_based_split(feature_df: pd.DataFrame, val_fraction: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits by match_order (chronological order); a match's original/mirrored rows always stay together."""
    unique_orders = np.sort(feature_df["match_order"].unique())
    n_val = max(1, int(len(unique_orders) * val_fraction))
    val_orders = set(unique_orders[-n_val:])
    train_df = feature_df[~feature_df["match_order"].isin(val_orders)].copy()
    val_df = feature_df[feature_df["match_order"].isin(val_orders)].copy()
    return train_df, val_df


def _evaluate(model: MatchPredictor, X_val, y_win_val, y_margin_val) -> dict:
    proba = model.predict_proba(X_val)
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    margin_pred = model.predict_margin(X_val)
    residuals = np.asarray(y_margin_val) - np.asarray(margin_pred)
    return {
        "auc": roc_auc_score(y_win_val, proba),
        "logloss": log_loss(y_win_val, proba),
        "brier": brier_score_loss(y_win_val, proba),
        "margin_mae": mean_absolute_error(y_margin_val, margin_pred),
        "margin_residual_std": float(np.std(residuals)),
        "n_val": len(y_win_val),
    }


def train_and_select(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    val_fraction: float = 0.15,
    refit_on_full_data: bool = True,
) -> tuple[MatchPredictor, TrainReport]:
    """
    Trains both LogisticRegression and GBDT candidates, splits out a
    chronological validation set, and picks the better classifier by
    logloss (primary) + AUC (secondary). The margin regressor is picked
    independently by MAE. By default, the chosen model is refit on the
    full dataset before being returned as the final artifact (the
    validation-stage model is only used for model selection).
    """
    train_df, val_df = time_based_split(feature_df, val_fraction)
    X_train, y_win_train, y_margin_train = (
        train_df[feature_columns],
        train_df["y_win"],
        train_df["y_margin"],
    )
    X_val, y_win_val, y_margin_val = val_df[feature_columns], val_df["y_win"], val_df["y_margin"]

    report = TrainReport(
        train_match_orders=sorted(train_df["match_order"].unique().tolist()),
        val_match_orders=sorted(val_df["match_order"].unique().tolist()),
    )

    fitted = {}
    for name, cls in MODEL_REGISTRY.items():
        model = cls().fit(X_train, y_win_train, y_margin_train)
        metrics = _evaluate(model, X_val, y_win_val, y_margin_val)
        report.candidate_metrics[name] = metrics
        fitted[name] = model

    # Lower logloss is better — use it to select the classifier with better calibrated probabilities
    chosen_name = min(report.candidate_metrics, key=lambda n: report.candidate_metrics[n]["logloss"])
    report.chosen_model_name = chosen_name

    if refit_on_full_data:
        final_model = MODEL_REGISTRY[chosen_name]().fit(
            feature_df[feature_columns], feature_df["y_win"], feature_df["y_margin"]
        )
    else:
        final_model = fitted[chosen_name]

    return final_model, report