"""
elo.py — dynamic Elo rating: turns "team strength" into a feature that
updates over time, instead of a static historical win rate. Naturally
handles roster changes (mean-reversion applied at the start of each new
season), and weights the update magnitude by margin of victory (a blowout
vs. a narrow win carries a different strength signal).

Usage:
    ratings = EloRatingSystem()
    matches_with_elo = ratings.fit_transform(league_matches)   # rolls through matches in match_order
    ratings.rating_of("MRC")                                    # get a team's "current" (latest) rating
    ratings.get_all_ratings()                                   # get all teams' current ratings, fed into season simulation
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class EloRatingSystem:
    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 24.0,
        season_regress: float = 0.3,
        margin_scale: float = 12.0,
    ):
        """
        initial_rating: starting rating for a new team
        k_factor: base step size for per-match rating adjustments
        season_regress: fraction by which ratings regress toward
                         initial_rating at the start of each new season
                         (0 = no regression/carry over last season's
                         strength entirely, 1 = full reset). Used to model
                         "rosters can change, so being strong last season
                         doesn't guarantee being strong this season".
        margin_scale: how much the score margin amplifies K — a cleaner win
                       produces a bigger rating adjustment.
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.season_regress = season_regress
        self.margin_scale = margin_scale
        self.ratings: dict[str, float] = {}
        self._last_season: str | None = None

    def rating_of(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)

    def get_all_ratings(self) -> dict[str, float]:
        return dict(self.ratings)

    def _maybe_regress_new_season(self, season: str) -> None:
        if self._last_season is not None and season != self._last_season:
            for team in list(self.ratings.keys()):
                self.ratings[team] = (
                    self.ratings[team] * (1 - self.season_regress)
                    + self.initial_rating * self.season_regress
                )
        self._last_season = season

    def _margin_multiplier(self, small_score_diff: float) -> float:
        """Larger score margins carry a stronger signal for this match's
        rating update; log-compressed to avoid extreme matches dominating."""
        if pd.isna(small_score_diff):
            return 1.0
        return 1.0 + np.log1p(abs(small_score_diff)) / self.margin_scale

    def update_one(self, team_a: str, team_b: str, a_wins: bool, small_score_diff: float) -> tuple[float, float]:
        """
        Update ratings for one match; returns the pre-match ratings
        (elo_a_pre, elo_b_pre). small_score_diff is passed from team_a's
        perspective, i.e. team_a's small score minus team_b's.
        """
        ra = self.rating_of(team_a)
        rb = self.rating_of(team_b)

        expected_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        actual_a = 1.0 if a_wins else 0.0

        mult = self._margin_multiplier(small_score_diff)
        delta = self.k_factor * mult * (actual_a - expected_a)

        self.ratings[team_a] = ra + delta
        self.ratings[team_b] = rb - delta
        return ra, rb

    def fit_transform(self, league_matches: pd.DataFrame) -> pd.DataFrame:
        """
        Rolls through the league match table in match_order and returns a
        copy with three new columns: elo_a_pre / elo_b_pre / elo_diff_pre —
        all "pre-match" ratings, i.e. values that are legitimately available
        when predicting this match's outcome (they don't contain this
        match's own result, so there's no leakage).
        """
        df = league_matches.sort_values("match_order").reset_index(drop=True).copy()
        elo_a_pre = np.empty(len(df))
        elo_b_pre = np.empty(len(df))

        for i, row in df.iterrows():
            self._maybe_regress_new_season(row["season"])
            a_wins = row["winner"] == row["team_a"]
            ra, rb = self.update_one(row["team_a"], row["team_b"], a_wins, row.get("small_score_diff", np.nan))
            elo_a_pre[i] = ra
            elo_b_pre[i] = rb

        df["elo_a_pre"] = elo_a_pre
        df["elo_b_pre"] = elo_b_pre
        df["elo_diff_pre"] = df["elo_a_pre"] - df["elo_b_pre"]
        return df