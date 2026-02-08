"""
Elo Rating System for Multi-Sport Betting Predictions.

Implements a configurable Elo rating engine with sport-specific parameters,
margin-of-victory adjustments, home-court advantage, and season regression.

Supported sports: NBA, NFL, MLB, NHL, Soccer.
"""
from __future__ import annotations

import json
import math
import os
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sport-specific configuration
# ---------------------------------------------------------------------------
# k_factor        – maximum rating change per game
# home_advantage   – Elo points added to home team's effective rating
# mean_rating      – league-average Elo (new teams start here)
# season_regression – fraction regressed toward the mean between seasons
#                     e.g. 0.33 means 1/3 of the way back to the mean
# ---------------------------------------------------------------------------
SPORT_CONFIG: Dict[str, Dict[str, float]] = {
    "nba": {
        "k_factor": 20.0,
        "home_advantage": 100.0,
        "mean_rating": 1500.0,
        "season_regression": 0.25,
    },
    "nfl": {
        "k_factor": 20.0,
        "home_advantage": 48.0,
        "mean_rating": 1500.0,
        "season_regression": 0.33,
    },
    "mlb": {
        "k_factor": 4.0,
        "home_advantage": 24.0,
        "mean_rating": 1500.0,
        "season_regression": 0.40,
    },
    "nhl": {
        "k_factor": 6.0,
        "home_advantage": 33.0,
        "mean_rating": 1500.0,
        "season_regression": 0.35,
    },
    "soccer": {
        "k_factor": 30.0,
        "home_advantage": 65.0,
        "mean_rating": 1500.0,
        "season_regression": 0.20,
    },
}


class EloRatingSystem:
    """Full Elo rating engine with sport-aware tuning.

    Parameters
    ----------
    sport : str
        One of ``"nba"``, ``"nfl"``, ``"mlb"``, ``"nhl"``, ``"soccer"``.
    ratings_dir : str, optional
        Directory where rating JSON snapshots are persisted.
        Defaults to ``"./data/elo_ratings"``.
    """

    def __init__(self, sport: str, ratings_dir: str = "./data/elo_ratings") -> None:
        sport = sport.lower()
        if sport not in SPORT_CONFIG:
            raise ValueError(
                f"Unsupported sport '{sport}'. Choose from: {list(SPORT_CONFIG.keys())}"
            )

        self.sport: str = sport
        self.config: Dict[str, float] = SPORT_CONFIG[sport]
        self.ratings: Dict[str, float] = {}  # team_id -> current Elo
        self.ratings_dir: str = ratings_dir
        self._game_count: Dict[str, int] = {}  # team_id -> games played

    # ------------------------------------------------------------------
    # Core Elo math
    # ------------------------------------------------------------------

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Return the expected score (win probability) for player/team A.

        Uses the standard logistic Elo formula:

        .. math::
            E_A = \\frac{1}{1 + 10^{(R_B - R_A) / 400}}

        Parameters
        ----------
        rating_a : float
            Elo rating of side A.
        rating_b : float
            Elo rating of side B.

        Returns
        -------
        float
            Probability that A wins (0-1).
        """
        exponent = (rating_b - rating_a) / 400.0
        return 1.0 / (1.0 + math.pow(10.0, exponent))

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def _get_rating(self, team_id: str) -> float:
        """Return current Elo for *team_id*, initializing at the mean if new."""
        if team_id not in self.ratings:
            self.ratings[team_id] = self.config["mean_rating"]
            self._game_count[team_id] = 0
        return self.ratings[team_id]

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def predict_game(
        self, home_team_id: str, away_team_id: str
    ) -> Dict[str, float]:
        """Predict win probabilities for a head-to-head matchup.

        Home-court advantage is applied by boosting the home team's
        effective rating before computing the expected score.

        Parameters
        ----------
        home_team_id : str
            Identifier for the home team.
        away_team_id : str
            Identifier for the away team.

        Returns
        -------
        dict
            ``home_win_prob`` – probability the home team wins (0-1).
            ``away_win_prob`` – probability the away team wins (0-1).
            ``elo_diff``      – raw Elo difference (home - away) *before*
            home advantage adjustment.
        """
        home_elo = self._get_rating(home_team_id)
        away_elo = self._get_rating(away_team_id)

        # Apply home-court advantage to effective rating
        effective_home = home_elo + self.config["home_advantage"]
        home_win_prob = self.expected_score(effective_home, away_elo)
        away_win_prob = 1.0 - home_win_prob

        return {
            "home_win_prob": round(home_win_prob, 6),
            "away_win_prob": round(away_win_prob, 6),
            "elo_diff": round(home_elo - away_elo, 2),
        }

    # ------------------------------------------------------------------
    # Rating updates
    # ------------------------------------------------------------------

    @staticmethod
    def _margin_of_victory_multiplier(
        elo_diff: float, margin: float
    ) -> float:
        """Compute a margin-of-victory (MOV) multiplier for the K-factor.

        Follows the FiveThirtyEight-style adjustment that rewards blowouts
        but dampens the effect when the favourite wins big (since that is
        more "expected").

        .. math::
            MOV = \\ln(|margin| + 1) \\times \\frac{2.2}{(elo\\_diff \\times 0.001) + 2.2}

        Parameters
        ----------
        elo_diff : float
            Elo rating of the winner minus the loser.
        margin : float
            Absolute score margin (always positive).

        Returns
        -------
        float
            Multiplier >= 1.0 (clamped at the lower end).
        """
        if margin <= 0:
            return 1.0
        log_margin = math.log(abs(margin) + 1.0)
        # Dampener: bigger leads by the favourite produce smaller boosts
        dampener = 2.2 / (elo_diff * 0.001 + 2.2)
        return max(log_margin * dampener, 1.0)

    def update_ratings(
        self,
        home_id: str,
        away_id: str,
        home_score: float,
        away_score: float,
    ) -> Dict[str, float]:
        """Update Elo ratings after a completed game.

        Parameters
        ----------
        home_id : str
            Home team identifier.
        away_id : str
            Away team identifier.
        home_score : float
            Points / goals scored by the home team.
        away_score : float
            Points / goals scored by the away team.

        Returns
        -------
        dict
            ``home_new`` – updated home Elo.
            ``away_new`` – updated away Elo.
            ``home_delta`` – change applied to the home team.
            ``away_delta`` – change applied to the away team.
        """
        home_elo = self._get_rating(home_id)
        away_elo = self._get_rating(away_id)

        # Effective ratings (home advantage)
        effective_home = home_elo + self.config["home_advantage"]

        # Expected scores
        home_expected = self.expected_score(effective_home, away_elo)
        away_expected = 1.0 - home_expected

        # Actual result (1 = win, 0.5 = draw, 0 = loss)
        if home_score > away_score:
            home_actual = 1.0
            away_actual = 0.0
        elif home_score < away_score:
            home_actual = 0.0
            away_actual = 1.0
        else:
            home_actual = 0.5
            away_actual = 0.5

        # Margin-of-victory multiplier
        margin = abs(home_score - away_score)
        winner_elo_diff = (home_elo - away_elo) if home_actual >= 0.5 else (away_elo - home_elo)
        mov_mult = self._margin_of_victory_multiplier(winner_elo_diff, margin)

        k = self.config["k_factor"]

        home_delta = k * mov_mult * (home_actual - home_expected)
        away_delta = k * mov_mult * (away_actual - away_expected)

        # Apply
        self.ratings[home_id] = home_elo + home_delta
        self.ratings[away_id] = away_elo + away_delta

        # Track games played
        self._game_count[home_id] = self._game_count.get(home_id, 0) + 1
        self._game_count[away_id] = self._game_count.get(away_id, 0) + 1

        logger.debug(
            "Elo update [%s]: %s %.1f -> %.1f (%+.1f) | %s %.1f -> %.1f (%+.1f)",
            self.sport,
            home_id,
            home_elo,
            self.ratings[home_id],
            home_delta,
            away_id,
            away_elo,
            self.ratings[away_id],
            away_delta,
        )

        return {
            "home_new": round(self.ratings[home_id], 2),
            "away_new": round(self.ratings[away_id], 2),
            "home_delta": round(home_delta, 2),
            "away_delta": round(away_delta, 2),
        }

    # ------------------------------------------------------------------
    # Season regression
    # ------------------------------------------------------------------

    def season_regression(self) -> Dict[str, float]:
        """Regress all ratings toward the mean between seasons.

        Each team's rating is pulled back toward ``mean_rating`` by the
        sport-specific ``season_regression`` fraction.

        Returns
        -------
        dict
            Mapping of team_id -> new Elo after regression.
        """
        mean = self.config["mean_rating"]
        regression_pct = self.config["season_regression"]

        regressed: Dict[str, float] = {}
        for team_id, rating in self.ratings.items():
            new_rating = rating - regression_pct * (rating - mean)
            self.ratings[team_id] = round(new_rating, 2)
            regressed[team_id] = self.ratings[team_id]

        # Reset game counts for the new season
        self._game_count = {tid: 0 for tid in self._game_count}

        logger.info(
            "Season regression applied for %s (%d teams, %.0f%% toward %.0f)",
            self.sport,
            len(regressed),
            regression_pct * 100,
            mean,
        )
        return regressed

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _ratings_path(self) -> str:
        """Return the full file path for this sport's ratings JSON."""
        return os.path.join(self.ratings_dir, f"elo_{self.sport}.json")

    def save_ratings(self) -> str:
        """Persist current ratings to a JSON file.

        Returns
        -------
        str
            Path to the saved file.
        """
        os.makedirs(self.ratings_dir, exist_ok=True)
        path = self._ratings_path()

        payload: Dict[str, Any] = {
            "sport": self.sport,
            "config": self.config,
            "ratings": self.ratings,
            "game_counts": self._game_count,
        }

        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

        logger.info("Saved %d Elo ratings to %s", len(self.ratings), path)
        return path

    def load_ratings(self) -> bool:
        """Load ratings from disk if the file exists.

        Returns
        -------
        bool
            ``True`` if ratings were loaded, ``False`` if the file was
            not found.
        """
        path = self._ratings_path()
        if not os.path.exists(path):
            logger.warning("No ratings file found at %s", path)
            return False

        with open(path, "r") as fh:
            payload = json.load(fh)

        self.ratings = payload.get("ratings", {})
        self._game_count = payload.get("game_counts", {})
        logger.info("Loaded %d Elo ratings from %s", len(self.ratings), path)
        return True

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_rankings(self, top_n: Optional[int] = None) -> list:
        """Return teams sorted by Elo descending.

        Parameters
        ----------
        top_n : int, optional
            If given, return only the top *n* teams.

        Returns
        -------
        list of tuple
            ``(team_id, elo, games_played)`` tuples.
        """
        ranked = sorted(self.ratings.items(), key=lambda t: t[1], reverse=True)
        result = [
            (tid, round(elo, 2), self._game_count.get(tid, 0))
            for tid, elo in ranked
        ]
        if top_n is not None:
            result = result[:top_n]
        return result

    def __repr__(self) -> str:
        return (
            f"EloRatingSystem(sport={self.sport!r}, "
            f"teams={len(self.ratings)}, "
            f"k={self.config['k_factor']})"
        )
