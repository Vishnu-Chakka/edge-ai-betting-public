"""
Poisson Scoring Model for Low-Scoring Sports.

Builds team-level attack and defense strength parameters from historical
results and uses a bivariate (independent) Poisson model to generate
score-line probability matrices, match outcome probabilities, and
over/under totals.

Best suited for: Soccer, NHL, MLB.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import poisson

logger = logging.getLogger(__name__)

# Default home-advantage goal/run boost per sport
DEFAULT_HOME_ADVANTAGE: Dict[str, float] = {
    "soccer": 0.25,
    "nhl": 0.15,
    "mlb": 0.10,
}

# Maximum goals/runs modelled in the score matrix
DEFAULT_MAX_GOALS: Dict[str, int] = {
    "soccer": 8,
    "nhl": 10,
    "mlb": 15,
}


class PoissonScoringModel:
    """Poisson-based match outcome predictor.

    Parameters
    ----------
    sport : str
        One of ``"soccer"``, ``"nhl"``, ``"mlb"``.
    home_advantage : float, optional
        Additive boost (in expected goals/runs) applied to the home side.
        If ``None``, uses the built-in default for *sport*.
    max_goals : int, optional
        Upper bound of the score matrix dimension. If ``None``, uses the
        built-in default for *sport*.
    """

    def __init__(
        self,
        sport: str,
        home_advantage: Optional[float] = None,
        max_goals: Optional[int] = None,
    ) -> None:
        sport = sport.lower()
        if sport not in DEFAULT_HOME_ADVANTAGE:
            raise ValueError(
                f"Unsupported sport '{sport}'. Choose from: "
                f"{list(DEFAULT_HOME_ADVANTAGE.keys())}"
            )

        self.sport: str = sport
        self.home_advantage: float = (
            home_advantage
            if home_advantage is not None
            else DEFAULT_HOME_ADVANTAGE[sport]
        )
        self.max_goals: int = (
            max_goals if max_goals is not None else DEFAULT_MAX_GOALS[sport]
        )

        # Fitted parameters (populated by .fit())
        self.attack_ratings: Dict[str, float] = {}
        self.defense_ratings: Dict[str, float] = {}
        self.league_avg_goals: float = 0.0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, game_history: List[Dict[str, Any]]) -> None:
        """Estimate attack / defense strengths from historical results.

        Uses an iterative approach (simplified Dixon-Coles style) that
        converges attack and defense parameters so that:

        * ``attack_i``  captures how many goals/runs team *i* scores
          relative to the league average.
        * ``defense_i`` captures how many goals/runs team *i* concedes
          relative to the league average.

        Parameters
        ----------
        game_history : list of dict
            Each dict must contain:
            ``home_team_id``, ``away_team_id``, ``home_score``, ``away_score``.

        Raises
        ------
        ValueError
            If fewer than 10 games are supplied.
        """
        if len(game_history) < 10:
            raise ValueError(
                f"Need at least 10 games to fit; got {len(game_history)}"
            )

        # ---- Aggregate per-team statistics ----
        home_scored: Dict[str, List[float]] = defaultdict(list)
        home_conceded: Dict[str, List[float]] = defaultdict(list)
        away_scored: Dict[str, List[float]] = defaultdict(list)
        away_conceded: Dict[str, List[float]] = defaultdict(list)

        total_home_goals = 0.0
        total_away_goals = 0.0
        n_games = len(game_history)

        for game in game_history:
            hid = str(game["home_team_id"])
            aid = str(game["away_team_id"])
            hs = float(game["home_score"])
            as_ = float(game["away_score"])

            home_scored[hid].append(hs)
            home_conceded[hid].append(as_)
            away_scored[aid].append(as_)
            away_conceded[aid].append(hs)

            total_home_goals += hs
            total_away_goals += as_

        avg_home_goals = total_home_goals / n_games
        avg_away_goals = total_away_goals / n_games
        self.league_avg_goals = (avg_home_goals + avg_away_goals) / 2.0

        # Collect all teams
        all_teams = set(home_scored.keys()) | set(away_scored.keys())

        # ---- Iterative parameter estimation ----
        # Initialize all ratings at 1.0 (league average)
        attack: Dict[str, float] = {t: 1.0 for t in all_teams}
        defense: Dict[str, float] = {t: 1.0 for t in all_teams}

        n_iterations = 20
        for iteration in range(n_iterations):
            new_attack: Dict[str, float] = {}
            new_defense: Dict[str, float] = {}

            for team in all_teams:
                # Attack: average goals scored / expected goals given
                # opponents' defense ratings
                goals_scored: List[float] = []
                opp_defense_faced: List[float] = []

                for g in home_scored.get(team, []):
                    goals_scored.append(g)
                for g in away_scored.get(team, []):
                    goals_scored.append(g)

                # Opponents faced at home (away teams)
                for game in game_history:
                    hid = str(game["home_team_id"])
                    aid = str(game["away_team_id"])
                    if hid == team:
                        opp_defense_faced.append(defense.get(aid, 1.0))
                    elif aid == team:
                        opp_defense_faced.append(defense.get(hid, 1.0))

                if goals_scored and opp_defense_faced:
                    avg_scored = sum(goals_scored) / len(goals_scored)
                    avg_opp_def = sum(opp_defense_faced) / len(opp_defense_faced)
                    raw_attack = avg_scored / (self.league_avg_goals * max(avg_opp_def, 0.01))
                    new_attack[team] = max(raw_attack, 0.1)
                else:
                    new_attack[team] = 1.0

                # Defense: average goals conceded / expected goals given
                # opponents' attack ratings
                goals_conceded: List[float] = []
                opp_attack_faced: List[float] = []

                for g in home_conceded.get(team, []):
                    goals_conceded.append(g)
                for g in away_conceded.get(team, []):
                    goals_conceded.append(g)

                for game in game_history:
                    hid = str(game["home_team_id"])
                    aid = str(game["away_team_id"])
                    if hid == team:
                        opp_attack_faced.append(attack.get(aid, 1.0))
                    elif aid == team:
                        opp_attack_faced.append(attack.get(hid, 1.0))

                if goals_conceded and opp_attack_faced:
                    avg_conceded = sum(goals_conceded) / len(goals_conceded)
                    avg_opp_att = sum(opp_attack_faced) / len(opp_attack_faced)
                    raw_defense = avg_conceded / (self.league_avg_goals * max(avg_opp_att, 0.01))
                    new_defense[team] = max(raw_defense, 0.1)
                else:
                    new_defense[team] = 1.0

            # Normalize so the mean attack and defense are 1.0
            mean_attack = sum(new_attack.values()) / len(new_attack) if new_attack else 1.0
            mean_defense = sum(new_defense.values()) / len(new_defense) if new_defense else 1.0

            attack = {t: v / mean_attack for t, v in new_attack.items()}
            defense = {t: v / mean_defense for t, v in new_defense.items()}

        self.attack_ratings = attack
        self.defense_ratings = defense
        self._fitted = True

        logger.info(
            "PoissonModel [%s] fitted on %d games, %d teams, league avg %.2f goals",
            self.sport,
            n_games,
            len(all_teams),
            self.league_avg_goals,
        )

    # ------------------------------------------------------------------
    # Expected goals helpers
    # ------------------------------------------------------------------

    def _expected_goals(
        self, home_id: str, away_id: str
    ) -> Tuple[float, float]:
        """Return ``(home_xG, away_xG)`` for a matchup.

        Raises ``RuntimeError`` if the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

        home_att = self.attack_ratings.get(home_id, 1.0)
        home_def = self.defense_ratings.get(home_id, 1.0)
        away_att = self.attack_ratings.get(away_id, 1.0)
        away_def = self.defense_ratings.get(away_id, 1.0)

        # home xG = league_avg * home_attack * away_defense + home_advantage
        home_xg = self.league_avg_goals * home_att * away_def + self.home_advantage
        # away xG = league_avg * away_attack * home_defense
        away_xg = self.league_avg_goals * away_att * home_def

        # Clamp to prevent degenerate Poisson means
        home_xg = max(home_xg, 0.05)
        away_xg = max(away_xg, 0.05)

        return home_xg, away_xg

    # ------------------------------------------------------------------
    # Score matrix
    # ------------------------------------------------------------------

    def _build_score_matrix(
        self, home_xg: float, away_xg: float
    ) -> np.ndarray:
        """Build an (max_goals+1) x (max_goals+1) joint probability matrix.

        Assumes independence between home and away scoring (standard
        bivariate Poisson assumption).

        Returns
        -------
        numpy.ndarray
            ``matrix[i][j]`` = P(home scores *i*, away scores *j*).
        """
        n = self.max_goals + 1
        home_probs = poisson.pmf(np.arange(n), home_xg)
        away_probs = poisson.pmf(np.arange(n), away_xg)

        # Outer product gives the joint probability matrix
        matrix = np.outer(home_probs, away_probs)
        return matrix

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def predict_game(
        self, home_id: str, away_id: str
    ) -> Dict[str, Any]:
        """Predict match outcome probabilities and expected totals.

        Parameters
        ----------
        home_id : str
            Home team identifier.
        away_id : str
            Away team identifier.

        Returns
        -------
        dict
            ``home_win_prob`` – P(home wins).
            ``draw_prob``     – P(draw / tie).
            ``away_win_prob`` – P(away wins).
            ``expected_total``– expected combined goals / runs.
            ``home_xg``       – home expected goals / runs.
            ``away_xg``       – away expected goals / runs.
            ``score_matrix``  – 2-D numpy array of score-line probabilities.
        """
        home_xg, away_xg = self._expected_goals(home_id, away_id)
        matrix = self._build_score_matrix(home_xg, away_xg)

        n = self.max_goals + 1

        # Home wins: sum of lower-triangle
        home_win_prob = float(np.sum(np.tril(matrix, k=-1)))
        # Draws: sum of diagonal
        draw_prob = float(np.sum(np.diag(matrix)))
        # Away wins: sum of upper-triangle
        away_win_prob = float(np.sum(np.triu(matrix, k=1)))

        # Normalize to handle truncation residual
        total_prob = home_win_prob + draw_prob + away_win_prob
        if total_prob > 0:
            home_win_prob /= total_prob
            draw_prob /= total_prob
            away_win_prob /= total_prob

        return {
            "home_win_prob": round(home_win_prob, 6),
            "draw_prob": round(draw_prob, 6),
            "away_win_prob": round(away_win_prob, 6),
            "expected_total": round(home_xg + away_xg, 3),
            "home_xg": round(home_xg, 3),
            "away_xg": round(away_xg, 3),
            "score_matrix": matrix,
        }

    def predict_total_prob(
        self,
        home_id: str,
        away_id: str,
        line: float,
    ) -> Dict[str, float]:
        """Compute over/under probabilities relative to a total-goals line.

        Parameters
        ----------
        home_id : str
            Home team identifier.
        away_id : str
            Away team identifier.
        line : float
            The bookmaker total line (e.g. 2.5 for soccer).

        Returns
        -------
        dict
            ``over_prob``  – P(total goals > line).
            ``under_prob`` – P(total goals < line).
            ``push_prob``  – P(total goals == line) (non-zero only for
            integer lines).
            ``expected_total`` – expected combined goals.
        """
        home_xg, away_xg = self._expected_goals(home_id, away_id)
        matrix = self._build_score_matrix(home_xg, away_xg)
        n = self.max_goals + 1

        over_prob = 0.0
        under_prob = 0.0
        push_prob = 0.0

        for i in range(n):
            for j in range(n):
                total = i + j
                p = float(matrix[i, j])
                if total > line:
                    over_prob += p
                elif total < line:
                    under_prob += p
                else:
                    # Exact match (push) – only possible for integer lines
                    push_prob += p

        # Normalize
        total_p = over_prob + under_prob + push_prob
        if total_p > 0:
            over_prob /= total_p
            under_prob /= total_p
            push_prob /= total_p

        return {
            "over_prob": round(over_prob, 6),
            "under_prob": round(under_prob, 6),
            "push_prob": round(push_prob, 6),
            "expected_total": round(home_xg + away_xg, 3),
        }

    # ------------------------------------------------------------------
    # Most-likely scorelines
    # ------------------------------------------------------------------

    def most_likely_scores(
        self,
        home_id: str,
        away_id: str,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the *top_n* most probable exact score-lines.

        Parameters
        ----------
        home_id : str
            Home team identifier.
        away_id : str
            Away team identifier.
        top_n : int
            How many score-lines to return.

        Returns
        -------
        list of dict
            Each dict: ``home_goals``, ``away_goals``, ``probability``.
        """
        home_xg, away_xg = self._expected_goals(home_id, away_id)
        matrix = self._build_score_matrix(home_xg, away_xg)
        n = self.max_goals + 1

        scores: List[Tuple[int, int, float]] = []
        for i in range(n):
            for j in range(n):
                scores.append((i, j, float(matrix[i, j])))

        scores.sort(key=lambda x: x[2], reverse=True)

        return [
            {
                "home_goals": s[0],
                "away_goals": s[1],
                "probability": round(s[2], 6),
            }
            for s in scores[:top_n]
        ]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_team_ratings(self, team_id: str) -> Dict[str, float]:
        """Return fitted attack and defense ratings for a single team.

        Parameters
        ----------
        team_id : str
            The team identifier.

        Returns
        -------
        dict
            ``attack`` and ``defense`` ratings. Values > 1.0 are
            above-average; < 1.0 below-average.
        """
        return {
            "attack": round(self.attack_ratings.get(team_id, 1.0), 4),
            "defense": round(self.defense_ratings.get(team_id, 1.0), 4),
        }

    def __repr__(self) -> str:
        return (
            f"PoissonScoringModel(sport={self.sport!r}, "
            f"teams={len(self.attack_ratings)}, "
            f"fitted={self._fitted})"
        )
