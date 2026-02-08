"""
Feature engineering module for the EDGE AI sports betting system.

Computes team-level and game-level feature vectors used by the ML and
Bayesian models in the ensemble.
"""

import math
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Builds feature vectors from raw game histories and team stats."""

    # Default Elo parameters
    ELO_K = 20.0
    ELO_HOME_ADVANTAGE = 100.0
    ELO_DEFAULT = 1500.0

    # ------------------------------------------------------------------ #
    #  Team-level features
    # ------------------------------------------------------------------ #

    def compute_team_features(
        self,
        team_id: str,
        game_history: List[Dict[str, Any]],
        sport: str,
    ) -> Dict[str, Any]:
        """Compute a full set of team-level features from recent game history.

        Parameters
        ----------
        team_id : str
            Identifier for the team.
        game_history : list of dict
            Most-recent-first list of completed games.  Each entry should
            contain at least:
                - ``date``             : ISO-8601 or datetime
                - ``team_id``          : str
                - ``opponent_id``      : str
                - ``is_home``          : bool
                - ``team_score``       : int
                - ``opponent_score``   : int
                - ``team_off_rating``  : float (points per 100 possessions for NBA,
                                                goals per 90 for soccer, etc.)
                - ``team_def_rating``  : float
                - ``pace``             : float (possessions per 48 min for NBA)
            Optional:
                - ``injury_impact``    : float (estimated point-swing from injuries)

        sport : str
            Sport code (e.g. "nba", "nfl", "mlb", "nhl", "soccer").

        Returns
        -------
        dict  – Feature vector for the team.
        """
        if not game_history:
            return self._empty_team_features(team_id, sport)

        # Sort most-recent-first (defensive copy)
        history = sorted(
            game_history,
            key=lambda g: self._parse_date(g["date"]),
            reverse=True,
        )

        # ---- Win percentages ----
        win_pct_all = self._win_pct(history)
        home_games = [g for g in history if g.get("is_home", False)]
        away_games = [g for g in history if not g.get("is_home", True)]
        win_pct_home = self._win_pct(home_games) if home_games else 0.5
        win_pct_away = self._win_pct(away_games) if away_games else 0.5

        # ---- Rolling offensive / defensive ratings ----
        off_5 = self._rolling_mean(history, "team_off_rating", 5)
        off_10 = self._rolling_mean(history, "team_off_rating", 10)
        off_20 = self._rolling_mean(history, "team_off_rating", 20)

        def_5 = self._rolling_mean(history, "team_def_rating", 5)
        def_10 = self._rolling_mean(history, "team_def_rating", 10)
        def_20 = self._rolling_mean(history, "team_def_rating", 20)

        # ---- Net rating (offensive - defensive, lower def is better) ----
        net_5 = off_5 - def_5
        net_10 = off_10 - def_10
        net_20 = off_20 - def_20

        # ---- Pace ----
        pace_avg = self._rolling_mean(history, "pace", 10)

        # ---- Rest days & back-to-back ----
        rest_days = self._rest_days(history)
        is_b2b = rest_days == 0

        # ---- Streaks ----
        streak = self._current_streak(history)

        # ---- Injury impact (average of recent reported values) ----
        injury_impacts = [
            g.get("injury_impact", 0.0) for g in history[:5]
        ]
        avg_injury_impact = (
            sum(injury_impacts) / len(injury_impacts)
            if injury_impacts else 0.0
        )

        # ---- Elo (cumulative from history) ----
        elo = self._compute_elo_from_history(team_id, history)

        return {
            "team_id": team_id,
            "sport": sport,
            "elo": round(elo, 2),
            "win_pct": round(win_pct_all, 4),
            "win_pct_home": round(win_pct_home, 4),
            "win_pct_away": round(win_pct_away, 4),
            "off_rating_5g": round(off_5, 4),
            "off_rating_10g": round(off_10, 4),
            "off_rating_20g": round(off_20, 4),
            "def_rating_5g": round(def_5, 4),
            "def_rating_10g": round(def_10, 4),
            "def_rating_20g": round(def_20, 4),
            "net_rating_5g": round(net_5, 4),
            "net_rating_10g": round(net_10, 4),
            "net_rating_20g": round(net_20, 4),
            "pace": round(pace_avg, 4),
            "rest_days": rest_days,
            "is_b2b": is_b2b,
            "streak": streak,
            "injury_impact": round(avg_injury_impact, 4),
            "games_played": len(history),
        }

    # ------------------------------------------------------------------ #
    #  Game-level features
    # ------------------------------------------------------------------ #

    def compute_game_features(
        self,
        game: Dict[str, Any],
        team_features_home: Dict[str, Any],
        team_features_away: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute the full game-level feature vector by combining home/away
        team features with game-specific context.

        Parameters
        ----------
        game : dict
            Game metadata.  Expected keys:
                - ``game_id``        : str
                - ``sport``          : str
                - ``home_team``      : str
                - ``away_team``      : str
            Optional:
                - ``market_home_prob``  : float (no-vig market implied)
                - ``line_movement``     : dict  (output of LineMovementAnalyzer)
                - ``h2h_home_wins``     : int
                - ``h2h_away_wins``     : int
                - ``h2h_draws``         : int
                - ``weather``           : dict  (outdoor sports)
        team_features_home : dict
            Output of ``compute_team_features`` for the home team.
        team_features_away : dict
            Output of ``compute_team_features`` for the away team.

        Returns
        -------
        dict  – Full feature vector suitable for model input.
        """
        h = team_features_home
        a = team_features_away

        # ---- Differential features ----
        elo_diff = h.get("elo", self.ELO_DEFAULT) - a.get("elo", self.ELO_DEFAULT)

        off_diff_5 = h.get("off_rating_5g", 0) - a.get("off_rating_5g", 0)
        off_diff_10 = h.get("off_rating_10g", 0) - a.get("off_rating_10g", 0)
        def_diff_5 = a.get("def_rating_5g", 0) - h.get("def_rating_5g", 0)  # inverted: lower is better
        def_diff_10 = a.get("def_rating_10g", 0) - h.get("def_rating_10g", 0)

        net_diff_5 = h.get("net_rating_5g", 0) - a.get("net_rating_5g", 0)
        net_diff_10 = h.get("net_rating_10g", 0) - a.get("net_rating_10g", 0)
        net_diff_20 = h.get("net_rating_20g", 0) - a.get("net_rating_20g", 0)

        pace_avg = (h.get("pace", 0) + a.get("pace", 0)) / 2.0

        # ---- Rest advantage ----
        rest_diff = h.get("rest_days", 1) - a.get("rest_days", 1)
        home_b2b = 1 if h.get("is_b2b", False) else 0
        away_b2b = 1 if a.get("is_b2b", False) else 0

        # ---- Win-pct differential ----
        win_pct_diff = h.get("win_pct", 0.5) - a.get("win_pct", 0.5)
        home_away_split = h.get("win_pct_home", 0.5) - a.get("win_pct_away", 0.5)

        # ---- Injury ----
        injury_diff = h.get("injury_impact", 0) - a.get("injury_impact", 0)

        # ---- Head-to-head ----
        h2h_home_wins = game.get("h2h_home_wins", 0)
        h2h_away_wins = game.get("h2h_away_wins", 0)
        h2h_draws = game.get("h2h_draws", 0)
        h2h_total = h2h_home_wins + h2h_away_wins + h2h_draws
        h2h_home_rate = (
            h2h_home_wins / h2h_total if h2h_total > 0 else 0.5
        )

        # ---- Market features ----
        market_home_prob = game.get("market_home_prob", 0.5)

        # ---- Line movement ----
        lm = game.get("line_movement", {})
        lm_prob_shift = lm.get("prob_shift", 0.0)
        lm_velocity = lm.get("velocity", 0.0)
        lm_is_steam = 1 if lm.get("is_steam", False) else 0
        lm_is_rlm = 1 if lm.get("is_rlm", False) else 0

        # ---- Streak differential ----
        streak_diff = h.get("streak", 0) - a.get("streak", 0)

        return {
            "game_id": game.get("game_id", ""),
            "sport": game.get("sport", ""),
            "home_team": game.get("home_team", ""),
            "away_team": game.get("away_team", ""),
            # Elo
            "elo_diff": round(elo_diff, 2),
            "elo_home": round(h.get("elo", self.ELO_DEFAULT), 2),
            "elo_away": round(a.get("elo", self.ELO_DEFAULT), 2),
            # Offensive ratings
            "off_rating_diff_5g": round(off_diff_5, 4),
            "off_rating_diff_10g": round(off_diff_10, 4),
            # Defensive ratings
            "def_rating_diff_5g": round(def_diff_5, 4),
            "def_rating_diff_10g": round(def_diff_10, 4),
            # Net ratings
            "net_rating_diff_5g": round(net_diff_5, 4),
            "net_rating_diff_10g": round(net_diff_10, 4),
            "net_rating_diff_20g": round(net_diff_20, 4),
            # Pace
            "pace_avg": round(pace_avg, 4),
            # Rest
            "rest_diff": rest_diff,
            "home_b2b": home_b2b,
            "away_b2b": away_b2b,
            # Win percentage
            "win_pct_diff": round(win_pct_diff, 4),
            "home_away_split": round(home_away_split, 4),
            # Injuries
            "injury_diff": round(injury_diff, 4),
            "injury_home": round(h.get("injury_impact", 0), 4),
            "injury_away": round(a.get("injury_impact", 0), 4),
            # Head-to-head
            "h2h_home_rate": round(h2h_home_rate, 4),
            "h2h_total_games": h2h_total,
            # Market
            "market_implied_home": round(market_home_prob, 6),
            # Line movement
            "line_movement_shift": round(lm_prob_shift, 6),
            "line_movement_velocity": round(lm_velocity, 6),
            "is_steam_move": lm_is_steam,
            "is_rlm": lm_is_rlm,
            # Streak
            "streak_diff": streak_diff,
            "streak_home": h.get("streak", 0),
            "streak_away": a.get("streak", 0),
        }

    # ================================================================== #
    #  Private helpers
    # ================================================================== #

    @staticmethod
    def _parse_date(date_val: Any) -> datetime:
        """Parse a date value to datetime."""
        if isinstance(date_val, datetime):
            return date_val
        if isinstance(date_val, str):
            cleaned = date_val.replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(cleaned)
            except ValueError:
                return datetime.strptime(cleaned[:10], "%Y-%m-%d")
        raise TypeError("Unsupported date type: {}".format(type(date_val)))

    @staticmethod
    def _win_pct(games: List[Dict[str, Any]]) -> float:
        """Win percentage from a list of game dicts."""
        if not games:
            return 0.5
        wins = sum(
            1 for g in games
            if g.get("team_score", 0) > g.get("opponent_score", 0)
        )
        return wins / len(games)

    @staticmethod
    def _rolling_mean(
        history: List[Dict[str, Any]],
        key: str,
        window: int,
    ) -> float:
        """Compute rolling mean of *key* over the most recent *window* games."""
        vals = [
            g[key] for g in history[:window]
            if key in g and g[key] is not None
        ]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    @staticmethod
    def _rest_days(history: List[Dict[str, Any]]) -> int:
        """Days between the most recent game and the one before it.

        Returns 0 if back-to-back, or -1 if insufficient data.
        """
        if len(history) < 2:
            return -1

        dates = []
        for g in history[:2]:
            d = g.get("date")
            if isinstance(d, str):
                d = d.replace("Z", "+00:00")
                try:
                    d = datetime.fromisoformat(d)
                except ValueError:
                    d = datetime.strptime(d[:10], "%Y-%m-%d")
            dates.append(d)

        delta = (dates[0] - dates[1]).days
        return max(delta - 1, 0)  # subtract game day itself

    @staticmethod
    def _current_streak(history: List[Dict[str, Any]]) -> int:
        """Compute current win/loss streak.

        Positive = winning streak, negative = losing streak.
        """
        if not history:
            return 0

        first_win = history[0].get("team_score", 0) > history[0].get("opponent_score", 0)
        streak = 0

        for g in history:
            is_win = g.get("team_score", 0) > g.get("opponent_score", 0)
            if is_win == first_win:
                streak += 1
            else:
                break

        return streak if first_win else -streak

    def _compute_elo_from_history(
        self,
        team_id: str,
        history: List[Dict[str, Any]],
    ) -> float:
        """Walk through the game history (oldest-first) and compute a
        running Elo rating for *team_id*.

        Uses a simple Elo model with K-factor and home-court advantage.
        """
        # Process oldest-first
        games = list(reversed(history))
        elo = self.ELO_DEFAULT
        opponent_elos = {}  # type: Dict[str, float]

        for g in games:
            opp_id = g.get("opponent_id", "unknown")
            opp_elo = opponent_elos.get(opp_id, self.ELO_DEFAULT)
            is_home = g.get("is_home", True)

            # Expected score
            diff = elo - opp_elo
            if is_home:
                diff += self.ELO_HOME_ADVANTAGE

            expected = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

            # Actual result
            team_score = g.get("team_score", 0)
            opp_score = g.get("opponent_score", 0)
            if team_score > opp_score:
                actual = 1.0
            elif team_score < opp_score:
                actual = 0.0
            else:
                actual = 0.5

            # Update
            elo += self.ELO_K * (actual - expected)

            # Track opponent Elo (simple mirror update)
            opp_elo += self.ELO_K * (expected - actual)
            opponent_elos[opp_id] = opp_elo

        return elo

    @staticmethod
    def _empty_team_features(team_id: str, sport: str) -> Dict[str, Any]:
        """Return a zeroed-out feature dict when no history is available."""
        return {
            "team_id": team_id,
            "sport": sport,
            "elo": 1500.0,
            "win_pct": 0.5,
            "win_pct_home": 0.5,
            "win_pct_away": 0.5,
            "off_rating_5g": 0.0,
            "off_rating_10g": 0.0,
            "off_rating_20g": 0.0,
            "def_rating_5g": 0.0,
            "def_rating_10g": 0.0,
            "def_rating_20g": 0.0,
            "net_rating_5g": 0.0,
            "net_rating_10g": 0.0,
            "net_rating_20g": 0.0,
            "pace": 0.0,
            "rest_days": -1,
            "is_b2b": False,
            "streak": 0,
            "injury_impact": 0.0,
            "games_played": 0,
        }
