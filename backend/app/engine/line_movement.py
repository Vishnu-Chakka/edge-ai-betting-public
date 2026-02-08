"""
Line movement analysis module for the EDGE AI sports betting system.

Detects steam moves, reverse line movement (RLM), and computes line velocity
from timestamped odds snapshots.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LineMovementAnalyzer:
    """Analyses timestamped odds history to detect meaningful line signals."""

    # Steam detection thresholds
    STEAM_MIN_BOOKS = 3          # minimum sportsbooks moving in the same direction
    STEAM_WINDOW_MINUTES = 15    # time window to detect synchronised moves
    STEAM_MIN_SHIFT = 0.005      # minimum implied-probability shift per book

    # RLM threshold: line moves opposite to the public side
    RLM_PUBLIC_THRESHOLD = 0.60  # fraction of bets on one side to consider "public"

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def analyze_movement(
        self,
        game_id: str,
        odds_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyse line movement for a single game.

        Parameters
        ----------
        game_id : str
            Unique game identifier.
        odds_history : list of dict
            Each entry must contain at least:
                - ``timestamp`` : ISO-8601 string or datetime
                - ``book``      : sportsbook name
                - ``home_odds`` : decimal odds for home
                - ``away_odds`` : decimal odds for away
            Optional:
                - ``draw_odds`` : decimal odds for draw
                - ``public_home_pct`` : fraction of public bets on home (0-1)

        Returns
        -------
        dict
            has_movement : bool   – any notable movement detected
            prob_shift   : float  – net implied-probability shift for the home side
            direction    : str    – "home", "away", or "neutral"
            magnitude    : float  – absolute size of probability shift
            is_steam     : bool
            steam_books  : list   – books involved in steam
            is_rlm       : bool
            velocity     : float  – implied-probability change per hour
        """
        if len(odds_history) < 2:
            return self._empty_result(game_id)

        # Ensure entries are sorted by timestamp
        history = self._normalise_history(odds_history)

        # Compute net probability shift (opening -> latest)
        opening = history[0]
        latest = history[-1]
        open_implied_home = 1.0 / opening["home_odds"]
        latest_implied_home = 1.0 / latest["home_odds"]
        prob_shift = latest_implied_home - open_implied_home

        direction = "neutral"
        if prob_shift > 0.005:
            direction = "home"
        elif prob_shift < -0.005:
            direction = "away"

        magnitude = abs(prob_shift)

        # Sub-analyses
        steam_result = self._detect_steam(history)
        is_rlm = self._detect_rlm(history, prob_shift)
        velocity = self._compute_velocity(history)

        has_movement = magnitude > 0.01 or steam_result["detected"] or is_rlm

        return {
            "game_id": game_id,
            "has_movement": has_movement,
            "prob_shift": round(prob_shift, 6),
            "direction": direction,
            "magnitude": round(magnitude, 6),
            "is_steam": steam_result["detected"],
            "steam_books": steam_result["books"],
            "is_rlm": is_rlm,
            "velocity": round(velocity, 6),
        }

    # ------------------------------------------------------------------ #
    #  Steam move detection
    # ------------------------------------------------------------------ #

    def _detect_steam(
        self,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Detect a steam move: 3+ books moving in the same direction within
        a 15-minute window.

        Parameters
        ----------
        history : list of dict
            Normalised, time-sorted odds snapshots.

        Returns
        -------
        dict  with ``detected`` (bool) and ``books`` (list of str)
        """
        if len(history) < 2:
            return {"detected": False, "books": []}

        # Build per-book movement events: (timestamp, book, delta_implied_home)
        events = []  # type: List[Tuple[datetime, str, float]]
        book_prev = {}  # type: Dict[str, Dict[str, Any]]

        for snap in history:
            book = snap["book"]
            implied_home = 1.0 / snap["home_odds"]

            if book in book_prev:
                prev_implied = 1.0 / book_prev[book]["home_odds"]
                delta = implied_home - prev_implied
                if abs(delta) >= self.STEAM_MIN_SHIFT:
                    events.append((snap["timestamp"], book, delta))

            book_prev[book] = snap

        if not events:
            return {"detected": False, "books": []}

        # Sliding window: for each event, look forward within STEAM_WINDOW
        window = timedelta(minutes=self.STEAM_WINDOW_MINUTES)
        best_books = set()  # type: set

        for i, (ts_i, book_i, delta_i) in enumerate(events):
            # Determine direction
            going_home = delta_i > 0
            cluster_books = {book_i}

            for j in range(i + 1, len(events)):
                ts_j, book_j, delta_j = events[j]
                if ts_j - ts_i > window:
                    break
                same_direction = (delta_j > 0) == going_home
                if same_direction and book_j not in cluster_books:
                    cluster_books.add(book_j)

            if len(cluster_books) >= self.STEAM_MIN_BOOKS:
                if len(cluster_books) > len(best_books):
                    best_books = cluster_books

        detected = len(best_books) >= self.STEAM_MIN_BOOKS
        return {
            "detected": detected,
            "books": sorted(best_books) if detected else [],
        }

    # ------------------------------------------------------------------ #
    #  Reverse line movement (RLM)
    # ------------------------------------------------------------------ #

    def _detect_rlm(
        self,
        history: List[Dict[str, Any]],
        prob_shift: float,
    ) -> bool:
        """Detect reverse line movement.

        RLM occurs when the line moves *against* the side receiving the
        majority of public bets.

        Parameters
        ----------
        history : list of dict
        prob_shift : float  – net implied-prob shift for the home side

        Returns
        -------
        bool
        """
        # Collect the most recent public-betting percentages
        public_home_pcts = [
            snap["public_home_pct"]
            for snap in history
            if snap.get("public_home_pct") is not None
        ]

        if not public_home_pcts:
            return False

        avg_public_home = sum(public_home_pcts) / len(public_home_pcts)

        # Public favours home
        if avg_public_home >= self.RLM_PUBLIC_THRESHOLD:
            # RLM if the line moved *towards away* (prob_shift < 0)
            return prob_shift < -0.005

        # Public favours away
        if avg_public_home <= (1.0 - self.RLM_PUBLIC_THRESHOLD):
            # RLM if the line moved *towards home* (prob_shift > 0)
            return prob_shift > 0.005

        return False

    # ------------------------------------------------------------------ #
    #  Velocity
    # ------------------------------------------------------------------ #

    def _compute_velocity(
        self,
        history: List[Dict[str, Any]],
    ) -> float:
        """Compute the implied-probability change per hour for the home side.

        Uses a simple linear approach: total shift / total hours elapsed.

        Returns
        -------
        float  – implied-probability change per hour (can be negative)
        """
        if len(history) < 2:
            return 0.0

        first = history[0]
        last = history[-1]

        elapsed = (last["timestamp"] - first["timestamp"]).total_seconds()
        if elapsed <= 0:
            return 0.0

        hours = elapsed / 3600.0
        open_implied = 1.0 / first["home_odds"]
        close_implied = 1.0 / last["home_odds"]

        return (close_implied - open_implied) / hours

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise_history(
        raw: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return a copy of *raw* with timestamps parsed and sorted."""
        normalised = []
        for entry in raw:
            rec = dict(entry)
            ts = rec["timestamp"]
            if isinstance(ts, str):
                # Accept ISO-8601 with or without microseconds / Z
                ts = ts.replace("Z", "+00:00")
                try:
                    rec["timestamp"] = datetime.fromisoformat(ts)
                except ValueError:
                    # Fallback: try strptime for common formats
                    rec["timestamp"] = datetime.strptime(
                        ts[:19], "%Y-%m-%dT%H:%M:%S"
                    )
            normalised.append(rec)

        normalised.sort(key=lambda x: x["timestamp"])
        return normalised

    @staticmethod
    def _empty_result(game_id: str) -> Dict[str, Any]:
        return {
            "game_id": game_id,
            "has_movement": False,
            "prob_shift": 0.0,
            "direction": "neutral",
            "magnitude": 0.0,
            "is_steam": False,
            "steam_books": [],
            "is_rlm": False,
            "velocity": 0.0,
        }
