"""
Expected Value calculator and bet ranking engine for the EDGE AI sports betting system.

Provides EV computation, Kelly criterion stake sizing, bet ranking with composite
scoring, edge classification, and odds format conversions.
"""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EVCalculator:
    """Core expected-value calculator with Kelly sizing and bet ranking."""

    # ------------------------------------------------------------------ #
    #  Odds format helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Convert American odds to decimal odds.

        +150 -> 2.50   (risk 100 to win 150, total return 250)
        -150 -> 1.6667  (risk 150 to win 100, total return ~166.67)
        """
        if american_odds > 0:
            return 1.0 + (american_odds / 100.0)
        elif american_odds < 0:
            return 1.0 + (100.0 / abs(american_odds))
        else:
            raise ValueError("American odds cannot be zero")

    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        """Convert decimal odds to American odds.

        2.50 -> +150
        1.6667 -> -150
        """
        if decimal_odds < 1.0:
            raise ValueError("Decimal odds must be >= 1.0")
        if decimal_odds == 1.0:
            return 0
        if decimal_odds >= 2.0:
            return int(round((decimal_odds - 1.0) * 100))
        else:
            return int(round(-100.0 / (decimal_odds - 1.0)))

    # ------------------------------------------------------------------ #
    #  Core EV calculation
    # ------------------------------------------------------------------ #

    def calculate_ev(
        self,
        fair_prob: float,
        decimal_odds: float,
    ) -> Dict[str, Any]:
        """Calculate expected value and edge for a single bet.

        Parameters
        ----------
        fair_prob : float
            Model-estimated fair probability of the outcome (0-1).
        decimal_odds : float
            Decimal odds offered by the sportsbook (e.g. 2.10).

        Returns
        -------
        dict
            ev_pct        – expected value as a percentage of stake
            edge_pct      – edge = fair_prob - implied_prob
            fair_prob     – echo back
            implied_prob  – 1 / decimal_odds
            is_positive_ev – bool
        """
        if not (0.0 < fair_prob < 1.0):
            raise ValueError("fair_prob must be between 0 and 1 exclusive")
        if decimal_odds <= 1.0:
            raise ValueError("decimal_odds must be > 1.0")

        implied_prob = 1.0 / decimal_odds
        ev_pct = (fair_prob * decimal_odds - 1.0) * 100.0
        edge_pct = (fair_prob - implied_prob) * 100.0

        return {
            "ev_pct": round(ev_pct, 4),
            "edge_pct": round(edge_pct, 4),
            "fair_prob": round(fair_prob, 6),
            "implied_prob": round(implied_prob, 6),
            "is_positive_ev": ev_pct > 0.0,
        }

    # ------------------------------------------------------------------ #
    #  Kelly criterion stake sizing
    # ------------------------------------------------------------------ #

    def compute_kelly_stake(
        self,
        fair_prob: float,
        decimal_odds: float,
        bankroll: float,
        kelly_fraction: float = 0.25,
        max_bet_pct: float = 0.05,
    ) -> Dict[str, float]:
        """Compute Kelly criterion stake.

        Parameters
        ----------
        fair_prob : float
            Model-estimated fair probability (0-1).
        decimal_odds : float
            Decimal odds offered.
        bankroll : float
            Current bankroll in currency units.
        kelly_fraction : float
            Fractional Kelly multiplier (default quarter-Kelly = 0.25).
        max_bet_pct : float
            Hard cap on bet size as fraction of bankroll (default 5 %).

        Returns
        -------
        dict
            full_kelly         – full Kelly fraction of bankroll (can be negative)
            adjusted           – fractional Kelly fraction, clamped to [0, max_bet_pct]
            recommended_amount – dollar amount to wager
            recommended_units  – wager expressed in units (1 unit = 1 % of bankroll)
        """
        if bankroll <= 0:
            raise ValueError("bankroll must be positive")

        b = decimal_odds - 1.0  # net payout per unit staked
        q = 1.0 - fair_prob

        if b <= 0:
            full_kelly = 0.0
        else:
            # Kelly formula: f* = (bp - q) / b
            full_kelly = (b * fair_prob - q) / b

        adjusted = full_kelly * kelly_fraction
        adjusted = max(0.0, min(adjusted, max_bet_pct))

        recommended_amount = round(adjusted * bankroll, 2)
        unit_size = bankroll * 0.01  # 1 unit = 1 % of bankroll
        recommended_units = round(adjusted * bankroll / unit_size, 2) if unit_size > 0 else 0.0

        return {
            "full_kelly": round(full_kelly, 6),
            "adjusted": round(adjusted, 6),
            "recommended_amount": recommended_amount,
            "recommended_units": recommended_units,
        }

    # ------------------------------------------------------------------ #
    #  Bet ranking
    # ------------------------------------------------------------------ #

    def rank_bets(
        self,
        candidates: List[Dict[str, Any]],
        user_prefs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Rank a list of candidate bets by composite score.

        Composite score weights
        -----------------------
        * EV percentage           : 50 %
        * Model confidence        : 25 %
        * Uncertainty penalty      : 15 % (inverted – lower uncertainty = higher score)
        * Correlation penalty      : 10 % (inverted – less correlation = higher score)

        Each candidate dict is expected to contain at least:
            ev_pct, confidence, uncertainty

        Optional fields: game_id, team, sport, existing_bets (list).

        Parameters
        ----------
        candidates : list of dict
        user_prefs : dict, optional
            May contain ``min_ev``, ``max_uncertainty``, ``sports_filter``,
            ``existing_bets`` (for correlation calculation).

        Returns
        -------
        list of dict  – sorted descending by ``composite_score``, each
        enriched with ``composite_score``, ``confidence_tier``, ``rank``.
        """
        if user_prefs is None:
            user_prefs = {}

        min_ev = user_prefs.get("min_ev", 0.0)
        max_uncertainty = user_prefs.get("max_uncertainty", 1.0)
        sports_filter = user_prefs.get("sports_filter", None)
        existing_bets = user_prefs.get("existing_bets", [])

        scored = []  # type: List[Dict[str, Any]]

        for bet in candidates:
            ev_pct = bet.get("ev_pct", 0.0)
            confidence = bet.get("confidence", 0.5)
            uncertainty = bet.get("uncertainty", 0.2)
            sport = bet.get("sport", "")

            # ---- Pre-filters ----
            if ev_pct < min_ev:
                continue
            if uncertainty > max_uncertainty:
                continue
            if sports_filter and sport not in sports_filter:
                continue

            # ---- Normalise components to 0-1 ----
            # EV: cap contribution at 30 % EV for normalisation purposes
            ev_norm = min(ev_pct / 30.0, 1.0) if ev_pct > 0 else 0.0

            # Confidence already in 0-1
            conf_norm = max(0.0, min(confidence, 1.0))

            # Uncertainty penalty (lower is better) – invert
            unc_norm = 1.0 - max(0.0, min(uncertainty, 1.0))

            # Correlation penalty
            corr_penalty = self._compute_correlation_penalty(bet, existing_bets)
            corr_norm = 1.0 - max(0.0, min(corr_penalty, 1.0))

            composite = (
                0.50 * ev_norm
                + 0.25 * conf_norm
                + 0.15 * unc_norm
                + 0.10 * corr_norm
            )

            enriched = dict(bet)
            enriched["composite_score"] = round(composite, 6)
            enriched["confidence_tier"] = self._confidence_tier(confidence)
            scored.append(enriched)

        # Sort descending by composite score
        scored.sort(key=lambda x: x["composite_score"], reverse=True)

        for idx, item in enumerate(scored, start=1):
            item["rank"] = idx

        return scored

    # ------------------------------------------------------------------ #
    #  Edge classification
    # ------------------------------------------------------------------ #

    @staticmethod
    def classify_edge_type(bet: Dict[str, Any]) -> str:
        """Classify the source / type of the detected edge.

        Heuristic priority:
        1. steam_move    – steam move detected in line_movement data
        2. sharp_signal  – sharp book movement without broad steam
        3. closing_line   – edge derived from closing-line-value analysis
        4. model_driven  – default: edge comes from model disagreement with market

        Parameters
        ----------
        bet : dict
            Must contain at least some of: ``is_steam``, ``is_rlm``,
            ``sharp_books_moved``, ``clv_edge``, ``model_edge``.

        Returns
        -------
        str  – one of "steam_move", "sharp_signal", "closing_line", "model_driven"
        """
        if bet.get("is_steam", False):
            return "steam_move"

        sharp_moved = bet.get("sharp_books_moved", 0)
        is_rlm = bet.get("is_rlm", False)
        if sharp_moved >= 2 or is_rlm:
            return "sharp_signal"

        clv_edge = bet.get("clv_edge", 0.0)
        if clv_edge > 0.0:
            return "closing_line"

        return "model_driven"

    # ------------------------------------------------------------------ #
    #  Correlation penalty (private)
    # ------------------------------------------------------------------ #

    def _compute_correlation_penalty(
        self,
        new_bet: Dict[str, Any],
        existing: List[Dict[str, Any]],
    ) -> float:
        """Compute a 0-1 correlation penalty for *new_bet* given *existing* bets.

        Rules
        -----
        * Same game_id → +0.35 per overlapping bet
        * Same team    → +0.20 per overlapping bet
        * Same sport on same day → +0.05 per overlapping bet

        The penalty is clamped to [0, 1].
        """
        if not existing:
            return 0.0

        penalty = 0.0
        new_game = new_bet.get("game_id")
        new_team = new_bet.get("team")
        new_sport = new_bet.get("sport")
        new_date = new_bet.get("game_date")

        for ex in existing:
            if new_game and ex.get("game_id") == new_game:
                penalty += 0.35
            elif new_team and (
                ex.get("team") == new_team
                or ex.get("opponent") == new_team
                or new_team in (ex.get("home_team", ""), ex.get("away_team", ""))
            ):
                penalty += 0.20
            elif (
                new_sport
                and ex.get("sport") == new_sport
                and new_date
                and ex.get("game_date") == new_date
            ):
                penalty += 0.05

        return min(penalty, 1.0)

    # ------------------------------------------------------------------ #
    #  Confidence tier helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def _confidence_tier(confidence: float) -> str:
        """Map a 0-1 confidence value to a letter tier.

        A : >= 0.75
        B : >= 0.55
        C : < 0.55
        """
        if confidence >= 0.75:
            return "A"
        elif confidence >= 0.55:
            return "B"
        else:
            return "C"
