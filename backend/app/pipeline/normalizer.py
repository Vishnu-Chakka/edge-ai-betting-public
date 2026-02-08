"""
Odds and statistics normalisation for the EDGE AI sports betting system.

OddsNormalizer  – odds format conversion, no-vig probabilities, consensus lines.
StatsNormalizer – pace-adjusted, per-minute, and z-score standardisation.
"""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ====================================================================== #
#  OddsNormalizer
# ====================================================================== #

class OddsNormalizer:
    """Convert between odds formats, remove vig, and compute consensus lines."""

    # Sharpness weights for sportsbooks (higher = sharper / more respected).
    # Values are relative; they will be normalised at runtime.
    SHARPNESS_WEIGHTS = {
        # Sharp / market-making books
        "pinnacle": 10.0,
        "circa": 9.0,
        "bookmaker": 8.5,
        "betcris": 8.0,
        "5dimes": 7.5,
        "heritage": 7.0,
        # Mid-tier
        "bet365": 6.0,
        "betonline": 5.5,
        "bovada": 5.0,
        "williamhill": 5.0,
        "betway": 4.5,
        "unibet": 4.5,
        # Retail / recreational
        "draftkings": 4.0,
        "fanduel": 4.0,
        "caesars": 3.5,
        "betmgm": 3.5,
        "pointsbet": 3.0,
        "barstool": 3.0,
        "wynn": 3.0,
        "betrivers": 3.0,
        "foxbet": 2.5,
        "superbook": 5.0,
    }

    # ------------------------------------------------------------------ #
    #  Odds conversions
    # ------------------------------------------------------------------ #

    @staticmethod
    def american_to_decimal(american: int) -> float:
        """Convert American odds to decimal odds.

        +150 -> 2.50
        -200 -> 1.50
        """
        if american > 0:
            return 1.0 + american / 100.0
        elif american < 0:
            return 1.0 + 100.0 / abs(american)
        else:
            raise ValueError("American odds cannot be zero")

    @staticmethod
    def decimal_to_implied(decimal_odds: float) -> float:
        """Convert decimal odds to implied probability (0-1).

        2.00 -> 0.50
        """
        if decimal_odds <= 0:
            raise ValueError("Decimal odds must be positive")
        return 1.0 / decimal_odds

    @staticmethod
    def implied_to_american(implied: float) -> int:
        """Convert implied probability (0-1) to American odds.

        0.50 -> -100
        0.40 -> +150
        0.667 -> -200
        """
        if implied <= 0 or implied >= 1:
            raise ValueError("Implied probability must be between 0 and 1 exclusive")

        if implied >= 0.50:
            # Favourite: negative American
            return int(round(-100.0 * implied / (1.0 - implied)))
        else:
            # Underdog: positive American
            return int(round(100.0 * (1.0 - implied) / implied))

    # ------------------------------------------------------------------ #
    #  No-vig probabilities
    # ------------------------------------------------------------------ #

    def compute_no_vig_probs(
        self,
        home_dec: float,
        away_dec: float,
        draw_dec: Optional[float] = None,
    ) -> Dict[str, float]:
        """Remove the vig using the multiplicative (normalisation) method.

        Parameters
        ----------
        home_dec : float  – decimal odds for home
        away_dec : float  – decimal odds for away
        draw_dec : float, optional – decimal odds for draw (three-way markets)

        Returns
        -------
        dict with ``home``, ``away``, and optionally ``draw`` fair probabilities,
        plus ``overround`` (the total implied prob before normalisation).
        """
        implied_home = 1.0 / home_dec
        implied_away = 1.0 / away_dec

        if draw_dec is not None and draw_dec > 0:
            implied_draw = 1.0 / draw_dec
            overround = implied_home + implied_away + implied_draw
            return {
                "home": round(implied_home / overround, 6),
                "away": round(implied_away / overround, 6),
                "draw": round(implied_draw / overround, 6),
                "overround": round(overround, 6),
            }
        else:
            overround = implied_home + implied_away
            return {
                "home": round(implied_home / overround, 6),
                "away": round(implied_away / overround, 6),
                "overround": round(overround, 6),
            }

    # ------------------------------------------------------------------ #
    #  Consensus line
    # ------------------------------------------------------------------ #

    def compute_consensus_line(
        self,
        snapshots: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute a sharpness-weighted consensus implied probability.

        Parameters
        ----------
        snapshots : list of dict
            Each snapshot must contain:
                - ``book``      : str  (sportsbook name, lowercase)
                - ``home_odds`` : float (decimal odds for home)
                - ``away_odds`` : float (decimal odds for away)
            Optional:
                - ``draw_odds`` : float

        Returns
        -------
        dict
            consensus_home : float  – weighted no-vig home probability
            consensus_away : float  – weighted no-vig away probability
            consensus_draw : float or None
            num_books      : int
            sharpest_book  : str
            overround_avg  : float
        """
        if not snapshots:
            raise ValueError("snapshots list must not be empty")

        weighted_home = 0.0
        weighted_away = 0.0
        weighted_draw = 0.0
        total_weight = 0.0
        has_draw = False
        overrounds = []  # type: List[float]
        sharpest_book = ""
        sharpest_weight = -1.0

        for snap in snapshots:
            book = snap["book"].lower()
            home_dec = snap["home_odds"]
            away_dec = snap["away_odds"]
            draw_dec = snap.get("draw_odds")

            weight = self.SHARPNESS_WEIGHTS.get(book, 2.0)

            # Track sharpest book
            if weight > sharpest_weight:
                sharpest_weight = weight
                sharpest_book = book

            # No-vig probabilities for this book
            nv = self.compute_no_vig_probs(home_dec, away_dec, draw_dec)
            overrounds.append(nv["overround"])

            weighted_home += weight * nv["home"]
            weighted_away += weight * nv["away"]

            if "draw" in nv:
                has_draw = True
                weighted_draw += weight * nv["draw"]

            total_weight += weight

        if total_weight <= 0:
            raise ValueError("Total sharpness weight is zero; check book names")

        consensus_home = weighted_home / total_weight
        consensus_away = weighted_away / total_weight
        overround_avg = sum(overrounds) / len(overrounds)

        result = {
            "consensus_home": round(consensus_home, 6),
            "consensus_away": round(consensus_away, 6),
            "consensus_draw": None,
            "num_books": len(snapshots),
            "sharpest_book": sharpest_book,
            "overround_avg": round(overround_avg, 6),
        }  # type: Dict[str, Any]

        if has_draw:
            result["consensus_draw"] = round(weighted_draw / total_weight, 6)

        return result


# ====================================================================== #
#  StatsNormalizer
# ====================================================================== #

class StatsNormalizer:
    """Normalize raw team / player stats for cross-team comparisons."""

    # ------------------------------------------------------------------ #
    #  Pace-adjusted stats (NBA)
    # ------------------------------------------------------------------ #

    @staticmethod
    def per_100_possessions(
        stats: Dict[str, float],
        pace: float,
    ) -> Dict[str, float]:
        """Normalize counting stats to per-100-possessions for NBA.

        Parameters
        ----------
        stats : dict
            Raw counting stats for the team / player.  Keys are stat names,
            values are totals (or per-game averages).
        pace : float
            Team pace (possessions per 48 minutes).

        Returns
        -------
        dict  – Same keys with values scaled to per-100-possessions.
        """
        if pace <= 0:
            raise ValueError("pace must be positive")

        factor = 100.0 / pace
        return {
            k: round(v * factor, 4)
            for k, v in stats.items()
        }

    # ------------------------------------------------------------------ #
    #  Per-90-minute stats (soccer) — expressed per 600 minutes (approx.
    #  every ~6.67 full matches) for smoother normalisation.
    # ------------------------------------------------------------------ #

    @staticmethod
    def per_600_minutes(
        stats: Dict[str, float],
    ) -> Dict[str, float]:
        """Normalize soccer stats to per-600-minutes rate.

        Each stat dict must include a ``minutes_played`` key.  All other
        numeric values are scaled.

        Parameters
        ----------
        stats : dict
            Must include ``minutes_played`` (int/float).  Other keys are
            counting stats.

        Returns
        -------
        dict  – Scaled stats (``minutes_played`` key removed).
        """
        minutes = stats.get("minutes_played", 0)
        if minutes <= 0:
            raise ValueError("minutes_played must be positive")

        factor = 600.0 / minutes
        return {
            k: round(v * factor, 4)
            for k, v in stats.items()
            if k != "minutes_played"
        }

    # ------------------------------------------------------------------ #
    #  Z-score standardisation
    # ------------------------------------------------------------------ #

    @staticmethod
    def z_score_standardize(
        values: List[float],
    ) -> List[float]:
        """Return z-score standardised values.

        Parameters
        ----------
        values : list of float

        Returns
        -------
        list of float  – z-scores (mean = 0, std = 1).
        """
        n = len(values)
        if n < 2:
            return [0.0] * n

        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(variance) if variance > 0 else 1.0

        return [round((v - mean) / std, 6) for v in values]
