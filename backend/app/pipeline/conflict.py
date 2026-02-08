"""
Data conflict resolution for the EDGE AI sports betting system.

When the same data point is reported differently by multiple sources,
DataConflictResolver arbitrates using source reliability weights,
weighted averaging (numeric), and priority selection (categorical).
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class DataConflictResolver:
    """Resolve conflicting data from heterogeneous sports-data sources."""

    # ------------------------------------------------------------------ #
    #  Source reliability ratings (0-10, higher = more trustworthy)
    # ------------------------------------------------------------------ #

    SOURCE_RELIABILITY = {
        # Official feeds
        "official_api": 10.0,
        "nba_api": 9.5,
        "nfl_api": 9.5,
        "mlb_api": 9.5,
        "nhl_api": 9.5,
        "mls_api": 9.0,
        "ncaa_api": 8.5,
        # Premium data providers
        "sportradar": 9.0,
        "stats_perform": 8.5,
        "genius_sports": 8.5,
        "second_spectrum": 8.5,
        # Odds providers
        "pinnacle": 9.0,
        "don_best": 8.5,
        "the_odds_api": 7.5,
        "odds_api": 7.5,
        # Reputable aggregators
        "espn": 7.5,
        "basketball_reference": 8.0,
        "baseball_reference": 8.0,
        "pro_football_reference": 8.0,
        "hockey_reference": 8.0,
        "fbref": 8.0,
        "fangraphs": 8.0,
        "covers": 7.0,
        # Community / scraped
        "rotowire": 6.5,
        "rotoworld": 6.5,
        "numberfire": 6.0,
        "fantasypros": 6.0,
        "action_network": 7.0,
        "oddshark": 6.5,
        # Social / least reliable
        "twitter": 4.0,
        "reddit": 3.5,
        "user_input": 5.0,
        "unknown": 2.0,
    }

    def __init__(
        self,
        custom_reliability: Optional[Dict[str, float]] = None,
    ) -> None:
        """Optionally merge in custom source reliability overrides."""
        self.reliability = dict(self.SOURCE_RELIABILITY)
        if custom_reliability:
            self.reliability.update(custom_reliability)

    # ------------------------------------------------------------------ #
    #  Numeric conflict resolution
    # ------------------------------------------------------------------ #

    def resolve_numeric(
        self,
        values: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Resolve conflicting numeric values via reliability-weighted average.

        Parameters
        ----------
        values : list of dict
            Each entry must contain:
                - ``source`` : str  (key into SOURCE_RELIABILITY)
                - ``value``  : float

        Returns
        -------
        dict
            resolved_value : float
            confidence     : float  (0-1, based on agreement among sources)
            sources_used   : int
            is_anomalous   : bool   (True if any single value is an outlier)
            details        : list   (per-source weight and value)
        """
        if not values:
            raise ValueError("values list must not be empty")

        total_weight = 0.0
        weighted_sum = 0.0
        details = []  # type: List[Dict[str, Any]]

        for entry in values:
            source = entry["source"].lower()
            val = float(entry["value"])
            w = self.reliability.get(source, self.reliability["unknown"])

            weighted_sum += w * val
            total_weight += w
            details.append({
                "source": source,
                "value": val,
                "weight": w,
            })

        if total_weight <= 0:
            resolved = sum(e["value"] for e in details) / len(details)
        else:
            resolved = weighted_sum / total_weight

        # Confidence: inverse of coefficient of variation among values
        raw_values = [entry["value"] for entry in values]
        confidence = self._agreement_confidence(raw_values)

        # Anomaly check
        is_anomalous = self.detect_anomaly(raw_values, threshold=2.5)

        return {
            "resolved_value": round(resolved, 6),
            "confidence": round(confidence, 6),
            "sources_used": len(values),
            "is_anomalous": is_anomalous,
            "details": details,
        }

    # ------------------------------------------------------------------ #
    #  Categorical conflict resolution
    # ------------------------------------------------------------------ #

    def resolve_categorical(
        self,
        values: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Resolve conflicting categorical values by highest-priority source.

        Parameters
        ----------
        values : list of dict
            Each entry must contain:
                - ``source`` : str
                - ``value``  : any (str, bool, etc.)

        Returns
        -------
        dict
            resolved_value : any
            source         : str  (the winning source)
            confidence     : float (1.0 if all agree, scaled down otherwise)
            sources_used   : int
            all_values     : list  (for transparency)
        """
        if not values:
            raise ValueError("values list must not be empty")

        # Sort by reliability descending
        ranked = sorted(
            values,
            key=lambda e: self.reliability.get(
                e["source"].lower(), self.reliability["unknown"]
            ),
            reverse=True,
        )

        best = ranked[0]
        best_source = best["source"].lower()
        best_value = best["value"]

        # Confidence: fraction of sources that agree with the winner
        agree_count = sum(
            1 for e in values if e["value"] == best_value
        )
        confidence = agree_count / len(values)

        return {
            "resolved_value": best_value,
            "source": best_source,
            "confidence": round(confidence, 6),
            "sources_used": len(values),
            "all_values": [
                {"source": e["source"].lower(), "value": e["value"]}
                for e in ranked
            ],
        }

    # ------------------------------------------------------------------ #
    #  Anomaly detection
    # ------------------------------------------------------------------ #

    def detect_anomaly(
        self,
        values: List[float],
        threshold: float = 2.5,
    ) -> bool:
        """Detect whether any value in *values* is an outlier.

        Uses a z-score approach: if any value deviates more than
        *threshold* standard deviations from the mean, it is anomalous.

        Parameters
        ----------
        values : list of float
        threshold : float  (default 2.5 standard deviations)

        Returns
        -------
        bool  – True if at least one outlier detected.
        """
        if len(values) < 2:
            return False

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std == 0.0:
            return False

        for v in values:
            z = abs(v - mean) / std
            if z > threshold:
                return True

        return False

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _agreement_confidence(values: List[float]) -> float:
        """Compute a 0-1 confidence score reflecting how much the values agree.

        Uses 1 / (1 + CV) where CV is the coefficient of variation.
        Perfect agreement returns 1.0.
        """
        if len(values) < 2:
            return 1.0

        mean = sum(values) / len(values)
        if mean == 0:
            return 1.0

        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)
        cv = abs(std / mean)

        return 1.0 / (1.0 + cv)
