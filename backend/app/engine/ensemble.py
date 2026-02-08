"""
Model ensemble module for the EDGE AI sports betting system.

Combines Elo, ML, Bayesian, and market-consensus models with sport-specific
weights to produce fair probabilities and full bet recommendations.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from backend.app.engine.ev_calculator import EVCalculator
from backend.app.engine.line_movement import LineMovementAnalyzer

logger = logging.getLogger(__name__)


class ModelEnsemble:
    """Weighted ensemble of probability models with sport-specific configs."""

    # ------------------------------------------------------------------ #
    #  Sport-specific model weights
    # ------------------------------------------------------------------ #

    SPORT_WEIGHTS = {
        "nba": {
            "ml": 0.40,
            "market": 0.25,
            "bayesian": 0.20,
            "elo": 0.15,
        },
        "nfl": {
            "ml": 0.35,
            "market": 0.30,
            "bayesian": 0.15,
            "elo": 0.20,
        },
        "mlb": {
            "ml": 0.35,
            "market": 0.25,
            "bayesian": 0.25,
            "elo": 0.15,
        },
        "nhl": {
            "ml": 0.30,
            "market": 0.30,
            "bayesian": 0.25,
            "elo": 0.15,
        },
        "soccer": {
            "ml": 0.30,
            "market": 0.35,
            "bayesian": 0.20,
            "elo": 0.15,
        },
        "ncaab": {
            "ml": 0.30,
            "market": 0.20,
            "bayesian": 0.20,
            "elo": 0.30,
        },
        "ncaaf": {
            "ml": 0.25,
            "market": 0.25,
            "bayesian": 0.20,
            "elo": 0.30,
        },
        # Fallback / default
        "default": {
            "ml": 0.35,
            "market": 0.25,
            "bayesian": 0.20,
            "elo": 0.20,
        },
    }

    def __init__(self) -> None:
        self.ev_calculator = EVCalculator()
        self.line_analyzer = LineMovementAnalyzer()

    # ------------------------------------------------------------------ #
    #  Fair probability computation
    # ------------------------------------------------------------------ #

    def compute_fair_probability(
        self,
        game_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute the ensemble fair probability for the home side.

        Parameters
        ----------
        game_data : dict
            Must contain:
                - ``sport``       : str (e.g. "nba")
                - ``elo_prob``    : float, Elo-derived home-win probability
                - ``ml_prob``     : float, ML-model home-win probability
                - ``bayesian_prob``: float, Bayesian-model home-win probability
                - ``market_prob`` : float, no-vig market-implied home-win probability
            Optional:
                - ``model_confidences`` : dict mapping model name to 0-1 confidence
                - ``features``          : dict of feature importances from the ML model

        Returns
        -------
        dict
            fair_prob       : float  – weighted ensemble probability
            model_breakdown : dict   – per-model contribution
            top_features    : list   – top-5 features from ML model
            uncertainty     : float  – 0-1 measure of model disagreement
        """
        sport = game_data.get("sport", "default").lower()
        weights = dict(self.SPORT_WEIGHTS.get(sport, self.SPORT_WEIGHTS["default"]))

        model_probs = {
            "elo": game_data.get("elo_prob", 0.5),
            "ml": game_data.get("ml_prob", 0.5),
            "bayesian": game_data.get("bayesian_prob", 0.5),
            "market": game_data.get("market_prob", 0.5),
        }

        # Optional: adjust weights by model confidence
        confidences = game_data.get("model_confidences", {})
        if confidences:
            weights = self._adjust_weights_by_confidence(weights, confidences)

        # Weighted combination
        fair_prob = sum(
            weights[model] * model_probs[model] for model in weights
        )

        # Clamp to valid probability range
        fair_prob = max(0.01, min(0.99, fair_prob))

        # Per-model contribution breakdown
        model_breakdown = {}
        for model in weights:
            model_breakdown[model] = {
                "weight": round(weights[model], 4),
                "prob": round(model_probs[model], 6),
                "contribution": round(weights[model] * model_probs[model], 6),
            }

        # Uncertainty: standard deviation of model probabilities
        probs_list = list(model_probs.values())
        mean_prob = sum(probs_list) / len(probs_list)
        variance = sum((p - mean_prob) ** 2 for p in probs_list) / len(probs_list)
        uncertainty = math.sqrt(variance)
        # Normalise to 0-1 range (max possible std dev for probs is 0.5)
        uncertainty_norm = min(uncertainty / 0.5, 1.0)

        # Top features from ML model
        features = game_data.get("features", {})
        top_features = self._extract_top_features(features, n=5)

        return {
            "fair_prob": round(fair_prob, 6),
            "model_breakdown": model_breakdown,
            "top_features": top_features,
            "uncertainty": round(uncertainty_norm, 6),
        }

    # ------------------------------------------------------------------ #
    #  Full recommendation
    # ------------------------------------------------------------------ #

    def generate_recommendation(
        self,
        game_data: Dict[str, Any],
        odds_data: Dict[str, Any],
        user_prefs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a complete bet recommendation.

        Parameters
        ----------
        game_data : dict
            Same as ``compute_fair_probability`` input, plus:
                - ``game_id``    : str
                - ``home_team``  : str
                - ``away_team``  : str
                - ``odds_history`` : list of snapshot dicts (for line movement)
        odds_data : dict
            Must contain:
                - ``home_odds``  : decimal odds for home
                - ``away_odds``  : decimal odds for away
            Optional:
                - ``draw_odds``  : decimal odds for draw
        user_prefs : dict, optional
            - ``bankroll``       : float (default 1000)
            - ``kelly_fraction`` : float (default 0.25)
            - ``max_bet_pct``    : float (default 0.05)
            - ``min_ev``         : float (default 2.0)
            - ``min_confidence`` : str   (default "C")

        Returns
        -------
        dict  – Full recommendation with EV, Kelly sizing, edge classification,
                confidence tier, line movement, and model breakdown.
        """
        if user_prefs is None:
            user_prefs = {}

        bankroll = user_prefs.get("bankroll", 1000.0)
        kelly_fraction = user_prefs.get("kelly_fraction", 0.25)
        max_bet_pct = user_prefs.get("max_bet_pct", 0.05)
        min_ev = user_prefs.get("min_ev", 2.0)
        min_confidence = user_prefs.get("min_confidence", "C")

        # 1. Compute fair probability
        ensemble_result = self.compute_fair_probability(game_data)
        fair_prob = ensemble_result["fair_prob"]
        uncertainty = ensemble_result["uncertainty"]

        # 2. Determine which side to bet
        home_odds = odds_data["home_odds"]
        away_odds = odds_data["away_odds"]

        # EV for home bet
        ev_home = self.ev_calculator.calculate_ev(fair_prob, home_odds)
        # EV for away bet
        away_fair_prob = 1.0 - fair_prob
        ev_away = self.ev_calculator.calculate_ev(away_fair_prob, away_odds)

        # Pick the better side
        if ev_home["ev_pct"] >= ev_away["ev_pct"]:
            chosen_side = "home"
            chosen_ev = ev_home
            chosen_odds = home_odds
            chosen_prob = fair_prob
            chosen_team = game_data.get("home_team", "Home")
        else:
            chosen_side = "away"
            chosen_ev = ev_away
            chosen_odds = away_odds
            chosen_prob = away_fair_prob
            chosen_team = game_data.get("away_team", "Away")

        # 3. Kelly sizing
        kelly = self.ev_calculator.compute_kelly_stake(
            fair_prob=chosen_prob,
            decimal_odds=chosen_odds,
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
            max_bet_pct=max_bet_pct,
        )

        # 4. Line movement analysis
        odds_history = game_data.get("odds_history", [])
        game_id = game_data.get("game_id", "unknown")
        line_movement = self.line_analyzer.analyze_movement(game_id, odds_history)

        # 5. Confidence score (inverse of uncertainty, boosted by EV)
        confidence = self._compute_confidence(
            ev_pct=chosen_ev["ev_pct"],
            uncertainty=uncertainty,
            line_movement=line_movement,
        )
        confidence_tier = EVCalculator._confidence_tier(confidence)

        # 6. Edge classification
        edge_info = {
            "is_steam": line_movement.get("is_steam", False),
            "is_rlm": line_movement.get("is_rlm", False),
            "sharp_books_moved": len(line_movement.get("steam_books", [])),
            "clv_edge": game_data.get("clv_edge", 0.0),
            "model_edge": chosen_ev["edge_pct"],
        }
        edge_type = EVCalculator.classify_edge_type(edge_info)

        # 7. Apply minimum-EV and minimum-confidence filters
        passes_filters = True
        rejection_reasons = []  # type: List[str]

        if chosen_ev["ev_pct"] < min_ev:
            passes_filters = False
            rejection_reasons.append(
                "EV {:.2f}% below minimum {:.2f}%".format(chosen_ev["ev_pct"], min_ev)
            )

        tier_order = {"A": 3, "B": 2, "C": 1}
        if tier_order.get(confidence_tier, 0) < tier_order.get(min_confidence, 0):
            passes_filters = False
            rejection_reasons.append(
                "Confidence tier {} below minimum {}".format(confidence_tier, min_confidence)
            )

        return {
            "game_id": game_id,
            "sport": game_data.get("sport", "unknown"),
            "home_team": game_data.get("home_team", ""),
            "away_team": game_data.get("away_team", ""),
            "recommended_side": chosen_side,
            "recommended_team": chosen_team,
            "decimal_odds": chosen_odds,
            "american_odds": EVCalculator.decimal_to_american(chosen_odds),
            "ev": chosen_ev,
            "kelly": kelly,
            "fair_prob": fair_prob,
            "model_breakdown": ensemble_result["model_breakdown"],
            "top_features": ensemble_result["top_features"],
            "uncertainty": uncertainty,
            "confidence": round(confidence, 6),
            "confidence_tier": confidence_tier,
            "edge_type": edge_type,
            "line_movement": line_movement,
            "passes_filters": passes_filters,
            "rejection_reasons": rejection_reasons,
        }

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _adjust_weights_by_confidence(
        base_weights: Dict[str, float],
        confidences: Dict[str, float],
    ) -> Dict[str, float]:
        """Scale model weights by each model's reported confidence, then
        re-normalise so they still sum to 1.0.

        Parameters
        ----------
        base_weights : dict  – model_name -> base weight
        confidences  : dict  – model_name -> 0-1 confidence

        Returns
        -------
        dict  – adjusted weights summing to 1.0
        """
        adjusted = {}
        for model, w in base_weights.items():
            c = confidences.get(model, 1.0)
            adjusted[model] = w * c

        total = sum(adjusted.values())
        if total <= 0:
            return dict(base_weights)

        return {m: v / total for m, v in adjusted.items()}

    @staticmethod
    def _extract_top_features(
        features: Dict[str, float],
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the top-*n* features sorted by absolute importance."""
        if not features:
            return []

        ranked = sorted(
            features.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )

        return [
            {"feature": name, "importance": round(imp, 6)}
            for name, imp in ranked[:n]
        ]

    @staticmethod
    def _compute_confidence(
        ev_pct: float,
        uncertainty: float,
        line_movement: Dict[str, Any],
    ) -> float:
        """Compute an overall confidence score in [0, 1].

        Factors:
        * Base confidence from inverse uncertainty (40 %)
        * EV magnitude (30 %)
        * Line-movement confirmation (30 %)
        """
        # Inverse uncertainty: 1.0 = no disagreement among models
        inv_unc = 1.0 - min(uncertainty, 1.0)

        # EV contribution: cap at 15 % EV for normalisation
        ev_norm = min(max(ev_pct, 0.0) / 15.0, 1.0)

        # Line-movement confirmation bonus
        lm_score = 0.0
        if line_movement.get("is_steam", False):
            lm_score += 0.4
        if line_movement.get("is_rlm", False):
            lm_score += 0.4
        if line_movement.get("has_movement", False):
            lm_score += 0.2
        lm_score = min(lm_score, 1.0)

        confidence = 0.40 * inv_unc + 0.30 * ev_norm + 0.30 * lm_score
        return max(0.0, min(confidence, 1.0))
