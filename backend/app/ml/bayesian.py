"""
Bayesian Probability Updater Using Beta Distributions.

Maintains per-team Beta(alpha, beta) posteriors that begin from an Elo-based
prior and are updated after every game result or piece of soft evidence
(injuries, travel fatigue, etc.).

Monte Carlo sampling from the posteriors produces matchup-level win
probabilities with full uncertainty quantification.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# Default number of Monte Carlo samples for matchup simulation
DEFAULT_MC_SAMPLES: int = 50_000

# Default RNG seed for reproducibility (can be overridden)
DEFAULT_SEED: int = 42


class BetaTeamState:
    """Container for a single team's Beta-distribution posterior.

    Attributes
    ----------
    alpha : float
        "Win-equivalent" shape parameter.
    beta : float
        "Loss-equivalent" shape parameter.
    """

    __slots__ = ("alpha", "beta")

    def __init__(self, alpha: float, beta: float) -> None:
        self.alpha = alpha
        self.beta = beta

    @property
    def mean(self) -> float:
        """Posterior mean: alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Posterior variance."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def std(self) -> float:
        """Posterior standard deviation."""
        return math.sqrt(self.variance)

    def credible_interval(self, level: float = 0.90) -> Tuple[float, float]:
        """Return a symmetric credible interval at the given level.

        Parameters
        ----------
        level : float
            Probability mass inside the interval (default 0.90 = 90% CI).

        Returns
        -------
        tuple of float
            ``(lower, upper)`` bounds.
        """
        tail = (1.0 - level) / 2.0
        lower = float(sp_stats.beta.ppf(tail, self.alpha, self.beta))
        upper = float(sp_stats.beta.ppf(1.0 - tail, self.alpha, self.beta))
        return lower, upper

    def __repr__(self) -> str:
        return (
            f"BetaTeamState(alpha={self.alpha:.2f}, beta={self.beta:.2f}, "
            f"mean={self.mean:.4f})"
        )


class BayesianUpdater:
    """Bayesian win-probability tracker for a pool of teams.

    Parameters
    ----------
    mc_samples : int
        Number of Monte Carlo draws used in ``get_matchup_probability``.
    seed : int, optional
        Random seed for reproducible MC sampling.
    """

    def __init__(
        self,
        mc_samples: int = DEFAULT_MC_SAMPLES,
        seed: Optional[int] = DEFAULT_SEED,
    ) -> None:
        self.mc_samples: int = mc_samples
        self.seed: Optional[int] = seed
        self._rng: np.random.RandomState = np.random.RandomState(seed)
        self.teams: Dict[str, BetaTeamState] = {}

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_from_elo(
        self,
        team_id: str,
        elo_win_prob: float,
        equivalent_games: int = 20,
    ) -> BetaTeamState:
        """Create a Beta prior for *team_id* from an Elo-derived win probability.

        The prior is parameterized as if the team had played
        *equivalent_games* games with a win rate equal to *elo_win_prob*.
        A higher ``equivalent_games`` expresses more confidence in the Elo
        estimate (tighter prior).

        Parameters
        ----------
        team_id : str
            Unique team identifier.
        elo_win_prob : float
            Win probability derived from the Elo system (0-1).
        equivalent_games : int
            Pseudo-count of games the prior is "worth".

        Returns
        -------
        BetaTeamState
            The newly created posterior.
        """
        elo_win_prob = max(0.01, min(0.99, elo_win_prob))
        alpha = elo_win_prob * equivalent_games
        beta = (1.0 - elo_win_prob) * equivalent_games

        state = BetaTeamState(alpha=alpha, beta=beta)
        self.teams[team_id] = state

        logger.debug(
            "Initialized %s: alpha=%.2f, beta=%.2f (win_prob=%.3f, n=%d)",
            team_id,
            alpha,
            beta,
            elo_win_prob,
            equivalent_games,
        )
        return state

    def _ensure_team(self, team_id: str) -> BetaTeamState:
        """Return the team's state, initializing with a flat prior if new."""
        if team_id not in self.teams:
            # Uninformative (flat) prior: Beta(1, 1) = Uniform(0, 1)
            self.teams[team_id] = BetaTeamState(alpha=1.0, beta=1.0)
            logger.debug("Auto-initialized %s with flat prior Beta(1,1)", team_id)
        return self.teams[team_id]

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update_with_result(self, team_id: str, won: bool) -> BetaTeamState:
        """Update a team's posterior after a game result.

        Parameters
        ----------
        team_id : str
            Team that played.
        won : bool
            ``True`` if the team won, ``False`` if it lost.

        Returns
        -------
        BetaTeamState
            Updated posterior.
        """
        state = self._ensure_team(team_id)

        if won:
            state.alpha += 1.0
        else:
            state.beta += 1.0

        logger.debug(
            "Result update %s (won=%s): alpha=%.2f, beta=%.2f, mean=%.4f",
            team_id,
            won,
            state.alpha,
            state.beta,
            state.mean,
        )
        return state

    def update_with_evidence(
        self,
        team_id: str,
        strength: float,
        direction: str,
    ) -> BetaTeamState:
        """Incorporate soft (non-game) evidence into a team's posterior.

        This is used for contextual signals such as injuries, rest days,
        travel fatigue, or roster changes. The *strength* parameter
        controls how many "pseudo-observations" the evidence is worth.

        Parameters
        ----------
        team_id : str
            The team affected by the evidence.
        strength : float
            Magnitude of the update in pseudo-observations. Typically in
            ``[0.1, 2.0]``. A value of 1.0 is roughly equivalent to one
            game result.
        direction : str
            ``"positive"`` or ``"negative"``.

            * ``"positive"`` – evidence that the team is *stronger* than
              the current posterior suggests (e.g., key player returns).
            * ``"negative"`` – evidence that the team is *weaker*
              (e.g., star player injured).

        Returns
        -------
        BetaTeamState
            Updated posterior.

        Raises
        ------
        ValueError
            If *direction* is not ``"positive"`` or ``"negative"``.
        """
        direction = direction.lower().strip()
        if direction not in ("positive", "negative"):
            raise ValueError(
                f"direction must be 'positive' or 'negative', got '{direction}'"
            )

        strength = max(0.0, min(5.0, strength))  # clamp to prevent extremes
        state = self._ensure_team(team_id)

        if direction == "positive":
            state.alpha += strength
        else:
            state.beta += strength

        logger.debug(
            "Evidence update %s (dir=%s, str=%.2f): alpha=%.2f, beta=%.2f, mean=%.4f",
            team_id,
            direction,
            strength,
            state.alpha,
            state.beta,
            state.mean,
        )
        return state

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_probability(self, team_id: str) -> Dict[str, float]:
        """Return summary statistics for a team's current win-probability posterior.

        Parameters
        ----------
        team_id : str
            The team to query.

        Returns
        -------
        dict
            ``mean``       – posterior mean win probability.
            ``std``        – posterior standard deviation.
            ``ci_lower``   – lower bound of the 90 % credible interval.
            ``ci_upper``   – upper bound of the 90 % credible interval.
            ``alpha``      – current alpha parameter.
            ``beta``       – current beta parameter.
        """
        state = self._ensure_team(team_id)
        ci_lower, ci_upper = state.credible_interval(0.90)

        return {
            "mean": round(state.mean, 6),
            "std": round(state.std, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "alpha": round(state.alpha, 4),
            "beta": round(state.beta, 4),
        }

    def get_matchup_probability(
        self,
        home_id: str,
        away_id: str,
        home_advantage: float = 0.0,
    ) -> Dict[str, float]:
        """Estimate P(home wins) via Monte Carlo sampling from Beta posteriors.

        For each MC sample we draw a "strength" value for each team from
        their respective Beta posteriors, add optional home advantage to
        the home team's sample, and compare.

        Parameters
        ----------
        home_id : str
            Home team identifier.
        away_id : str
            Away team identifier.
        home_advantage : float
            Additive boost applied to every home-team sample. Should be
            a small positive value (e.g. 0.03 for a ~3 % boost).
            Defaults to 0.0 (no extra advantage beyond what is encoded
            in the posteriors).

        Returns
        -------
        dict
            ``home_win_prob`` – Monte Carlo estimate of P(home wins).
            ``away_win_prob`` – Monte Carlo estimate of P(away wins).
            ``draw_prob``     – Probability both samples are within a
            tiny tolerance of each other (effectively 0 for
            continuous distributions but included for completeness).
            ``home_mean``     – Posterior mean for home team.
            ``away_mean``     – Posterior mean for away team.
            ``samples_used``  – Number of MC samples drawn.
        """
        home_state = self._ensure_team(home_id)
        away_state = self._ensure_team(away_id)

        n = self.mc_samples

        home_samples = self._rng.beta(home_state.alpha, home_state.beta, size=n)
        away_samples = self._rng.beta(away_state.alpha, away_state.beta, size=n)

        # Apply home advantage
        home_samples = home_samples + home_advantage

        home_wins = int(np.sum(home_samples > away_samples))
        away_wins = int(np.sum(away_samples > home_samples))
        draws = n - home_wins - away_wins

        home_win_prob = home_wins / n
        away_win_prob = away_wins / n
        draw_prob = draws / n

        return {
            "home_win_prob": round(home_win_prob, 6),
            "away_win_prob": round(away_win_prob, 6),
            "draw_prob": round(draw_prob, 6),
            "home_mean": round(home_state.mean, 6),
            "away_mean": round(away_state.mean, 6),
            "samples_used": n,
        }

    # ------------------------------------------------------------------
    # Bulk helpers
    # ------------------------------------------------------------------

    def initialize_all_from_elo(
        self,
        elo_probs: Dict[str, float],
        equivalent_games: int = 20,
    ) -> int:
        """Initialize Beta priors for many teams at once.

        Parameters
        ----------
        elo_probs : dict
            Mapping of team_id -> Elo-derived win probability (0-1).
        equivalent_games : int
            Pseudo-count for the prior (shared across all teams).

        Returns
        -------
        int
            Number of teams initialized.
        """
        for team_id, prob in elo_probs.items():
            self.initialize_from_elo(team_id, prob, equivalent_games)
        logger.info(
            "Bulk-initialized %d teams with equivalent_games=%d",
            len(elo_probs),
            equivalent_games,
        )
        return len(elo_probs)

    def decay_posteriors(self, factor: float = 0.95) -> None:
        """Shrink all posteriors toward the flat prior (increase uncertainty).

        Useful for between-season resets or when historical data ages.

        Parameters
        ----------
        factor : float
            Multiplicative factor applied to both alpha and beta.
            A value of 0.95 gently widens the distributions.
        """
        factor = max(0.5, min(1.0, factor))
        for team_id, state in self.teams.items():
            state.alpha = max(1.0, state.alpha * factor)
            state.beta = max(1.0, state.beta * factor)

        logger.info("Decayed posteriors for %d teams (factor=%.3f)", len(self.teams), factor)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BayesianUpdater(teams={len(self.teams)}, "
            f"mc_samples={self.mc_samples})"
        )
