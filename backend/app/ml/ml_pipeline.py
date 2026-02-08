"""
ML Ensemble Pipeline for Sports Betting Predictions.

Combines XGBoost, LightGBM, and Random Forest classifiers into a
calibrated, weighted ensemble. The pipeline handles sport-specific
feature sets, time-series-aware cross-validation, probability
calibration (isotonic regression), and automatic weight optimization
via log-loss minimization.
"""
from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

# Suppress noisy convergence warnings during calibration
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------------------------------------------------------------------------
# Sport-specific feature column definitions
# ---------------------------------------------------------------------------
SPORT_FEATURES: Dict[str, List[str]] = {
    "nba": [
        "home_elo",
        "away_elo",
        "elo_diff",
        "home_win_pct",
        "away_win_pct",
        "home_avg_pts",
        "away_avg_pts",
        "home_avg_pts_allowed",
        "away_avg_pts_allowed",
        "home_off_rating",
        "away_off_rating",
        "home_def_rating",
        "away_def_rating",
        "home_pace",
        "away_pace",
        "home_rest_days",
        "away_rest_days",
        "home_streak",
        "away_streak",
        "home_away_record",
        "away_away_record",
        "h2h_win_pct",
        "home_b2b",
        "away_b2b",
        "home_travel_miles",
        "away_travel_miles",
        "home_injury_impact",
        "away_injury_impact",
    ],
    "nfl": [
        "home_elo",
        "away_elo",
        "elo_diff",
        "home_win_pct",
        "away_win_pct",
        "home_pts_per_game",
        "away_pts_per_game",
        "home_pts_allowed",
        "away_pts_allowed",
        "home_yards_per_game",
        "away_yards_per_game",
        "home_yards_allowed",
        "away_yards_allowed",
        "home_turnover_diff",
        "away_turnover_diff",
        "home_third_down_pct",
        "away_third_down_pct",
        "home_redzone_pct",
        "away_redzone_pct",
        "home_sacks",
        "away_sacks",
        "home_rest_days",
        "away_rest_days",
        "spread_line",
        "total_line",
        "home_injury_impact",
        "away_injury_impact",
        "is_divisional",
        "is_primetime",
    ],
    "mlb": [
        "home_elo",
        "away_elo",
        "elo_diff",
        "home_win_pct",
        "away_win_pct",
        "home_runs_per_game",
        "away_runs_per_game",
        "home_runs_allowed",
        "away_runs_allowed",
        "home_team_era",
        "away_team_era",
        "home_starter_era",
        "away_starter_era",
        "home_starter_whip",
        "away_starter_whip",
        "home_team_ops",
        "away_team_ops",
        "home_bullpen_era",
        "away_bullpen_era",
        "home_batting_avg",
        "away_batting_avg",
        "home_streak",
        "away_streak",
        "home_last10_record",
        "away_last10_record",
        "home_rest_bullpen",
        "away_rest_bullpen",
    ],
    "nhl": [
        "home_elo",
        "away_elo",
        "elo_diff",
        "home_win_pct",
        "away_win_pct",
        "home_goals_per_game",
        "away_goals_per_game",
        "home_goals_allowed",
        "away_goals_allowed",
        "home_pp_pct",
        "away_pp_pct",
        "home_pk_pct",
        "away_pk_pct",
        "home_shots_per_game",
        "away_shots_per_game",
        "home_save_pct",
        "away_save_pct",
        "home_corsi_pct",
        "away_corsi_pct",
        "home_rest_days",
        "away_rest_days",
        "home_streak",
        "away_streak",
        "home_b2b",
        "away_b2b",
        "home_goalie_sv_pct",
        "away_goalie_sv_pct",
    ],
    "soccer": [
        "home_elo",
        "away_elo",
        "elo_diff",
        "home_win_pct",
        "away_win_pct",
        "home_goals_per_game",
        "away_goals_per_game",
        "home_goals_conceded",
        "away_goals_conceded",
        "home_xg",
        "away_xg",
        "home_xga",
        "away_xga",
        "home_possession_pct",
        "away_possession_pct",
        "home_shots_on_target",
        "away_shots_on_target",
        "home_form_last5",
        "away_form_last5",
        "home_clean_sheets_pct",
        "away_clean_sheets_pct",
        "home_rest_days",
        "away_rest_days",
        "h2h_home_win_pct",
        "is_derby",
        "is_cup_match",
    ],
}


def _get_xgboost_model() -> Any:
    """Build an XGBoost classifier with sensible betting defaults."""
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )


def _get_lightgbm_model() -> Any:
    """Build a LightGBM classifier with sensible betting defaults."""
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        num_leaves=31,
        objective="binary",
        metric="binary_logloss",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )


def _get_random_forest_model() -> RandomForestClassifier:
    """Build a Random Forest classifier with sensible betting defaults."""
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


class SportsBettingMLPipeline:
    """Calibrated ensemble of XGBoost, LightGBM, and Random Forest.

    Parameters
    ----------
    sport : str
        One of ``"nba"``, ``"nfl"``, ``"mlb"``, ``"nhl"``, ``"soccer"``.
    model_dir : str
        Directory for saving / loading serialized model artifacts.
    n_cv_splits : int
        Number of splits for ``TimeSeriesSplit`` cross-validation.
    calibration_method : str
        ``"isotonic"`` (default) or ``"sigmoid"`` for probability calibration.
    """

    MODEL_NAMES: List[str] = ["xgboost", "lightgbm", "random_forest"]

    def __init__(
        self,
        sport: str,
        model_dir: str = "./data/ml_models",
        n_cv_splits: int = 5,
        calibration_method: str = "isotonic",
    ) -> None:
        sport = sport.lower()
        if sport not in SPORT_FEATURES:
            raise ValueError(
                f"Unsupported sport '{sport}'. Choose from: "
                f"{list(SPORT_FEATURES.keys())}"
            )

        self.sport: str = sport
        self.feature_columns: List[str] = SPORT_FEATURES[sport]
        self.model_dir: str = model_dir
        self.n_cv_splits: int = n_cv_splits
        self.calibration_method: str = calibration_method

        # Models (populated by .train())
        self.base_models: Dict[str, Any] = {}
        self.calibrated_models: Dict[str, CalibratedClassifierCV] = {}
        self.ensemble_weights: Dict[str, float] = {
            "xgboost": 0.45,
            "lightgbm": 0.35,
            "random_forest": 0.20,
        }
        self.feature_importances: Optional[Dict[str, float]] = None
        self._trained: bool = False
        self._cv_scores: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Feature validation
    # ------------------------------------------------------------------

    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure required feature columns exist, filling missing with 0.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.

        Returns
        -------
        pd.DataFrame
            DataFrame with exactly the expected feature columns (in order).
        """
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            logger.warning(
                "Missing %d feature columns for %s (filling with 0): %s",
                len(missing),
                self.sport,
                missing,
            )
            for col in missing:
                X[col] = 0.0

        return X[self.feature_columns].copy()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        optimize_weights: bool = True,
    ) -> Dict[str, Any]:
        """Train all ensemble members with time-series cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. Must be sorted chronologically (oldest first).
        y : array-like
            Binary labels (1 = home win, 0 = away win / loss).
        optimize_weights : bool
            If ``True``, run weight optimization after training.

        Returns
        -------
        dict
            ``cv_scores``        – per-model mean CV log-loss.
            ``ensemble_weights`` – final ensemble weights.
            ``n_samples``        – number of training samples.
            ``n_features``       – number of features used.
        """
        X_clean = self._validate_features(X)
        y_arr = np.asarray(y, dtype=np.float64)

        if len(X_clean) < 50:
            raise ValueError(
                f"Need at least 50 samples to train; got {len(X_clean)}"
            )

        logger.info(
            "Training %s pipeline: %d samples, %d features",
            self.sport,
            len(X_clean),
            len(self.feature_columns),
        )

        # Build fresh base models
        model_factories = {
            "xgboost": _get_xgboost_model,
            "lightgbm": _get_lightgbm_model,
            "random_forest": _get_random_forest_model,
        }

        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        oof_predictions: Dict[str, np.ndarray] = {
            name: np.full(len(y_arr), np.nan) for name in self.MODEL_NAMES
        }
        cv_scores: Dict[str, List[float]] = {name: [] for name in self.MODEL_NAMES}

        # ---- Cross-validation ----
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
            X_train = X_clean.iloc[train_idx]
            y_train = y_arr[train_idx]
            X_val = X_clean.iloc[val_idx]
            y_val = y_arr[val_idx]

            for name in self.MODEL_NAMES:
                model = model_factories[name]()

                # Fit with early stopping where supported
                if name in ("xgboost", "lightgbm"):
                    fit_params: Dict[str, Any] = {}
                    if name == "xgboost":
                        fit_params["eval_set"] = [(X_val, y_val)]
                        fit_params["verbose"] = False
                    elif name == "lightgbm":
                        fit_params["eval_set"] = [(X_val, y_val)]
                        fit_params["callbacks"] = [
                            __import__("lightgbm").early_stopping(
                                stopping_rounds=50, verbose=False
                            ),
                            __import__("lightgbm").log_evaluation(period=-1),
                        ]
                    model.fit(X_train, y_train, **fit_params)
                else:
                    model.fit(X_train, y_train)

                # Out-of-fold predictions
                preds = model.predict_proba(X_val)[:, 1]
                oof_predictions[name][val_idx] = preds

                fold_loss = log_loss(y_val, preds)
                cv_scores[name].append(fold_loss)

                logger.debug(
                    "  Fold %d/%d %s log_loss=%.4f",
                    fold_idx + 1,
                    self.n_cv_splits,
                    name,
                    fold_loss,
                )

        self._cv_scores = cv_scores

        # ---- Train final models on full data ----
        for name in self.MODEL_NAMES:
            model = model_factories[name]()
            model.fit(X_clean, y_arr)
            self.base_models[name] = model

        # ---- Calibration ----
        for name in self.MODEL_NAMES:
            calibrated = CalibratedClassifierCV(
                self.base_models[name],
                method=self.calibration_method,
                cv="prefit",
            )
            # Use the last 30 % of data for calibration fitting
            cal_start = int(len(X_clean) * 0.7)
            X_cal = X_clean.iloc[cal_start:]
            y_cal = y_arr[cal_start:]

            if len(X_cal) >= 20:
                calibrated.fit(X_cal, y_cal)
                self.calibrated_models[name] = calibrated
            else:
                logger.warning(
                    "Not enough calibration data for %s, skipping calibration",
                    name,
                )
                # Wrap base model in a pass-through calibrator
                self.calibrated_models[name] = calibrated
                calibrated.fit(X_clean, y_arr)

        # ---- Weight optimization ----
        if optimize_weights:
            self._optimize_weights(oof_predictions, y_arr)

        # ---- Feature importances ----
        self._compute_feature_importances(X_clean)

        self._trained = True

        mean_cv = {
            name: round(float(np.mean(scores)), 4)
            for name, scores in cv_scores.items()
        }
        logger.info(
            "Training complete. CV log-loss: %s | Weights: %s",
            mean_cv,
            {k: round(v, 3) for k, v in self.ensemble_weights.items()},
        )

        return {
            "cv_scores": mean_cv,
            "ensemble_weights": dict(self.ensemble_weights),
            "n_samples": len(X_clean),
            "n_features": len(self.feature_columns),
        }

    # ------------------------------------------------------------------
    # Weight optimization
    # ------------------------------------------------------------------

    def _optimize_weights(
        self,
        oof_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> Dict[str, float]:
        """Find ensemble weights that minimize out-of-fold log-loss.

        Uses ``scipy.optimize.minimize`` with the SLSQP method under the
        constraint that weights sum to 1 and each weight is in [0, 1].

        Parameters
        ----------
        oof_predictions : dict
            Mapping of model name -> out-of-fold prediction array.
        y_true : np.ndarray
            True labels.

        Returns
        -------
        dict
            Optimized weights per model.
        """
        # Filter to rows that have valid OOF predictions for all models
        valid_mask = np.ones(len(y_true), dtype=bool)
        for name in self.MODEL_NAMES:
            valid_mask &= ~np.isnan(oof_predictions[name])

        if valid_mask.sum() < 20:
            logger.warning("Not enough valid OOF predictions for weight optimization")
            return dict(self.ensemble_weights)

        y_valid = y_true[valid_mask]
        preds_matrix = np.column_stack(
            [oof_predictions[name][valid_mask] for name in self.MODEL_NAMES]
        )

        def objective(weights: np.ndarray) -> float:
            blended = np.clip(preds_matrix @ weights, 1e-7, 1 - 1e-7)
            return log_loss(y_valid, blended)

        n_models = len(self.MODEL_NAMES)
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * n_models
        initial_weights = np.array([1.0 / n_models] * n_models)

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        if result.success:
            optimized = result.x
            self.ensemble_weights = {
                name: round(float(w), 4)
                for name, w in zip(self.MODEL_NAMES, optimized)
            }
            logger.info(
                "Optimized ensemble weights: %s (log_loss=%.4f)",
                self.ensemble_weights,
                result.fun,
            )
        else:
            logger.warning(
                "Weight optimization did not converge: %s", result.message
            )

        return dict(self.ensemble_weights)

    # ------------------------------------------------------------------
    # Feature importance aggregation
    # ------------------------------------------------------------------

    def _compute_feature_importances(self, X: pd.DataFrame) -> None:
        """Aggregate feature importances across ensemble members.

        XGBoost and LightGBM use gain-based importance; Random Forest uses
        Gini importance. All are normalized to [0, 1] before averaging
        with the ensemble weights.
        """
        importance_matrix: Dict[str, np.ndarray] = {}

        for name, model in self.base_models.items():
            if hasattr(model, "feature_importances_"):
                raw = np.array(model.feature_importances_, dtype=np.float64)
                total = raw.sum()
                if total > 0:
                    raw = raw / total  # normalize to sum to 1
                importance_matrix[name] = raw

        if not importance_matrix:
            self.feature_importances = {col: 0.0 for col in self.feature_columns}
            return

        # Weighted average across models
        combined = np.zeros(len(self.feature_columns), dtype=np.float64)
        total_weight = 0.0
        for name, imp in importance_matrix.items():
            w = self.ensemble_weights.get(name, 1.0 / len(importance_matrix))
            combined += imp * w
            total_weight += w

        if total_weight > 0:
            combined /= total_weight

        self.feature_importances = {
            col: round(float(val), 6)
            for col, val in zip(self.feature_columns, combined)
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        X: pd.DataFrame,
        return_model_probs: bool = True,
        top_n_features: int = 10,
    ) -> Dict[str, Any]:
        """Generate ensemble predictions for new data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (same schema as training data).
        return_model_probs : bool
            If ``True``, include per-model probabilities in the output.
        top_n_features : int
            Number of top feature importances to include.

        Returns
        -------
        dict
            ``ensemble_prob``  – numpy array of blended home-win probabilities.
            ``model_probs``    – dict of model_name -> probability array
            (if *return_model_probs* is True).
            ``top_features``   – list of (feature_name, importance) tuples.
        """
        if not self._trained:
            raise RuntimeError("Pipeline has not been trained. Call .train() first.")

        X_clean = self._validate_features(X)

        model_probs: Dict[str, np.ndarray] = {}
        for name in self.MODEL_NAMES:
            if name in self.calibrated_models:
                preds = self.calibrated_models[name].predict_proba(X_clean)[:, 1]
            else:
                preds = self.base_models[name].predict_proba(X_clean)[:, 1]
            model_probs[name] = preds

        # Weighted ensemble blend
        weights = np.array(
            [self.ensemble_weights[name] for name in self.MODEL_NAMES]
        )
        pred_matrix = np.column_stack(
            [model_probs[name] for name in self.MODEL_NAMES]
        )
        ensemble_prob = np.clip(pred_matrix @ weights, 0.0, 1.0)

        # Top features
        top_features: List[Tuple[str, float]] = []
        if self.feature_importances:
            sorted_feats = sorted(
                self.feature_importances.items(), key=lambda x: x[1], reverse=True
            )
            top_features = sorted_feats[:top_n_features]

        result: Dict[str, Any] = {
            "ensemble_prob": ensemble_prob,
            "top_features": top_features,
        }

        if return_model_probs:
            result["model_probs"] = {
                name: probs for name, probs in model_probs.items()
            }

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _model_path(self, filename: str) -> str:
        """Return full path for a model artifact."""
        return os.path.join(self.model_dir, self.sport, filename)

    def save_models(self) -> str:
        """Persist all models and metadata to disk using joblib.

        Returns
        -------
        str
            Path to the directory where models were saved.
        """
        if not self._trained:
            raise RuntimeError("Cannot save untrained pipeline.")

        save_dir = os.path.join(self.model_dir, self.sport)
        os.makedirs(save_dir, exist_ok=True)

        # Save each base model
        for name, model in self.base_models.items():
            joblib.dump(model, self._model_path(f"base_{name}.joblib"))

        # Save calibrated models
        for name, cal_model in self.calibrated_models.items():
            joblib.dump(cal_model, self._model_path(f"calibrated_{name}.joblib"))

        # Save metadata
        metadata = {
            "sport": self.sport,
            "feature_columns": self.feature_columns,
            "ensemble_weights": self.ensemble_weights,
            "feature_importances": self.feature_importances,
            "cv_scores": {
                k: [float(v) for v in vals] for k, vals in self._cv_scores.items()
            },
            "calibration_method": self.calibration_method,
            "n_cv_splits": self.n_cv_splits,
        }
        joblib.dump(metadata, self._model_path("metadata.joblib"))

        logger.info("Saved %s pipeline to %s", self.sport, save_dir)
        return save_dir

    def load_models(self) -> bool:
        """Load a previously saved pipeline from disk.

        Returns
        -------
        bool
            ``True`` if the models were loaded successfully, ``False`` if
            the directory or files were not found.
        """
        save_dir = os.path.join(self.model_dir, self.sport)
        meta_path = self._model_path("metadata.joblib")

        if not os.path.exists(meta_path):
            logger.warning("No saved pipeline found at %s", meta_path)
            return False

        # Load metadata
        metadata = joblib.load(meta_path)
        self.feature_columns = metadata.get("feature_columns", self.feature_columns)
        self.ensemble_weights = metadata.get("ensemble_weights", self.ensemble_weights)
        self.feature_importances = metadata.get("feature_importances")
        self._cv_scores = metadata.get("cv_scores", {})
        self.calibration_method = metadata.get("calibration_method", "isotonic")

        # Load base models
        for name in self.MODEL_NAMES:
            base_path = self._model_path(f"base_{name}.joblib")
            if os.path.exists(base_path):
                self.base_models[name] = joblib.load(base_path)
            else:
                logger.warning("Base model not found: %s", base_path)

        # Load calibrated models
        for name in self.MODEL_NAMES:
            cal_path = self._model_path(f"calibrated_{name}.joblib")
            if os.path.exists(cal_path):
                self.calibrated_models[name] = joblib.load(cal_path)
            else:
                logger.warning("Calibrated model not found: %s", cal_path)

        self._trained = bool(self.base_models)
        logger.info(
            "Loaded %s pipeline from %s (%d base models, %d calibrated)",
            self.sport,
            save_dir,
            len(self.base_models),
            len(self.calibrated_models),
        )
        return True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_cv_summary(self) -> Dict[str, Any]:
        """Return a summary of cross-validation performance.

        Returns
        -------
        dict
            Per-model mean and std of CV log-loss, plus ensemble weights.
        """
        summary: Dict[str, Any] = {"ensemble_weights": dict(self.ensemble_weights)}
        for name, scores in self._cv_scores.items():
            if scores:
                summary[name] = {
                    "mean_log_loss": round(float(np.mean(scores)), 4),
                    "std_log_loss": round(float(np.std(scores)), 4),
                    "n_folds": len(scores),
                }
        return summary

    def __repr__(self) -> str:
        return (
            f"SportsBettingMLPipeline(sport={self.sport!r}, "
            f"trained={self._trained}, "
            f"features={len(self.feature_columns)})"
        )
