"""Classical ML branch: Random Forest and Gradient Boosting regressors that
operate on the engineered feature vector defined in `features.py`."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config


def build_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=config.SEED,
    )


def build_gradient_boost() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.85,
        random_state=config.SEED,
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(y_pred, 0.0, 1.0)
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "mse":  float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
    }
