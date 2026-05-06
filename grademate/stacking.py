"""Stacking meta-regressor that fuses predictions from the ML branch (RF + GB)
and the DL branch into a single calibrated score in [0, 1].

The meta-model is a small Ridge regressor — weights are interpretable, so the
report can quote the contribution of each base learner directly."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from . import config


META_FEATURE_NAMES = ["rf_pred", "gb_pred", "dl_pred"]


def build_meta() -> Ridge:
    return Ridge(alpha=1.0, random_state=config.SEED)


def stack_predictions(rf_pred: np.ndarray, gb_pred: np.ndarray, dl_pred: np.ndarray) -> np.ndarray:
    """Stack three (N,) arrays into a (N, 3) matrix in the canonical order."""
    return np.column_stack([rf_pred, gb_pred, dl_pred]).astype(np.float32)
