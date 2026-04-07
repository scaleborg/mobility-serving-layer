"""
Sklearn-compatible wrapper around a LightGBM Booster.

This class must be importable at the path where pickle expects it,
so the model.pkl artifact is serialized with this module as its origin.
"""

import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore[assignment]


class LGBMServingModel:
    """Wraps a LightGBM Booster to expose a sklearn-compatible predict() interface."""

    def __init__(self, booster: "lgb.Booster"):
        self._booster = booster
        self.feature_names = booster.feature_name()
        self.num_features = booster.num_feature()

    def predict(self, X) -> np.ndarray:
        """Return P(target=1) for each row."""
        if hasattr(X, "values"):
            X = X.values
        return self._booster.predict(X)
