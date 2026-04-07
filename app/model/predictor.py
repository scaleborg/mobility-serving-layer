"""
Thin wrapper around the loaded sklearn estimator.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def run_prediction(model: Any, feature_vector: np.ndarray) -> float:
    """
    Calls model.predict on the (1, n_features) feature vector and returns
    a scalar float prediction.
    """
    raw = model.predict(feature_vector)
    prediction = float(raw[0])
    logger.debug("Prediction: %f", prediction)
    return prediction
