"""
Loads the pickled sklearn model from disk.
The model is loaded once at startup and stored in AppState.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_model(model_path: Path) -> Any:
    """
    Deserialise and return the model object.
    Raises FileNotFoundError (caught by startup) if path does not exist.
    """
    logger.info("Loading model from %s", model_path)
    with model_path.open("rb") as fh:
        model = pickle.load(fh)  # noqa: S301 — trusted internal artifact from P4
    logger.info("Model loaded: %s", type(model).__name__)
    return model
