"""
Domain exceptions with associated HTTP status codes.
All are caught in main.py and converted to JSON error responses.
"""

from dataclasses import dataclass


@dataclass
class ServingError(Exception):
    """Base class for all serving-layer domain errors."""
    message: str
    http_status: int = 500

    def __str__(self) -> str:
        return self.message


@dataclass
class EntityNotFoundError(ServingError):
    message: str = "Entity not found in feature store."
    http_status: int = 404


@dataclass
class FeatureParityError(ServingError):
    message: str = "Feature schema parity check failed."
    http_status: int = 422


@dataclass
class ModelNotReadyError(ServingError):
    message: str = "Model is not loaded."
    http_status: int = 503


@dataclass
class StartupError(ServingError):
    message: str = "Startup sanity check failed."
    http_status: int = 500
