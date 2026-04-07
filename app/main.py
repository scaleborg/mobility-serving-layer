"""
Application entry point.

Lifecycle:
  startup  → run_startup() populates app_state; fatal errors abort launch
  request  → routers delegate to services
  shutdown → DuckDB connection closed
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import health, metadata, predict
from app.core.config import settings
from app.core.logging import configure_logging
from app.core.startup import run_startup
from app.core.state import app_state
from app.schemas.errors import ServingError

configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Running startup checks...")
    run_startup(app_state, settings)
    yield
    if app_state.db_conn is not None:
        app_state.db_conn.close()
        logger.info("DuckDB connection closed.")


app = FastAPI(
    title="Mobility Serving Layer",
    description="P5 — Online prediction service for the mobility ML platform.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(ServingError)
async def serving_error_handler(request: Request, exc: ServingError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": type(exc).__name__, "detail": exc.message},
    )


app.include_router(health.router)
app.include_router(metadata.router)
app.include_router(predict.router)
