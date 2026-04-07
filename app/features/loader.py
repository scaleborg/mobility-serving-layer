"""
Loads the latest feature row for a given entity_id from the P3 parquet snapshot
using DuckDB in read-only mode.

Contract:
- The snapshot contains one latest row per entity_key (from feature_schema.json).
- A 'snapshot_ts' column must be present for timestamp extraction.
"""

import logging
from pathlib import Path

import duckdb

from app.schemas.errors import EntityNotFoundError

logger = logging.getLogger(__name__)

TIMESTAMP_COLUMN = "snapshot_ts"


def open_snapshot(snapshot_path: Path) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB in-memory connection and register the snapshot as a view."""
    conn = duckdb.connect(database=":memory:", read_only=False)
    conn.execute(
        f"CREATE VIEW online_features AS SELECT * FROM read_parquet('{snapshot_path}')"
    )
    logger.info("DuckDB view registered for snapshot: %s", snapshot_path)
    return conn


def load_entity_row(
    conn: duckdb.DuckDBPyConnection,
    entity_id: str,
    entity_key: str,
) -> dict:
    """
    Returns a dict of {column: value} for the requested entity.
    entity_key is driven by feature_schema.json, not hardcoded.
    Raises EntityNotFoundError if the entity is not in the snapshot.
    """
    result = conn.execute(
        f"SELECT * FROM online_features WHERE \"{entity_key}\" = ? LIMIT 1",
        [entity_id],
    ).fetchdf()

    if result.empty:
        raise EntityNotFoundError(
            message=f"Entity '{entity_id}' not found in feature snapshot."
        )

    row = result.iloc[0].to_dict()
    logger.debug("Loaded feature row for entity '%s'", entity_id)
    return row


def probe_snapshot(conn: duckdb.DuckDBPyConnection) -> None:
    """Raise if the snapshot view is not readable (used during startup)."""
    conn.execute("SELECT COUNT(*) FROM online_features").fetchone()
