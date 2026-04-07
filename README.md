# mobility-serving-layer (P5)

Online prediction serving layer for the mobility ML platform.

Consumes **online features** from P3 and **trained model artifacts** from P4 to serve real-time predictions with strict offline/online feature parity.

---

## Role in the ML Platform

P5 is the online inference layer of the platform.

It exposes trained models as real-time prediction services by consuming:

- features computed and materialized in P3 (Feature Store)
- models trained and packaged in P4 (Training Orchestrator)

This layer is responsible for:

- enforcing strict feature parity between training and serving
- serving low-latency predictions
- exposing a stable API for downstream systems

---

## Architecture

P3 (parquet snapshot)  ──→  DuckDB (read-only)  ──→  Feature Loader  
P4 (model.pkl + schema + metadata)  ──→  Model Loader  
                                                  │  
                                        Inference Service  
                                                  │  
                                          FastAPI Endpoints  
                                      /predict  /health  /metadata  

- Model loaded once at startup, held in memory  
- DuckDB reads the P3 parquet snapshot (one latest row per entity key)  
- Entity key is driven by `feature_schema.json`, not hardcoded  
- Feature parity validated on every prediction: presence, order, dtype  
- No feature engineering in P5 — all features are pre-materialized by P3  

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Predict for an entity. Body: `{"entity_id": "...", "timestamp": "..."}` (timestamp optional) |
| GET | `/health` | Model and feature store status |
| GET | `/metadata` | Model version, feature schema version, expected features |

---

## Required artifacts

These artifacts are required at runtime and are produced by upstream layers (P3, P4).

Artifacts live under a versioned directory: `models/<model_name>/<version>/`

| Source | File | Description |
|--------|------|-------------|
| P3 | `data/online_features.parquet` | Latest feature snapshot per entity |
| P4 | `models/<name>/<ver>/model.pkl` | Trained sklearn-compatible model |
| P4 | `models/<name>/<ver>/feature_schema.json` | Feature contract (schema + order + types) |
| P4 | `models/<name>/<ver>/model_metadata.json` | Model metadata (version, training info) |

---

## Artifact ownership (P3 → P4 → P5)

This service does not generate its own data or models.

Artifacts are produced upstream:

- **P3 (Feature Store)** → produces:
  - `data/online_features.parquet` (latest feature snapshot per entity)

- **P4 (Training Orchestrator)** → produces:
  - `model.pkl`
  - `feature_schema.json`
  - `model_metadata.json`

P5 strictly **consumes** these artifacts at runtime.

They are **not committed by default** in this repository.

---

## feature_schema.json contract

{
  "version": "v1",
  "entity_key": "station_id",
  "features": [
    {"name": "feat_a", "dtype": "float64"},
    {"name": "feat_b", "dtype": "int64"}
  ],
  "target": "demand_next_hour"
}

---

## Setup

python -m venv .venv && source .venv/bin/activate  
pip install -e ".[dev]"

---

## Local development (example)

To run the service locally, you must provide artifacts from P3 and P4.

# copy or mount artifacts  
cp /path/to/p3/online_features.parquet data/  
cp -r /path/to/p4/model_bundle models/mobility/v1/  

Then start the service:

uvicorn app.main:app --reload

---

## Configuration

Environment variables (prefix: SERVING_):

SERVING_MODEL_PATH = models/mobility/v1/model.pkl  
SERVING_FEATURE_SCHEMA_PATH = models/mobility/v1/feature_schema.json  
SERVING_MODEL_METADATA_PATH = models/mobility/v1/model_metadata.json  
SERVING_FEATURE_SNAPSHOT_PATH = data/online_features.parquet  
SERVING_INCLUDE_FEATURES_USED = false  
SERVING_LOG_LEVEL = INFO  

---

## Run

uvicorn app.main:app --host 0.0.0.0 --port 8000

---

## Test

pytest

---

## Startup checks

On launch the service verifies:

1. All artifact files exist  
2. feature_schema.json is valid and features are unique  
3. model_metadata.json is valid  
4. Model loads successfully  
5. Parquet snapshot is accessible via DuckDB  

Any failure aborts startup.

---

## Production expectations

- Latency: < 50ms per prediction (excluding network)
- Stateless service (horizontal scaling ready)
- Deterministic feature contracts
- No runtime feature computation
- Startup fails fast on any contract violation

---

## What this layer proves in the system

- Online inference using real feature store outputs  
- Strict training/serving feature parity  
- Deterministic, contract-driven ML serving  
- Clear separation between data (P3), training (P4), and serving (P5)
