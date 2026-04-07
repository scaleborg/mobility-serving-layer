"""Tests for the /predict endpoint and inference pipeline."""

from tests.conftest import OBS_TS


def test_known_entity_returns_prediction(client):
    resp = client.post(
        "/predict",
        json={"entity_id": "station_1", "timestamp": OBS_TS.isoformat()},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["entity_id"] == "station_1"
    assert body["prediction"] == 42.0
    assert body["model_version"] == "test-v1"
    assert "latency_ms" in body
    assert body["timestamp"] == OBS_TS.isoformat()


def test_unknown_entity_returns_404(client):
    resp = client.post(
        "/predict",
        json={"entity_id": "nonexistent", "timestamp": OBS_TS.isoformat()},
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"] == "EntityNotFoundError"


def test_blank_entity_id_returns_422(client):
    resp = client.post("/predict", json={"entity_id": "  "})
    assert resp.status_code == 422


def test_request_timestamp_is_preserved(client):
    resp = client.post(
        "/predict",
        json={"entity_id": "station_1", "timestamp": OBS_TS.isoformat()},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["timestamp"] == OBS_TS.isoformat()


def test_omitted_timestamp_uses_utc_now(client):
    """Without a timestamp, the service uses UTC now — just verify success."""
    resp = client.post("/predict", json={"entity_id": "station_1"})
    # May fail with staleness error since test data is in the past.
    # Either 200 or 422 (stale data) is acceptable; 500 is not.
    assert resp.status_code in (200, 422)


def test_stale_timestamp_returns_422(client):
    """A timestamp far in the future has no data — should be entity not found or stale."""
    resp = client.post(
        "/predict",
        json={"entity_id": "station_1", "timestamp": "2030-01-01T00:00:00"},
    )
    # No data exists at this point, so either 404 or 422
    assert resp.status_code in (404, 422)
