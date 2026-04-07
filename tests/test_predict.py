"""Tests for the /predict endpoint and inference pipeline."""


def test_known_entity_returns_prediction(client):
    resp = client.post("/predict", json={"entity_id": "station_1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["entity_id"] == "station_1"
    assert body["prediction"] == 42.0
    assert body["model_version"] == "test-v1"
    assert "latency_ms" in body
    assert body["timestamp"] == "2025-06-01T12:00:00Z"


def test_unknown_entity_returns_404(client):
    resp = client.post("/predict", json={"entity_id": "nonexistent"})
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"] == "EntityNotFoundError"


def test_blank_entity_id_returns_422(client):
    resp = client.post("/predict", json={"entity_id": "  "})
    assert resp.status_code == 422


def test_request_timestamp_is_preserved(client):
    resp = client.post(
        "/predict",
        json={"entity_id": "station_1", "timestamp": "2026-01-01T00:00:00Z"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["timestamp"] == "2026-01-01T00:00:00Z"


def test_omitted_timestamp_uses_snapshot(client):
    resp = client.post("/predict", json={"entity_id": "station_2"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["timestamp"] == "2025-06-01T12:00:00Z"
