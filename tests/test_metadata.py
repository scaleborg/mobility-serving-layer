"""Tests for the /metadata endpoint."""


def test_metadata_returns_versions(client):
    resp = client.get("/metadata")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "test-v1"
    assert body["feature_schema_version"] == "test-schema-v1"
    assert body["expected_features"] == ["feat_a", "feat_b", "feat_c"]
    assert "artifact_paths" in body
