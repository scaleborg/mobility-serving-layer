"""Tests for the /metadata endpoint."""


def test_metadata_returns_versions(client):
    resp = client.get("/metadata")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "test-v1"
    assert body["feature_schema_version"] == "test-schema-v1"
    assert body["expected_features"] == [
        "ft_bikes_available",
        "ft_docks_available",
        "ft_availability_ratio",
        "ft_bikes_available_lag_15m",
        "ft_bikes_available_lag_30m",
        "ft_bikes_available_lag_60m",
        "ft_bikes_available_lag_24h",
        "ft_avg_bikes_60m",
        "ft_min_bikes_60m",
        "ft_max_bikes_60m",
        "ft_avg_ratio_60m",
        "ft_avg_bikes_24h",
        "ft_min_bikes_24h",
        "ft_max_bikes_24h",
        "ft_low_avail_freq_24h",
        "ft_hour_of_day",
        "ft_day_of_week",
        "ft_is_weekend",
        "ft_capacity",
        "ft_pct_bikes_of_capacity",
        "ft_pct_docks_of_capacity",
        "ft_bikes_delta_60m",
    ]
    assert "artifact_paths" in body
