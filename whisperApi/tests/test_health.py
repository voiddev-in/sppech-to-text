def test_health_endpoint_no_auth(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "loaded_models" in data
    assert "device" in data
    assert "queue_depth" in data
    assert isinstance(data["loaded_models"], list)
    assert isinstance(data["queue_depth"], int)
