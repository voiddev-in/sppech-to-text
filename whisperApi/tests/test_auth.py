def test_missing_auth_header(client):
    response = client.get("/v1/models")
    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "invalid_api_key"
    assert data["error"]["type"] == "invalid_request_error"


def test_malformed_auth_header(client):
    response = client.get("/v1/models", headers={"Authorization": "Basic somecredentials"})
    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "invalid_api_key"


def test_invalid_api_key(client):
    response = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "invalid_api_key"
    assert "Incorrect API key" in data["error"]["message"]


def test_valid_api_key_success(client):
    # Both keys in test-secret-key-1,test-secret-key-2 should work
    res1 = client.get("/v1/models", headers={"Authorization": "Bearer test-secret-key-1"})
    assert res1.status_code == 200

    res2 = client.get("/v1/models", headers={"Authorization": "Bearer test-secret-key-2"})
    assert res2.status_code == 200
