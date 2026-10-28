def test_list_models_format(client, auth_headers):
    response = client.get("/v1/models", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data.get("object") == "list"
    models = data.get("data")
    assert isinstance(models, list)
    assert len(models) >= 2

    model_ids = [m["id"] for m in models]
    assert "whisper-1" in model_ids
    assert "tiny" in model_ids

    for m in models:
        assert m.get("object") == "model"
        assert m.get("owned_by") == "self-hosted"
        assert isinstance(m.get("created"), int)
