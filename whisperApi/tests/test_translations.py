def test_translation_default_json(client, auth_headers, wav_audio):
    response = client.post(
        "/v1/audio/translations",
        headers=auth_headers,
        data={"model": "whisper-1"},
        files={"file": ("test.wav", wav_audio, "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert isinstance(data["text"], str)


def test_translation_verbose_json(client, auth_headers, wav_audio):
    response = client.post(
        "/v1/audio/translations",
        headers=auth_headers,
        data={"model": "whisper-1", "response_format": "verbose_json"},
        files={"file": ("test.wav", wav_audio, "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("task") == "translate"
    assert "language" in data
    assert "text" in data
    assert "segments" in data
    assert "usage" in data
