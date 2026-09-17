from fastapi.testclient import TestClient

from backend.main import app
from backend.summarizers.extractive import sentence_tokenize


client = TestClient(app)


def test_health_does_not_eagerly_load_model() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["abstractive"]["loaded"] is False


def test_extractive_api_returns_summary_and_stats() -> None:
    text = (
        "Urban trees cool neighborhoods during heat waves. "
        "Trees also absorb carbon dioxide and filter air pollution. "
        "Cities are expanding tree cover in neighborhoods with little shade."
    )
    response = client.post(
        "/summarize",
        json={"text": text, "method": "extractive", "length": 20},
    )
    body = response.json()
    assert response.status_code == 200
    assert all(sentence in text for sentence in sentence_tokenize(body["summary"]))
    assert body["stats"]["original_words"] == len(text.split())
    assert 0 < body["stats"]["compression_ratio"] <= 1


def test_blank_text_is_rejected() -> None:
    response = client.post("/summarize", json={"text": "  ", "method": "extractive"})
    assert response.status_code == 422
