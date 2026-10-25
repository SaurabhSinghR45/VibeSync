from api.main import app
from fastapi.testclient import TestClient
from src.translator import analyze_line_emotion, translate_lyrics

client = TestClient(app)


def test_analyze_line_emotion():
    assert analyze_line_emotion("aansu beh gaye tanhai me") == "sadness"
    assert analyze_line_emotion("nachde ne saare bhangra party") == "energy"
    assert analyze_line_emotion("dil vich tu hi tu vasdi mera pyar") == "romance"
    assert analyze_line_emotion("random calm ambient text") == "neutral"


def test_translate_curated_hindi():
    res = translate_lyrics("Sun Le Sada", "Arijit Singh")
    assert res["language"] == "Hindi 🇮🇳"
    assert len(res["heatmap_lines"]) > 0
    assert "Listen to my prayer" in res["heatmap_lines"][0]["translated"]


def test_translate_curated_punjabi():
    res = translate_lyrics("Khalbali", "Arijit Singh")
    assert "Punjabi" in res["language"]
    assert len(res["heatmap_lines"]) > 0


def test_lyrics_translate_endpoint():
    payload = {"song_name": "Sun Le Sada", "artist": "Arijit Singh", "target_language": "English"}
    response = client.post("/api/v1/lyrics/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["song_name"] == "Sun Le Sada"
    assert "heatmap_lines" in data
    assert len(data["heatmap_lines"]) > 0
