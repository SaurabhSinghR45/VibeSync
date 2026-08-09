import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check_endpoint():
    """Verify GET /api/v1/health returns status 200 and valid schema."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "database_connected" in data
    assert data["database_connected"] is True

def test_metrics_endpoint():
    """Verify GET /api/v1/metrics returns status 200 and metrics counters."""
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_tracks_indexed" in data
    assert "uptime_seconds" in data

def test_recommendation_endpoint():
    """Verify POST /api/v1/recommend returns recommendations with Explainable AI reasoning."""
    payload = {
        "input": "late night lo-fi chill beats",
        "top_k": 5
    }
    response = client.post("/api/v1/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "late night lo-fi chill beats"
    assert "detected_emotion" in data
    assert "latency_ms" in data
    assert isinstance(data["results"], list)

    if len(data["results"]) > 0:
        track = data["results"][0]
        assert "track_id" in track
        assert "song_name" in track
        assert "artist" in track
        assert "similarity_score" in track
        assert "xai" in track
        assert "vibe_match_score" in track["xai"]
        assert "vibe_explanation" in track["xai"]

def test_liked_songs_routes():
    """Verify GET, POST, and DELETE routes for user liked songs."""
    # Add liked song
    payload = {
        "track_id": "TRK_TEST_888",
        "song_name": "Starboy Test",
        "artist": "The Weeknd"
    }
    post_res = client.post("/api/v1/likes", json=payload)
    assert post_res.status_code == 200

    # Get liked songs
    get_res = client.get("/api/v1/likes")
    assert get_res.status_code == 200
    likes = get_res.json()["liked_songs"]
    assert any(s["track_id"] == "TRK_TEST_888" for s in likes)

    # Delete liked song
    del_res = client.delete("/api/v1/likes/TRK_TEST_888")
    assert del_res.status_code == 200
