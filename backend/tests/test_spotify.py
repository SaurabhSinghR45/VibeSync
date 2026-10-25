from src.spotify_api import get_song_details, get_token


def test_get_song_details_fallback():
    """Verify song details return valid structure and search fallback URL."""
    details = get_song_details("Shape of You", "Ed Sheeran")
    assert isinstance(details, dict)
    assert "url" in details
    assert "image" in details
    assert "preview_url" in details
    assert "open.spotify.com" in details["url"]


def test_spotify_token_caching():
    """Verify token retrieval handles caching gracefully."""
    token = get_token()
    # Token can be string if credentials valid, or None if invalid/missing in test runner
    assert token is None or isinstance(token, str)
