import os
import pytest
from src.config import settings

def test_config_loading():
    """Verify application configuration loads cleanly from settings."""
    assert settings.PROJECT_NAME == "VibeSync AI"
    assert settings.ENVIRONMENT in ["development", "production", "testing"]
    assert settings.EMBEDDING_MODEL_NAME == "all-MiniLM-L6-v2"
    assert settings.RATE_LIMIT_PER_MINUTE > 0

def test_spotify_credentials_from_env():
    """Verify Spotify client ID and secret are retrieved from environment/settings."""
    assert hasattr(settings, "SPOTIFY_CLIENT_ID")
    assert hasattr(settings, "SPOTIFY_CLIENT_SECRET")
