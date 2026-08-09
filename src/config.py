import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Attempt to load dotenv if available
try:
    from dotenv import load_dotenv
    env_path = BASE_DIR / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

class Settings:
    """Production application configuration loader."""
    
    PROJECT_NAME: str = "VibeSync AI"
    PROJECT_VERSION: str = "2.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Spotify Credentials
    SPOTIFY_CLIENT_ID: str = os.getenv("SPOTIFY_CLIENT_ID", "")
    SPOTIFY_CLIENT_SECRET: str = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    
    # Data & Storage Paths
    BASE_DIR: Path = BASE_DIR
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    VECTOR_INDEX_PATH: Path = BASE_DIR / os.getenv("VECTOR_INDEX_PATH", "models/vibe_index.faiss")
    TRACK_METADATA_PATH: Path = BASE_DIR / os.getenv("TRACK_METADATA_PATH", "data/processed/cleaned_songs.parquet")
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/data/vibesync.db")
    
    # ML Model Config
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMOTION_MODEL_NAME: str = "j-hartmann/emotion-english-distilroberta-base"
    
    # Security & Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

settings = Settings()
