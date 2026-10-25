import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Attempt to load dotenv from backend/ or project root
try:
    from dotenv import load_dotenv

    for env_file in [BASE_DIR / ".env", BASE_DIR.parent / ".env"]:
        if env_file.exists():
            load_dotenv(dotenv_path=env_file)
            break
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

    # Ensure local storage directories exist
    (BASE_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "models").mkdir(parents=True, exist_ok=True)

    VECTOR_INDEX_PATH: Path = BASE_DIR / os.getenv("VECTOR_INDEX_PATH", "models/vibe_index.faiss").replace(
        "backend/", ""
    )
    TRACK_METADATA_PATH: Path = BASE_DIR / os.getenv(
        "TRACK_METADATA_PATH", "data/processed/cleaned_songs.parquet"
    ).replace("backend/", "")

    _raw_db: str = os.getenv("DATABASE_URL", "")
    if _raw_db.startswith("sqlite:///") and not os.path.isabs(_raw_db.replace("sqlite:///", "")):
        _clean_rel = _raw_db.replace("sqlite:///", "").replace("backend/", "")
        DATABASE_URL: str = f"sqlite:///{BASE_DIR / _clean_rel}"
    elif not _raw_db:
        DATABASE_URL: str = f"sqlite:///{BASE_DIR}/data/vibesync.db"
    else:
        DATABASE_URL: str = _raw_db

    # ML Model Config
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMOTION_MODEL_NAME: str = "j-hartmann/emotion-english-distilroberta-base"

    # Security & Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))


settings = Settings()
