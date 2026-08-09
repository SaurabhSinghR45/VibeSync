import os
from datetime import datetime, timezone
from typing import Optional, List
from sqlmodel import Field, SQLModel, create_engine, Session, select
from src.config import settings

# Database Engine Setup
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class SearchHistory(SQLModel, table=True):
    """Database model for tracking user AI search queries."""
    id: Optional[int] = Field(default=None, primary_key=True)
    query: str = Field(index=True)
    input_type: str
    detected_emotion: str
    timestamp: datetime = Field(default_factory=utc_now)

class LikedSong(SQLModel, table=True):
    """Database model for user favorite tracks."""
    id: Optional[int] = Field(default=None, primary_key=True)
    track_id: str = Field(index=True, unique=True)
    song_name: str
    artist: str
    album_art: Optional[str] = None
    spotify_url: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)

class VibePlaylist(SQLModel, table=True):
    """Database model for user curated vibe playlists."""
    id: Optional[int] = Field(default=None, primary_key=True)
    playlist_name: str = Field(index=True)
    track_ids_json: str  # JSON formatted list of track_ids
    created_at: datetime = Field(default_factory=utc_now)

class FeedbackEvent(SQLModel, table=True):
    """Database model for user implicit/explicit feedback events."""
    id: Optional[int] = Field(default=None, primary_key=True)
    track_id: str = Field(index=True)
    event_type: str = Field(index=True)  # 'like', 'dislike', 'play', 'skip'
    timestamp: datetime = Field(default_factory=utc_now)

def init_db():
    """Initialize database tables."""
    SQLModel.metadata.create_all(engine)

def get_session():
    """FastAPI Dependency for database sessions."""
    with Session(engine) as session:
        yield session

# Initialize tables on import
init_db()
