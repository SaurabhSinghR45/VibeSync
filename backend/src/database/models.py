import hashlib
import logging
import secrets
from datetime import datetime, timezone
from typing import Generator, Optional

import pandas as pd
from sqlmodel import Field, Session, SQLModel, create_engine, select
from src.config import settings

logger = logging.getLogger("vibesync.database")

# Database Engine Setup
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================================
# Core Normalized Relational Tables (Phase 3 Architecture)
# ============================================================================


class Track(SQLModel, table=True):
    """Normalized master table storing catalog track metadata."""

    __tablename__ = "tracks"
    __table_args__ = {"extend_existing": True}

    id: str = Field(primary_key=True, index=True)
    title: str = Field(index=True)
    artist: str = Field(index=True)
    release_year: Optional[int] = Field(default=None, index=True)
    popularity: int = Field(default=50, index=True)
    genre: Optional[str] = Field(default="general", index=True)
    lyrics: Optional[str] = Field(default=None)


class AudioFeature(SQLModel, table=True):
    """Normalized table storing Spotify audio acoustic descriptors and cluster IDs."""

    __tablename__ = "audio_features"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    track_id: str = Field(foreign_key="tracks.id", index=True, unique=True)
    valence: float = Field(index=True)
    energy: float = Field(index=True)
    tempo: float = Field(index=True)
    danceability: float = Field(index=True)
    acousticness: float = Field(default=0.5)
    acoustic_cluster_id: Optional[int] = Field(default=0, index=True)


class NLPEmbedding(SQLModel, table=True):
    """Normalized table storing NLP multi-class emotion alignments and vector hashes."""

    __tablename__ = "nlp_embeddings"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    track_id: str = Field(foreign_key="tracks.id", index=True, unique=True)
    dominant_emotion: str = Field(index=True)
    emotion_confidence: float = Field(default=0.85)
    embedding_vector: Optional[str] = Field(default=None)


class QueryAnalyticsLog(SQLModel, table=True):
    """High-throughput query observability table logging telemetry & latency metrics."""

    __tablename__ = "query_analytics_logs"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    query_text: str = Field(index=True)
    detected_mood: str = Field(index=True)
    latency_ms: float
    results_count: int = Field(default=10)
    mean_similarity: Optional[float] = Field(default=None)
    timestamp: datetime = Field(default_factory=utc_now, index=True)


# ============================================================================
# User Authentication & State Tables
# ============================================================================


def hash_password(password: str) -> str:
    """Hash password using PBKDF2 HMAC SHA-256 with cryptographic salt."""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"{salt}:{key.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against salted PBKDF2 hash."""
    try:
        salt, key = hashed.split(":", 1)
        computed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
        return secrets.compare_digest(computed.hex(), key)
    except Exception:
        return False


class User(SQLModel, table=True):
    """User account model for authentication, personal library, and history."""

    __tablename__ = "users"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    email: str = Field(index=True, unique=True)
    password_hash: str
    created_at: datetime = Field(default_factory=utc_now)


class UserSession(SQLModel, table=True):
    """Active user authentication session token."""

    __tablename__ = "user_sessions"
    __table_args__ = {"extend_existing": True}

    token: str = Field(primary_key=True, index=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    created_at: datetime = Field(default_factory=utc_now)


class SearchHistory(SQLModel, table=True):
    """Database model for tracking user AI search queries."""

    __tablename__ = "searchhistory"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="users.id", index=True)
    query: str = Field(index=True)
    input_type: str
    detected_emotion: str
    timestamp: datetime = Field(default_factory=utc_now)


class LikedSong(SQLModel, table=True):
    """Database model for user favorite tracks."""

    __tablename__ = "likedsong"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="users.id", index=True)
    track_id: str = Field(index=True)
    song_name: str
    artist: str
    album_art: Optional[str] = None
    spotify_url: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class VibePlaylist(SQLModel, table=True):
    """Database model for user curated vibe playlists."""

    __tablename__ = "vibeplaylist"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="users.id", index=True)
    playlist_name: str = Field(index=True)
    track_ids_json: str
    created_at: datetime = Field(default_factory=utc_now)


class FeedbackEvent(SQLModel, table=True):
    """Database model for user implicit/explicit feedback events."""

    __tablename__ = "feedbackevent"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    track_id: str = Field(index=True)
    event_type: str = Field(index=True)  # 'like', 'dislike', 'play', 'skip'
    timestamp: datetime = Field(default_factory=utc_now)


def init_db() -> None:
    """Initialize database tables."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """FastAPI Dependency for database sessions."""
    with Session(engine) as session:
        yield session


def seed_tracks_from_dataframe(df: pd.DataFrame, limit: int = 500) -> int:
    """
    Seed normalized SQLite tables from cleaned DataFrame if empty.
    Returns count of newly inserted tracks.
    """
    init_db()
    with Session(engine) as session:
        existing = session.exec(select(Track).limit(1)).first()
        if existing:
            return 0

        logger.info(f"Seeding normalized database tables with {min(len(df), limit)} tracks...")
        count = 0
        for _, row in df.head(limit).iterrows():
            t_id = str(row.get("track_id", f"TRK_{count:06d}"))
            title = str(row.get("song_name", "Unknown Title"))
            artist = str(row.get("artist", "Unknown Artist"))
            lyrics = str(row.get("lyrics", ""))
            year = int(row.get("release_year", 2018))
            pop = int(row.get("popularity", 50))
            genre = str(row.get("genre", "Pop / Bollywood"))

            valence = float(row.get("valence", 0.5))
            energy = float(row.get("energy", 0.5))
            if valence > 0.6 and energy > 0.6:
                emotion = "joy"
            elif valence < 0.4 and energy < 0.5:
                emotion = "sadness"
            elif energy > 0.75:
                emotion = "energy"
            elif valence > 0.5:
                emotion = "romance"
            else:
                emotion = "calm"

            track_obj = Track(
                id=t_id,
                title=title,
                artist=artist,
                release_year=year,
                popularity=pop,
                genre=genre,
                lyrics=lyrics[:1000],
            )
            session.add(track_obj)

            audio_obj = AudioFeature(
                track_id=t_id,
                valence=valence,
                energy=energy,
                tempo=float(row.get("tempo", 120.0)),
                danceability=float(row.get("danceability", 0.55)),
                acousticness=float(row.get("acousticness", 0.35)),
                acoustic_cluster_id=int(row.get("acoustic_cluster_id", count % 5)),
            )
            session.add(audio_obj)

            nlp_obj = NLPEmbedding(
                track_id=t_id,
                dominant_emotion=emotion,
                emotion_confidence=round(0.70 + ((count % 25) / 100.0), 2),
            )
            session.add(nlp_obj)
            count += 1

        session.commit()
        logger.info(f"Seeded {count} tracks into relational database.")
        return count
