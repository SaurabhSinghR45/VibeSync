from typing import List, Optional

from pydantic import BaseModel, Field


class XAIBreakdown(BaseModel):
    """Explainable AI (XAI) transparent reasoning breakdown schema."""

    vibe_match_score: int = Field(..., description="Cosine match percentage (0-100%)")
    emotion_tag: str = Field(..., description="Formatted emotion pill badge")
    key_semantic_triggers: List[str] = Field(..., description="Key extracted query concepts")
    vibe_explanation: str = Field(..., description="Human readable justification text")


class TrackItem(BaseModel):
    """Recommended track response schema."""

    track_id: str
    song_name: str
    artist: str
    similarity_score: float
    hybrid_score: Optional[float] = None
    semantic_score: Optional[float] = None
    emotion_score: Optional[float] = None
    acoustic_score: Optional[float] = None
    acoustic_cluster_id: Optional[int] = None
    spotify_url: Optional[str] = None
    album_art: Optional[str] = None
    preview_url: Optional[str] = None
    spotify_id: Optional[str] = None
    detected_emotion: str
    xai: XAIBreakdown


class RecommendRequest(BaseModel):
    """Search request schema."""

    input: str = Field(..., min_length=1, max_length=500, json_schema_extra={"example": "rainy day acoustic vibe"})
    top_k: int = Field(default=10, ge=1, le=50, json_schema_extra={"example": 10})


class RecommendResponse(BaseModel):
    """Search response schema."""

    query: str
    detected_emotion: str
    total_found: int
    latency_ms: float
    results: List[TrackItem]


class FeedbackRequest(BaseModel):
    """User feedback event request schema."""

    track_id: str
    event_type: str = Field(..., pattern="^(like|dislike|play|skip)$")


class HealthResponse(BaseModel):
    """Health check status schema."""

    status: str
    version: str
    environment: str
    vector_index_size: int
    database_connected: bool


class MetricResponse(BaseModel):
    """Observability telemetry metrics schema."""

    total_tracks_indexed: int
    total_searches_logged: int
    total_liked_tracks: int
    uptime_seconds: float


class SpotifyExportRequest(BaseModel):
    """Spotify playlist export schema."""

    playlist_name: str = Field(default="VibeSync AI Playlist", max_length=100)
    description: str = Field(default="Created via VibeSync AI Neural Recommender Engine")
    track_ids: List[str]
    user_token: Optional[str] = None


class LyricTranslationRequest(BaseModel):
    """Hindi, Punjabi and English lyric translation request schema."""

    song_name: str
    artist: str
    target_language: Optional[str] = "English"
    custom_lyrics: Optional[List[str]] = None
