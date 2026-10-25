from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class RawTrackRecord(BaseModel):
    """Schema representing raw ingested track fields prior to transformation."""

    track_id: Optional[str] = None
    song_name: str
    artist: str
    lyrics: Optional[str] = None
    release_year: Optional[int] = None
    popularity: Optional[int] = None
    genre: Optional[str] = None


class AudioFeaturesRecord(BaseModel):
    """Schema validating Spotify audio acoustic features."""

    track_id: str
    valence: float = Field(ge=0.0, le=1.0, description="Musical positiveness (0.0 to 1.0)")
    energy: float = Field(ge=0.0, le=1.0, description="Perceptual intensity and activity")
    danceability: float = Field(ge=0.0, le=1.0, description="Suitability for dancing")
    tempo: float = Field(gt=0.0, le=300.0, description="Overall estimated tempo in BPM")
    acousticness: float = Field(ge=0.0, le=1.0, default=0.5, description="Confidence measure of acousticness")
    speechiness: Optional[float] = Field(default=0.05, ge=0.0, le=1.0)
    loudness: Optional[float] = Field(default=-8.0, description="Overall loudness in decibels (dB)")
    acoustic_cluster_id: Optional[int] = Field(default=0, description="Assigned K-Means acoustic cluster")

    @field_validator("valence", "energy", "danceability", "acousticness", mode="before")
    @classmethod
    def clamp_normalized_bounds(cls, v: Any) -> float:
        if v is None:
            return 0.5
        val = float(v)
        return max(0.0, min(1.0, val))

    @field_validator("tempo", mode="before")
    @classmethod
    def validate_tempo(cls, v: Any) -> float:
        if v is None or float(v) <= 0:
            return 120.0
        return float(v)


class ValidatedTrackRecord(BaseModel):
    """Production validated track record with strict quality constraints."""

    track_id: str
    title: str = Field(min_length=1)
    artist: str = Field(min_length=1)
    lyrics: str
    release_year: Optional[int] = Field(default=None, ge=1900, le=2100)
    popularity: int = Field(default=50, ge=0, le=100)
    genre: Optional[str] = Field(default="general")
    audio_features: Optional[AudioFeaturesRecord] = None


class DataQualityReport(BaseModel):
    """Comprehensive data quality scorecard and validation audit trail."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_records_read: int
    duplicates_removed: int
    null_values_imputed: Dict[str, int]
    valid_records_passed: int
    completeness_score: float = Field(ge=0.0, le=100.0, description="Completeness percentage across all attributes")
    validity_score: float = Field(ge=0.0, le=100.0, description="Validity rate conforming to schema rules")
    feature_means: Dict[str, float] = Field(default_factory=dict)
    feature_stds: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
