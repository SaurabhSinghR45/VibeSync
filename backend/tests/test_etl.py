import pandas as pd
from src.etl.cleaner import DataCleaner
from src.etl.normalizer import FeatureNormalizer
from src.etl.pipeline import ETLPipeline
from src.etl.schemas import AudioFeaturesRecord


def test_data_cleaner_sanitization():
    cleaner = DataCleaner()
    dirty_text = "Hello &amp; World [Chorus] with   extra   spaces"
    sanitized = cleaner.sanitize_text(dirty_text)
    assert sanitized == "Hello & World with extra spaces"

    dirty_artist = "Taylor Swift Lyrics"
    cleaned_artist = cleaner.clean_artist_name(dirty_artist)
    assert cleaned_artist == "Taylor Swift"


def test_data_cleaner_deduplication():
    cleaner = DataCleaner()
    df = pd.DataFrame(
        [
            {"song_name": "Starboy", "artist": "The Weeknd", "lyrics": "I'm a starboy"},
            {"song_name": "STARBOY", "artist": "the weeknd ", "lyrics": "Duplicate lyrics"},
            {"song_name": "Blinding Lights", "artist": "The Weeknd", "lyrics": "Lights"},
        ]
    )
    deduped_df, removed = cleaner.deduplicate(df)
    assert len(deduped_df) == 2
    assert removed == 1


def test_feature_normalizer_zscore():
    normalizer = FeatureNormalizer(feature_cols=["valence", "energy"])
    df = pd.DataFrame(
        {
            "valence": [0.2, 0.4, 0.6, 0.8],
            "energy": [0.5, 0.5, 0.5, 0.5],
        }
    )
    normalizer.fit(df)
    transformed = normalizer.transform_zscore(df)
    assert "valence_zscore" in transformed.columns
    assert "energy_zscore" in transformed.columns
    # Check zero mean for zscore
    assert abs(transformed["valence_zscore"].mean()) < 1e-5


def test_pydantic_audio_features_validation():
    # Valid record
    valid = AudioFeaturesRecord(
        track_id="TRK_001",
        valence=0.75,
        energy=0.85,
        danceability=0.60,
        tempo=128.0,
        acousticness=0.15,
    )
    assert valid.valence == 0.75
    assert valid.tempo == 128.0

    # Auto-clamping out-of-bounds features
    clamped = AudioFeaturesRecord(
        track_id="TRK_002",
        valence=1.5,
        energy=-0.2,
        danceability=0.5,
        tempo=-50.0,
    )
    assert clamped.valence == 1.0
    assert clamped.energy == 0.0
    assert clamped.tempo == 120.0  # Default fallback tempo


def test_etl_pipeline_execution():
    pipeline = ETLPipeline()
    df_clean, report = pipeline.run(max_rows=50, save=False)

    assert isinstance(df_clean, pd.DataFrame)
    assert len(df_clean) > 0
    assert "valence_zscore" in df_clean.columns
    assert "energy_norm" in df_clean.columns
    assert report.completeness_score >= 95.0
    assert report.validity_score >= 95.0
    assert report.valid_records_passed == len(df_clean)
