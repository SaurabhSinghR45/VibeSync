import pytest
import pandas as pd
from src.recommender import HybridRecommender

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame([
        {"track_id": "TRK_001", "song_name": "Blinding Lights", "artist": "The Weeknd", "lyrics": "I said, ooh, I'm blinded by the lights"},
        {"track_id": "TRK_002", "song_name": "Someone Like You", "artist": "Adele", "lyrics": "Never mind, I'll find someone like you"},
        {"track_id": "TRK_003", "song_name": "Shape of You", "artist": "Ed Sheeran", "lyrics": "The club isn't the best place to find a lover"}
    ])

def test_xai_explanation_generation(sample_dataframe):
    """Verify Explainable AI (XAI) output structure."""
    recommender = HybridRecommender(df=sample_dataframe)
    xai = recommender.generate_xai_explanation(
        query="sad melancholic breakup song",
        song_name="Someone Like You",
        artist="Adele",
        similarity_score=0.88,
        detected_emotion="sad"
    )

    assert isinstance(xai, dict)
    assert "vibe_match_score" in xai
    assert "emotion_tag" in xai
    assert "key_semantic_triggers" in xai
    assert "vibe_explanation" in xai
    assert xai["vibe_match_score"] == 88
    assert "Melancholic" in xai["emotion_tag"]
