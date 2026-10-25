import numpy as np
import pandas as pd
import pytest
from src.ml.clustering import AcousticClusterer
from src.ml.evaluation import evaluate_retrieval_performance
from src.recommender import HybridRecommender


@pytest.fixture
def sample_catalog_dataframe():
    return pd.DataFrame(
        [
            {
                "track_id": "TRK_001",
                "song_name": "Blinding Lights",
                "artist": "The Weeknd",
                "lyrics": "I said, ooh, I'm blinded by the lights",
                "valence": 0.38,
                "energy": 0.73,
                "danceability": 0.51,
                "tempo": 171.0,
                "acousticness": 0.001,
            },
            {
                "track_id": "TRK_002",
                "song_name": "Someone Like You",
                "artist": "Adele",
                "lyrics": "Never mind, I'll find someone like you, I wish nothing but the best for you",
                "valence": 0.28,
                "energy": 0.33,
                "danceability": 0.38,
                "tempo": 135.0,
                "acousticness": 0.89,
            },
            {
                "track_id": "TRK_003",
                "song_name": "Levitating",
                "artist": "Dua Lipa",
                "lyrics": "If you wanna run away with me, I know a galaxy, and I can take you for a ride",
                "valence": 0.91,
                "energy": 0.82,
                "danceability": 0.79,
                "tempo": 103.0,
                "acousticness": 0.008,
            },
        ]
    )


def test_acoustic_clustering_fit_and_similarity(sample_catalog_dataframe):
    clusterer = AcousticClusterer(n_clusters=3, random_state=42)
    clusterer.fit(sample_catalog_dataframe)

    assert clusterer._is_fitted is True

    test_vec = np.array([0.8, 0.8, 0.7, 0.6, 0.1], dtype=np.float32)
    cluster_id = clusterer.predict_cluster(test_vec)
    assert isinstance(cluster_id, int)

    sim = clusterer.compute_acoustic_similarity(test_vec, target_emotion="joy")
    assert 0.0 <= sim <= 1.0


def test_evaluate_retrieval_performance():
    mock_recs = [
        {"similarity_score": 0.92, "acoustic_cluster_id": 0, "detected_emotion": "joy"},
        {"similarity_score": 0.88, "acoustic_cluster_id": 0, "detected_emotion": "joy"},
        {"similarity_score": 0.81, "acoustic_cluster_id": 1, "detected_emotion": "energy"},
    ]

    metrics = evaluate_retrieval_performance(
        recommendations=mock_recs,
        query_text="upbeat party pop",
        detected_emotion="joy",
        latency_ms=12.4,
    )

    assert metrics["total_evaluated"] == 3
    assert 0.80 <= metrics["mean_cosine_similarity"] <= 0.95
    assert metrics["cluster_cohesion_score"] == pytest.approx(0.667, rel=1e-2)
    assert metrics["emotion_consistency_rate"] == pytest.approx(0.667, rel=1e-2)
    assert metrics["inference_latency_ms"] == 12.4


def test_hybrid_recommender_quantitative_output_shape(sample_catalog_dataframe):
    recommender = HybridRecommender(df=sample_catalog_dataframe)
    results = recommender.recommend("upbeat fast dance anthem", top_k=2)

    assert isinstance(results, list)
    if len(results) > 0:
        item = results[0]
        assert "track_id" in item
        assert "similarity_score" in item
        assert "hybrid_score" in item
        assert "semantic_score" in item
        assert "emotion_score" in item
        assert "acoustic_score" in item
        assert "acoustic_cluster_id" in item
        assert "xai" in item
        assert 0.0 <= item["hybrid_score"] <= 100.0
