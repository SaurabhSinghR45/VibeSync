import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

logger = logging.getLogger("vibesync.ml.clustering")


class AcousticClusterer:
    """
    Statistical Acoustic Clustering Engine using Scikit-Learn K-Means.
    Segments tracks into distinct acoustic profiles based on normalized audio features.
    """

    FEATURE_COLS = ["valence", "energy", "danceability", "tempo", "acousticness"]

    CLUSTER_DESCRIPTIONS = {
        0: "⚡ High-Energy / Upbeat Party (High tempo, driving rhythm)",
        1: "🌧️ Melancholic & Introspective (Low valence, high acoustic texture)",
        2: "💖 Romantic & Smooth Groove (Moderate energy, high danceability)",
        3: "🌌 Deep Atmospheric & Chill (Subtle dynamics, mid-tempo ambient)",
        4: "🔥 High-Intensity & Dynamic (Peak energy, loud percussive presence)",
    }

    EMOTION_TO_TARGET_ACOUSTICS = {
        "joy": np.array([0.80, 0.75, 0.70, 0.65, 0.20], dtype=np.float32),
        "sadness": np.array([0.20, 0.30, 0.35, 0.40, 0.75], dtype=np.float32),
        "sad": np.array([0.20, 0.30, 0.35, 0.40, 0.75], dtype=np.float32),
        "energy": np.array([0.65, 0.90, 0.75, 0.85, 0.15], dtype=np.float32),
        "anger": np.array([0.40, 0.95, 0.60, 0.80, 0.10], dtype=np.float32),
        "romance": np.array([0.60, 0.50, 0.65, 0.45, 0.55], dtype=np.float32),
        "love": np.array([0.60, 0.50, 0.65, 0.45, 0.55], dtype=np.float32),
        "fear": np.array([0.25, 0.60, 0.40, 0.50, 0.60], dtype=np.float32),
        "surprise": np.array([0.70, 0.70, 0.65, 0.60, 0.30], dtype=np.float32),
    }

    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans: Optional[KMeans] = None
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "AcousticClusterer":
        """Fit K-Means clustering model on normalized or raw audio feature matrix."""
        matrix = self._extract_feature_matrix(df)
        if len(matrix) < self.n_clusters:
            # Handle minimal testing datasets
            self.n_clusters = max(1, len(matrix))

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.kmeans.fit(matrix)
        self._is_fitted = True
        logger.info(f"Fitted AcousticClusterer K-Means with {self.n_clusters} clusters on {len(matrix)} tracks.")
        return self

    def predict_cluster(self, feature_vector: np.ndarray) -> int:
        """Predict acoustic cluster ID for feature vector."""
        if not self._is_fitted or self.kmeans is None:
            return 0
        vec = feature_vector.reshape(1, -1)
        return int(self.kmeans.predict(vec)[0])

    def compute_acoustic_similarity(self, track_features: np.ndarray, target_emotion: str) -> float:
        """
        Compute Euclidean distance in normalized acoustic space and convert to [0.0, 1.0] similarity score:
            Score = exp(-EuclideanDistance)
        """
        target_vec = self.EMOTION_TO_TARGET_ACOUSTICS.get(
            target_emotion.lower(), np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        )
        # Euclidean distance
        dist = np.linalg.norm(track_features - target_vec)
        # Map distance to similarity: dist=0 -> 1.0, dist=1 -> 0.367
        sim = float(np.exp(-dist))
        return max(0.0, min(1.0, sim))

    def _extract_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract standardized feature matrix from DataFrame."""
        matrix = []
        for _, row in df.iterrows():
            vec = [
                float(row.get("valence_norm", row.get("valence", 0.5))),
                float(row.get("energy_norm", row.get("energy", 0.6))),
                float(row.get("danceability_norm", row.get("danceability", 0.55))),
                float(row.get("tempo_norm", (float(row.get("tempo", 120.0)) - 60.0) / 120.0)),
                float(row.get("acousticness_norm", row.get("acousticness", 0.35))),
            ]
            matrix.append(vec)
        return np.array(matrix, dtype=np.float32)
