import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("vibesync.etl.normalizer")


class FeatureNormalizer:
    """
    Statistical Feature Normalizer implementing Z-score standardization:
        z = (x - mu) / sigma
    and bounded MinMax scaling for distance metrics and machine learning features.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None):
        self.feature_cols = feature_cols or ["valence", "energy", "danceability", "tempo", "acousticness"]
        self.stats: Dict[str, Dict[str, float]] = {}
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "FeatureNormalizer":
        """Fit statistical parameters (mean, std, min, max) on training DataFrame."""
        self.stats = {}
        for col in self.feature_cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                mean_val = float(series.mean()) if not series.empty else 0.0
                std_val = float(series.std()) if not series.empty and series.std() > 0 else 1.0
                min_val = float(series.min()) if not series.empty else 0.0
                max_val = float(series.max()) if not series.empty and series.max() > min_val else min_val + 1.0

                self.stats[col] = {
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                }
            else:
                self.stats[col] = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0}

        self._is_fitted = True
        logger.info(
            f"Fitted FeatureNormalizer statistics on {len(self.feature_cols)} features: {list(self.stats.keys())}"
        )
        return self

    def transform_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Z-Score Standardization to DataFrame.
        Produces columns named `{col}_zscore`.
        """
        if not self._is_fitted:
            self.fit(df)

        result_df = df.copy()
        for col in self.feature_cols:
            if col in result_df.columns:
                mean = self.stats[col]["mean"]
                std = self.stats[col]["std"]
                result_df[f"{col}_zscore"] = (pd.to_numeric(result_df[col], errors="coerce").fillna(mean) - mean) / std
        return result_df

    def transform_minmax(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale features into [0.0, 1.0] interval.
        Produces columns named `{col}_norm`.
        """
        if not self._is_fitted:
            self.fit(df)

        result_df = df.copy()
        for col in self.feature_cols:
            if col in result_df.columns:
                min_val = self.stats[col]["min"]
                max_val = self.stats[col]["max"]
                diff = max_val - min_val if max_val > min_val else 1.0
                scaled = (pd.to_numeric(result_df[col], errors="coerce").fillna(min_val) - min_val) / diff
                result_df[f"{col}_norm"] = scaled.clip(0.0, 1.0)
        return result_df

    def normalize_vector(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """Transform a single sample feature dictionary into a normalized numpy vector."""
        vec = []
        for col in self.feature_cols:
            val = float(feature_dict.get(col, self.stats.get(col, {}).get("mean", 0.0)))
            if self._is_fitted and col in self.stats:
                mean = self.stats[col]["mean"]
                std = self.stats[col]["std"]
                z = (val - mean) / std
            else:
                z = 0.0
            vec.append(z)
        return np.array(vec, dtype=np.float32)

    def save_stats(self, file_path: Path) -> None:
        """Serialize normalization parameters to JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

    def load_stats(self, file_path: Path) -> bool:
        """Load normalization parameters from JSON file."""
        if not file_path.exists():
            return False
        with open(file_path, "r", encoding="utf-8") as f:
            self.stats = json.load(f)
            self._is_fitted = True
        return True
