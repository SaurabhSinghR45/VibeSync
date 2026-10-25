import html
import logging
import re
from typing import Dict, Tuple

import pandas as pd

logger = logging.getLogger("vibesync.etl.cleaner")


class DataCleaner:
    """
    Production Data Quality & Sanitization processor.
    Performs deterministic deduplication, text normalization, and null-value imputation.
    """

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text by removing HTML entities, redundant whitespace, and special artifacts."""
        if not isinstance(text, str):
            return ""
        # Unescape HTML entities
        clean = html.unescape(text)
        # Remove bracketed lyrics metadata e.g. [Chorus], [Verse 1]
        clean = re.sub(r"\[.*?\]", " ", clean)
        # Normalize whitespace
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    @staticmethod
    def clean_artist_name(artist: str) -> str:
        """Remove trailing 'Lyrics' or website artifacts from artist names."""
        if not isinstance(artist, str):
            return "Unknown Artist"
        cleaned = re.sub(r"\s+Lyrics$", "", artist.strip(), flags=re.IGNORECASE)
        return cleaned.strip() or "Unknown Artist"

    def deduplicate(
        self, df: pd.DataFrame, title_col: str = "song_name", artist_col: str = "artist"
    ) -> Tuple[pd.DataFrame, int]:
        """
        Deduplicate songs using normalized (title, artist) composite key.
        Returns cleaned DataFrame and count of removed duplicates.
        """
        initial_count = len(df)
        if initial_count == 0:
            return df, 0

        # Build composite deduplication key
        norm_title = df[title_col].astype(str).str.lower().str.strip()
        norm_artist = df[artist_col].astype(str).str.lower().str.strip()
        composite_key = norm_title + "___" + norm_artist

        # Keep first occurrence
        deduped_df = df.loc[~composite_key.duplicated(keep="first")].copy()
        removed = initial_count - len(deduped_df)
        logger.info(
            f"Deduplication removed {removed} duplicate tracks (Initial: {initial_count}, Remaining: {len(deduped_df)})"
        )
        return deduped_df, removed

    def impute_missing_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Impute missing values with domain-specific defaults and statistical medians.
        """
        imputed_counts: Dict[str, int] = {}
        cleaned_df = df.copy()

        # Text column imputation
        if "lyrics" in cleaned_df.columns:
            null_lyrics = cleaned_df["lyrics"].isnull() | (cleaned_df["lyrics"].astype(str).str.strip() == "")
            imputed_counts["lyrics"] = int(null_lyrics.sum())
            if imputed_counts["lyrics"] > 0:
                cleaned_df.loc[null_lyrics, "lyrics"] = (
                    cleaned_df.loc[null_lyrics, "song_name"].astype(str)
                    + " by "
                    + cleaned_df.loc[null_lyrics, "artist"].astype(str)
                )

        if "release_year" in cleaned_df.columns:
            # Coerce invalid years
            cleaned_df["release_year"] = pd.to_numeric(cleaned_df["release_year"], errors="coerce")
            null_years = (
                cleaned_df["release_year"].isnull()
                | (cleaned_df["release_year"] < 1920)
                | (cleaned_df["release_year"] > 2030)
            )
            imputed_counts["release_year"] = int(null_years.sum())
            # Default to median year or 2015
            median_year = cleaned_df.loc[~null_years, "release_year"].median() if (~null_years).any() else 2015
            cleaned_df.loc[null_years, "release_year"] = int(median_year)
            cleaned_df["release_year"] = cleaned_df["release_year"].astype(int)

        if "popularity" in cleaned_df.columns:
            cleaned_df["popularity"] = pd.to_numeric(cleaned_df["popularity"], errors="coerce").fillna(50)
            cleaned_df["popularity"] = cleaned_df["popularity"].clip(0, 100).astype(int)

        # Audio features defaults
        audio_cols = {
            "valence": 0.5,
            "energy": 0.6,
            "danceability": 0.55,
            "tempo": 120.0,
            "acousticness": 0.35,
        }
        for col, default_val in audio_cols.items():
            if col in cleaned_df.columns:
                null_audio = cleaned_df[col].isnull()
                imputed_counts[col] = int(null_audio.sum())
                median_val = cleaned_df[col].dropna().median() if not cleaned_df[col].dropna().empty else default_val
                cleaned_df[col] = cleaned_df[col].fillna(median_val)
            else:
                cleaned_df[col] = default_val
                imputed_counts[col] = len(cleaned_df)

        return cleaned_df, imputed_counts
