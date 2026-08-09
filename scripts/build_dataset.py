import os
import sys
from pathlib import Path
import pandas as pd
import logging

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("vibesync.etl")

def load_and_clean_raw_datasets() -> pd.DataFrame:
    """
    Ingest, combine, deduplicate, and sanitize raw song and lyric CSV files.
    """
    dfs = []
    raw_dir = settings.RAW_DATA_DIR

    # Helper function to map columns flexibly
    def extract_song_artist_lyrics(df_raw: pd.DataFrame, source_name: str) -> pd.DataFrame:
        df = df_raw.copy()
        df.columns = df.columns.str.strip().str.lower()
        
        # Rename possible column aliases
        column_map = {
            "song": "song_name",
            "track": "song_name",
            "track name": "song_name",
            "name": "song_name",
            "artist_name": "artist",
            "singer_name": "artist",
            "singer": "artist",
            "text": "lyrics"
        }
        df = df.rename(columns=column_map)

        if "song_name" not in df.columns or "artist" not in df.columns:
            return pd.DataFrame()

        # Clean artist name (remove 'Lyrics' suffix if present, e.g. 'Taylor Swift Lyrics')
        df["artist"] = df["artist"].astype(str).str.replace(r"\s+Lyrics$", "", regex=True)

        if "lyrics" not in df.columns or df["lyrics"].isnull().all():
            df["lyrics"] = df["song_name"].astype(str) + " " + df["artist"].astype(str)

        return df[["song_name", "artist", "lyrics"]]

    # 1. Ingest lyrics.csv
    lyrics_path = raw_dir / "lyrics.csv"
    if lyrics_path.exists():
        try:
            logger.info(f"Loading {lyrics_path.name}...")
            df1 = pd.read_csv(lyrics_path)
            cleaned1 = extract_song_artist_lyrics(df1, "lyrics.csv")
            if not cleaned1.empty:
                dfs.append(cleaned1)
                logger.info(f"Loaded {len(cleaned1)} rows from lyrics.csv")
        except Exception as exc:
            logger.error(f"Error reading lyrics.csv: {exc}")

    # 2. Ingest songs.csv
    songs_path = raw_dir / "songs.csv"
    if songs_path.exists():
        try:
            logger.info(f"Loading {songs_path.name}...")
            df2 = pd.read_csv(songs_path)
            cleaned2 = extract_song_artist_lyrics(df2, "songs.csv")
            if not cleaned2.empty:
                dfs.append(cleaned2)
                logger.info(f"Loaded {len(cleaned2)} rows from songs.csv")
        except Exception as exc:
            logger.error(f"Error reading songs.csv: {exc}")

    # 3. Ingest songs_details.csv
    songs_details_path = raw_dir / "songs_details.csv"
    if songs_details_path.exists():
        try:
            logger.info(f"Loading {songs_details_path.name}...")
            df3 = pd.read_csv(songs_details_path)
            cleaned3 = extract_song_artist_lyrics(df3, "songs_details.csv")
            if not cleaned3.empty:
                dfs.append(cleaned3)
                logger.info(f"Loaded {len(cleaned3)} rows from songs_details.csv")
        except Exception as exc:
            logger.error(f"Error reading songs_details.csv: {exc}")

    if not dfs:
        logger.warning("No valid raw datasets found. Creating placeholder dataset.")
        return pd.DataFrame(columns=["song_name", "artist", "lyrics", "track_id"])

    # Concatenate all datasets
    combined_df = pd.concat(dfs, ignore_index=True)

    # Sanitize and clean strings
    combined_df["song_name"] = combined_df["song_name"].astype(str).str.strip()
    combined_df["artist"] = combined_df["artist"].astype(str).str.strip()
    combined_df["lyrics"] = combined_df["lyrics"].astype(str).str.strip()

    # Drop nulls & empty strings
    combined_df = combined_df.dropna(subset=["song_name", "artist", "lyrics"])
    combined_df = combined_df[
        (combined_df["song_name"] != "") &
        (combined_df["artist"] != "") &
        (combined_df["lyrics"] != "") &
        (combined_df["song_name"] != "nan")
    ]

    # Deduplicate on (song_name, artist)
    combined_df = combined_df.drop_duplicates(subset=["song_name", "artist"])
    combined_df = combined_df.reset_index(drop=True)

    # Assign persistent ID
    combined_df["track_id"] = [f"TRK_{i:06d}" for i in range(len(combined_df))]

    logger.info(f"✅ Ingestion ETL complete! Final dataset contains {len(combined_df)} unique tracks.")
    return combined_df

def run_etl():
    """
    Run dataset build pipeline and save to processed directory.
    """
    processed_dir = settings.PROCESSED_DATA_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_clean_raw_datasets()

    csv_path = processed_dir / "cleaned_songs.csv"
    parquet_path = settings.TRACK_METADATA_PATH

    df.to_csv(csv_path, index=False)
    logger.info(f"Saved dataset to {csv_path}")

    try:
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved dataset to {parquet_path}")
    except Exception as exc:
        logger.warning(f"Could not save parquet file: {exc}")

if __name__ == "__main__":
    run_etl()
