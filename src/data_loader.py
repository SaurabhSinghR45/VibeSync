import logging
import pandas as pd
from typing import Optional
from src.config import settings

logger = logging.getLogger("vibesync.data_loader")

def load_all_datasets() -> pd.DataFrame:
    """
    Load cleaned dataset from processed storage (Parquet/CSV).
    If processed file does not exist, triggers ETL pipeline.
    """
    parquet_path = settings.PROCESSED_DATA_DIR / "cleaned_songs.parquet"
    csv_path = settings.PROCESSED_DATA_DIR / "cleaned_songs.csv"

    # Try loading Parquet
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            logger.info(f"Loaded {len(df)} tracks from Parquet storage.")
            return df
        except Exception as exc:
            logger.warning(f"Failed to read Parquet file ({exc}), falling back to CSV.")

    # Try loading CSV
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} tracks from CSV storage.")
            return df
        except Exception as exc:
            logger.error(f"Failed to read CSV file: {exc}")

    # Trigger inline ETL if processed files don't exist yet
    logger.info("Processed dataset missing. Executing ETL ingestion...")
    from scripts.build_dataset import load_and_clean_raw_datasets
    df = load_and_clean_raw_datasets()
    
    if not df.empty:
        settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        
    return df