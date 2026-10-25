import logging
import os
import sys
import time

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer
from src.config import settings
from src.data_loader import load_all_datasets
from src.vector_store import FaissVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("vibesync.build_index")


def run_build_vector_index():
    """
    Load dataset, encode track lyrics/metadata using SentenceTransformer,
    build FAISS vector index, and save binary file.
    """
    logger.info("📥 Loading cleaned dataset...")
    df = load_all_datasets()

    if df.empty:
        logger.error("Dataset is empty. Run scripts/build_dataset.py first.")
        return

    logger.info(f"Loaded {len(df)} tracks for embedding generation.")

    # Create composite embedding text (Song Title + Artist + Lyrics)
    texts_to_encode = []
    for _, row in df.iterrows():
        song = str(row.get("song_name", ""))
        artist = str(row.get("artist", ""))
        lyrics = str(row.get("lyrics", ""))[:300]  # Use representative snippet
        text = f"{song} by {artist}. {lyrics}".strip()
        texts_to_encode.append(text)

    logger.info(f"🤖 Initializing SentenceTransformer model ({settings.EMBEDDING_MODEL_NAME})...")
    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

    logger.info("🔄 Generating dense vector embeddings (batch_size=64)...")
    start_time = time.time()

    embeddings = model.encode(texts_to_encode, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    elapsed = time.time() - start_time
    logger.info(f"⚡ Embedding generation finished in {elapsed:.2f} seconds! Matrix shape: {embeddings.shape}")

    # Build FAISS index
    vector_store = FaissVectorStore(dimension=embeddings.shape[1])
    vector_store.build_index(embeddings)

    saved_path = vector_store.save()
    logger.info(f"✅ FAISS Vector Store build complete! Saved to {saved_path}")


if __name__ == "__main__":
    run_build_vector_index()
