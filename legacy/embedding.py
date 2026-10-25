import os
import pickle

from sentence_transformers import SentenceTransformer

MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_PATH = "models/embeddings.pkl"


# ✅ Load model (cached later in app)
def get_model():
    return SentenceTransformer(MODEL_NAME)


# ✅ Create embeddings (run once)
def create_embeddings(texts):
    model = get_model()

    vectors = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True
    )

    with open(EMBEDDING_PATH, "wb") as f:
        pickle.dump(vectors, f)

    return vectors


# ✅ Load embeddings (used in app)
def load_embeddings():
    if not os.path.exists(EMBEDDING_PATH):
        raise FileNotFoundError("❌ embeddings.pkl not found. Run generate_embeddings.py first.")

    with open(EMBEDDING_PATH, "rb") as f:
        return pickle.load(f)
