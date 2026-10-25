import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle

from sentence_transformers import SentenceTransformer

from src.data_loader import load_all_datasets

model = SentenceTransformer('all-MiniLM-L6-v2')

df = load_all_datasets()

# 🔥 Reduce text length
df['lyrics'] = df['lyrics'].astype(str).str[:200]

embeddings = model.encode(
    df['lyrics'].tolist(),
    show_progress_bar=True,
    batch_size=16
)

with open("models/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("✅ Embeddings updated!")
