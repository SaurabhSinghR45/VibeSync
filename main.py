import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

print("📥 Loading dataset...")

# Load dataset
df = pd.read_csv("data/raw/lyrics.csv")

# Clean columns
df.columns = df.columns.str.strip().str.lower()

# Remove useless column
df = df.drop(columns=['unnamed: 0'], errors='ignore')

# Remove null lyrics
df = df.dropna(subset=['lyrics'])

# 🔥 LIMIT DATA (IMPORTANT FOR YOUR LAPTOP)
df = df.head(200)

# Convert to string
df['lyrics'] = df['lyrics'].astype(str)

print("🤖 Loading model...")

# Load multilingual model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("🔄 Starting embedding...")

# 🔥 FIX: batch + progress bar
vectors = model.encode(
    df['lyrics'].tolist(),
    batch_size=16,
    show_progress_bar=True
)

print("💾 Saving embeddings...")

# Save embeddings
with open("models/embeddings.pkl", "wb") as f:
    pickle.dump(vectors, f)

# Save cleaned dataset
df.to_csv("data/processed/cleaned_songs.csv", index=False)

print("✅ NLP embeddings created successfully!")

print(df['lyrics'].head())
print(df['lyrics'].isnull().sum())