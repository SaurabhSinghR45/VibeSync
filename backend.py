from flask import Flask, request, jsonify
from flask_cors import CORS
from src.recommender import recommend
from src.embedding import load_embeddings
from src.data_loader import load_all_datasets
from src.emotion import detect_emotion
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app) 

# Load once
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
df = load_all_datasets()
vectors = load_embeddings()


@app.route("/recommend", methods=["POST"])
def get_recommendations():
    data = request.json
    user_input = data.get("input")

    results = recommend(user_input, model, df, vectors)

    songs = []
    for _, row in results.iterrows():
        songs.append({
            "song": row["song_name"],
            "artist": row["artist"]
        })

    return jsonify({
        "emotion": detect_emotion(user_input),
        "results": songs
    })


if __name__ == "__main__":
    app.run(debug=True)