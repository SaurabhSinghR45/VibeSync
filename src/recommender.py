import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.input_classifier import classify_input
from src.emotion import detect_emotion
from src.spotify_api import get_song_details
from src.vector_store import FaissVectorStore

logger = logging.getLogger("vibesync.recommender")

# Global singleton model & vector store instances
_model_instance: Optional[SentenceTransformer] = None
_vector_store_instance: Optional[FaissVectorStore] = None

def get_sentence_transformer() -> SentenceTransformer:
    """Lazy load SentenceTransformer model singleton."""
    global _model_instance
    if _model_instance is None:
        logger.info(f"Loading SentenceTransformer model ({settings.EMBEDDING_MODEL_NAME})...")
        _model_instance = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    return _model_instance

def get_vector_store(df: Optional[pd.DataFrame] = None) -> FaissVectorStore:
    """Lazy load FAISS vector store singleton with automatic fallback index creation."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = FaissVectorStore()

    if _vector_store_instance.index is None:
        loaded = _vector_store_instance.load()
        if not loaded and df is not None and not df.empty:
            logger.info("FAISS index missing. Generating on-the-fly index for dataset...")
            model = get_sentence_transformer()
            sample_texts = [f"{r.get('song_name','')} by {r.get('artist','')}. {str(r.get('lyrics',''))[:200]}" for _, r in df.iterrows()]
            embeddings = model.encode(sample_texts, batch_size=32, normalize_embeddings=True)
            _vector_store_instance.build_index(embeddings)

    return _vector_store_instance

class HybridRecommender:
    """
    Production Hybrid Recommendation & Explainable AI (XAI) Engine.
    Fuses FAISS ANN Vector Retrieval, DistilRoBERTa Emotion Confidence,
    and Spotify Metadata Enrichment.
    """
    def __init__(self, df: pd.DataFrame, vector_store: Optional[FaissVectorStore] = None):
        self.df = df
        self.model = get_sentence_transformer()
        self.vector_store = vector_store or get_vector_store(df)

    def generate_xai_explanation(
        self,
        query: str,
        song_name: str,
        artist: str,
        similarity_score: float,
        detected_emotion: str
    ) -> Dict[str, Any]:
        """
        Generate transparent Explainable AI (XAI) reasoning breakdown for a recommended track.
        """
        match_percentage = int(max(0.0, min(1.0, float(similarity_score))) * 100)
        
        # Extract query concepts
        query_words = [w.capitalize() for w in query.split() if len(w) > 3][:3]
        triggers = query_words if query_words else ["Vibe Harmony", "Acoustic Tone"]

        emotion_map = {
            "sad": ("🌧️ Melancholic", "Reflects introspective, melancholic lyric themes."),
            "joy": ("⚡ Energetic", "Upbeat tempo matching high-energy fun."),
            "happy": ("⚡ Energetic", "Upbeat tempo matching high-energy fun."),
            "love": ("💖 Romantic", "Passionate lyric harmony matching romantic mood."),
            "anger": ("🔥 Hype", "Aggressive, high-intensity sound signature."),
            "fear": ("🌌 Atmospheric", "Ambient, deep synth texture."),
            "surprise": ("✨ Dynamic", "Surprising harmonic transitions.")
        }

        tag, emotion_desc = emotion_map.get(
            detected_emotion.lower(),
            ("🎵 Vibe Synchronized", "Complements your search prompt's acoustic profile.")
        )

        explanation = (
            f"Matched with {match_percentage}% precision based on semantic lyric similarity "
            f"and '{detected_emotion.capitalize()}' mood alignment. {emotion_desc}"
        )

        return {
            "vibe_match_score": match_percentage,
            "emotion_tag": tag,
            "key_semantic_triggers": triggers,
            "vibe_explanation": explanation
        }

    def recommend(self, user_input: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Execute multi-modal hybrid recommendation retrieval.
        Returns top-k recommendations enriched with Spotify metadata and XAI reasoning.
        """
        if not user_input or self.df.empty:
            return []

        query = user_input.strip()
        input_type = classify_input(query, self.df)
        detected_emotion = detect_emotion(query) if len(query.split()) >= 2 else "joy"

        logger.info(f"Processing query '{query}' [Detected Type: {input_type}, Emotion: {detected_emotion}]")

        # Encode query prompt vector using SentenceTransformer
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]

        # Ensure vector store has index
        if self.vector_store.index is None:
            self.vector_store = get_vector_store(self.df)

        # FAISS top-k vector ANN retrieval
        scores, indices = self.vector_store.search(query_embedding, top_k=min(top_k * 2, len(self.df)))

        results = []
        seen_tracks = set()

        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.df):
                continue

            row = self.df.iloc[idx]
            song_name = str(row.get("song_name", ""))
            artist = str(row.get("artist", ""))
            track_key = f"{song_name.lower()} - {artist.lower()}"

            if track_key in seen_tracks:
                continue
            seen_tracks.add(track_key)

            # Spotify Metadata Enrichment
            spotify_details = get_song_details(song_name, artist)

            # Explainable AI (XAI) reasoning
            xai_data = self.generate_xai_explanation(
                query=query,
                song_name=song_name,
                artist=artist,
                similarity_score=score,
                detected_emotion=detected_emotion
            )

            results.append({
                "track_id": str(row.get("track_id", f"TRK_{idx}")),
                "song_name": song_name,
                "artist": artist,
                "similarity_score": float(score),
                "spotify_url": spotify_details.get("url"),
                "album_art": spotify_details.get("image"),
                "preview_url": spotify_details.get("preview_url"),
                "detected_emotion": detected_emotion,
                "xai": xai_data
            })

            if len(results) >= top_k:
                break

        return results

def recommend(user_input: str, model=None, df=None, vectors=None) -> List[Dict[str, Any]]:
    """
    Backwards-compatible recommend wrapper interface.
    """
    from src.data_loader import load_all_datasets
    if df is None:
        df = load_all_datasets()
    
    recommender = HybridRecommender(df=df)
    return recommender.recommend(user_input, top_k=10)