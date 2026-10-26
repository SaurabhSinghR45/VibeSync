import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.config import settings
from src.database import QueryAnalyticsLog, engine
from src.emotion import detect_emotion
from src.input_classifier import classify_input
from src.ml.clustering import AcousticClusterer
from src.spotify_api import get_song_details
from src.vector_store import FaissVectorStore

logger = logging.getLogger("vibesync.recommender")

# Global singleton model & vector store instances
_model_instance: Any = None
_vector_store_instance: Optional[FaissVectorStore] = None
_clusterer_instance: Optional[AcousticClusterer] = None


class FastEmbeddingWrapper:
    """Ultra-lightweight ONNX Runtime embedding engine using FastEmbed (~45MB RAM vs PyTorch's 380MB)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)

    def encode(self, texts: Any, normalize_embeddings: bool = True, batch_size: int = 32) -> np.ndarray:
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]
        embeddings = list(self.model.embed(texts, batch_size=batch_size))
        arr = np.array(embeddings, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=-1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr


def get_sentence_transformer() -> Any:
    """
    Lazy load embedding model singleton.
    Prefers FastEmbed (ONNX Runtime, ~45MB RAM) to operate safely on 512MB instances like Render.
    Falls back to SentenceTransformer if FastEmbed is unavailable.
    """
    global _model_instance
    if _model_instance is None:
        try:
            logger.info("Initializing ultra-lightweight ONNX FastEmbed embedding engine (~45MB RAM)...")
            fast_model = FastEmbeddingWrapper()
            fast_model.encode(["warmup"])
            _model_instance = fast_model
            logger.info("FastEmbed ONNX engine loaded successfully.")
            return _model_instance
        except Exception as exc:
            logger.warning(f"FastEmbed init notice ({exc}), falling back to SentenceTransformer...")

        logger.info(f"Loading SentenceTransformer model ({settings.EMBEDDING_MODEL_NAME})...")
        try:
            import torch

            torch.set_grad_enabled(False)
        except Exception:
            pass

        st_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        try:
            st_model.eval()
            for param in st_model.parameters():
                param.requires_grad = False
        except Exception:
            pass
        _model_instance = st_model
    return _model_instance


def get_vector_store(df: Optional[pd.DataFrame] = None) -> FaissVectorStore:
    """Lazy load FAISS vector store singleton with automatic fallback index creation."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = FaissVectorStore()

    if _vector_store_instance.index is None:
        loaded = _vector_store_instance.load()
        if not loaded:
            if df is None or df.empty:
                from src.data_loader import load_all_datasets

                df = load_all_datasets()

            if df is not None and not df.empty:
                logger.info("FAISS index missing. Generating on-the-fly index for dataset...")
                model: Any = get_sentence_transformer()
                sample_texts = [
                    f"{r.get('song_name','')} by {r.get('artist','')}. {str(r.get('lyrics',''))[:200]}"
                    for _, r in df.iterrows()
                ]
                embeddings = model.encode(sample_texts, batch_size=32, normalize_embeddings=True)
                _vector_store_instance.build_index(embeddings)
            else:
                logger.warning("Empty dataset. Generating minimal fallback index for test mode...")
                dummy_embeddings = np.random.randn(5, 384).astype(np.float32)
                _vector_store_instance.build_index(dummy_embeddings)

    return _vector_store_instance


def get_acoustic_clusterer(df: pd.DataFrame) -> AcousticClusterer:
    """Lazy load and fit K-Means acoustic clustering singleton."""
    global _clusterer_instance
    if _clusterer_instance is None or not _clusterer_instance._is_fitted:
        _clusterer_instance = AcousticClusterer()
        _clusterer_instance.fit(df)
    return _clusterer_instance


class HybridRecommender:
    """
    Production Hybrid Quantitative & Explainable AI (XAI) Recommendation Engine.
    Fuses:
      1. Semantic Similarity (50% weight via FAISS ANN Vector Retrieval)
      2. NLP Emotion Alignment (30% weight via DistilRoBERTa Multi-Class Distribution)
      3. Statistical Acoustic Clustering (20% weight via Scikit-Learn K-Means & Euclidean Distance)
    """

    def __init__(self, df: pd.DataFrame, vector_store: Optional[FaissVectorStore] = None):
        self.df = df
        self.model: Any = get_sentence_transformer()
        self.vector_store = vector_store or get_vector_store(df)
        self.clusterer = get_acoustic_clusterer(df) if not df.empty else AcousticClusterer()

    def generate_xai_explanation(
        self, query: str, song_name: str, artist: str, similarity_score: float, detected_emotion: str
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
            "sadness": ("🌧️ Melancholic", "Reflects introspective, melancholic lyric themes."),
            "joy": ("⚡ Energetic", "Upbeat tempo matching high-energy fun."),
            "happy": ("⚡ Energetic", "Upbeat tempo matching high-energy fun."),
            "love": ("💖 Romantic", "Passionate lyric harmony matching romantic mood."),
            "romance": ("💖 Romantic", "Passionate lyric harmony matching romantic mood."),
            "anger": ("🔥 Hype", "Aggressive, high-intensity sound signature."),
            "energy": ("🔥 High Energy", "Fast driving rhythm and upbeat dynamic."),
            "fear": ("🌌 Atmospheric", "Ambient, deep synth texture."),
            "surprise": ("✨ Dynamic", "Surprising harmonic transitions."),
        }

        tag, emotion_desc = emotion_map.get(
            detected_emotion.lower(), ("🎵 Vibe Synchronized", "Complements your search prompt's acoustic profile.")
        )

        explanation = (
            f"Matched with {match_percentage}% precision based on semantic lyric similarity "
            f"and '{detected_emotion.capitalize()}' mood alignment. {emotion_desc}"
        )

        return {
            "vibe_match_score": match_percentage,
            "emotion_tag": tag,
            "key_semantic_triggers": triggers,
            "vibe_explanation": explanation,
        }

    def _compute_emotion_alignment(self, query_emotion: str, track_row: pd.Series) -> float:
        """
        Compute NLP Emotion Alignment score (0.0 to 1.0) using emotion compatibility matrix.
        """
        q_emo = query_emotion.lower()
        t_valence = float(track_row.get("valence", 0.5))
        t_energy = float(track_row.get("energy", 0.5))

        # Check congruence with track's acoustic profile
        if q_emo in ["joy", "happy"]:
            score = (t_valence * 0.6) + (t_energy * 0.4)
        elif q_emo in ["sad", "sadness"]:
            score = ((1.0 - t_valence) * 0.6) + ((1.0 - t_energy) * 0.4)
        elif q_emo in ["energy", "anger"]:
            score = (t_energy * 0.8) + (t_valence * 0.2)
        elif q_emo in ["romance", "love"]:
            score = (t_valence * 0.7) + (0.3 if 0.3 <= t_energy <= 0.7 else 0.1)
        else:
            score = 0.75

        return float(max(0.0, min(1.0, score)))

    def recommend(self, user_input: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Execute multi-modal hybrid recommendation retrieval with 50/30/20 Quantitative Scoring:
          - 50% Semantic Cosine Similarity
          - 30% NLP Emotion Alignment
          - 20% Statistical Acoustic Clustering & Euclidean Proximity
        """
        start_time = time.time()
        if not user_input or self.df.empty:
            return []

        query = user_input.strip()
        input_type = classify_input(query, self.df)
        detected_emotion = detect_emotion(query) if len(query.split()) >= 2 else "joy"

        logger.info(f"Processing query '{query}' [Detected Type: {input_type}, Emotion: {detected_emotion}]")

        # 1. Encode query prompt vector using SentenceTransformer
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]

        # Ensure vector store has index
        if self.vector_store.index is None:
            self.vector_store = get_vector_store(self.df)

        # 2. FAISS top-k vector ANN candidate retrieval
        scores, indices = self.vector_store.search(query_embedding, top_k=min(top_k * 4, len(self.df)))

        candidate_pairs = list(zip(scores, indices))

        # Direct artist or song title matches to guarantee precision for title/artist queries
        query_lower = query.lower()
        if len(query_lower) >= 3 and "song_name" in self.df.columns and "artist" in self.df.columns:
            direct_match_mask = self.df["song_name"].str.lower().str.contains(
                query_lower, na=False, regex=False
            ) | self.df["artist"].str.lower().str.contains(query_lower, na=False, regex=False)
            direct_indices = self.df.index[direct_match_mask].tolist()[:top_k]
            for d_idx in reversed(direct_indices):
                candidate_pairs.insert(0, (0.96, d_idx))

        candidates = []
        seen_tracks = set()

        for score, idx in candidate_pairs:
            if idx < 0 or idx >= len(self.df):
                continue

            row = self.df.iloc[idx]
            song_name = str(row.get("song_name", ""))
            artist = str(row.get("artist", ""))
            track_key = f"{song_name.lower()} - {artist.lower()}"

            if track_key in seen_tracks:
                continue
            seen_tracks.add(track_key)

            # Component 1: Semantic Cosine Similarity (50% weight)
            semantic_score = float(max(0.0, min(1.0, float(score))))

            # Component 2: NLP Emotion Alignment (30% weight)
            emotion_score = self._compute_emotion_alignment(detected_emotion, row)

            # Component 3: Statistical Acoustic Clustering (20% weight)
            track_acoustic_vector = np.array(
                [
                    float(row.get("valence_norm", row.get("valence", 0.5))),
                    float(row.get("energy_norm", row.get("energy", 0.6))),
                    float(row.get("danceability_norm", row.get("danceability", 0.55))),
                    float(row.get("tempo_norm", 0.5)),
                    float(row.get("acousticness_norm", row.get("acousticness", 0.35))),
                ],
                dtype=np.float32,
            )

            acoustic_score = self.clusterer.compute_acoustic_similarity(track_acoustic_vector, detected_emotion)
            cluster_id = self.clusterer.predict_cluster(track_acoustic_vector)

            # Composite Quantitative Hybrid Formula (50 / 30 / 20)
            hybrid_score = (0.50 * semantic_score) + (0.30 * emotion_score) + (0.20 * acoustic_score)

            # Spotify Metadata Enrichment
            spotify_details = get_song_details(song_name, artist)

            # Explainable AI (XAI) reasoning
            xai_data = self.generate_xai_explanation(
                query=query,
                song_name=song_name,
                artist=artist,
                similarity_score=hybrid_score,
                detected_emotion=detected_emotion,
            )

            candidates.append(
                {
                    "track_id": str(row.get("track_id", f"TRK_{idx}")),
                    "song_name": song_name,
                    "artist": artist,
                    "similarity_score": round(semantic_score, 4),
                    "hybrid_score": round(hybrid_score * 100, 1),
                    "semantic_score": round(semantic_score * 100, 1),
                    "emotion_score": round(emotion_score * 100, 1),
                    "acoustic_score": round(acoustic_score * 100, 1),
                    "acoustic_cluster_id": cluster_id,
                    "spotify_url": spotify_details.get("url"),
                    "album_art": spotify_details.get("image"),
                    "preview_url": spotify_details.get("preview_url"),
                    "spotify_id": spotify_details.get("spotify_id"),
                    "detected_emotion": detected_emotion,
                    "xai": xai_data,
                }
            )

        # Re-rank candidates by Hybrid Quantitative Score
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top_results = candidates[:top_k]

        # Log query telemetry to database
        latency_ms = (time.time() - start_time) * 1000.0
        self._log_query_analytics(query, detected_emotion, latency_ms, top_results)

        return top_results

    def _log_query_analytics(
        self, query: str, detected_emotion: str, latency_ms: float, results: List[Dict[str, Any]]
    ) -> None:
        """Persist query execution telemetry to SQLite query_analytics_logs table."""
        try:
            from sqlmodel import Session

            mean_sim = float(np.mean([r["similarity_score"] for r in results])) if results else 0.85
            with Session(engine) as session:
                log_entry = QueryAnalyticsLog(
                    query_text=query,
                    detected_mood=detected_emotion,
                    latency_ms=round(latency_ms, 2),
                    results_count=len(results),
                    mean_similarity=round(mean_sim, 4),
                )
                session.add(log_entry)
                session.commit()
        except Exception as exc:
            logger.debug(f"Query analytics telemetry logging skipped: {exc}")


def recommend(user_input: str, model=None, df=None, vectors=None) -> List[Dict[str, Any]]:
    """Backwards-compatible recommend wrapper interface."""
    from src.data_loader import load_all_datasets

    if df is None:
        df = load_all_datasets()

    recommender = HybridRecommender(df=df)
    return recommender.recommend(user_input, top_k=10)
