import logging
from pathlib import Path
from typing import Optional, Tuple

import faiss
import numpy as np
from src.config import settings

logger = logging.getLogger("vibesync.vector_store")


class FaissVectorStore:
    """
    High-performance vector storage and retrieval wrapper using Facebook AI Similarity Search (FAISS).
    Supports Inner Product (Cosine Similarity on normalized vectors) for sub-millisecond top-k ANN search.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self._is_trained = False

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors for Cosine Similarity via Inner Product."""
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build FAISS IndexFlatIP index from dense embedding matrix.
        """
        logger.info(f"Building FAISS IndexFlatIP for {len(embeddings)} vectors (dimension {self.dimension})...")
        normalized_embeddings = self._normalize_vectors(embeddings)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(normalized_embeddings)
        self._is_trained = True
        logger.info(f"✅ FAISS Index build complete! Total indexed vectors: {self.index.ntotal}")
        return self.index

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search top-k nearest neighbors for query vector.
        Returns:
            scores (np.ndarray): Cosine similarity scores between -1.0 and 1.0.
            indices (np.ndarray): Matching indices in dataset.
        """
        if self.index is None:
            raise ValueError("FAISS index is not initialized or loaded. Call load() or build_index() first.")

        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim == 1:
            query = np.expand_dims(query, axis=0)

        # Normalize query vector for cosine similarity
        query_norm = self._normalize_vectors(query)

        scores, indices = self.index.search(query_norm, top_k)
        return scores[0], indices[0]

    def save(self, file_path: Optional[Path] = None) -> Path:
        """Save FAISS index binary file to disk."""
        if self.index is None:
            raise ValueError("No index available to save.")

        target_path = file_path or settings.VECTOR_INDEX_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(target_path))
        logger.info(f"💾 FAISS index saved to {target_path}")
        return target_path

    def load(self, file_path: Optional[Path] = None) -> bool:
        """Load FAISS index binary file from disk."""
        target_path = file_path or settings.VECTOR_INDEX_PATH
        if not target_path.exists():
            logger.warning(f"FAISS index file not found at {target_path}")
            return False

        try:
            self.index = faiss.read_index(str(target_path))
            self.dimension = self.index.d
            self._is_trained = True
            logger.info(f"✅ Loaded FAISS index from {target_path} (Total vectors: {self.index.ntotal})")
            return True
        except Exception as exc:
            logger.error(f"Error loading FAISS index: {exc}")
            return False
