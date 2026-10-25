import numpy as np
from src.vector_store import FaissVectorStore


def test_faiss_vector_store_build_and_search():
    """Verify FAISS vector store indexing and inner product cosine search."""
    dim = 64
    num_vectors = 100

    np.random.seed(42)
    fake_embeddings = np.random.randn(num_vectors, dim).astype(np.float32)

    store = FaissVectorStore(dimension=dim)
    store.build_index(fake_embeddings)

    assert store.index is not None
    assert store.index.ntotal == num_vectors

    # Query with first vector
    query = fake_embeddings[0]
    scores, indices = store.search(query, top_k=5)

    assert len(scores) == 5
    assert len(indices) == 5
    # The closest vector to query[0] should be index 0
    assert indices[0] == 0
    assert scores[0] >= 0.99  # Cosine similarity to self is ~1.0


def test_faiss_save_and_load(tmp_path):
    """Verify saving and loading FAISS index binary file."""
    dim = 32
    store = FaissVectorStore(dimension=dim)
    fake_embeddings = np.random.randn(20, dim).astype(np.float32)
    store.build_index(fake_embeddings)

    save_path = tmp_path / "test_index.faiss"
    store.save(save_path)
    assert save_path.exists()

    new_store = FaissVectorStore()
    loaded = new_store.load(save_path)

    assert loaded is True
    assert new_store.index is not None
    assert new_store.index.ntotal == 20
