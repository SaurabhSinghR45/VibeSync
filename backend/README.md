# 🚀 VibeSync AI — Backend Microservice & ML Architecture

The **VibeSync AI** backend is a production-grade **FastAPI microservice** with **FAISS vector search**, **DistilRoBERTa emotion classification**, **Scikit-Learn K-Means acoustic clustering**, and **SQLite/SQLModel relational persistence**.

---

## 📂 File Layout

```
backend/
├── api/                     # FastAPI controllers and contract schemas
│   ├── routes/              # recommendations, analytics, health, history, lyrics
│   ├── main.py              # Server entrypoint & frontend HTML serving
│   └── schemas.py           # Pydantic V2 schemas
├── src/                     # Core business logic, ML & ETL modules
│   ├── database/            # Normalized SQL models & analytics queries
│   ├── etl/                 # Ingestion, cleaning, Z-score normalization & quality audit
│   ├── ml/                  # K-Means acoustic clustering & retrieval metrics
│   ├── config.py            # Pydantic Settings & environment paths
│   ├── recommender.py       # 50/30/20 Quantitative Hybrid Recommender & XAI
│   ├── vector_store.py      # FAISS ANN vector store (IndexFlatIP)
│   ├── translator.py        # Safe batch lyric translator
│   └── spotify_api.py       # Spotify client credentials grant flow
├── data/                    # Raw CSVs & processed Parquet datasets
├── models/                  # FAISS index binary and precomputed embeddings
├── scripts/                 # CLI pipelines for building datasets & indices
├── tests/                   # 27 automated unit & integration tests
└── requirements.txt         # Pinned backend Python dependencies
```

---

## ⚡ Running the Backend

From the repository root:

```powershell
.\venv\Scripts\python.exe -m uvicorn api.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

Or from within `backend/`:

```powershell
cd backend
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive OpenAPI Swagger UI is available at `http://127.0.0.1:8000/docs`.
