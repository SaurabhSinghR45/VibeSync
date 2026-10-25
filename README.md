# VibeSync AI: Music Intelligence, Quantitative Modeling & Semantic Search Engine

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-00599C?style=for-the-badge&logo=facebook&logoColor=white)](https://github.com/facebookresearch/faiss)
[![SQLModel](https://img.shields.io/badge/SQLModel-SQLAlchemy%202.0-red?style=for-the-badge)](https://sqlmodel.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-K--Means%20Clustering-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pytest](https://img.shields.io/badge/Pytest-27%20Passed-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Ruff](https://img.shields.io/badge/Ruff-0%20Lint%20Errors-black?style=for-the-badge&logo=ruff&logoColor=white)](https://astral.sh/ruff)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

VibeSync AI is a production-grade music intelligence, quantitative data modeling, and semantic discovery platform. It integrates sub-millisecond **FAISS Approximate Nearest Neighbor (ANN)** vector retrieval, **DistilRoBERTa Multi-Class Emotion Classification**, and **Scikit-Learn K-Means Acoustic Clustering** into an Explainable AI (XAI) hybrid scoring engine.

---

## System Architecture

```mermaid
graph TD
    Client["Glassmorphic Web App (Single Page Application / Tailwind CSS)"] -->|HTTP REST / JSON| API["FastAPI Async Microservice (Uvicorn / Pydantic V2)"]

    subgraph Service & Security Layer
        API --> Auth["Authentication & Session Security (PBKDF2 HMAC SHA-256)"]
        API --> RL["Rate Limiter (SlowAPI Token Bucket)"]
        API --> CORS["CORS Security Middleware"]
        API --> LOG["Structured JSON Logger"]
        API --> AnalyticsAPI["Market Analytics Router (/api/v1/analytics/*)"]
    end

    subgraph ETL & Data Quality Pipeline
        RawData["Raw Catalogs (Lyrics, Songs, Audio Metadata)"] --> Cleaner["DataCleaner (Sanitize & Deduplicate)"]
        Cleaner --> Normalizer["FeatureNormalizer (Z-Score & MinMax)"]
        Normalizer --> Quality["Pydantic DataQualityReport (Completeness & Validity)"]
        Quality --> Processed["Processed Parquet / SQLite Database"]
    end

    subgraph Hybrid Quantitative Engine
        API --> Recommender["HybridRecommender (50 / 30 / 20 Scoring)"]
        Recommender -->|50% Weight| VectorStore["FAISS Vector Store (all-MiniLM-L6-v2 / 26,500+ Tracks)"]
        Recommender -->|30% Weight| Emotion["DistilRoBERTa Emotion Alignment Model"]
        Recommender -->|20% Weight| Clusterer["K-Means Acoustic Clustering & Euclidean Proximity"]
    end

    subgraph Persistence & Analytical SQL Engine
        API --> SQLService["SQL Analytics Service (CTEs & Window Functions)"]
        SQLService --> DB[("SQLite Relational Database")]
        DB --> T_Users["users & user_sessions"]
        DB --> T_Tracks["tracks"]
        DB --> T_Audio["audio_features"]
        DB --> T_NLP["nlp_embeddings"]
        DB --> T_History["search_history & liked_songs & playlists"]
    end
```

---

## Repository Structure

The codebase is organized into modular tiers separating the backend microservice from the client interface:

```text
VibeSync/
├── backend/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── analytics.py       # SQL CTE analytics and telemetry endpoints
│   │   │   ├── auth.py            # User authentication, session issuance & profile management
│   │   │   ├── health.py          # System health and index memory telemetry
│   │   │   ├── history.py         # Search history, favorites, and playlist management
│   │   │   ├── lyrics.py          # Dual-pane translation and sentiment heatmaps
│   │   │   └── recommendations.py # 50/30/20 hybrid vector recommendation search
│   │   ├── main.py                # FastAPI lifecycle, middleware, rate limiting & static mounts
│   │   └── schemas.py             # Pydantic V2 request/response contracts
│   ├── data/                      # Processed Parquet datasets and SQLite database
│   ├── models/                    # Serialized FAISS vector index (vibe_index.faiss)
│   ├── src/
│   │   ├── config.py              # Central environment configuration (Pydantic BaseSettings)
│   │   ├── data_loader.py         # Multi-source dataset ingestion with fallback synthesis
│   │   ├── database/              # SQLModel schema models, migrations and analytical queries
│   │   ├── emotion.py             # DistilRoBERTa emotion classification pipeline
│   │   ├── etl/                   # Modular data cleaning, normalizers, and schema validation
│   │   ├── ml/                    # Scikit-Learn K-Means acoustic clustering engine
│   │   ├── recommender.py         # Hybrid multi-model recommendation engine
│   │   ├── spotify_api.py         # Spotify Web API client with client-credentials caching
│   │   ├── translator.py          # Multilingual translation with emotion scoring
│   │   └── vector_store.py        # FAISS IndexFlatIP dense vector store wrapper
│   └── tests/                     # Comprehensive Pytest test suite (27 test cases)
├── frontend/
│   ├── favicon.svg                # Vector soundwave title bar icon for browser tabs
│   └── index.html                 # Glassmorphic dark UI, Web Audio visualizer, profile & modals
├── .github/workflows/ci.yml       # GitHub Actions automated lint and test CI pipeline
├── pyproject.toml                 # Build system, tool configurations (Ruff, Pytest, Coverage)
├── requirements.txt               # Pinned production dependencies
└── docker-compose.yml             # Containerized deployment manifest
```

---

## Normalized Relational Database Architecture

The persistence layer is structured using **SQLAlchemy 2.0 / SQLModel** with normalized relational tables:

```mermaid
erDiagram
    USERS ||--o{ USER_SESSIONS : "authenticates via"
    USERS ||--o{ SEARCH_HISTORY : "tracks personal"
    USERS ||--o{ LIKED_SONGS : "saves favorite"
    USERS ||--o{ VIBE_PLAYLISTS : "creates"
    TRACKS ||--|| AUDIO_FEATURES : "has acoustic profile"
    TRACKS ||--|| NLP_EMBEDDINGS : "has emotion alignment"
    TRACKS ||--o{ LIKED_SONGS : "referenced by"

    USERS {
        int id PK
        string username "Unique account username"
        string email "Unique email address"
        string password_hash "PBKDF2 HMAC SHA-256 hash"
        datetime created_at "Registration timestamp"
    }

    TRACKS {
        string id PK "Unique track identifier"
        string title "Song title"
        string artist "Artist name"
        int release_year "Year of release"
        int popularity "Track popularity score"
        string genre "Musical category"
        string lyrics "Sanitized lyrics text"
    }

    AUDIO_FEATURES {
        int id PK
        string track_id FK "References tracks.id"
        float valence "Musical positiveness (0.0 to 1.0)"
        float energy "Acoustic energy (0.0 to 1.0)"
        float tempo "Speed in BPM"
        float danceability "Rhythmic suitability (0.0 to 1.0)"
        float acousticness "Acoustic measure (0.0 to 1.0)"
        int acoustic_cluster_id "K-Means cluster ID (0 to 4)"
    }

    NLP_EMBEDDINGS {
        int id PK
        string track_id FK "References tracks.id"
        string dominant_emotion "Classified emotion"
        float emotion_confidence "Confidence probability (0.0 to 1.0)"
        string embedding_vector "Serialized vector representation"
    }

    VIBE_PLAYLISTS {
        int id PK
        int user_id FK "References users.id (Nullable for guests)"
        string playlist_name "Playlist title"
        string track_ids_json "JSON array of track IDs"
        datetime created_at "Creation timestamp"
    }
```

### Advanced SQL Analytical Views (`src/database/analytics_queries.sql`)

1. **Decadal Mood Trends (`get_decadal_mood_trends`)**:
   - Employs **Common Table Expressions (CTEs)** and **Window Functions** (`RANK() OVER (PARTITION BY decade ORDER BY track_count DESC)`) to analyze shifting emotional profiles and average acoustic metrics over decades.
2. **Sentiment-Acoustic Congruence (`get_acoustic_sentiment_correlations`)**:
   - Calculates statistical correlation coefficients evaluating harmony between lyrical sentiment and physical audio features.
3. **Query Telemetry & Latency Observability (`get_query_performance_metrics`)**:
   - Evaluates rolling percentiles measuring search throughput, p50/p95 latency, and recommendation relevance.

---

## Automated ETL & Data Quality Pipeline

The modular ETL subsystem (`src/etl/`) executes an automated data quality pipeline:

1. **Collection & Ingestion**: Collects track records, lyrics text, and Spotify audio features (`valence`, `energy`, `danceability`, `tempo`, `acousticness`).
2. **Sanitization & Deduplication (`DataCleaner`)**:
   - Strips metadata brackets, HTML entities, and redundant suffixes.
   - Applies composite-key deduplication (`title.lower() + "___" + artist.lower()`).
3. **Imputation & Normalization (`FeatureNormalizer`)**:
   - Statistical median imputation for missing values.
   - Applies **Z-score standardization**:
     $$z = \frac{x - \mu}{\sigma}$$
   - Applies bounded MinMax scaling for distance-based clustering algorithms.
4. **Schema Validation (`Pydantic V2`)**:
   - Validates all records against strict typing schemas (`ValidatedTrackRecord` and `AudioFeaturesRecord`).
   - Produces an automated `DataQualityReport` measuring row volume, completeness percentage (>99%), and validity rate (100%).

---

## Hybrid Quantitative Scoring Methodology

VibeSync AI determines recommendation relevance using a **Multi-Model Scoring Formula**:

$$\text{Hybrid Score} = 0.50 \cdot S_{\text{semantic}} + 0.30 \cdot S_{\text{emotion}} + 0.20 \cdot S_{\text{acoustic}}$$

### 1. Semantic Similarity (50% Weight)
- Encodes freeform natural language queries into 384-dimensional dense vectors using `SentenceTransformers` (`all-MiniLM-L6-v2`).
- Executes sub-15ms Approximate Nearest Neighbor (ANN) search via **FAISS** (`IndexFlatIP` with L2-normalized cosine distance):
  $$S_{\text{semantic}} = \frac{\mathbf{q} \cdot \mathbf{v}}{\|\mathbf{q}\| \|\mathbf{v}\|}$$

### 2. Emotion Alignment (30% Weight)
- Derives multi-class emotion probability distributions using Hugging Face Transformers (`distilroberta-base`).
- Computes compatibility between user mood intent and lyrical emotion vectors:
  $$S_{\text{emotion}} = \text{Compatibility}(\text{Mood}_{\text{query}}, \text{Mood}_{\text{track}})$$

### 3. Statistical Acoustic Clustering (20% Weight)
- Uses **Scikit-Learn K-Means** ($k=5$) trained on normalized acoustic feature profiles.
- Computes Euclidean distance $D_{\text{acoustic}}$ between the target acoustic archetype and candidate track features:
  $$S_{\text{acoustic}} = \exp\left(-\|\mathbf{a}_{\text{target}} - \mathbf{a}_{\text{track}}\|_2\right)$$

---

## API Documentation Reference

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### Authentication & Account Management

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/auth/signup` | Register a new user account |
| `POST` | `/api/v1/auth/login` | Authenticate user and issue session token |
| `GET` | `/api/v1/auth/me` | Retrieve authenticated user profile |
| `POST` | `/api/v1/auth/logout` | Revoke active session token |

### Recommendation & Music Services

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/recommend` | Execute 50/30/20 hybrid vector recommendation search |
| `POST` | `/api/v1/feedback` | Ingest user interaction telemetry (`like`, `play`, `skip`) |
| `POST` | `/api/v1/lyrics/translate` | Dual-pane lyric translation and emotion heatmap scoring |
| `GET` | `/api/v1/health` | System health check and database connection status |
| `GET` | `/api/v1/metrics` | FAISS index memory footprint and runtime statistics |
| `GET` | `/api/v1/history` | Retrieve user search history log (authenticated) |
| `GET` / `POST` | `/api/v1/likes` | Retrieve or save favorite tracks |
| `DELETE` | `/api/v1/likes/{track_id}` | Remove track from user favorites |
| `GET` / `POST` | `/api/v1/playlists` | Retrieve or create custom vibe playlists |
| `DELETE` | `/api/v1/playlists/{id}` | Delete a user vibe playlist |
| `POST` | `/api/v1/playlists/export-spotify` | Export playlist directly to Spotify user account |

### Market Analytics & Telemetry

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/analytics/trends` | Decadal mood progression and acoustic trends (SQL CTEs) |
| `GET` | `/api/v1/analytics/correlations` | Statistical sentiment-acoustic correlation matrix |
| `GET` | `/api/v1/analytics/performance` | Query telemetry, latency percentiles and similarity metrics |
| `GET` | `/api/v1/analytics/dataset-health` | ETL dataset quality scorecard and completeness metrics |
| `GET` | `/api/v1/analytics/export-csv` | Download complete market analytics report as CSV |

---

## Quick Start Guide

### Prerequisites
- Python 3.11, 3.12, or 3.13
- Git

### 1. Installation

```bash
# Clone repository
git clone https://github.com/SaurabhSinghR45/VibeSync.git
cd VibeSync

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Application Server

```powershell
# Run Uvicorn server from project root
python -m uvicorn api.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

- **Web Application**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive OpenAPI Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc API Reference**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 3. Run Automated Tests and Code Quality Checks

```powershell
# Run Ruff linting
python -m ruff check backend/

# Run complete Pytest test suite
python -m pytest -q
```

### 4. Containerized Deployment (Docker)

```bash
# Build and run with Docker Compose
docker compose up --build -d
```

---

## License

Distributed under the MIT License. See `LICENSE` for details.
