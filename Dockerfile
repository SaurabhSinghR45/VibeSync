# Stage 1: Build & Dependencies
FROM python:3.11-slim as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create isolated production virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install CPU-only PyTorch first (slashes image from ~4GB to ~300MB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements strictly preferring CPU wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Pre-bake FastEmbed ONNX model directly into image to prevent runtime download memory spikes
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2').embed(['build_warmup']))"

# Stage 2: Production Runner
FROM python:3.11-slim as runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    USE_HEAVY_EMOTION_MODEL=false \
    MALLOC_ARENA_MAX=2 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

# Copy virtual environment and pre-downloaded model cache from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/.cache /app/.cache

# Copy application codebase
COPY . .

# Create unprivileged runtime user and ensure data and cache directories are owned by appuser
RUN useradd -m -u 1000 appuser && \
    mkdir -p backend/data/processed backend/models && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 10000

# Built-in Python health check supporting dynamic PORT
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request, os; port = os.getenv('PORT', '8000'); urllib.request.urlopen(f'http://localhost:{port}/api/v1/health')" || exit 1

# Start Uvicorn with single worker to keep memory strictly under Render's 512MB RAM cap
CMD ["sh", "-c", "exec uvicorn api.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
