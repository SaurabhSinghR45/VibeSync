import gc
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Restrict multi-threading memory overhead in containerized hosts (Render 512MB RAM limit)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# Ensure backend root is on sys.path for direct script and modular execution
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.routes import analytics, auth, health, history, lyrics, recommendations
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from src.config import settings
from src.database import init_db
from src.recommender import get_sentence_transformer, get_vector_store

# Setup Logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vibesync.api")

# Rate Limiter setup (60 requests/minute per IP)
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.RATE_LIMIT_PER_MINUTE}/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Startup and Shutdown handler.
    Initializes database tables, pre-loads ML models, and FAISS vector index.
    """
    logger.info("🚀 Starting VibeSync Production Application...")
    start_time = time.time()

    # 1. Initialize DB tables
    logger.info("Initializing database schemas...")
    init_db()

    # 2. Pre-load FAISS vector store & SentenceTransformer model
    logger.info("Loading FAISS vector index & SentenceTransformer...")
    _ = get_vector_store()
    get_sentence_transformer()

    # Immediately release startup memory buffers
    gc.collect()

    elapsed = time.time() - start_time
    logger.info(f"✅ VibeSync Engine initialized successfully in {elapsed:.2f} seconds!")

    yield

    logger.info("🛑 Shutting down VibeSync Application...")


# Initialize FastAPI App
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="Production-grade AI Music Intelligence & Recommendation Engine API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add Rate Limiter State & Exception Handler
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Handle rate limit exceeded errors with structured JSON response."""
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )


# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production config can restrict to frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(auth.router)
app.include_router(health.router)
app.include_router(recommendations.router)
app.include_router(history.router)
app.include_router(lyrics.router)
app.include_router(analytics.router)

# Mount static files if static directory exists
static_dir = settings.BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Serve VibeSync glowing neon favicon."""
    fav_path = settings.BASE_DIR / "static" / "favicon.svg"
    if fav_path.exists():
        return FileResponse(fav_path, media_type="image/svg+xml")
    return HTMLResponse(content="", status_code=404)


@app.get("/", response_class=HTMLResponse)
def root_page():
    """Serve VibeSync Web UI homepage from frontend/index.html."""
    for candidate in [
        settings.BASE_DIR.parent / "frontend" / "index.html",
        settings.BASE_DIR / "frontend" / "index.html",
        settings.BASE_DIR / "index.html",
    ]:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return f.read()
    return "<h1>VibeSync AI API is running. Visit <a href='/docs'>/docs</a> for API specifications.</h1>"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True, app_dir=str(BACKEND_DIR))
