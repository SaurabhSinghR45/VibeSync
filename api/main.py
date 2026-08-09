import os
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.config import settings
from src.database import init_db
from src.recommender import get_vector_store, get_sentence_transformer
from api.routes import health, recommendations, history

# Setup Logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    vector_store = get_vector_store()
    get_sentence_transformer()
    
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
    lifespan=lifespan
)

# Add Rate Limiter State & Exception Handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production config can restrict to frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health.router)
app.include_router(recommendations.router)
app.include_router(history.router)

# Mount static files if static directory exists
static_dir = settings.BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
def root_page():
    """Serve VibeSync Web UI homepage."""
    code_html_path = settings.BASE_DIR / "code.html"
    if code_html_path.exists():
        with open(code_html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>VibeSync AI API is running. Visit <a href='/docs'>/docs</a> for API specifications.</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
