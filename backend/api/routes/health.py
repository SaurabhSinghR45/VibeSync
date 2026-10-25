import time

from api.schemas import HealthResponse, MetricResponse
from fastapi import APIRouter, Depends
from sqlmodel import Session, func, select
from src.config import settings
from src.database import LikedSong, SearchHistory, get_session
from src.recommender import get_vector_store

router = APIRouter(prefix="/api/v1", tags=["Health & Observability"])
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
def health_check(session: Session = Depends(get_session)):
    """System health check endpoint."""
    vector_store = get_vector_store()
    index_size = vector_store.index.ntotal if vector_store.index else 0

    # Check DB
    db_ok = True
    try:
        session.exec(select(func.count(SearchHistory.id))).first()
    except Exception:
        db_ok = False

    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        version=settings.PROJECT_VERSION,
        environment=settings.ENVIRONMENT,
        vector_index_size=index_size,
        database_connected=db_ok,
    )


@router.get("/metrics", response_model=MetricResponse)
def system_metrics(session: Session = Depends(get_session)):
    """System observability telemetry metrics."""
    vector_store = get_vector_store()
    index_size = vector_store.index.ntotal if vector_store.index else 0

    search_count = session.exec(select(func.count(SearchHistory.id))).first() or 0
    like_count = session.exec(select(func.count(LikedSong.id))).first() or 0
    uptime = time.time() - _start_time

    return MetricResponse(
        total_tracks_indexed=index_size,
        total_searches_logged=search_count,
        total_liked_tracks=like_count,
        uptime_seconds=round(uptime, 2),
    )
