import logging
import time
from typing import Optional

from api.routes.auth import get_current_user_optional
from api.schemas import FeedbackRequest, RecommendRequest, RecommendResponse
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from src.data_loader import load_all_datasets
from src.database import FeedbackEvent, SearchHistory, User, get_session
from src.recommender import HybridRecommender

logger = logging.getLogger("vibesync.api.recommend")
router = APIRouter(prefix="/api/v1", tags=["Recommendations & AI Engine"])

# Load dataset singleton for API routes
_df_cache = None


def get_dataset():
    global _df_cache
    if _df_cache is None:
        _df_cache = load_all_datasets()
    return _df_cache


@router.post("/recommend", response_model=RecommendResponse)
def get_recommendations(
    payload: RecommendRequest,
    current_user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """
    Execute AI Hybrid Recommendation Search.
    Fuses FAISS ANN Vector Search, Emotion Confidence, and Explainable AI (XAI) breakdown.
    Guests can search freely without storing personal history; logged-in users track account history.
    """
    start_time = time.time()
    df = get_dataset()

    if df.empty:
        raise HTTPException(status_code=500, detail="Track dataset is empty or uninitialized.")

    recommender = HybridRecommender(df=df)
    results = recommender.recommend(payload.input, top_k=payload.top_k)

    latency = (time.time() - start_time) * 1000.0
    detected_emotion = results[0]["detected_emotion"] if results else "general"

    # Only log search history for authenticated users (guest searches are not stored)
    if current_user:
        try:
            history_entry = SearchHistory(
                user_id=current_user.id,
                query=payload.input,
                input_type="hybrid",
                detected_emotion=detected_emotion,
            )
            session.add(history_entry)
            session.commit()
        except Exception as exc:
            logger.error(f"Failed to log search history: {exc}")

    return RecommendResponse(
        query=payload.input,
        detected_emotion=detected_emotion,
        total_found=len(results),
        latency_ms=round(latency, 2),
        results=results,
    )


@router.post("/feedback")
def record_feedback(payload: FeedbackRequest, session: Session = Depends(get_session)):
    """
    Record user feedback event (like, dislike, play, skip) for recommender analytics.
    """
    try:
        feedback = FeedbackEvent(track_id=payload.track_id, event_type=payload.event_type)
        session.add(feedback)
        session.commit()
        return {
            "status": "success",
            "message": f"Recorded feedback '{payload.event_type}' for track '{payload.track_id}'",
        }
    except Exception as exc:
        logger.error(f"Failed to record feedback: {exc}")
        raise HTTPException(status_code=500, detail="Failed to persist feedback event.")
