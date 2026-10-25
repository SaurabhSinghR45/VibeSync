from api.schemas import LyricTranslationRequest
from fastapi import APIRouter, HTTPException
from src.translator import translate_lyrics

router = APIRouter(prefix="/api/v1/lyrics", tags=["Lyrics & Emotion Heatmaps"])


@router.post("/translate")
def get_translated_lyrics(payload: LyricTranslationRequest):
    """
    Translate Hindi, Punjabi, or English lyrics into English with line-by-line sentiment emotion heatmaps.
    """
    try:
        res = translate_lyrics(
            song_name=payload.song_name,
            artist=payload.artist,
            target_language=payload.target_language or "English",
            custom_lyrics=payload.custom_lyrics,
        )
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lyric translation failed: {exc}")
