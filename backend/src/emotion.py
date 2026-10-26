import logging
import os
import re

logger = logging.getLogger("vibesync.emotion")

# Curated high-precision emotion lexicon for zero-RAM instant keyword inference
EMOTION_LEXICON = {
    "joy": {
        "happy",
        "joy",
        "joyful",
        "party",
        "dance",
        "celebrate",
        "uplifting",
        "vibes",
        "cheerful",
        "fun",
        "summer",
        "sunshine",
        "bliss",
        "excited",
        "groove",
        "ecstatic",
        "hype",
        "energetic",
        "club",
        "disco",
    },
    "sadness": {
        "sad",
        "sadness",
        "depressed",
        "depressing",
        "heartbreak",
        "heartbroken",
        "crying",
        "tears",
        "lonely",
        "alone",
        "miss",
        "grief",
        "pain",
        "broken",
        "melancholy",
        "melancholic",
        "cry",
        "hurt",
        "sorrow",
    },
    "anger": {
        "angry",
        "anger",
        "rage",
        "furious",
        "hate",
        "fight",
        "screaming",
        "mad",
        "rebel",
        "aggressive",
        "metal",
        "hardcore",
    },
    "fear": {
        "fear",
        "afraid",
        "dark",
        "scared",
        "horror",
        "eerie",
        "spooky",
        "anxious",
        "anxiety",
        "ghost",
        "tension",
        "thriller",
    },
    "surprise": {
        "surprise",
        "shock",
        "unexpected",
        "astonishing",
        "wild",
        "unreal",
        "magic",
        "wonder",
        "dynamic",
    },
    "love": {
        "love",
        "romance",
        "romantic",
        "kiss",
        "lovers",
        "crush",
        "passion",
        "sweetheart",
        "darling",
        "intimate",
        "affection",
    },
    "neutral": {
        "chill",
        "calm",
        "relax",
        "relaxing",
        "peace",
        "peaceful",
        "study",
        "ambient",
        "lofi",
        "quiet",
        "focus",
        "smooth",
        "drive",
        "night",
        "coffee",
    },
}

_lazy_hf_pipeline = None


def _get_hf_pipeline():
    """Lazy loader for optional heavy DistilRoBERTa model (only invoked if USE_HEAVY_EMOTION_MODEL=true)."""
    global _lazy_hf_pipeline
    if _lazy_hf_pipeline is None:
        from transformers import pipeline

        logger.info("Loading DistilRoBERTa emotion pipeline (high-memory instance)...")
        _lazy_hf_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    return _lazy_hf_pipeline


def detect_emotion(text: str) -> str:
    """
    Detect the dominant emotion from user prompt.
    Uses ultra-fast, zero-RAM lexical matching by default to operate comfortably under Render's 512MB RAM limit.
    """
    if not text or not text.strip():
        return "neutral"

    # Optional opt-in for instances with >1GB RAM
    if os.getenv("USE_HEAVY_EMOTION_MODEL", "false").lower() in ("true", "1", "yes"):
        try:
            pipe = _get_hf_pipeline()
            result = pipe(text)
            if isinstance(result, list):
                result = result[0]
            label = str(result.get("label", "joy")).lower()
            return label
        except Exception as exc:
            logger.warning(f"Heavy emotion pipeline failed ({exc}), falling back to lightweight detector.")

    # High-precision zero-RAM token matching
    words = set(re.findall(r"\b[a-z]+\b", text.lower()))
    for emotion, keywords in EMOTION_LEXICON.items():
        if words & keywords:
            return emotion

    # Check common 2-word phrases
    text_lower = text.lower()
    for emotion, keywords in EMOTION_LEXICON.items():
        for kw in keywords:
            if kw in text_lower:
                return emotion

    return "joy"
