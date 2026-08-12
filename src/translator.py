import logging
from typing import Any, Dict, List, Optional

from deep_translator import GoogleTranslator

logger = logging.getLogger("vibesync.translator")

# Emotion Color Mapping for Heatmap Visualization
EMOTION_COLOR_MAP = {
    "joy": {"color": "#ffd700", "label": "Joy / Uplifting 🟨", "bg": "rgba(255, 215, 0, 0.15)"},
    "sadness": {"color": "#00eefc", "label": "Melancholy / Sadness 🟦", "bg": "rgba(0, 238, 252, 0.15)"},
    "energy": {"color": "#ff6f7c", "label": "Hype / Energy 🟥", "bg": "rgba(255, 111, 124, 0.15)"},
    "romance": {"color": "#c59aff", "label": "Romance / Passion 🟪", "bg": "rgba(197, 154, 255, 0.15)"},
    "neutral": {"color": "#acaab1", "label": "Calm / Ambient ⬜", "bg": "rgba(172, 170, 177, 0.10)"},
}

# Curated Lyrics & Translations for Hindi, Punjabi, and English tracks
CURATED_LYRICS_DB = {
    "sun le sada": {
        "language": "Hindi 🇮🇳",
        "original": [
            "Sun le sada o mere khuda",
            "Dil ne pukara hai tu hai kahan",
            "Tanhaiyon me behte hain aansu",
            "Har saans me tera naam chhupa",
        ],
        "translated": [
            "Listen to my prayer, O my God",
            "My heart called out, where are you?",
            "Tears flow in these lonely nights",
            "Your name is hidden in every breath",
        ],
        "emotions": ["sadness", "sadness", "sadness", "romance"],
    },
    "dhuaan": {
        "language": "Hindi 🇮🇳",
        "original": [
            "Bujh gaya hai mera aasmaan",
            "Har taraf phaila hai ye dhuaan",
            "Yaadon ke mausam guzar na paaye",
            "Aankhon me bachi hai bas dastaan",
        ],
        "translated": [
            "My sky has gone dark and cold",
            "Smoke has spread all around me",
            "The seasons of memories refuse to pass",
            "Only stories remain in my eyes",
        ],
        "emotions": ["sadness", "sadness", "sadness", "romance"],
    },
    "khalbali": {
        "language": "Punjabi / Hindi 🇮🇳",
        "original": [
            "Tere naal naina mile te machi khalbali",
            "Nachde ne saare haan rang rali",
            "Dil vich mere tu hi tu vasdi",
            "Chad de tu gussa sun ho kudiye bhali",
        ],
        "translated": [
            "When my eyes met yours, madness broke out",
            "Everyone is dancing in joy and celebration",
            "Only you reside inside my heart",
            "Let go of anger and listen, sweet girl",
        ],
        "emotions": ["energy", "energy", "romance", "joy"],
    },
    "teray naal": {
        "language": "Punjabi 🇮🇳",
        "original": [
            "Teray naal jeena te teray naal marna",
            "Akhiyaan nu tere bina lagna nahi chaa",
            "Rab ton vi zyada tainu pyar assi karna",
            "Sada dil tere kadmaan ch dher ho gaya",
        ],
        "translated": [
            "To live with you and to die with you",
            "My eyes feel no peace without you",
            "I will love you even more than God",
            "My heart has fallen completely at your feet",
        ],
        "emotions": ["romance", "romance", "romance", "joy"],
    },
}


def analyze_line_emotion(line: str) -> str:
    """Analyze sentiment emotion tag for a lyric line."""
    line_lower = line.lower()
    if any(
        w in line_lower for w in ["sad", "tear", "cry", "lonely", "dark", "pain", "broken", "aansu", "bujh", "tanhai"]
    ):
        return "sadness"
    elif any(w in line_lower for w in ["love", "heart", "dil", "pyar", "passion", "saans", "naina", "sweet", "khuda"]):
        return "romance"
    elif any(w in line_lower for w in ["dance", "hype", "madness", "fire", "nachde", "khalbali", "party", "bhangra"]):
        return "energy"
    elif any(w in line_lower for w in ["god", "pray", "peace", "sky", "light", "joy", "happy", "rang"]):
        return "joy"
    return "neutral"


def translate_lyrics(
    song_name: str, artist: str, target_language: str = "English", custom_lyrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Translates Hindi, Punjabi, or English lyrics with line-by-line emotion heatmaps.
    """
    key = song_name.lower().strip()
    curated = CURATED_LYRICS_DB.get(key)

    if curated and not custom_lyrics:
        original_lines = curated["original"]
        translated_lines = curated["translated"]
        detected_lang = curated["language"]
        line_emotions = curated["emotions"]
    else:
        original_lines = custom_lyrics or [
            f"{song_name} - Line 1 (VibeSync Lyrics)",
            f"Melodic harmony by {artist}",
            "Rhythm matching emotion vector profile",
            "Pure acoustic atmosphere",
        ]
        detected_lang = (
            "Hindi 🇮🇳" if any(w in key for w in ["hindi", "arijit", "singh", "sada", "teray"]) else "English 🇬🇧"
        )
        translated_lines = []
        line_emotions = []

        translator = GoogleTranslator(source="auto", target="en")
        for line in original_lines:
            try:
                translated = translator.translate(line)
                translated_lines.append(translated or line)
            except Exception as e:
                logger.warning(f"Translation API fallback for '{line}': {e}")
                translated_lines.append(line)

            emotion = analyze_line_emotion(line)
            line_emotions.append(emotion)

    # Build Heatmap Line Objects
    heatmap_lines = []
    for orig, trans, emo in zip(original_lines, translated_lines, line_emotions):
        color_info = EMOTION_COLOR_MAP.get(emo, EMOTION_COLOR_MAP["neutral"])
        heatmap_lines.append(
            {
                "original": orig,
                "translated": trans,
                "emotion": emo,
                "color": color_info["color"],
                "label": color_info["label"],
                "bg": color_info["bg"],
            }
        )

    return {
        "song_name": song_name,
        "artist": artist,
        "language": detected_lang,
        "target_language": target_language,
        "heatmap_lines": heatmap_lines,
        "total_lines": len(heatmap_lines),
        "dominant_emotion": max(set(line_emotions), key=line_emotions.count) if line_emotions else "neutral",
    }
