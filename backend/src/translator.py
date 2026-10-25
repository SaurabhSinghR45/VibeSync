import logging
import re
from typing import Any, Dict, List, Optional, Union

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
    elif any(w in line_lower for w in ["god", "pray", "peace", "sky", "light", "joy", "happy", "rang", "smile"]):
        return "joy"
    return "neutral"


def _detect_language_name(text_lines: List[str], key: str) -> str:
    """Heuristic language detection for Indian & Global catalogs."""
    sample = " ".join(text_lines).lower() + " " + key.lower()
    if any(w in sample for w in ["nachde", "khalbali", "kudiye", "akhiyaan", "rab", "tainu", "bhangra", "punjabi"]):
        return "Punjabi 🇮🇳"
    elif any(w in sample for w in ["dil", "pyar", "khuda", "aansu", "sada", "hum", "tum", "mera", "meri", "hindi"]):
        return "Hindi 🇮🇳"
    elif any(w in sample for w in ["amor", "corazon", "vida", "noche", "fiesta", "adios", "cancion"]):
        return "Spanish 🇪🇸"
    return "English 🇬🇧"


def _extract_catalog_lyrics(song_name: str, artist: str) -> Optional[List[str]]:
    """Attempt to retrieve real lyric verses from the catalog dataset."""
    try:
        from src.data_loader import load_all_datasets

        df = load_all_datasets()
        if not df.empty and "lyrics" in df.columns:
            s_low = song_name.lower().strip()
            # Match song name
            matches = df[df["song_name"].str.lower().str.strip() == s_low]
            if matches.empty:
                # Substring match
                matches = df[df["song_name"].str.lower().str.contains(s_low, regex=False)]

            if not matches.empty:
                raw_lyrics = str(matches.iloc[0]["lyrics"])
                lines = [
                    re.sub(r"\[.*?\]", "", line).strip()
                    for line in raw_lyrics.splitlines()
                    if line.strip() and len(line.strip()) > 4 and not line.strip().startswith("[")
                ]
                if len(lines) >= 3:
                    return lines[:6]
    except Exception as exc:
        logger.debug(f"Catalog lyric extraction notice: {exc}")
    return None


def translate_lyrics(
    song_name: str,
    artist: str,
    target_language: str = "English",
    custom_lyrics: Optional[Union[List[str], str]] = None,
) -> Dict[str, Any]:
    """
    Translates Hindi, Punjabi, Spanish, or English lyrics with line-by-line emotion heatmaps.
    Features rate-limit safe batch execution and deterministic offline fallbacks.
    """
    key = song_name.lower().strip()
    curated = CURATED_LYRICS_DB.get(key)

    # 1. Normalize custom_lyrics parameter
    if isinstance(custom_lyrics, str):
        custom_lines = [line_text.strip() for line_text in custom_lyrics.splitlines() if line_text.strip()]
    elif isinstance(custom_lyrics, list):
        custom_lines = [str(line_text).strip() for line_text in custom_lyrics if str(line_text).strip()]
    else:
        custom_lines = None

    if curated and not custom_lines:
        original_lines = curated["original"]
        translated_lines = curated["translated"]
        detected_lang = curated["language"]
        line_emotions = curated["emotions"]
    else:
        # 2. Extract lines from custom input, catalog database, or melodic verses
        if custom_lines:
            original_lines = custom_lines[:8]
        else:
            catalog_lines = _extract_catalog_lyrics(song_name, artist)
            if catalog_lines:
                original_lines = catalog_lines
            else:
                original_lines = [
                    f"Close your eyes and feel the melody of {song_name}",
                    f"Echoing with harmonic passion by {artist}",
                    "Every beat synchronizes with your inner rhythm",
                    "Acoustic textures painting emotions through the night",
                    "Lost inside the soundscape of pure harmony",
                ]

        detected_lang = _detect_language_name(original_lines, key)
        line_emotions = [analyze_line_emotion(line) for line in original_lines]

        # 3. Batch Translate with Graceful Rate-Limit and Offline Handling
        translated_lines = []
        is_already_english = "English" in detected_lang

        if is_already_english:
            translated_lines = list(original_lines)
        else:
            try:
                # Use batch translation in a single network call to prevent 5 req/sec rate limiting
                translator = GoogleTranslator(source="auto", target="en")
                batch_res = translator.translate_batch(original_lines)
                if batch_res and len(batch_res) == len(original_lines):
                    translated_lines = [t or orig for t, orig in zip(batch_res, original_lines)]
                else:
                    translated_lines = list(original_lines)
            except Exception as exc:
                # Graceful fallback: Rate-limit or network disconnect
                logger.info(f"GoogleTranslator batch fallback for '{song_name}': {exc}")
                translated_lines = list(original_lines)

    # 4. Build Heatmap Line Objects
    heatmap_lines = []
    for orig, trans, emo in zip(original_lines, translated_lines, line_emotions):
        color_info = EMOTION_COLOR_MAP.get(emo, EMOTION_COLOR_MAP["neutral"])
        heatmap_lines.append(
            {
                "original": orig,
                "translated": trans or orig,
                "emotion": emo,
                "color": color_info["color"],
                "label": color_info["label"],
                "bg": color_info["bg"],
            }
        )

    # Calculate dominant emotion safely
    dominant_emo = "neutral"
    if line_emotions:
        valid_emos = [e for e in line_emotions if e != "neutral"] or line_emotions
        dominant_emo = max(set(valid_emos), key=valid_emos.count)

    return {
        "song_name": song_name,
        "artist": artist,
        "language": detected_lang,
        "target_language": target_language,
        "heatmap_lines": heatmap_lines,
        "total_lines": len(heatmap_lines),
        "dominant_emotion": dominant_emo,
    }
