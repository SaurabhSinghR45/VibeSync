def classify_input(text, df):
    text = text.lower().strip()

    emotion_keywords = ["sad", "happy", "love", "romantic", "angry", "chill"]

    artist_match = df["artist"].str.lower().str.contains(text, na=False).sum()
    song_match = df["song_name"].str.lower().str.contains(text, na=False).sum()

    # 🎵 Song priority
    if song_match > 0:
        return "song"

    # 🎤 Artist
    elif artist_match > 0:
        return "artist"

    # 😊 Emotion detection
    elif any(word in text for word in emotion_keywords):
        return "emotion"

    # 🧠 Lyrics
    elif len(text.split()) > 4:
        return "lyrics"

    else:
        return "emotion"
