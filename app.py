import sys
import os
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer

# 🔧 PATH FIX
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spotify_api import get_song_details
from src.recommender import recommend
from src.embedding import load_embeddings
from src.emotion import detect_emotion
from src.data_loader import load_all_datasets


# 🎧 SESSION STATE
if "current_track" not in st.session_state:
    st.session_state.current_track = None


# 🎧 PLAYER
def spotify_player(track_url):
    if track_url:
        if "search" in track_url:
            st.warning("⚠️ Open in Spotify")
            st.markdown(f"[▶ Open]({track_url})")
        else:
            embed_url = track_url.replace(
                "open.spotify.com/track",
                "open.spotify.com/embed/track"
            )

            st.markdown(f"""
            <iframe src="{embed_url}"
            width="100%" height="80"
            frameborder="0"
            allow="autoplay; encrypted-media">
            </iframe>
            """, unsafe_allow_html=True)


# ⚡ LOAD MODEL
@st.cache_resource
def load_model():
    return SentenceTransformer(
        'paraphrase-multilingual-MiniLM-L12-v2',
        device='cpu'
    )


# ⚡ LOAD DATA
@st.cache_data
def load_data():
    df = load_all_datasets()
    vectors = load_embeddings()
    return df, vectors


# 🎧 CONFIG
st.set_page_config(page_title="VibeSync", layout="wide")




# 📄 LOAD HTML TEMPLATE
with open("homepage.html") as f:
    html_template = f.read()

# 🎨 LOAD STITCH CSS
with open("styles.css") as f:
    css = f.read()

# Inject CSS into HTML
html_template = html_template.replace(
    "</head>",
    f"<style>{css}</style></head>"
)

# 🔍 INPUT (STILL STREAMLIT)
user_input = st.text_input("Search music...")


# LOAD DATA
df, vectors = load_data()
# model = load_model()


# 🎯 MAIN LOGIC
song_html = ""
player_html = ""

if user_input:

    with st.spinner("Finding your vibe 🎧..."):
        results = recommend(user_input, None, df, vectors)

    # 🎭 Emotion
    if len(user_input.split()) > 2:
        emotion = detect_emotion(user_input)
        st.markdown(f"### 🎭 Mood: {emotion.upper()}")

    if results.empty:
        song_html = "<p>No songs found 😔</p>"

    else:
        for idx, (_, row) in enumerate(results.head(10).iterrows()):
            song = row['song_name']
            artist = row['artist']

            try:
                details = get_song_details(song, artist)
            except:
                details = None

            image = ""
            url = ""

            if details:
                image = details.get("image") or "https://via.placeholder.com/300"
                url = details.get("url", "")

            song_html += f"""
            <div class="song-card">
                <img src="{image}" class="song-img"/>
                <div class="song-title">{song}</div>
                <div class="song-artist">{artist}</div>
            </div>
            """

            # 🎮 STREAMLIT BUTTON (outside HTML)
            if st.button(f"▶ Play {song}", key=f"{song}_{idx}"):
                st.session_state.current_track = url


# 🎧 PLAYER HTML
if st.session_state.current_track:
    track_url = st.session_state.current_track

    if "search" in track_url:
        player_html = f"""
        <a href="{track_url}" target="_blank">▶ Open in Spotify</a>
        """
    else:
        embed_url = track_url.replace(
            "open.spotify.com/track",
            "open.spotify.com/embed/track"
        )

        player_html = f"""
        <iframe src="{embed_url}" width="100%" height="80"></iframe>
        """


# 🔥 FINAL HTML RENDER
final_html = html_template.replace("{{SONGS}}", song_html)
final_html = final_html.replace("{{PLAYER}}", player_html)

import streamlit.components.v1 as components

components.html(final_html, height=1200, scrolling=True)