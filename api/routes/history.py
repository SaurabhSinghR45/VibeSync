import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from api.schemas import SpotifyExportRequest
from src.database import LikedSong, SearchHistory, VibePlaylist, get_session

router = APIRouter(prefix="/api/v1", tags=["User History & Playlists"])


class LikedSongCreate(BaseModel):
    track_id: str
    song_name: str
    artist: str
    album_art: Optional[str] = None
    spotify_url: Optional[str] = None


class PlaylistCreate(BaseModel):
    playlist_name: str
    track_ids: List[str]


@router.get("/history")
def get_search_history(limit: int = 20, session: Session = Depends(get_session)):
    """Retrieve recent AI search history."""
    statement = select(SearchHistory).order_by(SearchHistory.timestamp.desc()).limit(limit)
    history = session.exec(statement).all()
    return {"history": history}


@router.get("/likes")
def get_liked_songs(session: Session = Depends(get_session)):
    """Retrieve user favorite liked songs."""
    statement = select(LikedSong).order_by(LikedSong.created_at.desc())
    likes = session.exec(statement).all()
    return {"liked_songs": likes}


@router.post("/likes")
def add_liked_song(payload: LikedSongCreate, session: Session = Depends(get_session)):
    """Add a track to user liked songs."""
    existing = session.exec(select(LikedSong).where(LikedSong.track_id == payload.track_id)).first()
    if existing:
        return {"status": "exists", "message": "Song already in liked list."}

    song = LikedSong(
        track_id=payload.track_id,
        song_name=payload.song_name,
        artist=payload.artist,
        album_art=payload.album_art,
        spotify_url=payload.spotify_url,
    )
    session.add(song)
    session.commit()
    session.refresh(song)
    return {"status": "success", "liked_song": song}


@router.delete("/likes/{track_id}")
def remove_liked_song(track_id: str, session: Session = Depends(get_session)):
    """Remove a track from user liked songs."""
    song = session.exec(select(LikedSong).where(LikedSong.track_id == track_id)).first()
    if not song:
        raise HTTPException(status_code=404, detail="Track not found in liked list.")

    session.delete(song)
    session.commit()
    return {"status": "success", "message": f"Removed track '{track_id}' from liked list."}


@router.get("/playlists")
def get_playlists(session: Session = Depends(get_session)):
    """Retrieve user vibe playlists."""
    statement = select(VibePlaylist).order_by(VibePlaylist.created_at.desc())
    playlists = session.exec(statement).all()

    results = []
    for p in playlists:
        results.append(
            {
                "id": p.id,
                "playlist_name": p.playlist_name,
                "track_ids": json.loads(p.track_ids_json) if p.track_ids_json else [],
                "created_at": p.created_at,
            }
        )
    return {"playlists": results}


@router.post("/playlists")
def create_playlist(payload: PlaylistCreate, session: Session = Depends(get_session)):
    """Create a new user vibe playlist."""
    playlist = VibePlaylist(playlist_name=payload.playlist_name, track_ids_json=json.dumps(payload.track_ids))
    session.add(playlist)
    session.commit()
    session.refresh(playlist)
    return {"status": "success", "playlist": playlist}


@router.post("/playlists/export-spotify")
def export_to_spotify(payload: SpotifyExportRequest):
    """
    Export a VibeSync playlist directly to Spotify.
    If user_token is provided, uses user OAuth token to create playlist on user's account.
    """
    from src.spotify_api import create_spotify_playlist

    if not payload.user_token:
        # Fallback response when user token is not present
        return {
            "status": "requires_auth",
            "message": "Spotify OAuth login required to export directly to user account.",
            "auth_url": "https://accounts.spotify.com/authorize?client_id=5e5265a82d9040ec973ae86c1695834f&response_type=code&redirect_uri=http://127.0.0.1:8001/&scope=user-read-private%20user-read-email%20playlist-modify-public%20playlist-modify-private",
        }

    result = create_spotify_playlist(
        user_token=payload.user_token,
        playlist_name=payload.playlist_name,
        description=payload.description,
        track_ids=payload.track_ids,
    )
    return result


@router.get("/spotify/me")
def get_spotify_me(token: str):
    """Fetch profile info of currently authorized Spotify user."""
    from src.spotify_api import get_user_spotify_profile

    profile = get_user_spotify_profile(token)
    if not profile:
        raise HTTPException(status_code=401, detail="Invalid or expired Spotify user token.")
    return profile


@router.get("/spotify/token-exchange")
def token_exchange(code: str):
    """Exchange authorization code for valid access token."""
    from src.spotify_api import exchange_code_for_user_token

    token = exchange_code_for_user_token(code)
    if not token or token == code:
        raise HTTPException(status_code=400, detail="Failed to exchange authorization code for access token.")
    return {"access_token": token}
