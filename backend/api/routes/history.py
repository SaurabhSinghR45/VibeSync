import json
from typing import List, Optional

from api.routes.auth import get_current_user_optional
from api.schemas import SpotifyExportRequest
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, col, select
from src.database import LikedSong, SearchHistory, User, VibePlaylist, get_session

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
def get_search_history(
    limit: int = 20,
    current_user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """
    Retrieve search history.
    Logged-in users receive their personal history; guests return empty list.
    """
    if not current_user:
        return {"history": []}

    statement = (
        select(SearchHistory)
        .where(SearchHistory.user_id == current_user.id)
        .order_by(col(SearchHistory.timestamp).desc())
        .limit(limit)
    )
    history = session.exec(statement).all()
    return {"history": history}


@router.get("/likes")
def get_liked_songs(
    current_user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """Retrieve favorite liked songs scoped to user account."""
    if current_user:
        statement = (
            select(LikedSong).where(LikedSong.user_id == current_user.id).order_by(col(LikedSong.created_at).desc())
        )
    else:
        statement = select(LikedSong).where(col(LikedSong.user_id).is_(None)).order_by(col(LikedSong.created_at).desc())
    likes = session.exec(statement).all()
    return {"liked_songs": likes}


@router.post("/likes")
def add_liked_song(
    payload: LikedSongCreate,
    current_user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """Add a track to liked songs (linked to user account if authenticated)."""
    user_id = current_user.id if current_user else None

    query = select(LikedSong).where(LikedSong.track_id == payload.track_id)
    if user_id:
        query = query.where(LikedSong.user_id == user_id)
    else:
        query = query.where(col(LikedSong.user_id).is_(None))

    existing = session.exec(query).first()
    if existing:
        return {"status": "exists", "message": "Song already in liked list."}

    song = LikedSong(
        user_id=user_id,
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
def remove_liked_song(
    track_id: str,
    current_user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """Remove a track from liked songs."""
    query = select(LikedSong).where(LikedSong.track_id == track_id)
    if current_user:
        query = query.where(LikedSong.user_id == current_user.id)
    else:
        query = query.where(col(LikedSong.user_id).is_(None))

    song = session.exec(query).first()
    if not song:
        raise HTTPException(status_code=404, detail="Track not found in liked list.")

    session.delete(song)
    session.commit()
    return {"status": "success", "message": f"Removed track '{track_id}' from liked list."}


@router.get("/playlists")
def get_playlists(
    current_user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """Retrieve vibe playlists scoped to user account."""
    if current_user:
        statement = (
            select(VibePlaylist)
            .where(VibePlaylist.user_id == current_user.id)
            .order_by(col(VibePlaylist.created_at).desc())
        )
    else:
        statement = (
            select(VibePlaylist)
            .where(col(VibePlaylist.user_id).is_(None))
            .order_by(col(VibePlaylist.created_at).desc())
        )
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
def create_playlist(
    payload: PlaylistCreate,
    current_user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """Create a new user vibe playlist."""
    user_id = current_user.id if current_user else None
    playlist = VibePlaylist(
        user_id=user_id,
        playlist_name=payload.playlist_name,
        track_ids_json=json.dumps(payload.track_ids),
    )
    session.add(playlist)
    session.commit()
    session.refresh(playlist)
    return {"status": "success", "playlist": playlist}


@router.delete("/playlists/{playlist_id}")
def delete_playlist(
    playlist_id: int,
    current_user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """Delete a user vibe playlist."""
    user_id = current_user.id if current_user else None
    query = select(VibePlaylist).where(VibePlaylist.id == playlist_id)
    if user_id is not None:
        query = query.where(VibePlaylist.user_id == user_id)
    playlist = session.exec(query).first()
    if not playlist:
        raise HTTPException(status_code=404, detail="Playlist not found")
    session.delete(playlist)
    session.commit()
    return {"status": "success", "message": "Playlist deleted successfully."}


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
