import base64
import logging
import time
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests
from src.config import settings

logger = logging.getLogger("vibesync.spotify")

_token_cache: Optional[str] = None
_token_expiry: float = 0.0


def get_token() -> Optional[str]:
    """
    Fetch Spotify OAuth Client Credentials Access Token.
    Caches token in memory until 60 seconds before expiration.
    """
    global _token_cache, _token_expiry

    current_time = time.time()
    if _token_cache and current_time < _token_expiry:
        return _token_cache

    client_id = settings.SPOTIFY_CLIENT_ID
    client_secret = settings.SPOTIFY_CLIENT_SECRET

    if not client_id or not client_secret or client_id == "your_spotify_client_id_here":
        logger.warning("Spotify credentials missing in settings. Using fallback mode.")
        return None

    try:
        auth_string = f"{client_id}:{client_secret}"
        auth_bytes = auth_string.encode("utf-8")
        auth_base64 = base64.b64encode(auth_bytes).decode("utf-8")

        url = "https://accounts.spotify.com/api/token"
        headers = {"Authorization": f"Basic {auth_base64}", "Content-Type": "application/x-www-form-urlencoded"}
        data = {"grant_type": "client_credentials"}

        response = requests.post(url, headers=headers, data=data, timeout=5.0)

        if response.status_code != 200:
            logger.error(f"Spotify token authentication failed [Status {response.status_code}]: {response.text}")
            return None

        payload = response.json()
        access_token = payload.get("access_token")
        expires_in = payload.get("expires_in", 3600)

        if access_token:
            _token_cache = access_token
            # Cache until 60 seconds before token expires
            _token_expiry = current_time + (expires_in - 60)
            return access_token

    except requests.RequestException as exc:
        logger.error(f"Network error during Spotify token request: {exc}")
    except Exception as exc:
        logger.error(f"Unexpected error during Spotify token request: {exc}")

    return None


_song_details_cache: Dict[str, Dict[str, Any]] = {}
_spotify_api_disabled_until: float = 0.0


def get_song_details(song: str, artist: str) -> Dict[str, Any]:
    """
    Retrieve track URL, album art image, and audio preview URL from Spotify Web API.
    Provides reliable fallback search URL if API is unauthenticated or unavailable.
    """
    global _spotify_api_disabled_until
    query = f"{song} {artist}".strip()
    cache_key = query.lower()
    if cache_key in _song_details_cache:
        return _song_details_cache[cache_key]

    encoded_query = quote(query)
    search_url = f"https://open.spotify.com/search/{encoded_query}"
    fallback_response = {"url": search_url, "image": None, "preview_url": None, "spotify_id": None}

    if time.time() < _spotify_api_disabled_until:
        return fallback_response

    try:
        token = get_token()
        if not token:
            _song_details_cache[cache_key] = fallback_response
            return fallback_response

        headers = {"Authorization": f"Bearer {token}"}
        url = f"https://api.spotify.com/v1/search?q={encoded_query}&type=track&limit=1"

        response = requests.get(url, headers=headers, timeout=1.5)

        if response.status_code == 403:
            logger.warning(
                "Spotify API returned 403 Forbidden. Disabling remote queries for 300s to maintain instant latency."
            )
            _spotify_api_disabled_until = time.time() + 300.0
            _song_details_cache[cache_key] = fallback_response
            return fallback_response

        if response.status_code != 200:
            logger.warning(f"Spotify search returned non-200 status [{response.status_code}] for query: {query}")
            return fallback_response

        data = response.json()
        tracks = data.get("tracks", {}).get("items", [])

        if tracks:
            track = tracks[0]
            images = track.get("album", {}).get("images", [])
            image_url = images[0]["url"] if images else None
            preview_url = track.get("preview_url")
            external_url = track.get("external_urls", {}).get("spotify", search_url)

            res = {"url": external_url, "image": image_url, "preview_url": preview_url, "spotify_id": track.get("id")}
            _song_details_cache[cache_key] = res
            return res

    except requests.RequestException as exc:
        logger.error(f"Network exception searching Spotify for track '{query}': {exc}")
    except Exception as exc:
        logger.error(f"Unexpected exception searching Spotify for track '{query}': {exc}")

    _song_details_cache[cache_key] = fallback_response
    return fallback_response


def exchange_code_for_user_token(code_or_token: str, redirect_uri: str = "http://127.0.0.1:8001/") -> Optional[str]:
    """
    If code_or_token is a Spotify authorization code, exchange it for a User Access Token.
    """
    if not code_or_token:
        return None

    if code_or_token.startswith("BQ"):
        return code_or_token

    client_id = settings.SPOTIFY_CLIENT_ID
    client_secret = settings.SPOTIFY_CLIENT_SECRET

    if not client_id or not client_secret:
        return code_or_token

    auth_string = f"{client_id}:{client_secret}"
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = base64.b64encode(auth_bytes).decode("utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {"Authorization": f"Basic {auth_base64}", "Content-Type": "application/x-www-form-urlencoded"}

    uris_to_try = [
        redirect_uri,
        redirect_uri.rstrip("/"),
        redirect_uri.rstrip("/") + "/",
        "http://127.0.0.1:8001/",
        "http://127.0.0.1:8001",
    ]

    for uri in uris_to_try:
        try:
            data = {
                "grant_type": "authorization_code",
                "code": code_or_token,
                "redirect_uri": uri,
                "client_id": client_id,
                "client_secret": client_secret,
            }
            res = requests.post(url, headers=headers, data=data, timeout=5.0)
            if res.status_code == 200:
                token = res.json().get("access_token")
                if token:
                    return token
            else:
                logger.warning(f"Code exchange failed with URI '{uri}' [{res.status_code}]: {res.text}")
        except Exception as exc:
            logger.error(f"Error in token exchange: {exc}")

    return code_or_token


def get_user_spotify_profile(user_token: str) -> Optional[Dict[str, Any]]:
    """Fetch logged-in user profile details from Spotify API."""
    try:
        token = exchange_code_for_user_token(user_token)
        headers = {"Authorization": f"Bearer {token}"}
        res = requests.get("https://api.spotify.com/v1/me", headers=headers, timeout=5.0)
        if res.status_code == 200:
            return res.json()
    except Exception as exc:
        logger.error(f"Error fetching user Spotify profile: {exc}")
    return None


def create_spotify_playlist(user_token: str, playlist_name: str, description: str, track_ids: list) -> Dict[str, Any]:
    """
    Create a new playlist on user's Spotify account and add track IDs to it.
    Returns playlist URL and status details.
    """
    token = exchange_code_for_user_token(user_token)
    if not token:
        return {"success": False, "message": "Missing Spotify user authentication token."}

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Step 1: Create Playlist via Spotify Web API /v1/me/playlists
    create_url = "https://api.spotify.com/v1/me/playlists"
    payload = {"name": playlist_name, "description": description, "public": True}

    try:
        res = requests.post(create_url, headers=headers, json=payload, timeout=5.0)
        logger.info(f"Spotify create playlist status [{res.status_code}]: {res.text}")

        if res.status_code not in (200, 201):
            return {"success": False, "message": f"Spotify API error ({res.status_code}): {res.text}"}

        playlist_data = res.json()
        playlist_id = playlist_data.get("id")
        playlist_url = playlist_data.get("external_urls", {}).get("spotify")

        # Step 2: Format Track URIs
        track_uris = []
        for tid in track_ids:
            if not tid:
                continue
            if len(tid) == 22 and not tid.startswith("TRK_"):
                track_uris.append(f"spotify:track:{tid}")
            elif ":" in tid:
                track_uris.append(tid)

        if track_uris and playlist_id:
            add_url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
            add_res = requests.post(add_url, headers=headers, json={"uris": track_uris[:100]}, timeout=5.0)
            logger.info(f"Add tracks to Spotify playlist response [{add_res.status_code}]")

        return {
            "success": True,
            "playlist_id": playlist_id,
            "playlist_url": playlist_url,
            "access_token": token,
            "total_tracks_added": len(track_uris),
            "message": f"Successfully created playlist '{playlist_name}'!",
        }

    except Exception as exc:
        logger.error(f"Error creating Spotify playlist: {exc}")
        return {"success": False, "message": str(exc)}
