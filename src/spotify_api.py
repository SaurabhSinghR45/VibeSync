import requests
import base64
import time
import logging
from typing import Optional, Dict, Any
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
        headers = {
            "Authorization": f"Basic {auth_base64}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
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

def get_song_details(song: str, artist: str) -> Dict[str, Any]:
    """
    Retrieve track URL, album art image, and audio preview URL from Spotify Web API.
    Provides reliable fallback search URL if API is unauthenticated or unavailable.
    """
    query = f"{song} {artist}".strip()
    encoded_query = requests.utils.quote(query)
    search_url = f"https://open.spotify.com/search/{encoded_query}"

    fallback_response = {
        "url": search_url,
        "image": None,
        "preview_url": None,
        "spotify_id": None
    }

    try:
        token = get_token()
        if not token:
            return fallback_response

        headers = {"Authorization": f"Bearer {token}"}
        url = f"https://api.spotify.com/v1/search?q={encoded_query}&type=track&limit=1"

        response = requests.get(url, headers=headers, timeout=5.0)

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

            return {
                "url": external_url,
                "image": image_url,
                "preview_url": preview_url,
                "spotify_id": track.get("id")
            }

    except requests.RequestException as exc:
        logger.error(f"Network exception searching Spotify for track '{query}': {exc}")
    except Exception as exc:
        logger.error(f"Unexpected exception searching Spotify for track '{query}': {exc}")

    return fallback_response