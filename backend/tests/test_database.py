from sqlmodel import Session, select
from src.database import LikedSong, SearchHistory, engine, init_db


def setup_module():
    """Initialize database schemas before running DB tests."""
    init_db()


def test_search_history_persistence():
    """Verify writing and querying SearchHistory records."""
    with Session(engine) as session:
        entry = SearchHistory(query="cyberpunk synthwave workout", input_type="hybrid", detected_emotion="joy")
        session.add(entry)
        session.commit()
        session.refresh(entry)

        assert entry.id is not None
        assert entry.query == "cyberpunk synthwave workout"

        # Query back
        fetched = session.exec(select(SearchHistory).where(SearchHistory.id == entry.id)).first()
        assert fetched is not None
        assert fetched.detected_emotion == "joy"


def test_liked_song_persistence():
    """Verify writing and deleting LikedSong records."""
    with Session(engine) as session:
        song = LikedSong(track_id="TRK_TEST_999", song_name="Blinding Lights Test", artist="The Weeknd Test")
        session.add(song)
        session.commit()

        fetched = session.exec(select(LikedSong).where(LikedSong.track_id == "TRK_TEST_999")).first()
        assert fetched is not None
        assert fetched.song_name == "Blinding Lights Test"

        session.delete(fetched)
        session.commit()
