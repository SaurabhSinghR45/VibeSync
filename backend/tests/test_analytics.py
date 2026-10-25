from api.main import app
from fastapi.testclient import TestClient
from src.database.analytics import AnalyticsService

client = TestClient(app)


def test_analytics_service_queries():
    service = AnalyticsService()

    trends = service.get_decadal_mood_trends()
    assert isinstance(trends, list)
    if len(trends) > 0:
        first = trends[0]
        assert "decade" in first
        assert "dominant_emotion" in first
        assert "emotion_market_share_pct" in first
        assert "avg_valence" in first

    correlations = service.get_acoustic_sentiment_correlations()
    assert isinstance(correlations, list)
    if len(correlations) > 0:
        corr = correlations[0]
        assert "dominant_emotion" in corr
        assert "acoustic_sentiment_alignment_index" in corr

    perf = service.get_query_performance_metrics()
    assert isinstance(perf, dict)
    assert "avg_latency_ms" in perf

    csv_data = service.export_analytics_csv()
    assert isinstance(csv_data, str)
    assert "Decadal Mood Trends" in csv_data


def test_analytics_api_endpoints():
    res_trends = client.get("/api/v1/analytics/trends")
    assert res_trends.status_code == 200
    assert "trends" in res_trends.json()

    res_corr = client.get("/api/v1/analytics/correlations")
    assert res_corr.status_code == 200
    assert "correlations" in res_corr.json()

    res_perf = client.get("/api/v1/analytics/performance")
    assert res_perf.status_code == 200
    assert "telemetry" in res_perf.json()

    res_health = client.get("/api/v1/analytics/dataset-health")
    assert res_health.status_code == 200
    assert "completeness_score" in res_health.json()

    res_csv = client.get("/api/v1/analytics/export-csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["content-type"]
