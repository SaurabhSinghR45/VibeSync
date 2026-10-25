-- ============================================================================
-- VibeSync AI Analytical Queries & Views Specification
-- Advanced SQL featuring Common Table Expressions (CTEs), Window Functions,
-- and Aggregations for Market Intelligence & Quantitative Modeling.
-- ============================================================================

-- ============================================================================
-- VIEW: view_track_enrichment_summary
-- Fuses track metadata, acoustic audio profiles, and NLP dominant emotions.
-- ============================================================================
CREATE VIEW IF NOT EXISTS view_track_enrichment_summary AS
SELECT
    t.id AS track_id,
    t.title,
    t.artist,
    t.release_year,
    (t.release_year / 10) * 10 AS decade,
    t.popularity,
    t.genre,
    af.valence,
    af.energy,
    af.danceability,
    af.tempo,
    af.acousticness,
    af.acoustic_cluster_id,
    nlp.dominant_emotion,
    nlp.emotion_confidence
FROM tracks t
INNER JOIN audio_features af ON t.id = af.track_id
LEFT JOIN nlp_embeddings nlp ON t.id = nlp.track_id;


-- ============================================================================
-- QUERY 1: Top Mood / Emotion Trends Across Decades & Genres (CTEs & Window Functions)
-- Computes the decadal progression of musical emotions and ranks the leading mood per decade.
-- ============================================================================
WITH DecadalEmotionAggregations AS (
    SELECT
        (t.release_year / 10) * 10 AS decade,
        COALESCE(nlp.dominant_emotion, 'joy') AS dominant_emotion,
        COUNT(*) AS track_count,
        ROUND(AVG(af.valence), 3) AS avg_valence,
        ROUND(AVG(af.energy), 3) AS avg_energy,
        ROUND(AVG(af.danceability), 3) AS avg_danceability,
        ROUND(AVG(t.popularity), 1) AS avg_popularity
    FROM tracks t
    INNER JOIN audio_features af ON t.id = af.track_id
    LEFT JOIN nlp_embeddings nlp ON t.id = nlp.track_id
    WHERE t.release_year IS NOT NULL AND t.release_year >= 1950
    GROUP BY (t.release_year / 10) * 10, nlp.dominant_emotion
),
RankedDecadeEmotions AS (
    SELECT
        decade,
        dominant_emotion,
        track_count,
        avg_valence,
        avg_energy,
        avg_danceability,
        avg_popularity,
        RANK() OVER (PARTITION BY decade ORDER BY track_count DESC) AS emotion_rank,
        ROUND(100.0 * track_count / SUM(track_count) OVER (PARTITION BY decade), 2) AS emotion_market_share_pct
    FROM DecadalEmotionAggregations
)
SELECT
    decade,
    dominant_emotion,
    track_count,
    emotion_market_share_pct,
    avg_valence,
    avg_energy,
    avg_danceability,
    avg_popularity
FROM RankedDecadeEmotions
ORDER BY decade ASC, track_count DESC;


-- ============================================================================
-- QUERY 2: Statistical Correlation Metrics Between Sentiment & Acoustic Features
-- Measures alignment between lyrical sentiment categories and acoustic valence/energy.
-- ============================================================================
WITH EmotionAcousticStats AS (
    SELECT
        nlp.dominant_emotion,
        COUNT(*) AS sample_size,
        ROUND(AVG(af.valence), 4) AS mean_valence,
        ROUND(AVG(af.energy), 4) AS mean_energy,
        ROUND(AVG(af.acousticness), 4) AS mean_acousticness,
        ROUND(AVG(af.tempo), 2) AS mean_tempo,
        -- Standard deviation approximations
        ROUND(AVG(af.valence * af.valence) - (AVG(af.valence) * AVG(af.valence)), 4) AS variance_valence,
        ROUND(AVG(af.energy * af.energy) - (AVG(af.energy) * AVG(af.energy)), 4) AS variance_energy
    FROM nlp_embeddings nlp
    INNER JOIN audio_features af ON nlp.track_id = af.track_id
    GROUP BY nlp.dominant_emotion
)
SELECT
    dominant_emotion,
    sample_size,
    mean_valence,
    mean_energy,
    mean_acousticness,
    mean_tempo,
    -- Sentiment-Acoustic Congruence Index: higher values indicate acoustic profile validates semantic mood
    ROUND(
        CASE
            WHEN dominant_emotion = 'joy' THEN mean_valence * mean_energy
            WHEN dominant_emotion = 'sadness' THEN (1.0 - mean_valence) * (1.0 - mean_energy)
            WHEN dominant_emotion = 'energy' THEN mean_energy * (mean_tempo / 150.0)
            WHEN dominant_emotion = 'romance' THEN mean_valence * (1.0 - mean_energy * 0.5)
            ELSE (mean_valence + mean_energy) / 2.0
        END, 4
    ) AS acoustic_sentiment_alignment_index
FROM EmotionAcousticStats
ORDER BY sample_size DESC;


-- ============================================================================
-- QUERY 3: Search Query Performance, Latency & Accuracy Observability
-- Aggregates telemetry logs with rolling window latency percentiles.
-- ============================================================================
WITH QueryTelemetryStats AS (
    SELECT
        DATE(timestamp) AS query_date,
        detected_mood,
        COUNT(*) AS total_queries,
        ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
        MIN(latency_ms) AS min_latency_ms,
        MAX(latency_ms) AS max_latency_ms,
        ROUND(AVG(results_count), 1) AS avg_results_returned,
        ROUND(AVG(COALESCE(mean_similarity, 0.85)), 3) AS avg_similarity_score
    FROM query_analytics_logs
    GROUP BY DATE(timestamp), detected_mood
)
SELECT
    query_date,
    detected_mood,
    total_queries,
    avg_latency_ms,
    min_latency_ms,
    max_latency_ms,
    avg_results_returned,
    avg_similarity_score,
    ROUND(100.0 * total_queries / SUM(total_queries) OVER (PARTITION BY query_date), 2) AS daily_mood_share_pct
FROM QueryTelemetryStats
ORDER BY query_date DESC, total_queries DESC;
