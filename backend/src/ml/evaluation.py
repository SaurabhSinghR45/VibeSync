from typing import Any, Dict, List

import numpy as np


def evaluate_retrieval_performance(
    recommendations: List[Dict[str, Any]], query_text: str, detected_emotion: str, latency_ms: float
) -> Dict[str, Any]:
    """
    Quantitative evaluation utility calculating core retrieval benchmarks:
      1. Mean Cosine Similarity (MCS)
      2. Cluster Cohesion Score (CCS)
      3. Emotion Consistency Rate (ECR)
      4. Inference Latency (ms)
    """
    if not recommendations:
        return {
            "query": query_text,
            "detected_emotion": detected_emotion,
            "total_evaluated": 0,
            "mean_cosine_similarity": 0.0,
            "cluster_cohesion_score": 0.0,
            "emotion_consistency_rate": 0.0,
            "inference_latency_ms": round(latency_ms, 2),
            "status": "empty",
        }

    # 1. Mean Semantic Cosine Similarity
    sim_scores = [float(r.get("similarity_score", 0.0)) for r in recommendations]
    mean_cosine_sim = float(np.mean(sim_scores)) if sim_scores else 0.0

    # 2. Cluster Cohesion Score (concentration in dominant acoustic cluster)
    clusters = [r.get("acoustic_cluster_id", 0) for r in recommendations]
    if clusters:
        from collections import Counter

        counts = Counter(clusters)
        dominant_count = counts.most_common(1)[0][1]
        cluster_cohesion = round(dominant_count / len(clusters), 3)
    else:
        cluster_cohesion = 0.5

    # 3. Emotion Consistency Rate
    emotion_matches = 0
    for r in recommendations:
        track_emotion = r.get("detected_emotion", "").lower()
        if track_emotion == detected_emotion.lower() or (
            detected_emotion in ["joy", "happy"] and track_emotion in ["joy", "happy"]
        ):
            emotion_matches += 1
    emotion_consistency = round(emotion_matches / len(recommendations), 3)

    return {
        "query": query_text,
        "detected_emotion": detected_emotion,
        "total_evaluated": len(recommendations),
        "mean_cosine_similarity": round(mean_cosine_sim, 4),
        "cluster_cohesion_score": cluster_cohesion,
        "emotion_consistency_rate": emotion_consistency,
        "inference_latency_ms": round(latency_ms, 2),
        "status": "optimal",
    }
