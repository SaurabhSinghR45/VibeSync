from transformers import pipeline

emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")


def detect_emotion(text):
    result = emotion_model(text)

    # Handle both cases (list or dict)
    if isinstance(result, list):
        result = result[0]

    return result["label"]
