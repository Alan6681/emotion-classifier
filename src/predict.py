import sys
import numpy as np
from tensorflow.keras.models import load_model

from src.preprocessing import clean_text
from src.tokenizer import load_tokenizer, pad_sequences_fn
from src.exception.exception import PredictionException



MODEL_PATH = "saved_models/birnn_model.h5"
TOKENIZER_PATH = "saved_models/tokenizer.pkl"

MAX_SEQUENCE_LENGTH = 100
TOP_K = 3   # <-- Top 3 emotions


EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]


try:
    model = load_model(MODEL_PATH)
    tokenizer = load_tokenizer(TOKENIZER_PATH)
except Exception as e:
    raise PredictionException(
        error_message="Error loading model or tokenizer",
        error_detail=sys
    ) from e




def get_top_k_emotions(predictions, labels, k=3):
    """
    Returns top-k emotions sorted by confidence.
    """
    emotion_scores = list(zip(labels, predictions))
    emotion_scores.sort(key=lambda x: x[1], reverse=True)
    return emotion_scores[:k]



def predict_emotions(text: str):
    try:
        if not text.strip():
            raise ValueError("Empty input text")

        # 1. Clean text
        cleaned_text = clean_text(text)

        # 2. Tokenize
        sequence = tokenizer.texts_to_sequences([cleaned_text])

        # 3. Pad
        padded = pad_sequences_fn(
            sequence,
            max_len=MAX_SEQUENCE_LENGTH
        )

        # 4. Predict
        predictions = model.predict(padded, verbose=0)[0]

        # 5. Get top-k emotions
        top_emotions = get_top_k_emotions(
            predictions,
            EMOTION_LABELS,
            k=TOP_K
        )

        return top_emotions

    except Exception as e:
        raise PredictionException(
            error_message="Error during prediction",
            error_detail=sys
        ) from e



if __name__ == "__main__":
    print("\nEmotion Classifier (type 'q' to quit)\n")

    while True:
        text = input("Enter text: ")

        if text.lower() == "q":
            break

        results = predict_emotions(text)

        print("\nTop 3 Predicted Emotions:")
        for emotion, score in results:
            print(f"  {emotion}: {score:.2f}")
        print()
