import sys
import numpy as np
from tensorflow.keras.models import load_model

from src.preprocessing import clean_text
from src.tokenizer import load_tokenizer, pad_sequences_fn
from src.exception.exception import PredictionException


# Configurations

MODEL_PATH = "saved_models/birnn_model.h5"
TOKENIZER_PATH = "saved_models/tokenizer.pkl"

MAX_SEQUENCE_LENGTH = 100
THRESHOLD = 0.5


EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]


# Load Model and Tokenizer

try:
    model = load_model(MODEL_PATH)
    tokenizer = load_tokenizer(TOKENIZER_PATH)

except Exception as e:
    raise PredictionException(
        error_message="Error loading model or tokenizer",
        error_detail=sys
    ) from e



# Prediction Function

def predict_emotions(text: str):
    try:
        if not text.strip():
            raise ValueError("Empty input text")

        # 1. Preprocess
        cleaned_text = clean_text(text)

        # 2. Tokenize
        sequence = tokenizer.texts_to_sequences([cleaned_text])

        # 3. Pad
        padded = pad_sequences_fn(
            sequence,
            max_len=MAX_SEQUENCE_LENGTH
        )

        # 4. Predict
        preds = model.predict(padded)[0]

        # 5. Decode predictions
        results = {}
        for idx, score in enumerate(preds):
            if score >= THRESHOLD:
                results[EMOTION_LABELS[idx]] = float(score)

        return results

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

        emotions = predict_emotions(text)

        if not emotions:
            print("\nNo strong emotions detected.\n")
        else:
            print("\nPredicted emotions:")
            for emotion, score in emotions.items():
                print(f"  {emotion}: {score:.2f}")
            print()
