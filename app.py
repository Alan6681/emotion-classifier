import streamlit as st
import numpy as np

from src.predict import predict_emotions


# =========================
# Page config
# =========================

st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Emotion Classifier")

st.write(
    "This model predicts the **top 3 emotions** present in a piece of text. "
    "Multiple emotions can coexist because human feelings are complex."
)


# =========================
# Input
# =========================

text_input = st.text_area(
    "Enter text",
    placeholder="Type something emotional..."
)


# =========================
# Prediction
# =========================

if st.button("Analyze Emotion"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            results = predict_emotions(text_input)

        st.subheader("Top Predicted Emotions")

        # Convert to dict for charting
        emotion_dict = {}

        for emotion, score in results:
            st.write(f"**{emotion.capitalize()}** — {score:.2f}")
            emotion_dict[emotion.capitalize()] = score

        # Bar chart
        st.bar_chart(emotion_dict)


# =========================
# Footer
# =========================

st.markdown("---")
st.caption(
    "⚠️ This model is trained on the GoEmotions dataset and may reflect dataset biases. "
    "Predictions represent statistical associations, not absolute emotional truth."
)
