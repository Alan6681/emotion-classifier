import streamlit as st
import numpy as np

from src.predict import predict_emotions
from src.preprocessing import clean_text

st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Emotion Classifier")
st.write(
    "This model predicts **multiple emotions** from text. "
    "Emotions may overlap because human feelings are complex."
)

text_input = st.text_area(
    "Enter text",
    placeholder="Type something emotional..."
)

if st.button("Analyze Emotion"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            emotions = predict_emotions(text_input)

        if not emotions:
            st.info("No strong emotions detected.")
        else:
            st.subheader("Predicted Emotions")

            for emotion, score in emotions.items():
                st.write(f"**{emotion.capitalize()}** — {score:.2f}")

            # Bar chart
            st.bar_chart(emotions)

st.markdown("---")
st.caption(
    "⚠️ This model is trained on the GoEmotions dataset and may reflect dataset biases. "
    "Predictions represent statistical associations, not absolute emotional truth."
)
