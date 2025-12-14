# 🧠 Emotion Classifier (BiRNN)

A **multi-label emotion classification system** built using a  
**Bidirectional Recurrent Neural Network (BiRNN)** and trained on the  
**GoEmotions dataset**.

The model predicts **multiple overlapping emotions** from text, reflecting the
complexity of human emotional expression.

---

## 🚀 Key Features

- Multi-label emotion prediction (not single-class)
- 27 emotions + neutral (GoEmotions)
- Bidirectional RNN (BiRNN) architecture
- Custom preprocessing pipeline
- Tokenizer fitted and reused consistently
- Threshold-based emotion decoding
- GPU-accelerated training (Google Colab)
- Interactive Streamlit web app
- Modular, production-style project structure

---

## 🧩 Project Structure

```
Emotion-Classifier/
│
├── src/
│   ├── data_loader.py        # Dataset loading
│   ├── preprocessing.py     # Text cleaning & normalization
│   ├── tokenizer.py         # Tokenization & padding utilities
│   ├── utils.py             # Label building & dataset splitting
│   ├── model.py             # BiRNN model architecture
│   ├── train.py             # Training pipeline
│   ├── predict.py           # Inference logic
│   ├── app.py               # Streamlit application
│   └── exception/
│       └── exception.py     # Custom project exceptions
│
├── saved_models/
│   ├── birnn_model.h5        # Trained BiRNN model
│   └── tokenizer.pkl        # Saved tokenizer
│
├── data/
│   └── go_emotions_dataset.csv
│
├── requirements.txt
└── README.md
```

---

## 🧠 Supported Emotions

The model predicts the following emotions:

```
admiration, amusement, anger, annoyance, approval,
caring, confusion, curiosity, desire, disappointment,
disapproval, disgust, embarrassment, excitement, fear,
gratitude, grief, joy, love, nervousness,
optimism, pride, realization, relief, remorse,
sadness, surprise, neutral
```

> Multiple emotions may be predicted for a single input.

---

## 🔬 Model Architecture

- **Embedding Layer**
- **Bidirectional RNN**
- **Dense Output Layer (Sigmoid activation)**

### Why Sigmoid?

This is a **multi-label classification problem** — emotions are not mutually
exclusive, so each emotion is predicted independently.

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/Emotion-Classifier.git
cd Emotion-Classifier
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
source venv/bin/activate    # Linux / Mac
venv\Scripts\activate     # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏗️ Training the Model

```bash
python -m src.train
```

This pipeline will:
- Load and preprocess text
- Fit and save tokenizer
- Build label vectors
- Train the BiRNN model
- Save:
  - `saved_models/birnn_model.h5`
  - `saved_models/tokenizer.pkl`

---

## 🌐 Streamlit Web App

Run the app:

```bash
streamlit run src/app.py
```

Then open:

```
http://localhost:8501
```

---

## 👨‍💻 Author

**Amaegbe Alanabo**  
Data Scientist ML/AI Engineer
Computer Engineering Student  
Focus: Machine Learning & NLP

---

## 📜 Disclaimer

This model predicts statistical emotional associations, not emotional truth.
