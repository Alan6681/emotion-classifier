from src.data_loader import load_data
from src.preprocessing import preprocess_dataframe
from src.tokenizer import fit_tokenizer, save_tokenizer

# Load and preprocess text
df = load_data("data/go_emotions_dataset.csv")
df = preprocess_dataframe(df, text_column="text")

# Fit tokenizer
tokenizer = fit_tokenizer(df["text"], num_words=20000)

# Save tokenizer for later use
save_tokenizer(tokenizer, "saved_models/tokenizer.pkl")