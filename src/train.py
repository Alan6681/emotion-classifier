import os
import sys
from src.exception.exception import TrainingException
from src.logger.custom_logging import logging
from src.data_loader import load_data
from src.model import build_birnn_model
from src.preprocessing import preprocess_dataframe
from src.tokenizer import load_tokenizer, pad_sequences_fn, texts_to_sequence_fn
from src.utils import get_label_columns, build_labels, split_dataset
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def train_model(
        data_path:str, tokenizer_path:str, max_len:int=100, embedding_dim:int=128,
        lstm_units: int=128, batch_size: int=64, epochs: int=10, test_size: float=0.2,
        model_save_path: str ="saved_models/birnn_model.h5",
        text_column: str = "text"
):
    try:
        # Load the data
        df = load_data(data_path)
        logging.info(f"Data loaded from {data_path}. Shape: {df.shape}")

        # Preprocess text
        df = preprocess_dataframe(df, text_column=text_column)
        logging.info(f"Preprocessing completed for column {text_column}")

        # Build labels
        label_columns = get_label_columns(df, text_column=text_column)
        y = build_labels(df, label_columns)
        logging.info(f"Labels built. Number of classes: {len(label_columns)}")

        # Load tokenizer and convert the the text to sequences
        tokenizer = load_tokenizer(tokenizer_path)
        sequences = texts_to_sequence_fn(tokenizer, df[text_column])
        X = pad_sequences_fn(sequences, max_len=max_len)
        logging.info(f"Text tokenization and padding completed. X shape: {X.shape}")

        # Split dataset
        X_train, X_val, y_train, y_val = split_dataset(X, y, test_size=test_size)
        logging.info(f"Data split into train ({len(X_train)}) and validation ({len(X_val)}) sets")

        # Build BiRNN model
        vocab_size = len(tokenizer.word_index) + 1
        num_classes = len(label_columns)
        model = build_birnn_model(vocab_size, max_len, embedding_dim, num_classes, lstm_units)

        # Setting callbacks
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        checkpoints_tb = ModelCheckpoint(model_save_path, save_best_only=True, monitor="val_loss", mode="min")
        earlystop_cb = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size = batch_size,
            epochs = epochs,
            callbacks = [checkpoints_tb, earlystop_cb],
            verbose = 1

        )

        logging.info(f"Training completed successfully :))). Best model saved at {model_save_path}")

    except Exception as e:
        raise TrainingException(error_message="Error while training model", error_detail=sys) from e

if __name__ == "__main__":
    train_model(
        data_path = "data/go_emotions_dataset.csv",
        tokenizer_path="saved_models/tokenizer.pkl",
        max_len=100,
        embedding_dim=128,
        lstm_units=128,
        batch_size=64,
        epochs=10
    )
