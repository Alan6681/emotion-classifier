import sys
from src.exception.exception import ModelBuildException
from src.logger.custom_logging import logging
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall

def build_birnn_model(vocab_size: int,
                      max_len: int,
                      embedding_dim: int,
                      num_classes: int,
                      lstm_units: int = 128,
                      dropout_rate: float = 0.3) -> tf.keras.Model:
    try:
        inputs = Input(shape=(max_len,), name="input_layer")
        x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, name="embedding_layer")(inputs)
        x = Bidirectional(LSTM(units=lstm_units, return_sequences=False), name="bi_lstm_layer")(x)
        x = Dropout(dropout_rate, name="dropout_layer")(x)
        outputs = Dense(num_classes, activation="sigmoid", name="output_layer")(x)

        model = Model(inputs=inputs, outputs=outputs, name="BiRNN_Emotion_Classifier")
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", Precision(name="precision"), Recall(name="recall")])

        logging.info("Bidirectional RNN model built successfully :)")
        return model

    except Exception as e:
        raise ModelBuildException(error_message="Error building BiRNN model :(", error_detail=sys) from e
