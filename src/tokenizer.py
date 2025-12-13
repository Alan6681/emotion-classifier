import pickle
import sys
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.exception.exception import TokenizerException
from src.logger.custom_logging import logging


def fit_tokenizer(texts, num_words=20000, oov_token="<OOV>"):
    """
    Fits a tokenizer to a given text

    """
    try:

        tokenizer = Tokenizer(
            num_words=num_words,
            oov_token = oov_token
        )
        tokenizer.fit_on_texts(texts)
        logging.info("Fitted Tokenizer on texts")
        return tokenizer

    except Exception as e:
        raise TokenizerException(
            error_message="Error while fitting tokenizer on text",
            error_detail=sys
        ) from e
    
def texts_to_sequence_fn(tokenizer, texts):

    """
    Converts Texts to sequences of vectors
    
    """

    try:
        sequences = tokenizer.texts_to_sequences(texts)
        logging.info("Texts have been converted to sequences :)")
        return sequences
    except Exception as e:
        raise TokenizerException(error_message="Error Converting texts to Sequence", error_detail=sys) from e
    
def pad_sequences_fn(sequences, max_len=100, padding="post", truncating="post"):
    """
    Padding sequence to ensure same length 
    
    """
    try:
        padded = pad_sequences(
            sequences,
            maxlen=max_len,
            padding=padding,
            truncating=truncating
        )
        logging.info("Sequenceses have been padded successfully :)")
        return padded
    except Exception as e:
        raise TokenizerException(error_message="Error While Padding", error_detail=sys) from e
    
def save_tokenizer(tokenizer, file_path="saved_models/tokenizer.pkl"):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(tokenizer, file)
        logging.info("Successfully Saved Tokenizer :)")
    except Exception as e:
        raise TokenizerException(error_message="Error while saving tokenizer", error_detail=sys) from e
    

def load_tokenizer(file_path="saved_models/tokenizer.pkl"):
    try:
        with open(file_path, "rb") as file:
            tokenizer = pickle.load(file)
        logging.info(" Successfully loaded Tokenizer :)")

        return tokenizer
    except Exception as e:
        raise TokenizerException(error_message="Error while loading tokenizer", error_detail=sys) from e
    
    
