import sys
import os
import pandas as pd
import numpy as np
from src.exception.exception import UtilsException
from src.logger.custom_logging import logging
from sklearn.model_selection import train_test_split

def get_label_columns(df, text_column:str ="text") -> list:
    """
    Returns all emotion label columns from the dataset.

    """
    try:
        label_columns = [
            col for col in df.columns
            if col != text_column and pd.api.types.is_numeric_dtype(df[col])
        ]

        if not label_columns:
            raise ValueError("No numeric label columns found")

        logging.info(f"Detected label columns: {label_columns}")
        return label_columns

    except Exception as e:
        raise UtilsException(
            error_message="Error detecting label columns",
            error_detail=sys
        ) from e
    
def build_labels(df , label_columns) -> np.ndarray:
    """
    Convert labels to multi-hot encoded vectors
    """
    try:
        y = df[label_columns].values
        logging.info("Sucessfully built :)")

        return np.array(y, dtype=np.float32)
    except Exception as e:
        raise UtilsException(error_message="Error building label vectors:(", error_detail=sys) from e
    
def split_dataset(X, y, test_size=0.2, random_state=42) -> tuple:
    try:
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size,
            random_state=random_state
            )
        logging.info("Data successfully split into training and validation sets :)")

        return X_train, X_val, y_train, y_val
    except Exception as e:
        raise UtilsException(error_message="Error in splitting train and test data :(", error_detail=sys) from e