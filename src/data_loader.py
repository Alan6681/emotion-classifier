import pandas as pd
import os
import sys
from src.exception.exception import DataLoaderException

def load_data(file_path)-> pd.DataFrame:
    """
    Load a CSV dataset from a given file path with error handling.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    DataLoadingException
        If the file cannot be loaded.
    
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FilePath: {file_path} is not found")
        df = pd.read_csv(filepath_or_buffer=file_path, encoding="utf-8")
        return df

    except Exception as e:
        raise DataLoaderException(e,sys)

            
        

