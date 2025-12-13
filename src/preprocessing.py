from src.exception.exception import PreprocessingException
import pandas as pd
import sys
import re

def remove_emoji(text: str) -> str:
    """
    Docstring for remove_emoji
    
    :param text: Removes Emoticons, Symbols, pictographs, transport and map symbols and flags
    :type text: str
    :return: str
    
    """

    try:
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r"",text)
    except Exception as e:
        raise PreprocessingException(error_message="Error removing emojis",error_detail=sys) from e
    
def remove_urls(text:str) -> str:

    """
    Docstring for remove_urls
    
    :param text: Removes Urls in text file for preprocessing
    :type text: str
    :return: Returns a string
    """

    try:
        url_pattern = r"http[s]?://\S+|www\.\S+"
        return re.sub(url_pattern, '', text)
    except Exception as e:
        raise PreprocessingException(
            error_message="Error removing URLs",
            error_detail=sys
        ) from e
    
def clean_text(text: str) -> str:

    """
    Docstring for clean_text
    
    :param text: Convert text to lowercase, removes urls and emojis
    :type text: str
    :return: str
    """

    try:
        text = text.lower()
        text = remove_urls(text)
        text = remove_emoji(text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text) 
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except Exception as e:
        raise PreprocessingException(
            error_message="Error in while cleaning text",
            error_detail=sys
        ) from e
    
def preprocess_dataframe(df:pd.DataFrame, text_column:str) -> pd.DataFrame:
    """
    Docstring for preprocess_dataframe
    
    :param df: DataFrame containing text data to be preprocessed
    :type df: pd.DataFrame
    :param text_column: Name of the column containing text data
    :type text_column: str
    :return: Preprocessed DataFrame
    :rtype: pd.DataFrame
    """

    try:
        if text_column not in df.columns:
            raise PreprocessingException(error_message=f"Column '{text_column}' not found in DataFrame", error_detail=sys)
        
        df[text_column] = df[text_column].apply(clean_text)
        return df
    except Exception as e:
        raise PreprocessingException(
            error_message="Error preprocessing DataFrame",
            error_detail=sys
        ) from e