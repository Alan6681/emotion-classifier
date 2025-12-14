import sys

class EmotionClassifierException(Exception):
    def __init__(self, error_message, error_detail:sys) -> None:
        super().__init__(error_message)

        _,_,exc_tb = error_detail.exc_info()
        self.error_message = error_message
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.message = f"Error Occurred at filename: {self.file_name} at Line Number: {self.lineno} with error message: {self.error_message}"


    def __str__(self):
        return self.message
    
class DataLoaderException(EmotionClassifierException):
    pass

class EvaluationException(EmotionClassifierException):
    pass

class PreprocessingException(EmotionClassifierException):
    pass

class TokenizerException(EmotionClassifierException):
    pass

class TrainingException(EmotionClassifierException):
    pass

class ModelBuildException(EmotionClassifierException):
    pass

class UtilsException(EmotionClassifierException):
    pass

class PredictionException(EmotionClassifierException):
    pass


        

