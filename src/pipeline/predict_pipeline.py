import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict_percentage(self, features: pd.DataFrame):
        """
        features: Pandas DataFrame containing input features (NO target column)
        returns: Dropout percentage
        """
        try:
            logging.info("Loading preprocessor and trained model")

            preprocessor_path = os.path.join("artifacts", "proprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            if not os.path.exists(preprocessor_path):
                raise Exception("Preprocessor not found. Please train the model first.")

            if not os.path.exists(model_path):
                raise Exception("Model not found. Please train the model first.")


            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)

            logging.info("Applying preprocessing on input features")

            transformed_features = preprocessor.transform(features)

            logging.info("Starting prediction")

            probas = model.predict_proba(transformed_features)

            dropout_percentage = np.round(probas[:, 1] * 100, 2)

            logging.info("Prediction completed successfully")

            return dropout_percentage

        except Exception as e:
            raise CustomException(e, sys)
