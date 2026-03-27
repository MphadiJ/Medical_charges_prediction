import joblib
import logging
import pandas as pd
from typing import Union, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Predictor:
    def __init__(self, artifacts_path: str = "models/artifacts.pkl", debug: bool = False):
        self.artifacts_path = artifacts_path
        self.debug = debug

        self.model = None
        self.preprocessor = None

        self._load_artifacts()

    # Load model + preprocessor
    def _load_artifacts(self):
        try:
            bundle = joblib.load(self.artifacts_path)
            self.model = bundle["model"]
            self.preprocessor = bundle["preprocessor"]

            logging.info(f"Artifacts loaded from {self.artifacts_path}")

        except Exception as e:
            logging.error(f"Failed to load artifacts: {e}")
            raise

    # Predict from DataFrame
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            if self.debug:
                logging.info(f"Incoming data shape: {data.shape}")

            # Apply SAME preprocessing used in training
            processed_data = self.preprocessor.transform(data)

            if self.debug:
                logging.info(f"Processed data shape: {processed_data.shape}")

            predictions = self.model.predict(processed_data)

            # Attach predictions
            result = data.copy()
            result["prediction"] = predictions

            return result

        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise

    # Predict from dict
    def predict_from_dict(self, data: Union[Dict, List[Dict]]) -> List[Dict]:
        """
        Supports:
        - single record dict
        - list of dicts (batch prediction)
        """
        try:
            df = pd.DataFrame(data)
            result_df = self.predict(df)

            return result_df.to_dict(orient="records")

        except Exception as e:
            logging.error(f"Prediction from dict failed: {e}")
            raise
