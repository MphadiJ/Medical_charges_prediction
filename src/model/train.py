import os
import joblib
import logging
import numpy as np
from typing import Dict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelTrainer:
    def __init__(
        self,
        target_column: str,
        model_path: str = "models/random_forest.pkl",
    ):
        self.target_column = target_column
        self.model_path = model_path
        self.model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

    def split(self, df):
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")

        x = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        return x, y

    def train(self, x, y):
        logging.info("Training Random Forest model...")
        self.model.fit(x, y)
        logging.info("Training completed.")

    def evaluate(self, x, y, dataset_name: str = "Dataset") -> Dict:
        predictions = self.model.predict(x)

        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)

        logging.info(f"{dataset_name} RMSE: {rmse:.4f}")
        logging.info(f"{dataset_name} R2: {r2:.4f}")

        return {"rmse": rmse, "r2": r2}

    def save_model(self, model_path: str = None):
        target_path = model_path or self.model_path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        joblib.dump(self.model, target_path)
        logging.info(f"Model saved to {target_path}")

    # Compatibility alias for older callers.
    def save_artifacts(self):
        self.save_model()
