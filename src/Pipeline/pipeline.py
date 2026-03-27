import os
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from features.DataPreprocessor import DataPreprocessor
from model.train import ModelTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MLPipeline:
    def __init__(
        self,
        target_column: str,
        debug: bool = True,
        random_state: int = 42,
        artifacts_path: str = "models/artifacts.pkl"
    ):
        self.target_column = target_column
        self.debug = debug
        self.random_state = random_state
        self.artifacts_path = artifacts_path

        self.preprocessor = DataPreprocessor(target_column=target_column, debug=debug)
        self.trainer = ModelTrainer(target_column=target_column)

    def run(self, df: pd.DataFrame, test_size: float = 0.2, n_iter_search: int = 20):
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=self.random_state
        )

        self.preprocessor.fit(train_df)
        train_df = self.preprocessor.transform(train_df)
        test_df = self.preprocessor.transform(test_df)

        if self.debug:
            self.preprocessor.preview(train_df, "Train Data")
            self.preprocessor.preview(test_df, "Test Data")

        x_train, y_train = self.trainer.split(train_df)
        x_test, y_test = self.trainer.split(test_df)

        param_dist = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [5, 8, 12, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5, 0.8],
            "bootstrap": [True, False]
        }

        rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        logging.info("Starting hyperparameter tuning...")
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            cv=5,
            scoring="r2",
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1
        )
        random_search.fit(x_train, y_train)
        logging.info(f"Best params: {random_search.best_params_}")

        self.trainer.model = random_search.best_estimator_
        self.trainer.train(x_train, y_train)

        train_metrics = self.trainer.evaluate(x_train, y_train, "Train")
        test_metrics = self.trainer.evaluate(x_test, y_test, "Test")

        r2_gap = train_metrics["r2"] - test_metrics["r2"]
        logging.info(f"R2 gap (Train - Test): {r2_gap:.4f}")

        if r2_gap > 0.1:
            logging.warning("Potential overfitting detected")
        else:
            logging.info("Model generalization looks good")

        self._save_artifacts()

        return {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "r2_gap": r2_gap,
            "best_hyperparameters": random_search.best_params_
        }

    def _save_artifacts(self):
        os.makedirs(os.path.dirname(self.artifacts_path), exist_ok=True)
        bundle = {
            "model": self.trainer.model,
            "preprocessor": self.preprocessor
        }
        joblib.dump(bundle, self.artifacts_path)
        logging.info(f"Artifacts saved to {self.artifacts_path}")
