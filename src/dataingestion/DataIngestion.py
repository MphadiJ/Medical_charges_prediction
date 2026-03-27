import os
import pandas as pd
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, file_path: str, target_column: str):
        self.file_path = file_path
        self.target_column = target_column

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        df = pd.read_csv(self.file_path)
        print(f"Dataset shape: {df.shape}")
        return df

    def split_features_target(self, df: pd.DataFrame):
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")

        x = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        return x, y

    def train_validation_split(self, x: pd.DataFrame, y: pd.Series):
        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=0.3,
            random_state=42
        )

        return x_train, x_val, y_train, y_val

    def validate_split(self, x_train, x_val, y_train, y_val):
        print("\nSplit Validation:")
        print(f"X_train shape: {x_train.shape}")
        print(f"X_val shape: {x_val.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_val shape: {y_val.shape}")

    def preview_data(self, x_train, x_val, y_train, y_val):
        print("\nFirst 5 rows of X_train:")
        print(x_train.head())

        print("\nFirst 5 rows of X_val:")
        print(x_val.head())

        print("\nFirst 5 rows of y_train:")
        print(y_train.head())

        print("\nFirst 5 rows of y_val:")
        print(y_val.head())

    def run(self):
        df = self.load_data()
        x, y = self.split_features_target(df)
        x_train, x_val, y_train, y_val = self.train_validation_split(x, y)

        self.validate_split(x_train, x_val, y_train, y_val)
        self.preview_data(x_train, x_val, y_train, y_val)

        return x_train, x_val, y_train, y_val
