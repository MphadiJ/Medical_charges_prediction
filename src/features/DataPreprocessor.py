import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class DataPreprocessor:
    def __init__(self, target_column: str, skew_threshold: float = 0.75, debug: bool = True):
        self.target_column = target_column
        self.skew_threshold = skew_threshold
        self.debug = debug

        self.num_cols = []
        self.cat_cols = []
        self.binary_cols = []
        self.multi_cols = []
        self.skewed_cols = []

        self.encoder = None
        self.outlier_bounds = {}
        self.binary_mappings = {}
        self.is_fitted = False

    # ---------------- #
    # Identify columns #
    # ---------------- #
    def identify_columns(self, df):
        self.num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if self.target_column in self.num_cols:
            self.num_cols.remove(self.target_column)

    # ---------------- #
    # Detect skewness  #
    # ---------------- #
    def detect_skewness(self, df):
        skewness = df[self.num_cols].skew()
        self.skewed_cols = skewness[abs(skewness) > self.skew_threshold].index.tolist()
        if self.debug:
            print(f"Skewed columns detected: {self.skewed_cols}")

    # ---------------- #
    # Transform skewed #
    # ---------------- #
    def transform_skewed(self, df):
        for col in self.skewed_cols:
            if col not in df.columns or (df[col] < 0).any():
                continue
            df[col] = np.log1p(df[col])
        return df

    # ---------------- #
    # Fit outliers     #
    # ---------------- #
    def fit_outlier_bounds(self, df):
        self.outlier_bounds = {}
        for col in self.num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.outlier_bounds[col] = (lower, upper)

    # ---------------- #
    # Cap outliers     #
    # ---------------- #
    def cap_outliers(self, df):
        for col, (lower, upper) in self.outlier_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower, upper)
        return df

    # ------------------------- #
    # Identify categorical cols #
    # ------------------------- #
    def identify_categorical(self, df):
        self.binary_cols = [col for col in self.cat_cols if df[col].nunique() == 2]
        self.multi_cols = [col for col in self.cat_cols if df[col].nunique() > 2]

    # --------------------- #
    # Fit binary mappings   #
    # --------------------- #
    def fit_binary_mappings(self, df):
        self.binary_mappings = {
            col: {val: idx for idx, val in enumerate(sorted(df[col].dropna().unique()))}
            for col in self.binary_cols
        }

    # ----------------- #
    # Fit the encoder   #
    # ----------------- #
    def fit(self, df):
        self.identify_columns(df)
        self.detect_skewness(df)
        self.identify_categorical(df)
        self.fit_outlier_bounds(df)
        self.fit_binary_mappings(df)

        if self.multi_cols:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.encoder.fit(df[self.multi_cols])

        self.is_fitted = True
        return self

    def validate_input_columns(self, df):
        required_columns = set(self.num_cols + self.binary_cols + self.multi_cols)
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            raise ValueError(f"Missing required columns for preprocessing: {missing_columns}")

    # ----------------- #
    # Transform df      #
    # ----------------- #
    def transform(self, df):
        if not self.is_fitted:
            raise ValueError("DataPreprocessor must be fitted before calling transform().")

        df = df.copy()
        self.validate_input_columns(df)
        df = self.transform_skewed(df)
        df = self.cap_outliers(df)

        for col in self.binary_cols:
            mapping = self.binary_mappings[col]
            encoded_col = df[col].map(mapping)
            if encoded_col.isna().any() and df[col].notna().any():
                unknown_values = sorted(df.loc[encoded_col.isna(), col].dropna().unique())
                raise ValueError(f"Unseen categories in binary column '{col}': {unknown_values}")
            df[col] = encoded_col

        if self.multi_cols:
            encoded = self.encoder.transform(df[self.multi_cols])
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out(self.multi_cols),
                index=df.index
            )
            df = df.drop(columns=self.multi_cols)
            df = pd.concat([df, encoded_df], axis=1)

        return df

    # ---------------- #
    # Preview function #
    # ---------------- #
    def preview(self, df, name="DataFrame", n=5):
        if self.debug:
            print(f"\n{name} Preview (first {n} rows):")
            print(df.head(n))
            print(f"{name} Shape: {df.shape}")
