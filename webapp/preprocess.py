import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib

class Preprocessor:
    def __init__(self):
        self.num_cols = []
        self.cat_cols = []
        self.scaler = None
        self.encoder = None

    def fit(self, df, target_col="target"):
        df = df.copy()
        if target_col in df.columns:
            df = df.drop(columns=[target_col])

        self.num_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Fill missing
        df[self.num_cols] = df[self.num_cols].fillna(df[self.num_cols].median())
        df[self.cat_cols] = df[self.cat_cols].fillna("unknown")

        # Fit encoders
        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        scaled = self.scaler.fit_transform(df[self.num_cols])
        encoded = self.encoder.fit_transform(df[self.cat_cols])

        return np.concatenate([scaled, encoded], axis=1)

    def transform(self, df):
        df = df.copy()
        df[self.num_cols] = df[self.num_cols].fillna(df[self.num_cols].median())
        df[self.cat_cols] = df[self.cat_cols].fillna("unknown")

        scaled = self.scaler.transform(df[self.num_cols])
        encoded = self.encoder.transform(df[self.cat_cols])
        return np.concatenate([scaled, encoded], axis=1)

    def save(self, path="preprocessor.pkl"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="preprocessor.pkl"):
        return joblib.load(path)

    def get_feature_names(self):
        return self.num_cols + list(self.encoder.get_feature_names_out(self.cat_cols))
