"""
feature_engineering.py
Loads raw data, scales features, and returns a clean feature matrix.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ["feature_1", "feature_2", "feature_3", "feature_4"]


def load_data(path: str) -> pd.DataFrame:
    """Load data from Delta table (Databricks) or CSV (local)."""
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        return spark.read.format("delta").load(path).toPandas()
    except ImportError:
        csv_path = path.replace("/dbfs", "").strip("/") + ".csv"
        return pd.read_csv(csv_path)


def scale_features(df: pd.DataFrame, feature_cols: list = FEATURE_COLS) -> tuple[np.ndarray, StandardScaler]:
    """Fit a StandardScaler and return scaled array + fitted scaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].values)
    return X_scaled, scaler


def get_features(path: str) -> tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """End-to-end: load → validate → scale."""
    df = load_data(path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    X, scaler = scale_features(df)
    return X, scaler, df