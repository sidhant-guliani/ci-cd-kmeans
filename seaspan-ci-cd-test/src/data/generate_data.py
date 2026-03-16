"""
generate_data.py
Generates dummy clusterable data and saves it as a Delta table.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs


def generate_dummy_data(n_samples: int = 2000, n_features: int = 4,
                        centers: int = 4, random_state: int = 42) -> pd.DataFrame:
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=1.2,
        random_state=random_state,
    )
    cols = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["true_label"] = y  # kept for evaluation reference only
    return df


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """Save DataFrame as a CSV (local) or Delta table (Databricks)."""
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        sdf.write.format("delta").mode("overwrite").save(output_path)
        print(f"Data saved as Delta table at: {output_path}")
    except ImportError:
        csv_path = output_path.replace("/dbfs", "").strip("/") + ".csv"
        df.to_csv(csv_path, index=False)
        print(f"Spark not available. Data saved as CSV: {csv_path}")


if __name__ == "__main__":
    from src.utils.config import DATA_PATH
    df = generate_dummy_data()
    print(f"Generated {len(df)} rows with columns: {list(df.columns)}")
    save_data(df, DATA_PATH)