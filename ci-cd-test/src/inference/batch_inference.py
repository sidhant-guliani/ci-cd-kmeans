"""
batch_inference.py
Loads the Production model from the MLflow Unity Catalog registry,
assigns cluster labels to new data, and writes results to a Delta table.

Why batch inference (vs real-time)?
  For clustering use cases (customer segmentation, demand grouping, anomaly
  detection) predictions are typically consumed downstream in bulk — e.g.
  a daily marketing campaign or a BI dashboard. Batch is cheaper and simpler
  than keeping a REST endpoint alive 24/7 when sub-second latency isn't needed.
"""

import mlflow.sklearn
import pandas as pd
import pickle
import os

from src.features.feature_engineering import load_data, FEATURE_COLS
from src.utils.config import DATA_PATH, MODEL_NAME, PREDICTIONS_PATH
from src.utils.logger import get_logger

log = get_logger(__name__)


def load_scaler(run_id: str):
    """
    Download the scaler.pkl artifact from the MLflow run that produced
    the current Production model, then deserialise it.

    Why do we need the scaler from the training run specifically?
      StandardScaler stores the mean and std of the TRAINING data.
      Using any other scaler (e.g. refit on today's data) would transform
      features differently, making the model's cluster boundaries invalid.
      Tying the scaler to run_id guarantees we always use the paired scaler.
    """
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    # download_artifacts fetches the file from MLflow's artifact store
    # (S3, ADLS, DBFS, etc.) to a local temp path automatically.
    local_path = client.download_artifacts(run_id, "scaler/scaler.pkl")
    with open(local_path, "rb") as f:
        return pickle.load(f)


def run_inference(data_path: str = DATA_PATH,
                  output_path: str = PREDICTIONS_PATH) -> pd.DataFrame:
    """
    End-to-end batch inference pipeline:
      load data → scale → predict → write predictions to Delta.

    Args:
        data_path:   Path to the input Delta table (or CSV locally).
        output_path: Path where the predictions Delta table will be written.

    Returns:
        DataFrame with original features + a 'cluster' column.
    """
    # ── Load the Production model ─────────────────────────────────────────────
    # Loading by stage alias ("Production") means this script always uses
    # whatever model is currently in Production — no version number to update
    # when a new model is promoted.
    # Load by alias — @champion is set by evaluate.py when a model passes
    # the quality gate. UC aliases replace the old /Production stage syntax.
    model_uri = f"models:/{MODEL_NAME}@champion"
    log.info(f"Loading champion model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    # ── Retrieve the run_id so we can load the paired scaler ─────────────────
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    if not mv:
        raise RuntimeError("No @champion model found in registry. "
                           "Has the evaluation job promoted a model yet?")
    run_id = mv.run_id

    # ── Scale features using the training scaler ──────────────────────────────
    # Critical: we transform (not fit_transform) — we apply existing
    # scaler parameters, we do NOT relearn them from the new data.
    scaler = load_scaler(run_id)
    df = load_data(data_path)
    X_scaled = scaler.transform(df[FEATURE_COLS].values)

    # ── Predict cluster assignments ───────────────────────────────────────────
    # KMeans.predict() assigns each point to the nearest centroid.
    # This is fast (no retraining) — it's a single nearest-neighbour lookup.
    df["cluster"] = model.predict(X_scaled)
    log.info(f"Cluster distribution:\n{df['cluster'].value_counts().sort_index()}")

    # ── Write predictions to Delta ────────────────────────────────────────────
    # Writing to Delta (not CSV) means downstream consumers (BI tools,
    # downstream pipelines) get ACID-safe reads and we get a full audit trail
    # of every prediction batch via Delta's transaction log.
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        sdf = spark.createDataFrame(df)
        sdf.write.format("delta").mode("overwrite").save(output_path)
        log.info(f"Predictions written to Delta: {output_path}")
    except ImportError:
        # Local fallback for development/testing without Spark
        csv_out = output_path.replace("/dbfs", "").strip("/") + ".csv"
        df.to_csv(csv_out, index=False)
        log.info(f"Predictions written to CSV: {csv_out}")

    return df


if __name__ == "__main__":
    result_df = run_inference()
    print(result_df[["cluster"]].value_counts().sort_index())