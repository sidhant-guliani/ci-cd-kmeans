"""
config.py
Central config — override values via environment variables in each env.
"""

import os

# ── Data ──────────────────────────────────────────────────────────────────────
# config.py — update these
DATA_PATH = os.getenv("DATA_PATH", "/Volumes/dev_kmeans/clustering/data/raw")
PREDICTIONS_PATH = os.getenv("PREDICTIONS_PATH", "/Volumes/dev_kmeans/clustering/data/predictions")

# ── MLflow ────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "/Shared/kmeans-clustering")
MODEL_NAME = os.getenv("MODEL_NAME", "dev_kmeans.clustering.kmeans_model")

# ── Model Hyperparameters ─────────────────────────────────────────────────────
N_CLUSTERS = int(os.getenv("N_CLUSTERS", "4"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
MAX_ITER = int(os.getenv("MAX_ITER", "300"))
N_INIT = int(os.getenv("N_INIT", "10"))

# ── Evaluation ────────────────────────────────────────────────────────────────
SILHOUETTE_THRESHOLD = float(os.getenv("SILHOUETTE_THRESHOLD", "0.35"))

# ── Monitoring ────────────────────────────────────────────────────────────────
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.05"))  # KS p-value threshold