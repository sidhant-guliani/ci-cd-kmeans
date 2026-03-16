"""
train.py
Trains a KMeans clustering model, logs everything to MLflow,
and registers the model in the Unity Catalog Model Registry.

Why KMeans?
  KMeans is an unsupervised algorithm that groups data points into K clusters
  by minimising the distance between each point and its cluster centroid.
  It's fast, interpretable, and works well when clusters are roughly spherical.

Why MLflow?
  MLflow gives us experiment tracking (what params/metrics did each run have?),
  model versioning (which model is in production right now?), and a central
  registry so downstream jobs (inference, monitoring) always load the right model.
"""

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from mlflow.models.signature import infer_signature

from src.features.feature_engineering import get_features
from src.utils.config import (
    DATA_PATH, EXPERIMENT_NAME, MODEL_NAME,
    N_CLUSTERS, RANDOM_STATE, MAX_ITER, N_INIT,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


def train(
    n_clusters: int = N_CLUSTERS,
    random_state: int = RANDOM_STATE,
    max_iter: int = MAX_ITER,
    n_init: int = N_INIT,
) -> str:
    """
    Train KMeans and return the MLflow run_id.

    Args:
        n_clusters:   Number of clusters K. Should be chosen via elbow method
                      or domain knowledge. Here we default to 4 to match our
                      dummy data which was generated with 4 centres.
        random_state: Seed for reproducibility — same seed = same centroid
                      initialisation every run, making results comparable.
        max_iter:     Max iterations for the EM-style optimisation loop.
                      300 is usually more than enough for convergence.
        n_init:       How many times KMeans reruns with different centroid seeds.
                      The best result (lowest inertia) is kept. Higher = more
                      robust but slower. 10 is the sklearn default.

    Returns:
        MLflow run_id string — useful for linking downstream tasks back to
        the exact training run that produced the model.
    """

    # ── Point MLflow at Unity Catalog ─────────────────────────────────────────
    # By default MLflow uses the legacy workspace registry. Setting this URI
    # switches to Unity Catalog, which enforces governance, lineage, and
    # access controls at the catalog/schema level (dev_kmeans.clustering.*).
    mlflow.set_registry_uri("databricks-uc")

    # ── Create or reuse the experiment ────────────────────────────────────────
    # All runs for this project are grouped under one experiment so we can
    # compare metrics across runs in the MLflow UI. If it doesn't exist yet,
    # MLflow creates it automatically.
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        log.info(f"MLflow run started: {run_id}")

        # ── Load & scale features ─────────────────────────────────────────────
        # get_features() handles loading from Delta/CSV AND applies
        # StandardScaler. Scaling is critical for KMeans because the algorithm
        # uses Euclidean distance — unscaled features with large ranges would
        # dominate the distance calculation and distort cluster shapes.
        X, scaler, df = get_features(DATA_PATH)
        log.info(f"Loaded {len(df)} samples for training.")

        # ── Log hyperparameters ───────────────────────────────────────────────
        # Logging params links every metric and artifact to the exact config
        # that produced them. This is what makes runs reproducible and
        # comparable in the MLflow UI.
        params = dict(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            n_init=n_init,
        )
        mlflow.log_params(params)

        # ── Train ─────────────────────────────────────────────────────────────
        # KMeans.fit() runs the Lloyd's algorithm:
        #   1. Randomly initialise K centroids (repeated n_init times)
        #   2. Assign each point to its nearest centroid
        #   3. Recompute centroids as the mean of assigned points
        #   4. Repeat steps 2-3 until convergence or max_iter is reached
        model = KMeans(**params)
        model.fit(X)
        labels = model.labels_  # cluster assignment for each training point

        # ── Evaluate & log metrics ────────────────────────────────────────────
        # We log three complementary metrics because no single metric tells
        # the full story for clustering:

        # Inertia: sum of squared distances from each point to its centroid.
        # Lower = tighter clusters. BUT always decreases as K increases, so
        # don't use it alone to choose K (use the elbow method instead).
        inertia = model.inertia_

        # Silhouette score: measures how similar a point is to its own cluster
        # vs neighbouring clusters. Range [-1, 1]. Higher is better.
        # >0.5 is generally considered good. We use this as our promotion gate.
        sil = silhouette_score(X, labels, sample_size=min(2000, len(X)))

        # Davies-Bouldin score: ratio of within-cluster scatter to
        # between-cluster separation. Lower is better (0 is perfect).
        # Useful as a second opinion alongside silhouette.
        db = davies_bouldin_score(X, labels)

        mlflow.log_metrics({
            "inertia": inertia,
            "silhouette_score": sil,
            "davies_bouldin_score": db,
        })
        log.info(f"inertia={inertia:.2f}  silhouette={sil:.4f}  davies_bouldin={db:.4f}")

        # ── Persist the scaler as an MLflow artifact ──────────────────────────
        # The scaler MUST be saved alongside the model because inference must
        # apply the exact same scaling transformation that was used during
        # training. If we retrain with new data the scaler parameters (mean,
        # std) will change — saving it here ensures the paired scaler is always
        # retrievable via the run_id.
        import pickle, tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            scaler_path = os.path.join(tmp, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact(scaler_path, artifact_path="scaler")

        # ── Build an explicit model signature ────────────────────────────────
        # Unity Catalog requires BOTH input AND output signatures.
        # input_example alone sometimes fails to auto-infer the output for
        # KMeans because predict() returns integer cluster labels — MLflow
        # can't always deduce that automatically.
        # infer_signature(inputs, outputs) inspects the actual numpy arrays
        # and builds a typed schema UC will accept.
        input_example = X[:5]
        predictions_example = model.predict(input_example)  # shape (5,) int array
        signature = infer_signature(input_example, predictions_example)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=input_example,
            signature=signature,              # explicit signature — satisfies UC
        )
        log.info(f"Model registered as '{MODEL_NAME}'.")

        # ── Tag the new version as 'candidate' ───────────────────────────────
        # UC uses aliases instead of stages. We set 'candidate' here so
        # evaluate.py can always find the latest trained model without
        # knowing its version number. Think of it as "ready for review".
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        # The registered model version is the latest one after log_model
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest_version = max(versions, key=lambda v: int(v.version)).version
        client.set_registered_model_alias(MODEL_NAME, "candidate", latest_version)
        log.info(f"Alias '@candidate' set on version {latest_version}.")

    return run_id


if __name__ == "__main__":
    rid = train()
    print(f"Training complete. Run ID: {rid}")