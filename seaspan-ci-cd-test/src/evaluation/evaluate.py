"""
evaluate.py
Loads the latest 'candidate' model from the MLflow Unity Catalog registry,
evaluates it, and promotes it to 'champion' alias if it meets the threshold.

Alias convention used here:
  'candidate'  →  freshly trained model, awaiting evaluation  (was: Staging)
  'champion'   →  approved model, used by inference jobs       (was: Production)
"""

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from mlflow.tracking import MlflowClient

from src.features.feature_engineering import get_features
from src.utils.config import DATA_PATH, MODEL_NAME, SILHOUETTE_THRESHOLD
from src.utils.logger import get_logger

log = get_logger(__name__)
client = MlflowClient()


def get_candidate_version() -> str:
    """
    Fetch the model version currently tagged with the 'candidate' alias.

    The training job sets this alias immediately after logging the model,
    so evaluate.py always picks up the latest trained version automatically
    without needing to know the version number.
    """
    # get_model_version_by_alias is the UC-compatible replacement for
    # get_latest_versions(stages=["Staging"])
    mv = client.get_model_version_by_alias(MODEL_NAME, "candidate")
    return mv.version


def evaluate(promote: bool = True) -> dict:
    """
    Evaluate the 'candidate' model and promote to 'champion' if good enough.

    Args:
        promote: If True, set the 'champion' alias on passing models.
                 Set to False in unit tests or for dry-run evaluation.

    Returns:
        dict of evaluation metrics.
    """
    version = get_candidate_version()

    # Load by alias — always evaluates whatever is currently 'candidate'
    # without hardcoding a version number.
    model_uri = f"models:/{MODEL_NAME}@candidate"
    log.info(f"Loading model '{MODEL_NAME}' v{version} (@candidate)...")
    model = mlflow.sklearn.load_model(model_uri)

    X, _, df = get_features(DATA_PATH)
    labels = model.predict(X)

    sil     = silhouette_score(X, labels, sample_size=min(2000, len(X)))
    db      = davies_bouldin_score(X, labels)
    inertia = model.inertia_

    metrics = {
        "silhouette_score":    sil,
        "davies_bouldin_score": db,
        "inertia":             inertia,
    }
    log.info(f"Evaluation metrics: {metrics}")

    # Write eval metrics back to the original training run so everything
    # about this model version lives in one MLflow run.
    run_id = client.get_model_version(MODEL_NAME, version).run_id
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})

    if promote:
        if sil >= SILHOUETTE_THRESHOLD:
            # Set 'champion' alias on this version — inference jobs load
            # models:/MODEL_NAME@champion so they automatically pick up
            # the new champion without any code changes.
            client.set_registered_model_alias(MODEL_NAME, "champion", version)
            log.info(f"✅ v{version} is now '@champion' "
                     f"(silhouette={sil:.4f} >= {SILHOUETTE_THRESHOLD})")
        else:
            log.warning(f"❌ v{version} NOT promoted "
                        f"(silhouette={sil:.4f} < {SILHOUETTE_THRESHOLD}). "
                        "Consider tuning n_clusters or inspecting data quality.")

    return metrics


if __name__ == "__main__":
    result = evaluate(promote=True)
    print(f"Evaluation result: {result}")