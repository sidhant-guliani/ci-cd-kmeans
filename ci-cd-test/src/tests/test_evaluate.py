
# ─────────────────────────────────────────────────────────────────────────────
"""
tests/test_evaluate.py
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


@patch("src.evaluation.evaluate.client")
@patch("src.evaluation.evaluate.mlflow")
@patch("src.evaluation.evaluate.get_features")
def test_evaluate_returns_metrics(mock_features, mock_mlflow, mock_client):
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 4))
    mock_features.return_value = (X, MagicMock(), MagicMock())

    model = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
    mock_mlflow.sklearn.load_model.return_value = model

    version_mock = MagicMock()
    version_mock.version = "1"
    version_mock.run_id = "run-abc"
    mock_client.get_latest_versions.return_value = [version_mock]
    mock_client.get_model_version.return_value = version_mock

    mock_run = MagicMock()
    mock_run.__enter__ = lambda s: mock_run
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_mlflow.start_run.return_value = mock_run

    from src.evaluation.evaluate import evaluate
    metrics = evaluate(promote=False)

    assert "silhouette_score" in metrics
    assert "davies_bouldin_score" in metrics
    assert metrics["silhouette_score"] > 0