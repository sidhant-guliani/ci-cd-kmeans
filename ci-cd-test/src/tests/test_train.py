
"""
tests/test_train.py
"""
import numpy as np
from unittest.mock import patch, MagicMock
from src.training.train import train


@patch("src.training.train.mlflow")
@patch("src.training.train.get_features")
def test_train_returns_run_id(mock_features, mock_mlflow):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 4))
    mock_features.return_value = (X, MagicMock(), MagicMock())

    mock_run = MagicMock()
    mock_run.__enter__ = lambda s: mock_run
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_run.info.run_id = "test-run-123"
    mock_mlflow.start_run.return_value = mock_run
    mock_mlflow.sklearn = MagicMock()

    run_id = train(n_clusters=3)
    assert run_id == "test-run-123"


# ─────────────────────────────────────────────────────────────────────────────