
"""
tests/test_features.py
"""
import numpy as np
import pandas as pd
import pytest
from src.features.feature_engineering import scale_features, FEATURE_COLS


def make_df(n=100):
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((n, len(FEATURE_COLS))), columns=FEATURE_COLS)


def test_scale_features_shape():
    df = make_df()
    X, scaler = scale_features(df)
    assert X.shape == (100, len(FEATURE_COLS))


def test_scale_features_mean_std():
    df = make_df(500)
    X, _ = scale_features(df)
    assert abs(X.mean()) < 0.05
    assert abs(X.std() - 1.0) < 0.05


def test_missing_column_raises():
    df = make_df().drop(columns=["feature_1"])
    with pytest.raises(Exception):
        scale_features(df)