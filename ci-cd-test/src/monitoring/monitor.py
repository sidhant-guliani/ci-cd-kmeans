"""
monitor.py
Basic data drift detection: compares current feature distributions
against a reference (training) snapshot and logs a drift score to MLflow.
"""

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.features.feature_engineering import load_data, FEATURE_COLS
from src.utils.config import DATA_PATH, EXPERIMENT_NAME, DRIFT_THRESHOLD
from src.utils.logger import get_logger

log = get_logger(__name__)


def compute_drift(ref: pd.DataFrame, curr: pd.DataFrame,
                  feature_cols: list = FEATURE_COLS) -> dict:
    """
    KS-test per feature between reference and current datasets.
    Returns dict of {feature: p_value}.
    """
    results = {}
    for col in feature_cols:
        stat, p_val = ks_2samp(ref[col].dropna(), curr[col].dropna())
        results[col] = {"ks_stat": stat, "p_value": p_val}
        log.info(f"  {col}: KS={stat:.4f}  p={p_val:.4f}")
    return results


def run_monitoring(reference_path: str = DATA_PATH,
                   current_path: str = DATA_PATH) -> dict:
    """
    In production you'd point current_path to fresh incoming data.
    Here we use the same dataset split into halves as a demo.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    ref_df = load_data(reference_path)
    curr_df = load_data(current_path)

    # Demo: split into two halves to simulate reference vs current
    mid = len(ref_df) // 2
    ref_half = ref_df.iloc[:mid].reset_index(drop=True)
    curr_half = curr_df.iloc[mid:].reset_index(drop=True)

    log.info("Running drift detection...")
    drift_results = compute_drift(ref_half, curr_half)

    drifted = [f for f, v in drift_results.items() if v["p_value"] < DRIFT_THRESHOLD]

    with mlflow.start_run(run_name="monitoring"):
        for feat, vals in drift_results.items():
            mlflow.log_metric(f"drift_ks_{feat}", vals["ks_stat"])
            mlflow.log_metric(f"drift_pval_{feat}", vals["p_value"])
        mlflow.log_metric("n_drifted_features", len(drifted))

    if drifted:
        log.warning(f"⚠️  Drift detected in features: {drifted}. Consider retraining.")
    else:
        log.info("✅ No significant drift detected.")

    return {"drifted_features": drifted, "details": drift_results}


if __name__ == "__main__":
    result = run_monitoring()
    print(result)