"""
Unit tests for data loading, partitioning, scaling, and SMOTE resampling.
"""

import numpy as np
import pandas as pd
import pytest
from src.data import split_and_scale_data, apply_smote, compute_imbalance_weights


@pytest.fixture
def synthetic_fraud_df():
    """Generates synthetic dataset with realistic imbalance for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_fraud = 10  # 1% imbalance

    X_legit = np.random.randn(n_samples - n_fraud, 10)
    X_fraud = np.random.randn(n_fraud, 10) + 2.0

    X_all = np.vstack([X_legit, X_fraud])
    y_all = np.array([0] * (n_samples - n_fraud) + [1] * n_fraud)

    columns = [f"V{i+1}" for i in range(8)] + ["Time", "Amount"]
    df = pd.DataFrame(X_all, columns=columns)
    df["Class"] = y_all
    return df


def test_split_and_scale_data_shapes(synthetic_fraud_df):
    X_tr, X_val, X_te, y_tr, y_val, y_te, scaler = split_and_scale_data(
        synthetic_fraud_df, target_col="Class", test_size=0.10, val_size=0.10, random_state=42
    )

    total_len = len(X_tr) + len(X_val) + len(X_te)
    assert total_len == len(synthetic_fraud_df)
    assert len(X_te) == 100
    assert len(X_val) == 100
    assert len(X_tr) == 800

    # Ensure target class stratification
    assert y_tr.sum() > 0
    assert y_val.sum() > 0 or y_te.sum() > 0


def test_scaling_no_leakage(synthetic_fraud_df):
    X_tr, X_val, X_te, y_tr, y_val, y_te, scaler = split_and_scale_data(
        synthetic_fraud_df, target_col="Class", test_size=0.10, val_size=0.10, random_state=42
    )

    # Scaler mean must match training partition mean
    assert np.allclose(X_tr.mean().values, np.zeros(X_tr.shape[1]), atol=1e-2)


def test_apply_smote(synthetic_fraud_df):
    X_tr, _, _, y_tr, _, _, _ = split_and_scale_data(
        synthetic_fraud_df, target_col="Class", test_size=0.10, val_size=0.10, random_state=42
    )

    X_res, y_res = apply_smote(X_tr, y_tr, random_state=42, k_neighbors=3)
    counts = y_res.value_counts()

    assert counts[0] == counts[1]
    assert len(X_res) == counts[0] * 2


def test_compute_imbalance_weights():
    y = pd.Series([0] * 990 + [1] * 10)
    weights = compute_imbalance_weights(y, mode="sqrt")

    assert weights["imbalance_ratio"] == 99.0
    assert np.isclose(weights["keras_weights"][1], np.sqrt(99.0))
