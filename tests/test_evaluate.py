"""
Unit tests for evaluation metrics, threshold tuning, and cost matrix computations.
"""

import numpy as np
import pytest
from src.evaluate import (
    find_optimal_threshold,
    evaluate_model_performance,
    compute_financial_cost_matrix,
    build_comparison_dataframe
)


def test_find_optimal_threshold():
    y_val = np.array([0, 0, 0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.15, 0.4, 0.8, 0.9, 0.3, 0.75])

    thresh, best_f1 = find_optimal_threshold(y_val, y_proba)
    assert 0.0 <= thresh <= 1.0
    assert 0.0 <= best_f1 <= 1.0


def test_evaluate_model_performance():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.9, 0.4])

    metrics = evaluate_model_performance("TestModel", y_true, y_pred, y_proba)
    assert metrics["Model"] == "TestModel"
    assert "Precision" in metrics
    assert "Recall" in metrics
    assert "F1-Score" in metrics
    assert "ROC-AUC" in metrics
    assert "Avg Precision" in metrics


def test_compute_financial_cost_matrix():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])  # 1 FP, 1 FN

    cost = compute_financial_cost_matrix(y_true, y_pred, cost_false_negative=100.0, cost_false_positive=5.0)
    assert cost["False Positives"] == 1
    assert cost["False Negatives"] == 1
    assert cost["Total Cost (EUR)"] == 105.0


def test_build_comparison_dataframe():
    results = [
        {"Model": "M1", "Precision": 0.8, "Recall": 0.7, "F1-Score": 0.75, "ROC-AUC": 0.90, "Avg Precision": 0.70},
        {"Model": "M2", "Precision": 0.9, "Recall": 0.8, "F1-Score": 0.85, "ROC-AUC": 0.95, "Avg Precision": 0.82}
    ]
    df = build_comparison_dataframe(results)
    assert len(df) == 2
    assert df.iloc[0]["Model"] == "M2"  # Sorted by Avg Precision descending
