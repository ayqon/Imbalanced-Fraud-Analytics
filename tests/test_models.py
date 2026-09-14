"""
Unit tests for model instantiation, training, and inference.
"""

import numpy as np
import pytest
from src.models import (
    get_logistic_regression,
    get_random_forest,
    get_xgboost
)


@pytest.fixture
def sample_dataset():
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.random.choice([0, 1], size=200, p=[0.9, 0.1])
    return X, y


def test_logistic_regression(sample_dataset):
    X, y = sample_dataset
    model = get_logistic_regression(random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert preds.shape == (200,)
    assert probs.shape == (200, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_random_forest(sample_dataset):
    X, y = sample_dataset
    model = get_random_forest(n_estimators=10, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert preds.shape == (200,)
    assert probs.shape == (200, 2)


def test_xgboost(sample_dataset):
    X, y = sample_dataset
    model = get_xgboost(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert preds.shape == (200,)
    assert probs.shape == (200, 2)
