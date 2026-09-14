"""
Unit tests for explainability and local attribution.
"""

import numpy as np
import pandas as pd
import pytest
from src.models import get_xgboost
from src.explainability import get_tree_explainer, generate_local_explanation


def test_generate_local_explanation():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 6), columns=[f"feat_{i}" for i in range(6)])
    y = np.random.choice([0, 1], size=50, p=[0.8, 0.2])

    model = get_xgboost(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)

    explainer = get_tree_explainer(model)
    explanation = generate_local_explanation(explainer, X.iloc[0], top_n=3)

    assert "base_value" in explanation
    assert "top_contributions" in explanation
    assert len(explanation["top_contributions"]) <= 3
    assert "risk_factors" in explanation
    assert "protective_factors" in explanation
