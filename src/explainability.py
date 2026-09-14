"""
Explainability and regulatory compliance utilities using SHAP (Shapley Additive Explanations).
Alings with GDPR Article 22 explanation requirements.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def get_tree_explainer(model: Any) -> shap.TreeExplainer:
    """Initializes TreeExplainer for tree-based ensemble models (XGBoost, Random Forest)."""
    return shap.TreeExplainer(model)


def compute_sample_shap_values(
    explainer: shap.TreeExplainer,
    X_sample: pd.DataFrame
) -> np.ndarray:
    """Computes SHAP values on a provided DataFrame sample."""
    return explainer.shap_values(X_sample)


def generate_local_explanation(
    explainer: shap.TreeExplainer,
    transaction_series: pd.Series,
    top_n: int = 8
) -> Dict[str, Any]:
    """
    Generates structured local attribution explanation for an individual transaction.

    Returns:
        Dict[str, Any]: Top positive (risk increasing) and negative (risk decreasing) feature impacts.
    """
    sample_df = pd.DataFrame([transaction_series])
    shap_vals = explainer.shap_values(sample_df)

    if isinstance(shap_vals, list):
        # In case of binary classification returning [class 0, class 1]
        shap_array = shap_vals[1][0]
    else:
        shap_array = shap_vals[0]

    feature_impacts = pd.Series(shap_array, index=transaction_series.index)
    sorted_impacts = feature_impacts.sort_values(key=abs, ascending=False)

    top_features = sorted_impacts.head(top_n).to_dict()

    return {
        'base_value': float(explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1]),
        'top_contributions': top_features,
        'risk_factors': {k: float(v) for k, v in top_features.items() if v > 0},
        'protective_factors': {k: float(v) for k, v in top_features.items() if v < 0}
    }
