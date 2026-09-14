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
        shap_array = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
    elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
        shap_array = shap_vals[0, :, 1] if shap_vals.shape[2] > 1 else shap_vals[0, :, 0]
    elif hasattr(shap_vals, 'ndim') and shap_vals.ndim == 2:
        shap_array = shap_vals[0]
    else:
        shap_array = np.array(shap_vals).flatten()

    feature_impacts = pd.Series(shap_array, index=transaction_series.index)
    sorted_impacts = feature_impacts.sort_values(key=abs, ascending=False)

    top_features = sorted_impacts.head(top_n).to_dict()

    expected_val = explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)):
        base_val = float(expected_val[1]) if len(expected_val) > 1 else float(expected_val[0])
    else:
        base_val = float(expected_val)

    return {
        'base_value': base_val,
        'top_contributions': top_features,
        'risk_factors': {k: float(v) for k, v in top_features.items() if v > 0},
        'protective_factors': {k: float(v) for k, v in top_features.items() if v < 0}
    }
