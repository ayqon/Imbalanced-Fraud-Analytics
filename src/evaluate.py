"""
Evaluation metrics, threshold optimization, confusion matrix visualizers, and cost analysis.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)


def find_optimal_threshold(
    y_true_val: np.ndarray,
    y_proba_val: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Finds optimal classification threshold strictly on the validation partition.
    Prevents test-set contamination.

    Parameters:
        y_true_val (np.ndarray): Validation labels
        y_proba_val (np.ndarray): Predicted validation probabilities or anomaly scores
        metric (str): Optimization target ('f1')

    Returns:
        Tuple[float, float]: (optimal_threshold, best_metric_score)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true_val, y_proba_val)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    return best_threshold, best_f1


def evaluate_model_performance(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculates comprehensive metrics for imbalanced binary classification.

    Returns dictionary with metrics and raw predictions.
    """
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
    ap = average_precision_score(y_true, y_proba) if y_proba is not None else np.nan

    return {
        'Model': name,
        'Precision': round(float(prec), 4),
        'Recall': round(float(rec), 4),
        'F1-Score': round(float(f1), 4),
        'ROC-AUC': round(float(auc), 4),
        'Avg Precision': round(float(ap), 4),
        'y_pred': y_pred,
        'y_proba': y_proba
    }


def compute_financial_cost_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_false_negative: float = 100.0,
    cost_false_positive: float = 2.0
) -> Dict[str, Any]:
    """
    Calculates estimated financial business cost based on confusion matrix.
    - False Negative (missed fraud): High direct financial loss.
    - False Positive (blocked legit): Customer friction / verification cost.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    total_cost = (fn * cost_false_negative) + (fp * cost_false_positive)
    return {
        'True Positives': int(tp),
        'False Positives': int(fp),
        'True Negatives': int(tn),
        'False Negatives': int(fn),
        'Total Cost (EUR)': float(total_cost),
        'Cost Breakdown': {
            'Fraud Losses (FN)': float(fn * cost_false_negative),
            'Friction Cost (FP)': float(fp * cost_false_positive)
        }
    }


def build_comparison_dataframe(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generates a clean comparison table sorted by Average Precision."""
    records = [{
        'Model': r['Model'],
        'Precision': r['Precision'],
        'Recall': r['Recall'],
        'F1-Score': r['F1-Score'],
        'ROC-AUC': r['ROC-AUC'],
        'Avg Precision': r['Avg Precision']
    } for r in results_list]

    df = pd.DataFrame(records)
    return df.sort_values('Avg Precision', ascending=False).reset_index(drop=True)
