"""
Data ingestion, partitioning, feature scaling, and class imbalance handling.
"""

from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads credit card transaction dataset from CSV file.

    Parameters:
        file_path (str): Path to creditcard.csv

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    df = pd.read_csv(file_path)
    if 'Class' not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")
    return df


def split_and_scale_data(
    df: pd.DataFrame,
    target_col: str = 'Class',
    test_size: float = 0.10,
    val_size: float = 0.10,
    random_state: int = 42
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
    StandardScaler
]:
    """
    Partitions dataset into Train, Validation, and Test sets with stratification.
    StandardScaler is fitted strictly on the training set to prevent data leakage.

    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test, scaler
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()

    temp_size = test_size + val_size
    val_relative_ratio = val_size / temp_size

    # First split: Train vs (Val + Test)
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_size,
        random_state=random_state,
        stratify=y
    )

    # Second split: Val vs Test
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp,
        test_size=(1.0 - val_relative_ratio),
        random_state=random_state,
        stratify=y_temp
    )

    # Standardize features strictly using training statistics
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=feature_names)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_raw), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=feature_names)

    return (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
        y_test.reset_index(drop=True),
        scaler
    )


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    k_neighbors: int = 5
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies SMOTE oversampling strictly to the training set.

    Parameters:
        X_train (pd.DataFrame): Scaled training features
        y_train (pd.Series): Training labels
        random_state (int): Random seed
        k_neighbors (int): Nearest neighbors for interpolation

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Resampled features and labels
    """
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)


def compute_imbalance_weights(y_train: pd.Series, mode: str = 'sqrt') -> Dict[str, Any]:
    """
    Computes class weight ratios for algorithmic and deep learning cost adjustment.

    Parameters:
        y_train (pd.Series): Binary labels
        mode (str): 'full' for exact ratio (negative/positive) or 'sqrt' for dampened ratio

    Returns:
        Dict[str, Any]: Contains 'ratio', 'keras_weights', and 'scale_pos_weight'
    """
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())

    if pos_count == 0:
        raise ValueError("Positive class has 0 occurrences.")

    ratio = neg_count / pos_count
    sqrt_ratio = np.sqrt(ratio)

    weight_val = sqrt_ratio if mode == 'sqrt' else ratio
    keras_weights = {0: 1.0, 1: float(weight_val)}

    return {
        'negative_count': neg_count,
        'positive_count': pos_count,
        'imbalance_ratio': ratio,
        'scale_pos_weight': ratio,
        'keras_weights': keras_weights
    }
