"""
Model architectures and factories for classical ML, Deep Learning, and Anomaly Detection.
"""

from typing import List, Tuple, Optional, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_logistic_regression(
    class_weight: str = 'balanced',
    max_iter: int = 1000,
    random_state: int = 42
) -> LogisticRegression:
    """Returns configured Logistic Regression baseline classifier."""
    return LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        solver='lbfgs'
    )


def get_random_forest(
    n_estimators: int = 200,
    min_samples_leaf: int = 2,
    class_weight: str = 'balanced',
    random_state: int = 42,
    n_jobs: int = -1
) -> RandomForestClassifier:
    """Returns configured Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )


def get_xgboost(
    scale_pos_weight: float = 1.0,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    n_jobs: int = -1
) -> XGBClassifier:
    """Returns configured XGBoost classifier."""
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric='logloss',
        n_jobs=n_jobs
    )


def build_ann(
    input_dim: int,
    dropout_rate: float = 0.30,
    learning_rate: float = 1e-3
) -> Any:
    """
    Builds feedforward neural network with Batch Normalization and Dropout regularisation.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model

    inputs = layers.Input(shape=(input_dim,), name='ann_input')
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(16, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(8, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name='ANN')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_lstm(
    n_timesteps: int,
    learning_rate: float = 1e-3
) -> Any:
    """
    Builds LSTM recurrent neural network for sequence/reshaped tabular comparisons.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model

    inputs = layers.Input(shape=(n_timesteps, 1), name='lstm_input')
    x = layers.LSTM(32, return_sequences=True)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(16)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(8, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name='LSTM')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_autoencoder(
    input_dim: int,
    bottleneck_dim: int = 8,
    l1_reg: float = 1e-5
) -> Tuple[Any, Any]:
    """
    Builds unsupervised deep autoencoder for anomaly detection.

    Returns:
        Tuple[Model, Model]: (full_autoencoder, encoder_model)
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, regularizers

    input_layer = layers.Input(shape=(input_dim,), name='ae_input')
    encoder = layers.Dense(16, activation='relu', activity_regularizer=regularizers.l1(l1_reg))(input_layer)
    bottleneck = layers.Dense(bottleneck_dim, activation='relu', name='bottleneck')(encoder)

    decoder = layers.Dense(16, activation='relu')(bottleneck)
    output_layer = layers.Dense(input_dim, activation='linear', name='reconstruction')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=output_layer, name='Autoencoder')
    autoencoder.compile(optimizer='adam', loss='mse')

    encoder_model = Model(inputs=input_layer, outputs=bottleneck, name='Encoder')
    return autoencoder, encoder_model


class HybridAutoencoderRF:
    """
    Hybrid model: extracts compressed latent representation using an unsupervised
    autoencoder, followed by Random Forest classification on the bottleneck features.
    """
    def __init__(self, encoder: Any, rf_classifier: Optional[RandomForestClassifier] = None):
        self.encoder = encoder
        self.rf = rf_classifier or RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        encoded_features = self.encoder.predict(X_train, verbose=0)
        self.rf.fit(encoded_features, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        encoded_features = self.encoder.predict(X, verbose=0)
        return self.rf.predict(encoded_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        encoded_features = self.encoder.predict(X, verbose=0)
        return self.rf.predict_proba(encoded_features)


def get_training_callbacks(monitor: str = 'val_loss', patience: int = 6, lr_patience: int = 3) -> List[Any]:
    """Standard training callbacks for Keras models."""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    return [
        EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=lr_patience, min_lr=1e-6, verbose=0)
    ]
