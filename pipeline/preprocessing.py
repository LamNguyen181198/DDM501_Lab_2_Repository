"""Preprocessing module: scale features for model training."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pipeline.config import SCALER


def get_scaler(scaler_type: str = SCALER):
    """Return a scaler instance based on the configuration.

    Args:
        scaler_type: Type of scaler – "standard" or "minmax".

    Returns:
        A scikit-learn scaler instance.

    Raises:
        ValueError: If an unsupported scaler type is provided.
    """
    if scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler type: '{scaler_type}'")


def preprocess(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: str = SCALER,
) -> tuple[np.ndarray, np.ndarray, StandardScaler | MinMaxScaler]:
    """Fit a scaler on training data and transform both train and test sets.

    Args:
        X_train: Training feature DataFrame.
        X_test: Test feature DataFrame.
        scaler_type: Type of scaler to apply.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    scaler = get_scaler(scaler_type)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"[preprocessing] Applied '{scaler_type}' scaler.")
    return X_train_scaled, X_test_scaled, scaler
