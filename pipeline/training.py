"""Training module: train an ElasticNet model and log it with MLflow."""

import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import ElasticNet

from pipeline.config import (
    ALPHA,
    L1_RATIO,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = ALPHA,
    l1_ratio: float = L1_RATIO,
    random_state: int = RANDOM_STATE,
) -> ElasticNet:
    """Train an ElasticNet regression model.

    Args:
        X_train: Scaled training features.
        y_train: Training labels.
        alpha: Regularisation strength.
        l1_ratio: The ElasticNet mixing parameter.
        random_state: Random seed for reproducibility.

    Returns:
        Fitted ElasticNet model.
    """
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    model.fit(X_train, y_train)
    print(
        f"[training] Trained ElasticNet(alpha={alpha}, l1_ratio={l1_ratio})."
    )
    return model


def run_training(X_train: np.ndarray, y_train: np.ndarray) -> tuple[ElasticNet, str]:
    """Set up MLflow, train the model, and log parameters + model artifact.

    Args:
        X_train: Scaled training features.
        y_train: Training labels.

    Returns:
        Tuple of (fitted_model, mlflow_run_id).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_param("alpha", ALPHA)
        mlflow.log_param("l1_ratio", L1_RATIO)

        model = train_model(X_train, y_train)

        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id
        print(f"[training] MLflow run_id: {run_id}")

    return model, run_id
