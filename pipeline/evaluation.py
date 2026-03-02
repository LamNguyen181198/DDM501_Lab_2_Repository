"""Evaluation module: compute regression metrics and log them with MLflow."""

import numpy as np
import mlflow
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipeline.config import MLFLOW_TRACKING_URI


def compute_metrics(model: ElasticNet, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute MAE, RMSE, and R² for a fitted model.

    Args:
        model: A fitted scikit-learn estimator.
        X_test: Scaled test features.
        y_test: True test labels.

    Returns:
        Dictionary with keys 'mae', 'rmse', and 'r2'.
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_model(model: ElasticNet, X_test: np.ndarray, y_test: np.ndarray, run_id: str) -> dict:
    """Evaluate the model and log metrics to the active MLflow run.

    Args:
        model: A fitted scikit-learn estimator.
        X_test: Scaled test features.
        y_test: True test labels.
        run_id: MLflow run ID to log metrics against.

    Returns:
        Dictionary of evaluation metrics.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    metrics = compute_metrics(model, X_test, y_test)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)

    print(
        f"[evaluation] MAE={metrics['mae']:.4f}, "
        f"RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}"
    )
    return metrics
